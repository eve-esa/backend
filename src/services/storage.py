"""S3-compatible object storage service (AWS S3 / local MinIO).

The backend is S3-ready but environment-independent: leave ``S3_ENDPOINT_URL``
empty to talk to real AWS S3, or point it at ``http://minio:9000`` to use the
local MinIO container. No code changes are needed to switch between the two.
"""

import logging
import mimetypes
import uuid
from datetime import datetime
from typing import AsyncIterator, Optional

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from fastapi.concurrency import run_in_threadpool

from src.config import (
    S3_ACCESS_KEY_ID,
    S3_BUCKET_NAME,
    S3_ENDPOINT_URL,
    S3_PRESIGN_TTL_SECONDS,
    S3_REGION,
    S3_SECRET_ACCESS_KEY,
)

logger = logging.getLogger(__name__)

# Chunk size used when proxy-streaming an object back to the client.
STREAM_CHUNK_SIZE = 64 * 1024

# Error codes S3 and MinIO use for "the key isn't there". S3 answers GetObject
# with NoSuchKey, but a bucket the credentials can't enumerate answers 403-as-404
# (AccessDenied is deliberately not treated as missing), and HeadObject-style
# paths surface a bare 404/NotFound instead.
_MISSING_OBJECT_ERROR_CODES = {"NoSuchKey", "NoSuchBucket", "NotFound", "404"}


class ObjectNotFoundError(Exception):
    """Raised when an object key is absent from the bucket.

    Metadata lives in Mongo while the bytes live in the bucket, so the two can
    drift (a lifecycle rule, a manual cleanup, a failed upload that still wrote
    its document). Callers translate this into a 404 rather than letting a
    botocore ClientError escape as an unhandled 500.
    """


def sniff_image_type(header: bytes) -> Optional[str]:
    """Detect the image type from magic bytes, never trusting the Content-Type.

    Args:
        header (bytes): The leading bytes of the uploaded file.

    Returns:
        The image subtype ('png', 'jpeg', 'gif', 'webp') or None if unrecognized.
    """
    if header[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if header[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if header[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "webp"
    return None


# Maps every allowlistable artifact "type" key (the vocabulary used by
# ARTIFACT_UPLOAD_ALLOWED_TYPES) to its MIME content type.
# Image keys are handled by sniff_image_type; the rest are matched here.
ARTIFACT_TYPE_CONTENT_TYPES = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "pdf": "application/pdf",
    "csv": "text/csv",
    "txt": "text/plain",
    "json": "application/json",
    "geojson": "application/geo+json",
}

# Text-like artifact types that have no magic bytes to sniff. Detecting them
# safely requires BOTH a matching file extension AND a decodable, NUL-free
# UTF-8 payload (see sniff_artifact_type) so an uploader can't smuggle
# arbitrary binary content past the allowlist just by naming the file right.
_TEXT_EXTENSION_TYPES = {"csv", "txt", "json", "geojson"}


def _looks_like_text(data: bytes) -> bool:
    """True if `data` decodes as UTF-8 and contains no NUL bytes.

    NUL bytes are the cheapest binary tell (valid UTF-8 text never contains
    them); rejecting them catches most non-text payloads that would otherwise
    decode successfully (e.g. UTF-16 with mostly-ASCII content).
    """
    if b"\x00" in data:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def sniff_artifact_type(
    header: bytes, filename: Optional[str], full_data: bytes
) -> Optional[str]:
    """Detect an artifact's type key, generalizing sniff_image_type.

    Tries, in order:
      1. Image magic bytes (png/jpeg/gif/webp) via sniff_image_type.
      2. The PDF magic prefix ('%PDF-').
      3. Text-like types (csv, txt, json, geojson): these have no magic bytes,
         so both a matching file extension AND a decodable, NUL-free UTF-8
         payload are required before trusting the extension.

    Args:
        header (bytes): The leading bytes of the uploaded file (for magic-byte checks).
        filename (str | None): The client-supplied filename (for the text-type extension check).
        full_data (bytes): The full uploaded payload (for the UTF-8 decodability check).

    Returns:
        The detected type key (e.g. 'png', 'pdf', 'csv') -- look up
        ARTIFACT_TYPE_CONTENT_TYPES for the MIME content type -- or None if
        the file cannot be safely classified.
    """
    image_subtype = sniff_image_type(header)
    if image_subtype is not None:
        return image_subtype

    if header[:5] == b"%PDF-":
        return "pdf"

    ext = (filename or "").rsplit(".", 1)[-1].lower() if "." in (filename or "") else ""
    if ext in _TEXT_EXTENSION_TYPES and _looks_like_text(full_data):
        return ext

    return None


def _tag_set(tagging: str) -> list[dict[str, str]]:
    """Turn a PutObject ``Tagging`` query string into a TagSet.

    ``kind=tool-result`` becomes ``[{"Key": "kind", "Value": "tool-result"}]``.
    Empty parts are dropped. The object write does not depend on this parse.
    """
    from urllib.parse import unquote_plus

    tags: list[dict[str, str]] = []
    for part in tagging.split("&"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = unquote_plus(key).strip()
        if not key:
            continue
        tags.append({"Key": key, "Value": unquote_plus(value)})
    return tags


def guess_extension_from_content_type(content_type: Optional[str]) -> str:
    """Guess a filename extension (without the leading dot) from a MIME type.

    Falls back to "application/octet-stream" when no content type is given, and
    to ".bin" when the type is unrecognized (e.g. an unusual tool-declared type).
    Meant for artifact sources where the type isn't sniffed from magic bytes
    (e.g. MCP tool outputs), unlike the upload path which uses sniff_image_type.
    """
    ext = mimetypes.guess_extension(content_type or "application/octet-stream")
    return (ext or ".bin").lstrip(".")


class StorageService:
    """Thin async wrapper over a boto3 S3 client (compatible with MinIO)."""

    def __init__(self):
        """Initialize the storage service with a lazily built client."""
        self._s3 = None

    def _client(self):
        """Get or lazily create the boto3 S3 client (SigV4, MinIO endpoint override)."""
        if self._s3 is None:
            self._s3 = boto3.client(
                "s3",
                region_name=S3_REGION or None,
                endpoint_url=S3_ENDPOINT_URL or None,
                aws_access_key_id=S3_ACCESS_KEY_ID or None,
                aws_secret_access_key=S3_SECRET_ACCESS_KEY or None,
                config=BotoConfig(signature_version="s3v4"),
            )
        return self._s3

    @staticmethod
    def build_user_key(user_id: str, ext: str, prefix: Optional[str] = None) -> str:
        """Build the per-user object key: ``users/{user_id}/[{prefix}/]{uuid}.{ext}``.

        The per-user prefix scales naturally on S3 and matches the layout a
        future Cognito identity policy would authorize. ``prefix`` groups keys
        by artifact kind (e.g. ``"artifacts"``) without changing the per-user
        top-level layout.
        """
        clean_ext = ext.lstrip(".").lower()
        clean_prefix = prefix.strip("/") if prefix else None
        base = f"users/{user_id}/{clean_prefix}" if clean_prefix else f"users/{user_id}"
        return f"{base}/{uuid.uuid4().hex}.{clean_ext}"

    def put_object_sync(
        self,
        key: str,
        body: bytes,
        content_type: str,
        tagging: Optional[str] = None,
    ) -> None:
        """Store an object. ``tagging`` is an S3 ``Tagging`` query string.

        The bytes go up first, with no ``Tagging`` argument. Tags are a second
        call. A bucket policy that rejects ``PutObjectTagging`` must not fail
        the write: the warning is logged and the object stays.
        """
        self._client().put_object(
            Bucket=S3_BUCKET_NAME,
            Key=key,
            Body=body,
            ContentType=content_type,
        )
        if not tagging:
            return
        try:
            tag_set = _tag_set(tagging)
            if not tag_set:
                return
            self._client().put_object_tagging(
                Bucket=S3_BUCKET_NAME,
                Key=key,
                Tagging={"TagSet": tag_set},
            )
        except Exception:
            logger.warning("storage.tagging_failed key=%s", key, exc_info=True)

    async def put_object(
        self,
        key: str,
        body: bytes,
        content_type: str,
        tagging: Optional[str] = None,
    ) -> None:
        """Store an object in the bucket.

        Tagging uses the sync path: a rejected tag call logs a warning and
        this write still succeeds.
        """
        await run_in_threadpool(
            self.put_object_sync, key, body, content_type, tagging
        )

    def exists_sync(self, key: str) -> bool:
        """True when ``HeadObject`` finds ``key``.

        A missing key is false. Any other error propagates, so a denied head
        is not treated as "absent" and then overwritten.
        """
        try:
            self._client().head_object(Bucket=S3_BUCKET_NAME, Key=key)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in _MISSING_OBJECT_ERROR_CODES:
                return False
            raise
        return True

    async def exists(self, key: str) -> bool:
        """Async ``exists_sync``."""
        return await run_in_threadpool(self.exists_sync, key)

    def get_object_sync(self, key: str) -> dict:
        """Fetch an object. ``Body`` is a ``StreamingBody``.

        Raises:
            ObjectNotFoundError: if the key is absent from the bucket.
        """
        try:
            return self._client().get_object(Bucket=S3_BUCKET_NAME, Key=key)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in _MISSING_OBJECT_ERROR_CODES:
                raise ObjectNotFoundError(key) from exc
            raise

    async def get_object(self, key: str) -> dict:
        """Fetch an object; returns the raw boto3 response (Body is a StreamingBody).

        Raises:
            ObjectNotFoundError: if the key is absent from the bucket.
        """
        return await run_in_threadpool(self.get_object_sync, key)

    def list_key_records(
        self, prefix: str, max_keys: int, *, scan_limit: int = 1000
    ) -> tuple[list[dict], bool]:
        """Objects under ``prefix``, newest first, at most ``max_keys``.

        Each record is ``{"key", "size", "last_modified"}``. ListObjects
        order is lexicographic, so the scanned window is sorted by
        ``LastModified`` before the slice. A first page is not the newest
        set. The second value is true when the scan stopped early or more
        objects exist than ``max_keys``. Listing stops after ``scan_limit``
        keys so one prefix cannot page without bound.
        """
        if max_keys < 1:
            return [], False
        limit = max(scan_limit, max_keys)
        client = self._client()
        collected: list[dict] = []
        token = None
        stopped_early = False
        while len(collected) < limit:
            params = {
                "Bucket": S3_BUCKET_NAME,
                "Prefix": prefix,
                "MaxKeys": min(1000, limit - len(collected)),
            }
            if token:
                params["ContinuationToken"] = token
            page = client.list_objects_v2(**params)
            for item in page.get("Contents") or []:
                key = item.get("Key")
                if not key or key.endswith("/"):
                    continue
                collected.append(
                    {
                        "key": key,
                        "size": int(item.get("Size") or 0),
                        "last_modified": item.get("LastModified"),
                    }
                )
            if not page.get("IsTruncated"):
                break
            token = page.get("NextContinuationToken")
            if not token:
                stopped_early = True
                break
        else:
            stopped_early = True

        def _stamp(value: object) -> float:
            if isinstance(value, datetime):
                return value.timestamp()
            return 0.0

        collected.sort(key=lambda item: _stamp(item.get("last_modified")), reverse=True)
        truncated = stopped_early or len(collected) > max_keys
        return collected[:max_keys], truncated

    async def list_keys(self, prefix: str, max_keys: int) -> list[dict]:
        """Keys under ``prefix``, newest first, at most ``max_keys``.

        Each item is ``{"key", "size", "last_modified"}``.
        """
        records, _truncated = await run_in_threadpool(
            self.list_key_records, prefix, max_keys
        )
        return records

    async def stream_body(
        self, body, chunk_size: int = STREAM_CHUNK_SIZE
    ) -> AsyncIterator[bytes]:
        """Yield an object body in chunks, reading from S3 inside the threadpool.

        Args:
            body: A boto3 ``StreamingBody`` (from a ``get_object`` response).
            chunk_size (int): Bytes to read per chunk.
        """
        chunks = body.iter_chunks(chunk_size)
        try:
            while True:
                chunk = await run_in_threadpool(lambda: next(chunks, None))
                if chunk is None:
                    break
                yield chunk
        finally:
            await run_in_threadpool(body.close)

    async def delete_object(self, key: str) -> None:
        """Delete an object from the bucket."""
        await run_in_threadpool(
            lambda: self._client().delete_object(Bucket=S3_BUCKET_NAME, Key=key)
        )

    async def generate_presigned_get(
        self, key: str, expires_in: Optional[int] = None
    ) -> str:
        """Generate a short-TTL presigned GET URL (never persist it)."""
        ttl = expires_in or S3_PRESIGN_TTL_SECONDS
        return await run_in_threadpool(
            lambda: self._client().generate_presigned_url(
                "get_object",
                Params={"Bucket": S3_BUCKET_NAME, "Key": key},
                ExpiresIn=ttl,
            )
        )


# Module-level singleton (mirrors the `document_service` pattern; monkeypatchable in tests).
storage_service = StorageService()
