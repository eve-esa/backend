"""S3-compatible object storage service (AWS S3 / local MinIO).

The backend is S3-ready but environment-independent: leave ``S3_ENDPOINT_URL``
empty to talk to real AWS S3, or point it at ``http://minio:9000`` to use the
local MinIO container. No code changes are needed to switch between the two.
"""

import logging
import uuid
from typing import AsyncIterator, Optional

import boto3
from botocore.config import Config as BotoConfig
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
    def build_user_key(user_id: str, ext: str) -> str:
        """Build the per-user object key: ``users/{user_id}/{uuid}.{ext}``.

        The per-user prefix scales naturally on S3 and matches the layout a
        future Cognito identity policy would authorize.
        """
        return f"users/{user_id}/{uuid.uuid4().hex}.{ext.lstrip('.').lower()}"

    async def put_object(self, key: str, body: bytes, content_type: str) -> None:
        """Store an object in the bucket."""
        await run_in_threadpool(
            lambda: self._client().put_object(
                Bucket=S3_BUCKET_NAME,
                Key=key,
                Body=body,
                ContentType=content_type,
            )
        )

    async def get_object(self, key: str) -> dict:
        """Fetch an object; returns the raw boto3 response (Body is a StreamingBody)."""
        return await run_in_threadpool(
            lambda: self._client().get_object(Bucket=S3_BUCKET_NAME, Key=key)
        )

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

    async def generate_presigned_put(
        self,
        key: str,
        content_type: Optional[str] = None,
        expires_in: Optional[int] = None,
    ) -> str:
        """Generate a short-TTL presigned PUT URL (never persist it)."""
        ttl = expires_in or S3_PRESIGN_TTL_SECONDS
        params = {"Bucket": S3_BUCKET_NAME, "Key": key}
        if content_type:
            params["ContentType"] = content_type
        return await run_in_threadpool(
            lambda: self._client().generate_presigned_url(
                "put_object",
                Params=params,
                ExpiresIn=ttl,
            )
        )


# Module-level singleton (mirrors the `document_service` pattern; monkeypatchable in tests).
storage_service = StorageService()
