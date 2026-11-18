"""RESTful artifact endpoints: upload, owner-only serving, listing and deletion.

Artifacts generalize the former image-only storage to any file a user attaches
or an MCP tool produces (source.type "upload" vs "mcp_tool"). User uploads are
restricted to the sniffed/extension-checked allowlist below (images, pdf, and
a handful of text-like formats); MCP-produced artifacts are ingested by a
separate path (not this router) and are read-only here.
"""

import logging
import re
import unicodedata
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, HTTPException, Path, Query, UploadFile
from fastapi.responses import StreamingResponse
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from src import config
from src.config import (
    ARTIFACT_UPLOAD_ALLOWED_TYPES,
    ARTIFACT_UPLOAD_MAX_BYTES,
    ARTIFACT_UPLOADS_PER_DAY,
    S3_PRESIGN_TTL_SECONDS,
)
from src.database.models.artifact import Artifact, ArtifactSource
from src.database.models.user import User
from src.database.mongo import get_collection
from src.database.mongo_model import PaginatedResponse
from src.middlewares.auth import get_current_user
from src.schemas.common import Pagination
from src.services.artifact_cleanup import delete_artifact
from src.services.mcp.artifact_ingestion import (
    artifact_markdown_line,
    clean_filename,
    strip_format_chars,
)
from src.services.storage import (
    ARTIFACT_TYPE_CONTENT_TYPES,
    INLINE_CONTENT_TYPES,
    OPAQUE_CONTENT_TYPE,
    TOOL_INLINE_CONTENT_TYPES,
    ObjectNotFoundError,
    sniff_artifact_type,
    storage_service,
)
from src.services.request_rate_limiter import enforce_request_rate

router = APIRouter()
logger = logging.getLogger(__name__)

# Collection holding the atomic per-user-per-day upload counters.
QUOTA_COLLECTION = "image_upload_quota"

# Characters that never reach a filename in a header: controls (CR/LF split
# the header, the rest make uvicorn refuse it), quotes, backslashes and path
# separators.
_UNSAFE_FILENAME_CHARS = re.compile(r'[\x00-\x1f\x7f"\\/]+')


async def reserve_daily_quota_slot(user_id: str) -> None:
    """Atomically reserve one upload slot for the user's current UTC day.

    Uses an upserted per-user-per-day counter incremented in a single Mongo op,
    so parallel uploads cannot bypass the cap (fails closed under concurrency).
    The counter is monotonic: deletes never decrement it, because the quota limits
    uploads-per-day (a rate), not the number of currently stored objects. Counter
    docs are tiny; they simply accumulate and can be reaped with a TTL index on
    `day` if that ever becomes desirable.

    Two uploads that are the user's first of the day both upsert the same `_id`.
    MongoDB retries the loser server side, DocumentDB does not and raises a duplicate
    key error, so the upsert is retried once: the document now exists and the retry is
    a plain increment.

    Raises:
        HTTPException: 429 if the daily upload limit has been reached.
        DuplicateKeyError: if the upsert collides on both attempts.
    """
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    counter = get_collection(QUOTA_COLLECTION)
    for attempt in (1, 2):
        try:
            doc = await counter.find_one_and_update(
                {"_id": f"{user_id}:{day}"},
                {"$inc": {"count": 1}, "$setOnInsert": {"day": day, "user_id": user_id}},
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )
            break
        except DuplicateKeyError:
            if attempt == 2:
                raise
            logger.info("Daily upload quota upsert collided for user %s, retrying", user_id)
    if doc["count"] > ARTIFACT_UPLOADS_PER_DAY:
        raise HTTPException(status_code=429, detail="Daily upload limit reached")


async def get_owned_artifact(artifact_id: str, requesting_user: User) -> Artifact:
    """Fetch an artifact the requesting user owns.

    A missing id and another user's id both answer 404, so the answer never
    confirms that someone else's artifact exists.
    """
    artifact = await Artifact.find_by_id(artifact_id)
    if not artifact or artifact.user_id != requesting_user.id:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return artifact


def require_uploads_enabled() -> None:
    """404 while ``FEATURE_ARTIFACT_UPLOADS`` is off, as if the route did not exist.

    A dependency, not a conditional mount: GET /artifacts shares the path, so
    an unmounted POST would answer 405 and confirm the route exists. Listed
    first, so a closed route spends no rate limit token.
    """
    if not config.FEATURE_ARTIFACT_UPLOADS:
        raise HTTPException(status_code=404, detail="Not Found")


def serving_headers(artifact: Artifact) -> tuple[str, str]:
    """Media type and ``Content-Disposition`` an artifact is served with.

    Only the inline allowlist keeps its stored type: ``INLINE_CONTENT_TYPES``
    for a user's upload, ``TOOL_INLINE_CONTENT_TYPES`` (no PDF) for tool and
    sandbox output. Anything else, including rows stored as ``image/svg+xml``
    or ``text/html`` before the type came from the bytes, is served as
    ``application/octet-stream`` with ``attachment``, so neither a browser
    tab nor a blob URL renders it.
    """
    uploaded = getattr(artifact.source, "type", None) == "upload"
    inline = INLINE_CONTENT_TYPES if uploaded else TOOL_INLINE_CONTENT_TYPES
    if artifact.content_type in inline:
        return artifact.content_type, content_disposition("inline", artifact.filename)
    return OPAQUE_CONTENT_TYPE, content_disposition("attachment", artifact.filename)


def content_disposition(disposition: str, filename: Optional[str]) -> str:
    """``Content-Disposition`` value that any filename can travel in (RFC 6266).

    Header values are encoded as latin-1, so a raw CJK, emoji or U+202F name
    raised and the route answered 500. An ASCII name keeps the plain
    ``filename="..."`` form. Any other name gets an ASCII fallback (accents
    folded, the rest as ``_``) plus ``filename*=UTF-8''<percent-encoded>``.
    Control characters, quotes, backslashes and path separators become ``_``
    in both, so the header cannot be split or refused, nor the name used as
    a path. Unicode format characters (bidi overrides, zero width) go, so a
    saved file cannot show a reversed extension.
    """
    name = (
        _UNSAFE_FILENAME_CHARS.sub("_", strip_format_chars(filename or "")).strip()
        or "artifact"
    )
    if name.isascii():
        return f'{disposition}; filename="{name}"'
    fallback = "".join(
        char if char.isascii() and char.isprintable() else "_"
        for char in unicodedata.normalize("NFKD", name)
        if not unicodedata.combining(char)
    )
    return (
        f'{disposition}; filename="{fallback}"; '
        f"filename*=UTF-8''{quote(name, safe='')}"
    )


@router.post(
    "/artifacts",
    dependencies=[
        Depends(require_uploads_enabled),
        Depends(enforce_request_rate("upload")),
    ],
)
async def upload_artifact(
    file: UploadFile = File(...),
    requesting_user: User = Depends(get_current_user),
) -> dict:
    """
    Upload an artifact, validating quota, size and type before storing it.

    The Content-Type declared by the client is never trusted: images are
    sniffed from magic bytes, pdf from its magic prefix, and text-like types
    (csv, txt, json, geojson) require both a matching file extension and a
    decodable UTF-8 payload. The stored object lives under the per-user
    prefix ``users/{user_id}/artifacts/`` and is served through the stable
    ``/artifacts/{id}`` route.

    Args:
        file (UploadFile): The file to upload.
        requesting_user (User): Authenticated user injected by dependency.

    Returns:
        Upload result with the stable url and ready-to-embed markdown.

    Raises:
        HTTPException: 429 if the daily quota is exceeded; 413 if the file is too
        large; 415 if the file is not an allowed/recognized type.
    """
    # Read with a hard cap so oversize payloads never reach storage.
    data = await file.read(ARTIFACT_UPLOAD_MAX_BYTES + 1)
    if len(data) > ARTIFACT_UPLOAD_MAX_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds maximum size")

    type_key = sniff_artifact_type(data[:16], file.filename, data)
    if type_key is None or type_key not in ARTIFACT_UPLOAD_ALLOWED_TYPES:
        raise HTTPException(
            status_code=415, detail="Unsupported or unrecognized file type"
        )

    # Reserve a quota slot atomically once the payload is known-valid, so malformed
    # uploads don't burn quota and concurrent valid uploads can't exceed the cap.
    await reserve_daily_quota_slot(requesting_user.id)

    content_type = ARTIFACT_TYPE_CONTENT_TYPES[type_key]
    key = storage_service.build_user_key(
        requesting_user.id, type_key, prefix="artifacts"
    )
    await storage_service.put_object(key, data, content_type)

    artifact = await Artifact.create(
        user_id=requesting_user.id,
        key=key,
        # Cleaned and capped once here, so the list, the header and the
        # markdown all carry the same safe name.
        filename=clean_filename(file.filename) if file.filename else f"artifact.{type_key}",
        content_type=content_type,
        size_bytes=len(data),
        source=ArtifactSource(type="upload"),
    )

    url = f"/artifacts/{artifact.id}"
    return {
        "id": artifact.id,
        "url": url,
        "markdown": artifact_markdown_line(
            artifact_id=artifact.id,
            filename=artifact.filename,
            content_type=artifact.content_type,
            provenance=False,
        ),
        "filename": artifact.filename,
        "content_type": artifact.content_type,
        "size_bytes": artifact.size_bytes,
    }


@router.get("/artifacts", response_model=PaginatedResponse[Artifact])
async def list_artifacts(
    conversation_id: Optional[str] = Query(
        None, description="Filter to artifacts attached to this conversation"
    ),
    pagination: Pagination = Depends(),
    requesting_user: User = Depends(get_current_user),
) -> PaginatedResponse[Artifact]:
    """
    List the current user's artifacts, most recent first.

    Args:
        conversation_id (str | None): Optional filter to a single conversation.
        pagination (Pagination): Pagination parameters.
        requesting_user (User): Authenticated user injected by dependency.

    Returns:
        Paginated artifacts owned by the user.
    """
    filter_dict = {"user_id": requesting_user.id}
    if conversation_id:
        filter_dict["conversation_id"] = conversation_id

    return await Artifact.find_all_with_pagination(
        filter_dict=filter_dict,
        limit=pagination.limit,
        page=pagination.page,
        sort=[("timestamp", -1)],
    )


@router.get("/artifacts/{artifact_id}/url")
async def get_artifact_presigned_url(
    artifact_id: str = Path(..., description="Artifact ID"),
    requesting_user: User = Depends(get_current_user),
) -> dict:
    """
    Return a short-TTL presigned GET URL for an owned artifact.

    The presigned URL is short-lived and must never be persisted (e.g. in message
    markdown): use the stable ``/artifacts/{id}`` route for durable references.

    Args:
        artifact_id (str): Artifact identifier.
        requesting_user (User): Authenticated user injected by dependency.

    The type and disposition are signed into the URL, so storage serves the
    same headers as ``GET /artifacts/{id}`` whatever type the object carries.

    Returns:
        The presigned URL and its TTL in seconds.

    Raises:
        HTTPException: 404 if missing or owned by another user.
    """
    artifact = await get_owned_artifact(artifact_id, requesting_user)
    media_type, disposition = serving_headers(artifact)
    url = await storage_service.generate_presigned_get(
        artifact.key,
        response_content_type=media_type,
        response_content_disposition=disposition,
    )
    return {"url": url, "expires_in": S3_PRESIGN_TTL_SECONDS}


@router.get("/artifacts/{artifact_id}")
async def get_artifact(
    artifact_id: str = Path(..., description="Artifact ID"),
    requesting_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """
    Stream an owned artifact's bytes, proxying them from object storage.

    Serving is authorized on every render via the JWT, so the stable relative URL
    persisted in markdown/attachments never expires and is environment-independent.

    Args:
        artifact_id (str): Artifact identifier.
        requesting_user (User): Authenticated user injected by dependency.

    Returns:
        Streaming response with the served media type and caching headers.

    Raises:
        HTTPException: 404 if the record is missing, owned by another user, or
        its bytes are gone from storage.
    """
    artifact = await get_owned_artifact(artifact_id, requesting_user)
    try:
        response = await storage_service.get_object(artifact.key)
    except ObjectNotFoundError:
        # The record outlived its bytes. Report it as missing rather than as a
        # server fault, and log the key so the orphan can be reclaimed.
        logger.warning(
            f"Artifact {artifact_id} has no object at key {artifact.key}"
        )
        raise HTTPException(
            status_code=404, detail="Artifact content is no longer available"
        )

    media_type, disposition = serving_headers(artifact)

    headers = {
        # Per-user authorized payload: force revalidation so a shared browser
        # cache re-checks ownership, and key the cache on the token so one
        # user's cached bytes can't be replayed to another (cross-account bleed).
        "Cache-Control": "private, no-cache",
        "Vary": "Authorization",
        # Never let the browser MIME-sniff a mismatched/hostile type.
        "X-Content-Type-Options": "nosniff",
        "Content-Disposition": disposition,
    }
    etag = response.get("ETag")
    if etag:
        headers["ETag"] = etag
    content_length = response.get("ContentLength", artifact.size_bytes)
    if content_length is not None:
        headers["Content-Length"] = str(content_length)

    return StreamingResponse(
        storage_service.stream_body(response["Body"]),
        media_type=media_type,
        headers=headers,
    )


@router.delete("/artifacts/{artifact_id}")
async def delete_owned_artifact(
    artifact_id: str = Path(..., description="Artifact ID"),
    requesting_user: User = Depends(get_current_user),
) -> dict:
    """
    Delete an owned artifact: its object, its record and its message references.

    References are both MCP outputs (``messages.artifact_ids``) and attached
    uploads (``messages.attachments[].image_id``). The object goes first.
    When storage refuses, the record stays and the answer is 503, so the user
    can retry and nothing is orphaned. With versioning on, S3 keeps the bytes
    as a noncurrent version until the lifecycle rule expires it.

    Args:
        artifact_id (str): Artifact identifier.
        requesting_user (User): Authenticated user injected by dependency.

    Returns:
        Confirmation message.

    Raises:
        HTTPException: 404 if missing or owned by another user; 503 with
        ``detail.code`` ``artifact_delete_failed`` if storage refused.
    """
    # Primary: a delete right after an upload or an attach must find the row.
    # Another user's id answers 404 like a missing one.
    artifact = await Artifact.find_by_id_on_primary(artifact_id)
    if not artifact or artifact.user_id != requesting_user.id:
        raise HTTPException(status_code=404, detail="Artifact not found")
    try:
        await delete_artifact(artifact)
    except Exception as exc:  # noqa: BLE001 - mapped to a retryable 503
        logger.warning(
            "artifact.delete_failed user=%s artifact=%s error=%s",
            requesting_user.id,
            artifact.id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=503,
            detail={
                "code": "artifact_delete_failed",
                "message": "The file could not be deleted. Try again later.",
            },
        )
    return {"message": "Artifact deleted successfully"}
