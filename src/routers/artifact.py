"""RESTful artifact endpoints: upload, owner-only serving, listing and deletion.

Artifacts generalize the former image-only storage to any file a user attaches
or an MCP tool produces (source.type "upload" vs "mcp_tool"). User uploads are
restricted to the sniffed/extension-checked allowlist below (images, pdf, and
a handful of text-like formats); MCP-produced artifacts are ingested by a
separate path (not this router) and are read-only here.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Path, Query, UploadFile
from fastapi.responses import StreamingResponse
from pymongo import ReturnDocument

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
from src.services.storage import (
    ARTIFACT_TYPE_CONTENT_TYPES,
    ObjectNotFoundError,
    sniff_artifact_type,
    storage_service,
)

router = APIRouter()
logger = logging.getLogger(__name__)

# Collection holding the atomic per-user-per-day upload counters.
QUOTA_COLLECTION = "image_upload_quota"

# Content types allowed to be served with `Content-Disposition: inline`. Anything
# else (html, svg, unknown) is forced to `attachment` so a browser never
# executes a stored artifact as a page (stored-XSS defense).
INLINE_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
    "application/pdf",
    "text/plain",
    "application/json",
}


async def reserve_daily_quota_slot(user_id: str) -> None:
    """Atomically reserve one upload slot for the user's current UTC day.

    Uses an upserted per-user-per-day counter incremented in a single Mongo op,
    so parallel uploads cannot bypass the cap (fails closed under concurrency).
    The counter is monotonic: deletes never decrement it, because the quota limits
    uploads-per-day (a rate), not the number of currently stored objects. Counter
    docs are tiny; they simply accumulate and can be reaped with a TTL index on
    `day` if that ever becomes desirable.

    Raises:
        HTTPException: 429 if the daily upload limit has been reached.
    """
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    counter = get_collection(QUOTA_COLLECTION)
    doc = await counter.find_one_and_update(
        {"_id": f"{user_id}:{day}"},
        {"$inc": {"count": 1}, "$setOnInsert": {"day": day, "user_id": user_id}},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    if doc["count"] > ARTIFACT_UPLOADS_PER_DAY:
        raise HTTPException(status_code=429, detail="Daily upload limit reached")


async def get_owned_artifact(artifact_id: str, requesting_user: User) -> Artifact:
    """Fetch an artifact and validate that the requesting user owns it."""
    artifact = await Artifact.find_by_id(artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")

    if artifact.user_id != requesting_user.id:
        raise HTTPException(
            status_code=403, detail="You are not allowed to access this artifact"
        )

    return artifact


@router.post("/artifacts")
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
        filename=file.filename or f"artifact.{type_key}",
        content_type=content_type,
        size_bytes=len(data),
        source=ArtifactSource(type="upload"),
    )

    url = f"/artifacts/{artifact.id}"
    return {
        "id": artifact.id,
        "url": url,
        "markdown": f"![{artifact.filename}]({url})",
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

    Returns:
        The presigned URL and its TTL in seconds.

    Raises:
        HTTPException: 404 if not found; 403 if access is forbidden.
    """
    artifact = await get_owned_artifact(artifact_id, requesting_user)
    url = await storage_service.generate_presigned_get(artifact.key)
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
        Streaming response with the stored media type and caching headers.

    Raises:
        HTTPException: 404 if the record is missing or its bytes are gone from
        storage; 403 if access is forbidden.
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

    # Sanitize the filename for the header (display-only): drop CR/LF, quotes,
    # backslashes and path separators to prevent header injection.
    safe_filename = re.sub(r'[\r\n"\\/]+', "_", artifact.filename or "artifact")

    # Only a known-safe set of types render inline; everything else (html, svg,
    # unrecognized) is forced to download so a stored artifact is never executed
    # as a page by the browser.
    disposition = (
        "inline" if artifact.content_type in INLINE_CONTENT_TYPES else "attachment"
    )

    headers = {
        # Per-user authorized payload: force revalidation so a shared browser
        # cache re-checks ownership, and key the cache on the token so one
        # user's cached bytes can't be replayed to another (cross-account bleed).
        "Cache-Control": "private, no-cache",
        "Vary": "Authorization",
        # Never let the browser MIME-sniff a mismatched/hostile type.
        "X-Content-Type-Options": "nosniff",
        "Content-Disposition": f'{disposition}; filename="{safe_filename}"',
    }
    etag = response.get("ETag")
    if etag:
        headers["ETag"] = etag
    content_length = response.get("ContentLength", artifact.size_bytes)
    if content_length is not None:
        headers["Content-Length"] = str(content_length)

    return StreamingResponse(
        storage_service.stream_body(response["Body"]),
        media_type=artifact.content_type,
        headers=headers,
    )


@router.delete("/artifacts/{artifact_id}")
async def delete_artifact(
    artifact_id: str = Path(..., description="Artifact ID"),
    requesting_user: User = Depends(get_current_user),
) -> dict:
    """
    Delete an owned, user-uploaded artifact from object storage and the database.

    MCP-produced artifacts (source.type == "mcp_tool") cannot be deleted through
    this endpoint: they belong to the tool call trace, not the user's uploads.

    Args:
        artifact_id (str): Artifact identifier.
        requesting_user (User): Authenticated user injected by dependency.

    Returns:
        Confirmation message.

    Raises:
        HTTPException: 404 if not found; 403 if deletion is forbidden (including
        for MCP-generated artifacts); 500 if the object store delete fails (the
        DB record is kept so a retry can reclaim it).
    """
    artifact = await get_owned_artifact(artifact_id, requesting_user)

    if artifact.source.type != "upload":
        raise HTTPException(
            status_code=403, detail="MCP-generated artifacts cannot be deleted"
        )

    # Fail closed: only drop the DB record once the object is gone from storage.
    # Deleting the doc on a storage failure would orphan the bytes (no record to
    # ever reclaim them), so surface the error and let the client retry.
    try:
        await storage_service.delete_object(artifact.key)
    except Exception as e:  # noqa: BLE001 - surfaced below; never orphan the object
        logger.error(f"Failed to delete artifact object {artifact.key}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete artifact object")

    await artifact.delete()
    return {"message": "Artifact deleted successfully"}
