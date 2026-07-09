"""RESTful image endpoints: upload, owner-only serving, listing and deletion."""

import logging
import re
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, HTTPException, Path, UploadFile
from fastapi.responses import StreamingResponse
from pymongo import ReturnDocument

from src.config import (
    IMAGE_ALLOWED_TYPES,
    IMAGE_MAX_BYTES,
    IMAGE_UPLOADS_PER_DAY,
    S3_PRESIGN_TTL_SECONDS,
)
from src.database.models.image import Image
from src.database.models.user import User
from src.database.mongo import get_collection
from src.database.mongo_model import PaginatedResponse
from src.middlewares.auth import get_current_user
from src.schemas.common import Pagination
from src.services.storage import sniff_image_type, storage_service

router = APIRouter()
logger = logging.getLogger(__name__)

# Collection holding the atomic per-user-per-day upload counters.
QUOTA_COLLECTION = "image_upload_quota"


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
    if doc["count"] > IMAGE_UPLOADS_PER_DAY:
        raise HTTPException(status_code=429, detail="Daily image upload limit reached")


async def get_owned_image(image_id: str, requesting_user: User) -> Image:
    """Fetch an image and validate that the requesting user owns it."""
    image = await Image.find_by_id(image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    if image.user_id != requesting_user.id:
        raise HTTPException(
            status_code=403, detail="You are not allowed to access this image"
        )

    return image


@router.post("/images")
async def upload_image(
    file: UploadFile = File(...),
    requesting_user: User = Depends(get_current_user),
) -> dict:
    """
    Upload an image, validating quota, size and magic bytes before storing it.

    The Content-Type declared by the client is never trusted: the image type is
    sniffed from the file's magic bytes. The stored object lives under the
    per-user prefix ``users/{user_id}/`` and is served through the stable
    ``/images/{id}`` route.

    Args:
        file (UploadFile): The image to upload.
        requesting_user (User): Authenticated user injected by dependency.

    Returns:
        Upload result with the stable url and ready-to-embed markdown.

    Raises:
        HTTPException: 429 if the daily quota is exceeded; 413 if the file is too
        large; 415 if the file is not an allowed image type.
    """
    # Read with a hard cap so oversize payloads never reach storage.
    data = await file.read(IMAGE_MAX_BYTES + 1)
    if len(data) > IMAGE_MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds maximum size")

    subtype = sniff_image_type(data[:16])
    if subtype is None or subtype not in IMAGE_ALLOWED_TYPES:
        raise HTTPException(
            status_code=415, detail="Unsupported or unrecognized image type"
        )

    # Reserve a quota slot atomically once the payload is known-valid, so malformed
    # uploads don't burn quota and concurrent valid uploads can't exceed the cap.
    await reserve_daily_quota_slot(requesting_user.id)

    content_type = f"image/{subtype}"
    key = storage_service.build_user_key(requesting_user.id, subtype)
    await storage_service.put_object(key, data, content_type)

    image = await Image.create(
        user_id=requesting_user.id,
        key=key,
        filename=file.filename or f"image.{subtype}",
        content_type=content_type,
        size_bytes=len(data),
    )

    url = f"/images/{image.id}"
    return {
        "id": image.id,
        "url": url,
        "markdown": f"![{image.filename}]({url})",
        "filename": image.filename,
        "content_type": image.content_type,
        "size_bytes": image.size_bytes,
    }


@router.get("/images", response_model=PaginatedResponse[Image])
async def list_images(
    pagination: Pagination = Depends(),
    requesting_user: User = Depends(get_current_user),
) -> PaginatedResponse[Image]:
    """
    List the current user's images, most recent first.

    Args:
        pagination (Pagination): Pagination parameters.
        requesting_user (User): Authenticated user injected by dependency.

    Returns:
        Paginated images owned by the user.
    """
    return await Image.find_all_with_pagination(
        filter_dict={"user_id": requesting_user.id},
        limit=pagination.limit,
        page=pagination.page,
        sort=[("timestamp", -1)],
    )


@router.get("/images/{image_id}/url")
async def get_image_presigned_url(
    image_id: str = Path(..., description="Image ID"),
    requesting_user: User = Depends(get_current_user),
) -> dict:
    """
    Return a short-TTL presigned GET URL for an owned image.

    The presigned URL is short-lived and must never be persisted (e.g. in message
    markdown): use the stable ``/images/{id}`` route for durable references.

    Args:
        image_id (str): Image identifier.
        requesting_user (User): Authenticated user injected by dependency.

    Returns:
        The presigned URL and its TTL in seconds.

    Raises:
        HTTPException: 404 if not found; 403 if access is forbidden.
    """
    image = await get_owned_image(image_id, requesting_user)
    url = await storage_service.generate_presigned_get(image.key)
    return {"url": url, "expires_in": S3_PRESIGN_TTL_SECONDS}


@router.get("/images/{image_id}")
async def get_image(
    image_id: str = Path(..., description="Image ID"),
    requesting_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """
    Stream an owned image's bytes, proxying them from object storage.

    Serving is authorized on every render via the JWT, so the stable relative URL
    persisted in markdown/attachments never expires and is environment-independent.

    Args:
        image_id (str): Image identifier.
        requesting_user (User): Authenticated user injected by dependency.

    Returns:
        Streaming response with the sniffed media type and caching headers.

    Raises:
        HTTPException: 404 if not found; 403 if access is forbidden.
    """
    image = await get_owned_image(image_id, requesting_user)
    response = await storage_service.get_object(image.key)

    # Sanitize the filename for the header (display-only): drop CR/LF, quotes,
    # backslashes and path separators to prevent header injection.
    safe_filename = re.sub(r'[\r\n"\\/]+', "_", image.filename or "image")

    headers = {
        "Cache-Control": "private, max-age=3600",
        # Never let the browser MIME-sniff a mismatched/hostile type.
        "X-Content-Type-Options": "nosniff",
        "Content-Disposition": f'inline; filename="{safe_filename}"',
    }
    etag = response.get("ETag")
    if etag:
        headers["ETag"] = etag
    content_length = response.get("ContentLength", image.size_bytes)
    if content_length is not None:
        headers["Content-Length"] = str(content_length)

    return StreamingResponse(
        storage_service.stream_body(response["Body"]),
        media_type=image.content_type,
        headers=headers,
    )


@router.delete("/images/{image_id}")
async def delete_image(
    image_id: str = Path(..., description="Image ID"),
    requesting_user: User = Depends(get_current_user),
) -> dict:
    """
    Delete an owned image from object storage and the database.

    Args:
        image_id (str): Image identifier.
        requesting_user (User): Authenticated user injected by dependency.

    Returns:
        Confirmation message.

    Raises:
        HTTPException: 404 if not found; 403 if deletion is forbidden.
    """
    image = await get_owned_image(image_id, requesting_user)

    try:
        await storage_service.delete_object(image.key)
    except Exception as e:  # noqa: BLE001 - best-effort; still remove the DB record
        logger.error(f"Failed to delete image object {image.key}: {e}")

    await image.delete()
    return {"message": "Image deleted successfully"}
