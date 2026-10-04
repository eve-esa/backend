from __future__ import annotations

import logging

from bson import ObjectId
from fastapi import HTTPException

from src.constants import MAX_PRIVATE_DOCUMENTS
from src.database.models.document import Document
from src.database.models.user import User

logger = logging.getLogger(__name__)

_RELEASE_ATTEMPTS = 3


def _limit_reached_error(current_count: int) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail=(
            f"Private document limit reached. You can upload at most "
            f"{MAX_PRIVATE_DOCUMENTS} documents in total "
            f"({current_count} already uploaded)."
        ),
    )


async def _ensure_private_document_count_initialized(user_id: str) -> None:
    """Backfill the counter once for users created before this field existed."""
    user_doc = await User.get_collection().find_one(
        {"_id": ObjectId(user_id)},
        projection={"private_document_count": 1},
    )
    if user_doc is not None and "private_document_count" in user_doc:
        return

    actual_count = await Document.count_documents({"user_id": user_id})
    await User.get_collection().update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"private_document_count": actual_count}},
    )


async def reserve_private_document_slots(user_id: str, slot_count: int) -> None:
    """Atomically reserve upload slots against the per-user private document cap."""
    if slot_count <= 0:
        return

    await _ensure_private_document_count_initialized(user_id)

    updated = await User.get_collection().find_one_and_update(
        {
            "_id": ObjectId(user_id),
            "private_document_count": {
                "$lte": MAX_PRIVATE_DOCUMENTS - slot_count,
            },
        },
        {"$inc": {"private_document_count": slot_count}},
    )
    if updated is None:
        user = await User.find_by_id(user_id)
        current_count = int((user.private_document_count if user else 0) or 0)
        raise _limit_reached_error(current_count)


async def release_private_document_slots(user_id: str, slot_count: int) -> None:
    """Return reserved or deleted slots to the user's private document counter.

    Plain update operators only: DocumentDB 5.0 rejects pipeline updates
    (code 14), which local MongoDB accepts. The counter never goes below 0.
    """
    if slot_count <= 0:
        return

    users = User.get_collection()
    user_id_filter = {"_id": ObjectId(user_id)}
    for _ in range(_RELEASE_ATTEMPTS):
        decremented = await users.update_one(
            {**user_id_filter, "private_document_count": {"$gte": slot_count}},
            {"$inc": {"private_document_count": -slot_count}},
        )
        if decremented.matched_count:
            return
        # Fewer slots held than released, or no counter yet: floor at 0, but
        # only on the value just read, so a concurrent reserve is never erased.
        user_doc = await users.find_one(
            user_id_filter, projection={"private_document_count": 1}
        )
        if user_doc is None:
            return
        observed = user_doc.get("private_document_count")
        if observed is not None and observed >= slot_count:
            continue
        floored = await users.update_one(
            {**user_id_filter, "private_document_count": observed},
            {"$set": {"private_document_count": 0}},
        )
        if floored.matched_count:
            return
    logger.warning(
        "private_document_release_exhausted user_id=%s released=%s attempts=%s",
        user_id,
        slot_count,
        _RELEASE_ATTEMPTS,
    )
