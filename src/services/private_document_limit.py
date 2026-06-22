from __future__ import annotations

from bson import ObjectId
from fastapi import HTTPException

from src.constants import MAX_PRIVATE_DOCUMENTS
from src.database.models.document import Document
from src.database.models.user import User


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
    """Return reserved or deleted slots to the user's private document counter."""
    if slot_count <= 0:
        return

    await User.get_collection().update_one(
        {"_id": ObjectId(user_id)},
        [
            {
                "$set": {
                    "private_document_count": {
                        "$max": [
                            {
                                "$subtract": [
                                    {"$ifNull": ["$private_document_count", 0]},
                                    slot_count,
                                ]
                            },
                            0,
                        ]
                    }
                }
            }
        ],
    )
