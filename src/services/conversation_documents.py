"""Documents attached to one conversation (POST /conversations/{id}/documents).

They live in an ordinary private collection whose ``conversation_id`` is set:
its points sit in the shared private Qdrant collection like any other, chat in
that conversation searches it, and GET /collections does not list it.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Optional

import anyio
from bson import ObjectId
from fastapi import HTTPException
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from src import config
from src.core.vector_store_manager import VectorStoreManager
from src.database.models.collection import Collection
from src.database.models.document import Document
from src.services.private_document_limit import release_private_document_slots

logger = logging.getLogger(__name__)

CONVERSATION_COLLECTION_NAME = "Attached documents"


async def get_conversation_collection(
    user_id: str, conversation_id: str
) -> Optional[Collection]:
    """The collection holding a conversation's attached documents, read on the primary.

    Primary, because the chat message that follows an upload must see the
    collection the upload just created, and replica sets read secondaries.
    """
    doc = await Collection.get_primary_collection().find_one(
        {"user_id": user_id, "conversation_id": conversation_id}
    )
    return Collection.from_dict(doc) if doc else None


async def get_or_create_conversation_collection(
    user_id: str, conversation_id: str
) -> Collection:
    """One collection per conversation, created on its first upload.

    The collection's ``_id`` is the conversation id, so the unique ``_id`` index
    holds two first uploads racing each other to a single row: one inserts, the
    other matches it (or hits the duplicate key and reads the winner).
    """
    existing = await get_conversation_collection(user_id, conversation_id)
    if existing is not None:
        return existing
    on_insert = Collection(
        user_id=user_id,
        name=CONVERSATION_COLLECTION_NAME,
        conversation_id=conversation_id,
    ).to_dict()
    collections = Collection.get_primary_collection()
    try:
        doc = await collections.find_one_and_update(
            {"_id": ObjectId(conversation_id)},
            {"$setOnInsert": on_insert},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
    except DuplicateKeyError:
        doc = await collections.find_one({"_id": ObjectId(conversation_id)})
    return Collection.from_dict(doc)


async def collection_for_chat(
    user_id: str, conversation_id: Optional[str]
) -> Optional[Collection]:
    """The attached-documents collection chat should search, or None."""
    if not conversation_id or not config.FEATURE_CONVERSATION_DOCUMENTS:
        return None
    return await get_conversation_collection(user_id, conversation_id)


async def delete_conversation_documents(user_id: str, conversation_id: str) -> None:
    """Remove a conversation's attached documents: points, rows, collection, slots.

    Runs whatever the flag says, so documents attached while it was on are not
    left holding slots after it is turned off. Qdrant goes first and a failure
    there stops everything, like DELETE /collections/{id}: Mongo rows deleted
    over points that survive would leave those points with nothing to find them.
    """
    collection = await get_conversation_collection(user_id, conversation_id)
    if collection is None:
        return
    try:
        await anyio.to_thread.run_sync(
            partial(
                VectorStoreManager().delete_points_for_collection,
                user_id,
                collection.id,
            )
        )
    except Exception:
        logger.error(
            "Failed to delete Qdrant points for conversation collection %s",
            collection.id,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "conversation_document_cleanup_failed",
                "message": "The attached documents could not be deleted. Try again.",
            },
        )
    deleted = await Document.delete_many({"collection_id": collection.id})
    await collection.delete()
    await release_private_document_slots(user_id, deleted)
