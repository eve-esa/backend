"""Documents attached to one conversation (POST /conversations/{id}/documents).

They live in an ordinary private collection whose ``conversation_id`` is set:
its points sit in the shared private Qdrant collection like any other, chat in
that conversation searches it, and GET /collections does not list it.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any, Optional

import anyio
from bson import ObjectId
from bson.errors import InvalidId
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


async def delete_conversation_collection_if_empty(collection: Collection) -> None:
    """Drop a conversation collection a failed upload left without documents.

    Otherwise every chat turn in the conversation searches an empty collection.
    An upload racing this one recreates it under the same ``_id``, the
    conversation id, so documents it writes stay attached either way.
    """
    try:
        remaining = await Document.get_primary_collection().count_documents(
            {"collection_id": collection.id}
        )
        if remaining == 0:
            await Collection.get_primary_collection().delete_one(
                {"_id": ObjectId(collection.id)}
            )
    except Exception:
        logger.warning(
            "Failed to drop empty conversation collection %s",
            collection.id,
            exc_info=True,
        )


async def collection_for_chat(
    user_id: str, conversation_id: Optional[str]
) -> Optional[Collection]:
    """The attached-documents collection chat should search, or None."""
    if not conversation_id or not config.FEATURE_CONVERSATION_DOCUMENTS:
        return None
    return await get_conversation_collection(user_id, conversation_id)


def _ingested_documents(collection_id: str) -> dict[str, Any]:
    """Rows whose upload finished.

    An upload inserts its rows first and sets ``chunk_count`` once the points
    are written. Until then a row is neither listed nor deletable: a delete
    landing mid-upload would leave the points written after it with no row.
    """
    return {"collection_id": collection_id, "chunk_count": {"$gt": 0}}


async def list_conversation_documents(
    user_id: str, conversation_id: str
) -> list[Document]:
    """A conversation's documents, oldest first, read on the primary.

    Primary, because the composer lists right after an upload or a delete. Not
    paginated: the user's private document cap bounds the rows.
    """
    collection = await get_conversation_collection(user_id, conversation_id)
    if collection is None:
        return []
    rows = (
        await Document.get_primary_collection()
        .find(_ingested_documents(collection.id))
        .sort([("timestamp", 1), ("_id", 1)])
        .to_list(length=None)
    )
    return [Document.from_dict(row) for row in rows]


async def delete_conversation_document(
    user_id: str, conversation_id: str, document_id: str
) -> bool:
    """Remove one attached document: points, row, slot. False if there is none.

    Qdrant goes first and a failure there stops everything, like the whole
    conversation delete. The slot goes back only if this call removed the row,
    so two deletes of one document racing each other release it once. Removing
    the last document drops the collection, so chat stops searching it.
    """
    collection = await get_conversation_collection(user_id, conversation_id)
    if collection is None:
        return False
    try:
        row_filter = {
            "_id": ObjectId(document_id),
            **_ingested_documents(collection.id),
        }
    except InvalidId:
        return False
    documents = Document.get_primary_collection()
    if await documents.find_one(row_filter, projection={"_id": 1}) is None:
        return False
    try:
        await anyio.to_thread.run_sync(
            partial(
                VectorStoreManager().delete_private_docs,
                user_id=user_id,
                collection_id=collection.id,
                metadata={"metadata.document_id": document_id},
            )
        )
    except Exception:
        logger.error(
            "Failed to delete Qdrant points for conversation document %s",
            document_id,
            exc_info=True,
        )
        raise _cleanup_failed()
    try:
        removed = (await documents.delete_one(row_filter)).deleted_count
        if removed:
            await release_private_document_slots(user_id, removed)
    except Exception:
        logger.error(
            "Failed to delete Mongo row for conversation document %s",
            document_id,
            exc_info=True,
        )
        raise _cleanup_failed()
    if not removed:
        return False
    await delete_conversation_collection_if_empty(collection)
    return True


async def delete_conversation_documents(user_id: str, conversation_id: str) -> None:
    """Remove a conversation's attached documents: points, rows, collection, slots.

    Runs whatever the flag says, so documents attached while it was on are not
    left holding slots after it is turned off. Same order as
    DELETE /collections/{id}. Qdrant goes first and a failure there stops
    everything: Mongo rows deleted over points that survive would leave those
    points with nothing to find them. Slots go back right after their rows:
    a retry finds no rows, so it could never release them later.
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
        raise _cleanup_failed()
    try:
        deleted = await Document.delete_many({"collection_id": collection.id})
        await release_private_document_slots(user_id, deleted)
        await collection.delete()
    except Exception:
        logger.error(
            "Failed to delete Mongo rows for conversation collection %s",
            collection.id,
            exc_info=True,
        )
        raise _cleanup_failed()


def _cleanup_failed() -> HTTPException:
    return HTTPException(
        status_code=500,
        detail={
            "code": "conversation_document_cleanup_failed",
            "message": "The attached documents could not be deleted. Try again.",
        },
    )
