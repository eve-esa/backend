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
from src.constants import (
    CONVERSATION_DOCUMENT_DELETE_SECONDS,
    CONVERSATION_DOCUMENT_QDRANT_TIMEOUT_SECONDS,
)
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


async def keep_conversation_collection(collection: Collection) -> None:
    """Put a conversation collection's row back if a delete just dropped it.

    An idempotent upsert on the collection's own ``_id``: a no-op while the row
    exists. An upload calls it after inserting its rows, so rows written while
    the last document was being deleted keep a collection that list, chat and
    the deletes can find.
    """
    await Collection.get_primary_collection().update_one(
        {"_id": ObjectId(collection.id)},
        {"$setOnInsert": collection.to_dict()},
        upsert=True,
    )


async def delete_conversation_collection_if_empty(collection: Collection) -> None:
    """Drop a conversation collection left without documents.

    Otherwise every chat turn in the conversation searches an empty collection.
    An upload may insert its rows between the count and the drop, so the rows
    are counted again after it and the row comes back if any appeared. Together
    with :func:`keep_conversation_collection` in the upload, rows always end up
    under a collection row, whichever side runs first.
    """
    documents = Document.get_primary_collection()
    try:
        if await documents.count_documents({"collection_id": collection.id}):
            return
        await Collection.get_primary_collection().delete_one(
            {"_id": ObjectId(collection.id)}
        )
        if await documents.count_documents({"collection_id": collection.id}):
            await keep_conversation_collection(collection)
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


async def count_attached_documents(collection_id: str) -> int:
    """Documents counted against the per-conversation cap, read on the primary.

    Only finished uploads, like the list and the delete: rows an incomplete
    rollback left behind can be neither seen nor deleted, so they must not
    hold the cap either.
    """
    return await Document.get_primary_collection().count_documents(
        _ingested_documents(collection_id)
    )


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

    Qdrant goes first and a failure or a timeout there stops everything, like
    the whole conversation delete: the row stays and a retry finishes the job.
    The slot goes back only if this call removed the row, so two deletes of one
    document racing each other release it once. Once the row is gone the
    delete has happened: a failed release is logged and left to the reconcile
    command rather than answered with a 500 whose retry would find nothing.
    Removing the last document drops the collection, so chat stops searching it.
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
        await _delete_points(
            "delete_private_docs",
            user_id=user_id,
            collection_id=collection.id,
            metadata={"metadata.document_id": document_id},
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
    except Exception:
        logger.error(
            "Failed to delete Mongo row for conversation document %s",
            document_id,
            exc_info=True,
        )
        raise _cleanup_failed()
    if not removed:
        return False
    await _release_slots(user_id, removed)
    await delete_conversation_collection_if_empty(collection)
    return True


async def delete_conversation_documents(user_id: str, conversation_id: str) -> None:
    """Remove a conversation's attached documents: points, rows, collection, slots.

    Runs whatever the flag says, so documents attached while it was on are not
    left holding slots after it is turned off. Same order as
    DELETE /collections/{id}. Qdrant goes first and a failure there stops
    everything: Mongo rows deleted over points that survive would leave those
    points with nothing to find them. Slots go back right after their rows:
    a retry finds no rows, so it could never release them later. A failed
    release is logged and left to the reconcile command for the same reason.

    Without a collection row the cascade still runs on the conversation id,
    which is the ``_id`` of every collection created since #333: rows an upload
    wrote while its last document was being deleted are removed with the rest.
    """
    collection = await get_conversation_collection(user_id, conversation_id)
    collection_id = collection.id if collection else conversation_id
    row_filter = {"collection_id": collection_id, "user_id": user_id}
    if collection is None and not await Document.get_primary_collection().find_one(
        row_filter, projection={"_id": 1}
    ):
        return
    try:
        await _delete_points(
            "delete_points_for_collection", user_id=user_id, collection_id=collection_id
        )
    except Exception:
        logger.error(
            "Failed to delete Qdrant points for conversation collection %s",
            collection_id,
            exc_info=True,
        )
        raise _cleanup_failed()
    try:
        deleted = await Document.delete_many(row_filter)
    except Exception:
        logger.error(
            "Failed to delete Mongo rows for conversation collection %s",
            collection_id,
            exc_info=True,
        )
        raise _cleanup_failed()
    await _release_slots(user_id, deleted)
    if collection is None:
        return
    try:
        await collection.delete()
    except Exception:
        logger.error(
            "Failed to delete conversation collection %s",
            collection_id,
            exc_info=True,
        )
        raise _cleanup_failed()


async def _delete_points(method: str, **kwargs: Any) -> None:
    """Run a blocking Qdrant delete within CONVERSATION_DOCUMENT_DELETE_SECONDS.

    Each call gets the upload's short HTTP timeout, and the deadline bounds them
    together. A delete still running when the deadline passes is left to finish
    on its thread; the caller has not touched Mongo yet, so a retry redoes it.
    """
    store = VectorStoreManager(
        qdrant_timeout_s=CONVERSATION_DOCUMENT_QDRANT_TIMEOUT_SECONDS
    )
    with anyio.fail_after(CONVERSATION_DOCUMENT_DELETE_SECONDS):
        await anyio.to_thread.run_sync(
            partial(getattr(store, method), **kwargs), abandon_on_cancel=True
        )


async def _release_slots(user_id: str, count: int) -> None:
    """Give back slots whose rows are already gone; a failure is only logged."""
    if not count:
        return
    try:
        await release_private_document_slots(user_id, count)
    except Exception:
        logger.error(
            "private_document_slots_not_released user_id=%s count=%s; "
            "run python -m src.commands.reconcile_private_document_counters",
            user_id,
            count,
            exc_info=True,
        )


def _cleanup_failed() -> HTTPException:
    return HTTPException(
        status_code=500,
        detail={
            "code": "conversation_document_cleanup_failed",
            "message": "The attached documents could not be deleted. Try again.",
        },
    )
