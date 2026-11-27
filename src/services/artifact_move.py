"""Move an artifact's object into a conversation's folder.

The upload route and the conversation-delete cascade both call this. The
object and the row move together, so a later delete of that conversation
removes the bytes. The quota counter does not change: a move copies and then
deletes, and the bytes stay the same user's.
"""

from __future__ import annotations

import logging
from typing import Optional

from bson import ObjectId
from fastapi import HTTPException

from src.database.models.artifact import Artifact
from src.services.mcp.conversation_artifacts import file_path_for, object_key
from src.services.storage import ObjectNotFoundError, storage_service

logger = logging.getLogger(__name__)

# Rounds a move retries after another request moved the same row first.
MOVE_ATTEMPTS = 3


async def move_to_conversation(artifact: Artifact, conversation_id: str) -> None:
    """Bind an artifact to a conversation and keep its object in that conversation's folder.

    The object goes to
    ``users/{user_id}/conversations/{conversation_id}/artifacts/{artifact_id}/{filename}``,
    the layout MCP outputs use, so a new sandbox restores it under
    ``/artifacts/`` and the conversation's delete cascade finds it. The copy
    and the row are written first and the old object is deleted last, so a
    failure leaves a stray object, never a row without bytes. A failed copy
    keeps the old key: the attachment still renders from it.

    Two requests can attach the same file at once (an overlapping send, a
    page that moved on mid-upload). The row is updated only while it still
    holds the key this request copied from, never replaced whole: a request
    that lost the race rereads the row and moves from where the winner left
    it, so it can never write back a key whose object is already gone.

    Raises:
        HTTPException: 404 if the row was deleted meanwhile; 409 with
        ``detail.code`` ``artifact_move_contended`` when other requests kept moving it
        for ``MOVE_ATTEMPTS`` rounds (the client can retry).
    """
    for _ in range(MOVE_ATTEMPTS):
        key = object_key(
            artifact.user_id, conversation_id, file_path_for(artifact.id, artifact.filename)
        )
        if artifact.conversation_id == conversation_id and artifact.key == key:
            return
        old_key = artifact.key
        new_key = old_key
        if old_key != key:
            try:
                await storage_service.copy_object(old_key, key)
            except ObjectNotFoundError:
                if await _reload_moved(artifact, old_key):
                    continue
                # The bytes are gone for good: bind the row, keep its key.
            except Exception as exc:  # noqa: BLE001 - the old key still serves
                logger.warning(
                    "artifact.move_failed artifact=%s conversation=%s error=%s",
                    artifact.id,
                    conversation_id,
                    type(exc).__name__,
                )
            else:
                new_key = key
        result = await Artifact.get_collection().update_one(
            {"_id": ObjectId(artifact.id), "key": old_key},
            {"$set": {"key": new_key, "conversation_id": conversation_id}},
        )
        if result.matched_count:
            artifact.key = new_key
            artifact.conversation_id = conversation_id
            if new_key != old_key and await _confirm_copy(artifact, old_key, new_key):
                await _delete_stray(artifact.id, old_key)
            return
        # Another request moved the row first. Drop this copy unless the row
        # now points at it (both requests chose the same conversation).
        current = await Artifact.find_by_id_on_primary(artifact.id)
        if new_key != old_key and (current is None or current.key != new_key):
            await _delete_stray(artifact.id, new_key)
        _adopt(artifact, current)
    logger.warning(
        "artifact.move_contended artifact=%s conversation=%s", artifact.id, conversation_id
    )
    raise HTTPException(
        status_code=409,
        detail={
            "code": "artifact_move_contended",
            "message": "The file is being attached elsewhere. Try again.",
        },
    )


async def _confirm_copy(artifact: Artifact, old_key: str, new_key: str) -> bool:
    """Make sure the object the row was just bound to exists; False keeps ``old_key``.

    Attaches to the same conversation share the destination key, so a request
    that lost the race can delete it after the winner copied and before the
    winner binds the row. The old object is still there: copy again. When that
    fails too, the row goes back to ``old_key``, which still serves.
    """
    try:
        if await storage_service.exists(new_key):
            return True
        await storage_service.copy_object(old_key, new_key)
        return True
    except Exception as exc:  # noqa: BLE001 - the old key still serves
        logger.warning(
            "artifact.move_failed artifact=%s error=%s", artifact.id, type(exc).__name__
        )
    result = await Artifact.get_collection().update_one(
        {"_id": ObjectId(artifact.id), "key": new_key}, {"$set": {"key": old_key}}
    )
    if result.matched_count:
        artifact.key = old_key
    return False


async def _reload_moved(artifact: Artifact, old_key: str) -> bool:
    """True when the row moved off ``old_key`` meanwhile; ``artifact`` then holds the new state."""
    current = await Artifact.find_by_id_on_primary(artifact.id)
    if current is not None and current.key == old_key:
        return False
    _adopt(artifact, current)
    return True


def _adopt(artifact: Artifact, current: Optional[Artifact]) -> None:
    """Take the key and conversation of the row as stored now; 404 when it is gone."""
    if current is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    artifact.key = current.key
    artifact.conversation_id = current.conversation_id


async def _delete_stray(artifact_id: str, key: str) -> None:
    """Delete an object no row points to; a failure leaks it, logged with the artifact id."""
    try:
        await storage_service.delete_object(key)
    except Exception as exc:  # noqa: BLE001 - leaked, no job reclaims it
        logger.warning(
            "artifact.move_cleanup_failed artifact=%s error=%s",
            artifact_id,
            type(exc).__name__,
        )
