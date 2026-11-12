"""Erase a user's stored files: one artifact, or everything a conversation left.

Order for an artifact, the same as ``gc_mcp_artifacts``: the object first,
then the row, then the references in messages. A failed object delete keeps
the row, so nothing is orphaned and a later run can still find the bytes.

A message references an artifact in two places: ``artifact_ids`` (MCP tool
outputs) and ``attachments[].image_id`` (user uploads). Both are matched and
both are cleaned.

Logs carry ids and the exception class only: never a filename, a key, or the
provider's error text.
"""

from __future__ import annotations

import asyncio
import logging

from src.database.models.artifact import Artifact
from src.database.models.message import Message
from src.services.agents.core.tool_results import tool_result_prefix
from src.services.storage import storage_service

logger = logging.getLogger(__name__)

# The whole cascade, so a hung S3 call cannot hold a conversation delete.
# The threadpool call keeps running (bounded by the S3 client timeouts), the
# request does not wait for it.
CASCADE_TIMEOUT_S = 10
# Rows read per conversation and objects removed from its tool-result prefix.
CASCADE_MAX_ARTIFACTS = 1000
PREFIX_DELETE_MAX_KEYS = 10_000


def _referencing(artifact_ids: list[str]) -> dict:
    """Message filter for any reference to one of ``artifact_ids``."""
    return {
        "$or": [
            {"artifact_ids": {"$in": artifact_ids}},
            {"attachments.image_id": {"$in": artifact_ids}},
        ]
    }


async def pull_message_references(artifact_id: str) -> None:
    """Remove ``artifact_id`` from ``artifact_ids`` and ``attachments`` of every message.

    Two updates, each filtered on its own array: ``$pull`` on a field that is
    null (the default of both) is an error, and the filter guarantees an array.
    """
    messages = Message.get_collection()
    await messages.update_many(
        {"artifact_ids": artifact_id}, {"$pull": {"artifact_ids": artifact_id}}
    )
    await messages.update_many(
        {"attachments.image_id": artifact_id},
        {"$pull": {"attachments": {"image_id": artifact_id}}},
    )


async def delete_artifact(artifact: Artifact) -> None:
    """Delete one artifact: object, row, then message references.

    Raises whatever the storage delete or the row delete raised; the row is
    left in place when the object could not be deleted. A failed reference
    pull after the row is gone is logged and does not raise: the file is
    erased, and a dangling reference already answers 404 on read.
    """
    await storage_service.delete_object(artifact.key)
    await artifact.delete()
    try:
        await pull_message_references(artifact.id)
    except Exception as exc:  # noqa: BLE001 - the erasure itself succeeded
        logger.warning(
            "artifact.pull_failed artifact=%s error=%s",
            artifact.id,
            type(exc).__name__,
        )


async def delete_conversation_artifacts(user_id: str, conversation_id: str) -> None:
    """Remove a conversation's artifacts and offloaded tool results. Never raises.

    An artifact a message in another conversation still shows (as a tool
    output or as an attached upload) is kept and moved to that conversation,
    so a later delete of that one removes it. Every failure is a WARNING with
    ids only; the conversation delete goes on. Safe to run twice: a second
    run finds nothing to do.
    """
    try:
        async with asyncio.timeout(CASCADE_TIMEOUT_S):
            await _delete_artifacts(user_id, conversation_id)
            await _delete_tool_results(user_id, conversation_id)
    except TimeoutError:
        logger.warning(
            "artifact.cascade_failed reason=timeout user=%s conversation=%s",
            user_id,
            conversation_id,
        )
    except Exception as exc:  # noqa: BLE001 - best effort, logged
        logger.warning(
            "artifact.cascade_failed user=%s conversation=%s error=%s",
            user_id,
            conversation_id,
            type(exc).__name__,
        )


async def _delete_artifacts(user_id: str, conversation_id: str) -> None:
    # Primary: an attach a moment ago may have moved a row here, and a
    # secondary that has not caught up would miss it.
    docs = (
        await Artifact.get_primary_collection()
        .find({"user_id": user_id, "conversation_id": conversation_id})
        .limit(CASCADE_MAX_ARTIFACTS)
        .to_list(length=CASCADE_MAX_ARTIFACTS)
    )
    artifacts = [Artifact.from_dict(doc) for doc in docs]
    if not artifacts:
        return
    shown_elsewhere = await _other_conversations(
        [a.id for a in artifacts], conversation_id
    )
    for artifact in artifacts:
        other = shown_elsewhere.get(artifact.id)
        try:
            if other:
                artifact.conversation_id = other
                await artifact.save()
            else:
                await delete_artifact(artifact)
        except Exception as exc:  # noqa: BLE001 - best effort, logged
            logger.warning(
                "artifact.cascade_failed user=%s conversation=%s artifact=%s error=%s",
                user_id,
                conversation_id,
                artifact.id,
                type(exc).__name__,
            )


async def _other_conversations(
    artifact_ids: list[str], conversation_id: str
) -> dict[str, str]:
    """Artifact id to one other conversation whose messages still show it.

    Read on the primary: a secondary that missed a fresh attach would let
    the cascade delete a file another conversation shows.
    """
    query = _referencing(artifact_ids)
    query["conversation_id"] = {"$ne": conversation_id}
    cursor = Message.get_primary_collection().find(
        query, {"artifact_ids": 1, "attachments.image_id": 1, "conversation_id": 1}
    )
    wanted = set(artifact_ids)
    found: dict[str, str] = {}
    async for doc in cursor:
        referenced = list(doc.get("artifact_ids") or [])
        referenced += [
            a.get("image_id")
            for a in doc.get("attachments") or []
            if isinstance(a, dict)
        ]
        for artifact_id in referenced:
            if artifact_id in wanted and artifact_id not in found:
                found[artifact_id] = doc["conversation_id"]
    return found


async def _delete_tool_results(user_id: str, conversation_id: str) -> None:
    try:
        await storage_service.delete_prefix(
            tool_result_prefix(user_id, conversation_id),
            max_keys=PREFIX_DELETE_MAX_KEYS,
        )
    except Exception as exc:  # noqa: BLE001 - best effort, logged
        logger.warning(
            "artifact.cascade_failed kind=tool_results user=%s conversation=%s error=%s",
            user_id,
            conversation_id,
            type(exc).__name__,
        )
