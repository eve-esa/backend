"""Erase a user's stored files: one artifact, or everything a conversation left.

Order for one artifact (``delete_artifact``, as ``gc_mcp_artifacts``): the
object first, then the row, then the references in messages. A failed object
delete keeps the row, so nothing is orphaned and a later run can still find
the bytes. The conversation cascade deletes the row first, conditionally, so
an attach that moved it meanwhile keeps it, and puts it back when the object
delete fails.

A message references an artifact in two places: ``artifact_ids`` (MCP tool
outputs) and ``attachments[].image_id`` (user uploads). Both are matched and
both are cleaned.

Logs carry ids and the exception class only: never a filename, a key, or the
provider's error text.
"""

from __future__ import annotations

import asyncio
import logging

from bson import ObjectId

from src.database.models.artifact import Artifact
from src.database.models.conversation import Conversation
from src.database.models.message import Message
from src.services.agents.core.tool_results import tool_result_prefix
from src.services.agents.core.workspace import workspace_prefix, workspace_state_key
from src.services.storage import storage_service

logger = logging.getLogger(__name__)

# Each part of the cascade (artifacts, tool results, workspace) has its own
# deadline, so a hung S3 call cannot hold a conversation delete. The
# threadpool call keeps running (bounded by the S3 client timeouts), the
# request does not wait for it.
CASCADE_TIMEOUT_S = 10
# Rows read per round; the cascade reads rounds until none is left.
CASCADE_BATCH_SIZE = 1000
PREFIX_DELETE_MAX_KEYS = 10_000


class ArtifactCascadeIncomplete(Exception):
    """A conversation's files were not all erased; a retry finishes the job.

    ``failed`` names the parts that did not finish: any of ``artifacts``,
    ``tool_results`` and ``workspace``.
    """

    def __init__(self, failed: list[str]):
        super().__init__(",".join(failed))
        self.failed = failed


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
    """Remove a conversation's artifacts, offloaded tool results and workspace.

    The three parts run side by side, each with its own deadline, so one that
    fails or hangs never skips the others. An artifact a message in another of
    the owner's conversations still shows (as a tool output or as an attached
    upload) is kept and moved to that conversation, so a later delete of that
    one removes it. Every failure is a WARNING with ids only.

    Safe to run twice: a second run finds nothing to do, so the caller keeps
    the conversation and asks for a retry.

    Raises:
        ArtifactCascadeIncomplete: when either half failed or ran out of time.
    """
    results = await asyncio.gather(
        _bounded("artifacts", _delete_artifacts, user_id, conversation_id),
        _bounded("tool_results", _delete_tool_results, user_id, conversation_id),
        _bounded("workspace", _delete_workspace, user_id, conversation_id),
    )
    failed = [kind for kind, ok in results if not ok]
    if failed:
        raise ArtifactCascadeIncomplete(failed)


async def _bounded(kind: str, step, user_id: str, conversation_id: str) -> tuple[str, bool]:
    """Run one part of the cascade under its deadline; True when it finished."""
    try:
        async with asyncio.timeout(CASCADE_TIMEOUT_S):
            return kind, await step(user_id, conversation_id)
    except TimeoutError:
        logger.warning(
            "artifact.cascade_failed reason=timeout kind=%s user=%s conversation=%s",
            kind,
            user_id,
            conversation_id,
        )
    except Exception as exc:  # noqa: BLE001 - reported to the caller, logged
        logger.warning(
            "artifact.cascade_failed kind=%s user=%s conversation=%s error=%s",
            kind,
            user_id,
            conversation_id,
            type(exc).__name__,
        )
    return kind, False


async def _delete_artifacts(user_id: str, conversation_id: str) -> bool:
    """Delete or move every artifact of the conversation; False if one failed.

    Reads rounds of ``CASCADE_BATCH_SIZE`` until none is left. A row whose
    delete failed stays (with its object), so the loop stops after a round
    with a failure instead of reading the same rows again.
    """
    while True:
        # Primary: an attach a moment ago may have moved a row here, and a
        # secondary that has not caught up would miss it.
        docs = (
            await Artifact.get_primary_collection()
            .find({"user_id": user_id, "conversation_id": conversation_id})
            .limit(CASCADE_BATCH_SIZE)
            .to_list(length=CASCADE_BATCH_SIZE)
        )
        artifacts = [Artifact.from_dict(doc) for doc in docs]
        if not artifacts:
            return True
        shown_elsewhere = await _other_conversations(
            user_id, [a.id for a in artifacts], conversation_id
        )
        complete = True
        for artifact in artifacts:
            other = shown_elsewhere.get(artifact.id)
            try:
                if other:
                    # Only the conversation, and only while the row is still
                    # here: a full save would write back a key an attach just
                    # moved, and the row would point at a deleted object.
                    await Artifact.get_collection().update_one(
                        {"_id": ObjectId(artifact.id), "conversation_id": conversation_id},
                        {"$set": {"conversation_id": other}},
                    )
                else:
                    await _delete_unless_moved(artifact, conversation_id)
            except Exception as exc:  # noqa: BLE001 - reported, logged
                complete = False
                logger.warning(
                    "artifact.cascade_failed user=%s conversation=%s artifact=%s error=%s",
                    user_id,
                    conversation_id,
                    artifact.id,
                    type(exc).__name__,
                )
        if not complete:
            return False
        if len(artifacts) < CASCADE_BATCH_SIZE:
            return True


async def _delete_unless_moved(artifact: Artifact, conversation_id: str) -> None:
    """Delete an artifact of the conversation unless an attach moved it meanwhile.

    The row goes first and only while it still has the key and conversation
    the cascade read, so an attach that moved it elsewhere keeps its row and
    its new object. Then the object; when storage refuses, the row is put
    back so a retry still finds the file. Then the message references.
    """
    collection = Artifact.get_collection()
    row = await collection.find_one_and_delete(
        {
            "_id": ObjectId(artifact.id),
            "key": artifact.key,
            "conversation_id": conversation_id,
        }
    )
    if row is None:
        return
    try:
        await storage_service.delete_object(artifact.key)
    except Exception:
        await collection.insert_one(row)
        raise
    try:
        await pull_message_references(artifact.id)
    except Exception as exc:  # noqa: BLE001 - the erasure itself succeeded
        logger.warning(
            "artifact.pull_failed artifact=%s error=%s",
            artifact.id,
            type(exc).__name__,
        )


async def _other_conversations(
    user_id: str, artifact_ids: list[str], conversation_id: str
) -> dict[str, str]:
    """Artifact id to another of the owner's conversations whose messages still show it.

    Only the owner's conversations count, so a message elsewhere that carries
    the id can never keep the owner's file alive. Read on the primary: a
    secondary that missed a fresh attach would let the cascade delete a file
    another conversation shows.
    """
    owned = (
        await Conversation.get_primary_collection()
        .find({"user_id": user_id}, {"_id": 1})
        .to_list(length=None)
    )
    others = [str(doc["_id"]) for doc in owned if str(doc["_id"]) != conversation_id]
    if not others:
        return {}
    query = _referencing(artifact_ids)
    query["conversation_id"] = {"$in": others}
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


async def _delete_tool_results(user_id: str, conversation_id: str) -> bool:
    """Empty the conversation's tool-result prefix; False when the cap was hit.

    Tool results have no Mongo row: once the conversation is gone nothing
    lists them again, so a capped run reports itself incomplete.
    """
    deleted = await storage_service.delete_prefix(
        tool_result_prefix(user_id, conversation_id),
        max_keys=PREFIX_DELETE_MAX_KEYS,
    )
    return deleted < PREFIX_DELETE_MAX_KEYS


async def _delete_workspace(user_id: str, conversation_id: str) -> bool:
    """Empty the saved workspace and drop its state file; False when the cap was hit.

    Like tool results, the workspace has no Mongo row. The state file goes
    only once the prefix is empty, so a capped run leaves both for the retry.
    """
    deleted = await storage_service.delete_prefix(
        workspace_prefix(user_id, conversation_id),
        max_keys=PREFIX_DELETE_MAX_KEYS,
    )
    if deleted >= PREFIX_DELETE_MAX_KEYS:
        return False
    await storage_service.delete_object(workspace_state_key(user_id, conversation_id))
    return True
