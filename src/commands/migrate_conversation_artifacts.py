"""Move MCP artifacts into the one conversation store.

An older MCP file has its record pointed at ``users/{uid}/artifacts/{uuid}``
and, when a conversation copy was made, a second object at
``users/{uid}/conversations/{cid}/tool_artifacts/{id}/{name}``. This command
keeps one object, at
``users/{uid}/conversations/{cid}/artifacts/{id}/{name}``, and points the
record at it.

    python -m src.commands.migrate_conversation_artifacts [--apply]

Defaults to a dry run. ``--apply`` writes. A record with no conversation, a
user upload, and a record already on the conversation key are left alone.
"""

import argparse
import asyncio
import logging
import os

from src.config import configure_logging
from src.database.models.artifact import Artifact
from src.database.mongo import async_mongo_manager
from src.services.storage import ObjectNotFoundError, storage_service

configure_logging()
logger = logging.getLogger(__name__)

_LEGACY_KEY = r"^users/[^/]+/artifacts/"


def _basename(filename: str) -> str:
    raw = (filename or "").replace("\\", "/").strip()
    name = os.path.basename(raw) or "file"
    if name in {".", ".."}:
        return "file"
    return name


def _new_key(artifact: Artifact) -> str:
    name = _basename(artifact.filename)
    return (
        f"users/{artifact.user_id}/conversations/{artifact.conversation_id}"
        f"/artifacts/{artifact.id}/{name}"
    )


def _legacy_copy_key(artifact: Artifact) -> str:
    name = _basename(artifact.filename)
    return (
        f"users/{artifact.user_id}/conversations/{artifact.conversation_id}"
        f"/tool_artifacts/{artifact.id}/{name}"
    )


def _read_body(response: dict) -> bytes:
    body = response["Body"]
    try:
        read = getattr(body, "read", None)
        if read is None:
            return body.data
        return read()
    finally:
        close = getattr(body, "close", None)
        if close is not None:
            close()


async def _load(key: str):
    try:
        response = await storage_service.get_object(key)
    except ObjectNotFoundError:
        return None
    return await asyncio.to_thread(_read_body, response)


async def migrate_conversation_artifacts(apply: bool = False) -> None:
    """Copy each legacy MCP object, retarget the record, then delete the old keys.

    The record changes only after the new object is written. A failed save
    deletes that new object, so the old key keeps serving. A failed delete of
    an old key is logged and left for a later run.
    """
    if async_mongo_manager.database is None:
        await async_mongo_manager.connect(ensure_indexes=apply)

    candidates = await Artifact.find_all(
        filter_dict={
            "source.type": "mcp_tool",
            "conversation_id": {"$nin": [None, ""]},
            "key": {"$regex": _LEGACY_KEY},
        }
    )
    if not candidates:
        print("No legacy MCP artifacts to move.")
        return

    print(f"Found {len(candidates)} legacy MCP artifact(s).")
    for artifact in candidates:
        print(
            f"  {artifact.id}  {artifact.key}  ->  {_new_key(artifact)}"
        )
    if not apply:
        print("Dry run: nothing was written. Re-run with --apply to move them.")
        return

    moved = 0
    for artifact in candidates:
        if await _move(artifact):
            moved += 1
    print(f"Moved {moved} of {len(candidates)} artifact(s).")


async def _move(artifact: Artifact) -> bool:
    new_key = _new_key(artifact)
    previous = artifact.key
    copy_key = _legacy_copy_key(artifact)
    data = await _load(copy_key)
    if data is None:
        data = await _load(previous)
    if data is None:
        print(f"  SKIPPED (no bytes): {artifact.id}")
        return False
    await storage_service.put_object(
        new_key, data, artifact.content_type or "application/octet-stream"
    )
    artifact.key = new_key
    try:
        await artifact.save()
    except Exception:
        logger.exception("artifacts.migrate_save_failed id=%s", artifact.id)
        artifact.key = previous
        try:
            await storage_service.delete_object(new_key)
        except Exception:
            logger.exception("artifacts.migrate_cleanup_failed key=%s", new_key)
        print(f"  SKIPPED (record update failed): {artifact.id}")
        return False
    for stale in (previous, copy_key):
        if stale == new_key:
            continue
        try:
            await storage_service.delete_object(stale)
        except Exception:
            logger.exception("artifacts.migrate_delete_failed key=%s", stale)
            print(f"  KEPT old object (delete failed): {stale}")
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move MCP artifacts into the conversation artifact store."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the new objects and update records. Without this flag, dry-run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(migrate_conversation_artifacts(apply=args.apply))
