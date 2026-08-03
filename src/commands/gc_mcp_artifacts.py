"""Garbage-collect old MCP-produced artifacts (source.type == 'mcp_tool').

MCP tool outputs accumulate in object storage over time with no automatic
expiry (unlike user uploads, which are capped by a daily quota but never
auto-deleted either). This command reclaims storage for artifacts past a
retention window, run manually or from an external scheduler:

    python -m src.commands.gc_mcp_artifacts [--days N] [--apply]

Defaults to a dry run (prints what would be deleted); pass --apply to
actually delete. User uploads (source.type == "upload") are never touched --
this mirrors the DELETE /artifacts/{id} restriction that only MCP-generated
artifacts are eligible for this kind of automated reclaim.
"""

import argparse
import asyncio
import logging
from datetime import datetime, timedelta, timezone

from src.config import configure_logging
from src.database.models.artifact import Artifact
from src.database.models.message import Message
from src.database.mongo import async_mongo_manager
from src.services.storage import storage_service

configure_logging()
logger = logging.getLogger(__name__)

DEFAULT_RETENTION_DAYS = 30


async def gc_mcp_artifacts(days: int = DEFAULT_RETENTION_DAYS, apply: bool = False) -> None:
    """Delete mcp_tool artifacts older than `days` and scrub dangling references.

    Mirrors the fail-closed delete order used by DELETE /artifacts/{id}: the
    object is removed from storage FIRST, and the Mongo record is only
    dropped once that succeeds. If the storage delete fails, the artifact is
    skipped (logged) and left for a future run rather than dropping the
    record and orphaning the bytes with nothing left to reclaim them.

    Args:
        days (int): Retention window in days; artifacts older than this are eligible.
        apply (bool): When False (default), only report what would be deleted.
    """
    # Reuse an already-open connection (e.g. the test suite's isolated test
    # database) instead of unconditionally reconnecting to the default URI.
    if async_mongo_manager.database is None:
        await async_mongo_manager.connect()

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    candidates = await Artifact.find_all(
        filter_dict={"source.type": "mcp_tool", "timestamp": {"$lt": cutoff}}
    )

    if not candidates:
        print(f"No mcp_tool artifacts older than {days} days.")
        return

    print(f"Found {len(candidates)} mcp_tool artifact(s) older than {days} days.")
    for artifact in candidates:
        print(f"  {artifact.id}  key={artifact.key}  timestamp={artifact.timestamp.isoformat()}")

    if not apply:
        print("Dry run: no artifacts were deleted. Re-run with --apply to delete them.")
        return

    deleted_ids = []
    for artifact in candidates:
        try:
            await storage_service.delete_object(artifact.key)
        except Exception as e:  # noqa: BLE001 - never orphan the object; skip and retry later
            logger.error(f"Failed to delete artifact object {artifact.key}: {e}")
            print(f"  SKIPPED (storage delete failed): {artifact.id}")
            continue
        await artifact.delete()
        deleted_ids.append(artifact.id)

    if deleted_ids:
        result = await Message.get_collection().update_many(
            {"artifact_ids": {"$in": deleted_ids}},
            {"$pull": {"artifact_ids": {"$in": deleted_ids}}},
        )
        print(
            f"Deleted {len(deleted_ids)} artifact(s); "
            f"scrubbed references from {result.modified_count} message(s)."
        )
    else:
        print("Deleted 0 artifacts (all storage deletes failed; see log).")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Garbage-collect MCP-produced artifacts older than N days."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_RETENTION_DAYS,
        help=f"Retention window in days (default: {DEFAULT_RETENTION_DAYS})",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete matched artifacts. Without this flag, runs as a dry-run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(gc_mcp_artifacts(days=args.days, apply=args.apply))
