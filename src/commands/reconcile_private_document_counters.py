"""Set each user's private document counter to the documents they really own.

``users.private_document_count`` backs the MAX_PRIVATE_DOCUMENTS cap
(src/services/private_document_limit.py). Until backend #275 (2026-10-04) the
release used a pipeline update that DocumentDB rejects, so every delete and
rollback before it left the counter too high, and the cap refuses uploads the
user is entitled to. This recounts ``documents`` per user and writes the result:

    python -m src.commands.reconcile_private_document_counters [--apply]

Defaults to a dry run that prints one line per user whose counter is wrong:
user id, stored counter, document count. No email or other personal data: the
output is pasted into tasks and kept in terminal logs. A dry run works with the
reader credential: it connects without creating indexes. ``--apply`` writes,
and needs the writer.

Users whose counter was never set are left alone: the upload path counts their
documents the first time it needs the value (_ensure_private_document_count_initialized).

Each write is conditional on the counter still holding the value read a moment
earlier. An upload or delete that moves it in between wins, the user is
reported as skipped, and a second run picks them up. An upload reserves its
slots before its document rows exist, so one in flight at the moment of the
read is not counted: run it when uploads are quiet, and compare a second dry run.
"""

import argparse
import asyncio
import logging
import sys
from typing import Dict, List, Tuple

from bson import ObjectId

from src.config import configure_logging
from src.database.models.document import Document
from src.database.models.user import User
from src.database.mongo import async_mongo_manager

configure_logging()
logger = logging.getLogger(__name__)

# (user id, stored counter, document count)
Mismatch = Tuple[str, int, int]


async def _document_counts() -> Dict[str, int]:
    """Documents per user_id, in one aggregation."""
    pipeline = [{"$group": {"_id": "$user_id", "count": {"$sum": 1}}}]
    counts: Dict[str, int] = {}
    async for row in Document.get_collection().aggregate(pipeline):
        if row["_id"]:
            counts[str(row["_id"])] = row["count"]
    return counts


async def find_mismatches() -> Tuple[List[Mismatch], int, int, int]:
    """Users whose stored counter differs from their document count.

    Returns the mismatches, the number of users checked, the number whose
    counter was never set, and the number of documents whose user no longer
    exists (reported, never counted).
    """
    counts = await _document_counts()
    mismatches: List[Mismatch] = []
    checked = 0
    unset = 0
    cursor = User.get_collection().find(
        {}, projection={"private_document_count": 1}
    )
    async for row in cursor:
        checked += 1
        user_id = str(row["_id"])
        actual = counts.pop(user_id, 0)
        stored = row.get("private_document_count")
        if stored is None:
            unset += 1
        elif stored != actual:
            mismatches.append((user_id, stored, actual))
    orphaned = sum(counts.values())
    return mismatches, checked, unset, orphaned


async def _apply_one(user_id: str, stored: int) -> bool:
    """Write a fresh count if the counter still holds ``stored``; False if it moved.

    Counted on the primary: on a replica set the default read goes to a
    secondary, which can miss a document written a moment earlier (#281).
    """
    actual = await Document.get_primary_collection().count_documents(
        {"user_id": user_id}
    )
    result = await User.get_collection().update_one(
        {"_id": ObjectId(user_id), "private_document_count": stored},
        {"$set": {"private_document_count": actual}},
    )
    return result.matched_count == 1


async def reconcile_private_document_counters(apply: bool = False) -> int:
    """Print every wrong counter and, with ``apply``, correct it.

    Returns the number of users still wrong afterwards: every mismatch on a dry
    run, the skipped ones after ``--apply``.
    """
    # Reuse an already-open connection (the test suite's isolated database).
    if async_mongo_manager.database is None:
        await async_mongo_manager.connect(ensure_indexes=apply)

    mismatches, checked, unset, orphaned = await find_mismatches()

    print(
        f"Checked {checked} user(s); {len(mismatches)} counter(s) differ, "
        f"{unset} never set (counted on first upload, left alone)."
    )
    if orphaned:
        print(f"{orphaned} document(s) belong to users that no longer exist (left alone).")
    if mismatches:
        print("user_id\tstored\tdocuments")
    for user_id, stored, actual in mismatches:
        print(f"{user_id}\t{stored}\t{actual}")

    if not mismatches:
        return 0
    if not apply:
        print("Dry run: nothing written. Re-run with --apply to set these counters.")
        return len(mismatches)

    fixed = 0
    skipped: List[str] = []
    for user_id, stored, _actual in mismatches:
        if await _apply_one(user_id, stored):
            fixed += 1
        else:
            skipped.append(user_id)
    print(f"Set {fixed} counter(s).")
    if skipped:
        print(
            f"Skipped {len(skipped)} user(s) whose counter changed during the run; "
            "run again: " + ", ".join(skipped)
        )
    return len(skipped)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recount private documents per user and fix the stored counter."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the corrected counters. Without this flag, runs as a dry run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    remaining = asyncio.run(reconcile_private_document_counters(apply=args.apply))
    # A dry run always succeeds; an apply fails if it had to skip anyone.
    sys.exit(1 if args.apply and remaining else 0)
