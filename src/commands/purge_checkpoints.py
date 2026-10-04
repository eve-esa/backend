"""Delete LangGraph checkpoints older than the retention window.

The TTL index (src/services/checkpoint_retention.py) expires only documents
that carry ``created_at``, and the saver writes that field only since retention
was turned on. Everything written before stays until this command removes it:

    python -m src.commands.purge_checkpoints [--days N] [--apply] [--batch-size N]

Age is read from ``_id``: every saver document is created by an upsert, so its
``_id`` is a server-generated ObjectId whose timestamp is the insert time. That
makes the cutoff an index range on ``_id`` with no extra index and no scan of
``created_at``. Defaults to a dry run that counts per collection; ``--apply``
deletes in batches of ``--batch-size`` documents, each one a ``delete_many`` on
an ``_id`` range, so an interrupted run is resumed by running it again.
``--days`` defaults to ``CHECKPOINT_RETENTION_DAYS``; ``0`` does nothing.

On DocumentDB, run it before the first deploy with retention on: AWS asks for
expired documents to be deleted before a TTL index is created on an existing
collection. Every delete is billed I/O.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from bson import ObjectId

from src.config import CHECKPOINT_RETENTION_DAYS
from src.services.checkpoint_retention import CHECKPOINT_COLLECTIONS, CHECKPOINT_DB_NAME

DEFAULT_BATCH_SIZE = 1000


def purge_checkpoints(
    database: Any,
    days: int = CHECKPOINT_RETENTION_DAYS,
    apply: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    now: Optional[datetime] = None,
) -> Dict[str, int]:
    """Count (dry run) or delete checkpoint documents inserted more than ``days`` ago.

    Returns ``{collection: count}``: documents older than the cutoff on a dry
    run, documents deleted with ``apply``. Empty when ``days`` <= 0.
    """
    if days is None or days <= 0:
        print("Checkpoint retention is disabled (days <= 0): nothing to do.")
        return {}
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=days)
    older = {"_id": {"$lt": ObjectId.from_datetime(cutoff)}}
    result: Dict[str, int] = {}

    for name in CHECKPOINT_COLLECTIONS:
        collection = database[name]
        if not apply:
            result[name] = collection.count_documents(older)
            print(f"{name}: {result[name]} document(s) older than {days} days")
            continue

        deleted = 0
        while True:
            ids = [
                doc["_id"]
                for doc in collection.find(older, {"_id": 1})
                .sort("_id", 1)
                .limit(batch_size)
            ]
            if not ids:
                break
            # Every id in [first, last] is below the cutoff, so the range is safe.
            outcome = collection.delete_many({"_id": {"$gte": ids[0], "$lte": ids[-1]}})
            deleted += outcome.deleted_count
            print(f"{name}: deleted {deleted} so far")
        result[name] = deleted
        print(f"{name}: deleted {deleted} document(s) older than {days} days")

    if not apply:
        print("Dry run: nothing was deleted. Re-run with --apply to delete.")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete LangGraph checkpoints older than N days."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=CHECKPOINT_RETENTION_DAYS,
        help=f"Retention window in days (default CHECKPOINT_RETENTION_DAYS={CHECKPOINT_RETENTION_DAYS}; 0 does nothing)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete the matched documents. Without it the command only counts.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Documents per delete (default {DEFAULT_BATCH_SIZE})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    from pymongo import MongoClient

    from src.utils.helpers import get_mongodb_uri

    args = _parse_args()
    client = MongoClient(get_mongodb_uri())
    try:
        purge_checkpoints(
            client[CHECKPOINT_DB_NAME],
            days=args.days,
            apply=args.apply,
            batch_size=args.batch_size,
        )
    finally:
        client.close()
