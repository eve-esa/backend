"""Fill ``storage_bytes_used`` from the objects under ``users/{id}/``.

Files stored before counting started were never added. Run this once per
environment after the deploy and before ``FEATURE_STORAGE_QUOTA`` is turned
on. Run it again by hand when a user's number is reported wrong. There is
no scheduled recount.

Counting stays on while this runs, so the counter is never replaced with
``$set``. That would erase a write that happened during the listing. The
adjustment is ``$inc(listed - counter_read)``:

1. Read the counter.
2. List ``users/{id}/`` and sum the current object sizes.
3. Apply the difference.

A write that is both listed and already counted is added twice. A second
run converges. A write that lands after the listing is kept, because the
difference is applied on top of the counter as it is then, not written over it.

    python -m src.commands.reconcile_storage_usage [--user ID] [--apply]

Defaults to a dry run.
"""

import argparse
import asyncio
import logging

from bson import ObjectId
from bson.errors import InvalidId

from src.config import configure_logging
from src.database.models.user import User
from src.database.mongo import async_mongo_manager
from src.services.storage_quota import (
    apply_storage_listing_delta,
    displayed_bytes,
    read_storage_listing,
)

configure_logging()
logger = logging.getLogger(__name__)


async def reconcile_storage_usage(user_id: str | None = None, apply: bool = False) -> None:
    """Print each user's listed bytes against the counter, and apply the difference.

    Args:
        user_id: One user. ``None`` walks every user.
        apply: When False (default), only print. Nothing is written.
    """
    if async_mongo_manager.database is None:
        await async_mongo_manager.connect(ensure_indexes=False)

    query: dict = {}
    if user_id is not None:
        try:
            query = {"_id": ObjectId(user_id)}
        except (InvalidId, TypeError):
            print(f"Not a user id: {user_id}")
            return

    cursor = User.get_collection().find(query, {"storage_bytes_used": 1})
    seen = 0
    async for doc in cursor:
        seen += 1
        uid = str(doc["_id"])
        try:
            counter, listed = await read_storage_listing(uid)
        except Exception:
            logger.warning("storage.reconcile_failed user=%s", uid, exc_info=True)
            print(f"  SKIPPED (listing failed): {uid}")
            continue
        delta = listed - counter
        print(
            f"  {uid}  listed={listed}  counter={counter}  "
            f"delta={delta}  shown={displayed_bytes(counter)}"
        )
        if apply:
            await apply_storage_listing_delta(uid, counter, listed)
    if seen == 0:
        print("No users matched.")
        return
    if not apply:
        print("Dry run: no counters were changed. Re-run with --apply to apply them.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill storage_bytes_used from the objects under users/{id}/."
    )
    parser.add_argument("--user", default=None, help="One user id. Default: every user.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the difference. Without this flag, runs as a dry run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(reconcile_storage_usage(user_id=args.user, apply=args.apply))
