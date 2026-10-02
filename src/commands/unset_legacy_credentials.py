"""Remove the legacy credential fields from every stored user.

Sign-in belongs to the identity provider, and the bridge that let Cognito verify
an old password is gone, so ``password_hash``, ``is_active`` and
``activation_code`` are dead data. Dropping them from the ``User`` model erases
them from a document only when that user is next saved; this sweep reaches the
accounts nobody has used since.

    python -m src.commands.unset_legacy_credentials [--apply]

Dry run by default: it prints how many documents hold each field and writes
nothing. ``--apply`` unsets all three and reports matched and modified counts.
Only counts are ever printed, never a value. Idempotent: a second ``--apply``
matches nothing.
"""

import argparse
import asyncio
import logging

from src.config import configure_logging
from src.database.models.user import User
from src.database.mongo import async_mongo_manager

configure_logging()
logger = logging.getLogger(__name__)

LEGACY_FIELDS = ("password_hash", "is_active", "activation_code")


def _holds_any_legacy_field() -> dict:
    return {"$or": [{field: {"$exists": True}} for field in LEGACY_FIELDS]}


async def unset_legacy_credentials(*, apply: bool = False) -> dict[str, int]:
    """Count, and with ``apply`` unset, the legacy fields. Returns a summary dict."""
    # Reuse an already-open connection (the test suite's isolated database)
    # instead of reconnecting to the default URI.
    if async_mongo_manager.database is None:
        await async_mongo_manager.connect()

    collection = User.get_collection()
    summary: dict[str, int] = {}
    for field in LEGACY_FIELDS:
        summary[field] = await collection.count_documents({field: {"$exists": True}})
        print(f"{field}: {summary[field]} document(s)")
    summary["documents"] = await collection.count_documents(_holds_any_legacy_field())
    print(f"Users holding at least one legacy field: {summary['documents']}")

    if not apply:
        print("Dry run: nothing written. Pass --apply to unset the fields.")
        return summary

    result = await collection.update_many(
        _holds_any_legacy_field(),
        {"$unset": {field: "" for field in LEGACY_FIELDS}},
    )
    summary["matched"] = result.matched_count
    summary["modified"] = result.modified_count
    logger.info("Legacy credential sweep: %s", summary)
    print(f"Matched: {result.matched_count}  Modified: {result.modified_count}")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unset password_hash, is_active and activation_code on every stored user."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Unset the fields. Without it the command only counts.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(unset_legacy_credentials(apply=args.apply))
