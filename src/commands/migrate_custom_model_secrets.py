"""One-time migration: move legacy Secrets Manager-backed custom model API
keys into the envelope-encrypted ``encrypted_key`` blob on their Mongo row.

For each ``user_custom_models`` row that still has ``secret_arn`` set, this:

    1. Reads the plaintext API key from the Secrets Manager secret (unless
       ``encrypted_key`` is already populated, e.g. a retry after step 3
       failed on a previous run).
    2. Encrypts it into ``encrypted_key`` via the same cipher the app uses
       going forward (src/services/custom_model_cipher.py) and saves the row.
    3. Deletes the now-redundant Secrets Manager secret and clears
       ``secret_arn``.

Step 2 is only committed to Mongo *before* step 3 deletes anything external,
so a crash mid-run never loses a key: worst case, a secret lingers in
Secrets Manager and gets cleaned up on the next run. Idempotent either way --
safe to re-run after a partial failure.

Usage:

    python -m src.commands.migrate_custom_model_secrets [--dry-run]
"""

import argparse
import asyncio
import logging

from src.config import configure_logging
from src.database.models.user_custom_model import UserCustomModel
from src.database.mongo import async_mongo_manager
from src.services.custom_model_secrets import (
    create_custom_model_secret,
    delete_legacy_secret,
    read_legacy_secret_value,
)

configure_logging()
logger = logging.getLogger(__name__)


async def migrate_custom_model_secrets(*, dry_run: bool = False) -> dict[str, int]:
    """Re-encrypt every un-migrated custom model row. Returns a summary dict."""
    if async_mongo_manager.database is None:
        await async_mongo_manager.connect()

    collection = UserCustomModel.get_collection()
    cursor = collection.find({"secret_arn": {"$exists": True, "$nin": [None, ""]}})

    summary = {"migrated": 0, "failed": 0, "would_migrate": 0}
    async for doc in cursor:
        model = UserCustomModel.from_dict(doc)

        if dry_run:
            summary["would_migrate"] += 1
            logger.info("[dry-run] would migrate model_id=%s", model.id)
            continue

        try:
            if not model.encrypted_key:
                api_key = await read_legacy_secret_value(model.secret_arn)
                model.encrypted_key = await create_custom_model_secret(
                    user_id=model.user_id,
                    provider_id=model.provider_id,
                    model_id=model.id,
                    api_key=api_key,
                )
                await model.save()

            await delete_legacy_secret(model.secret_arn)
            model.secret_arn = None
            await model.save()

            summary["migrated"] += 1
            logger.info("Migrated custom model secret model_id=%s", model.id)
        except Exception:
            summary["failed"] += 1
            logger.exception("Failed to migrate custom model secret model_id=%s", model.id)

    logger.info("Migration summary: %s", summary)
    if dry_run:
        print(f"Would migrate: {summary['would_migrate']}")
    else:
        print(f"Migrated: {summary['migrated']}  Failed: {summary['failed']}")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-encrypt legacy Secrets Manager custom model API keys into "
            "encrypted_key and delete the now-redundant secrets."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log/count rows that would be migrated without writing or deleting anything.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(migrate_custom_model_secrets(dry_run=args.dry_run))
