"""Add the large JSC model to the picker of the database this connects to.

``provider_models.yaml`` seeds only empty collections, and the same file would
seed the row into every new environment, so a model offered in one environment
gets its row here. The row only controls what the picker lists: the chain still
needs ``JSC_LARGE_MODEL_NAME`` set, and without it /models hides the row.

    python -m src.commands.register_jsc_large_model [--apply] [--disable]

Defaults to a dry run that prints what would change, which works with the reader
credential. ``--apply`` writes and needs the writer. ``--disable`` hides the row
again. Re-running changes nothing. The picker refreshes within the catalog cache
TTL (PROVIDER_CATALOG_CACHE_TTL_SECONDS).

``--disable`` is not a rollback: while the alias is set, a client can still send
``llm_type: jsc_large``. To turn the model off, blank ``JSC_LARGE_MODEL_NAME``.
"""

import argparse
import asyncio
import logging
import sys

from pymongo.errors import OperationFailure

from src.config import configure_logging
from src.database.models.catalog_platform_model import CatalogPlatformModelDoc
from src.database.mongo import async_mongo_manager

configure_logging()
logger = logging.getLogger(__name__)

UNAUTHORIZED_CODE = 13
CATALOG_ID = "jsc-mistral-large"
LLM_TYPE = "jsc_large"
DISPLAY_NAME = "Mistral Large 4"
DESCRIPTION = "Mistral Large 4 served by the Jülich Supercomputing Centre."


async def register_jsc_large_model(apply: bool = False, disable: bool = False) -> None:
    """Create or update the catalog row; only reports unless ``apply``."""
    # Reuse an already-open connection (the test suite's isolated database).
    if async_mongo_manager.database is None:
        await async_mongo_manager.connect(ensure_indexes=apply)

    rows = CatalogPlatformModelDoc.get_primary_collection()
    existing = await rows.find_one({"catalog_id": CATALOG_ID})
    enabled = not disable
    database = async_mongo_manager.database.name

    if existing is not None and existing.get("enabled") == enabled:
        print(f"{CATALOG_ID} already enabled={enabled} in {database}: nothing to do.")
        return
    if existing is None and disable:
        print(f"{CATALOG_ID} is not in {database}: nothing to disable.")
        return

    action = "update" if existing else "create"
    print(f"Would {action} {CATALOG_ID} (llm_type={LLM_TYPE}, enabled={enabled}) in {database}.")
    if not apply:
        print("Dry run: nothing written. Re-run with --apply to write it.")
        return

    if existing:
        await rows.update_one(
            {"catalog_id": CATALOG_ID}, {"$set": {"enabled": enabled}}
        )
    else:
        # Last in the picker: the frontend defaults to the first entry.
        last = await rows.find_one({}, sort=[("sort_order", -1)])
        await CatalogPlatformModelDoc(
            catalog_id=CATALOG_ID,
            llm_type=LLM_TYPE,
            display_name=DISPLAY_NAME,
            description=DESCRIPTION,
            enabled=enabled,
            sort_order=int(last["sort_order"]) + 1 if last else 0,
        ).save()
    print(f"Done: {CATALOG_ID} enabled={enabled} in {database}.")


async def main(apply: bool = False, disable: bool = False) -> int:
    """Run the command; a missing permission is a message, not a traceback."""
    try:
        await register_jsc_large_model(apply=apply, disable=disable)
    except OperationFailure as exc:
        if exc.code != UNAUTHORIZED_CODE:
            raise
        print("Not authorized: --apply needs the writer credential.")
        return 1
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add the large JSC model to the picker.")
    parser.add_argument("--apply", action="store_true", help="Write. Default is a dry run.")
    parser.add_argument("--disable", action="store_true", help="Hide the row from the picker.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(asyncio.run(main(apply=args.apply, disable=args.disable)))
