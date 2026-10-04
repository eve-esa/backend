"""Retention for the LangGraph checkpoints kept in ``checkpointing_db``.

Both pipelines (classic in ``generate_answer``, agentic in the agents runner)
persist LangGraph state through ``langgraph-checkpoint-mongodb``. Its
``MongoDBSaver`` writes two collections and never deletes anything:

- ``checkpoints``: one document per graph step, keyed by ``thread_id``
  (the conversation id), ``checkpoint_ns`` and ``checkpoint_id``.
- ``checkpoint_writes``: the pending writes of each step, keyed the same way
  plus ``task_id`` and ``idx``.

The saver takes a ``ttl`` in seconds. With it set, every write also sets
``created_at`` to a BSON date and the saver asks for a TTL index on that field.
This module turns ``CHECKPOINT_RETENTION_DAYS`` into that ``ttl`` and owns the
TTL index, because the saver's own index check never matches an existing index
(it compares a list with a tuple) and calls ``create_index`` every time: a
changed retention then fails with an index options conflict, and the callers
silently fall back to an in-memory saver. Here a changed retention drops and
recreates the index, and ``0`` drops it.

Losing a checkpoint is safe: the app ``messages`` collection stays the source of
truth, and both pipelines rebuild history and the rolling summary from it on
every turn. A thread idle longer than the window starts from that context.

DocumentDB caveats:

- TTL deletes run in a background task, best effort: documents can outlive the
  window, more so under load, and every delete is billed I/O.
- AWS asks for expired documents to be deleted before a TTL index is created on
  an existing collection. Documents written before retention existed have no
  ``created_at`` and never expire through the index: remove them with
  ``python -m src.commands.purge_checkpoints --apply`` before the first deploy
  that turns retention on, then let the index keep up.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.config import CHECKPOINT_RETENTION_DAYS

logger = logging.getLogger(__name__)

CHECKPOINT_DB_NAME = "checkpointing_db"
CHECKPOINT_COLLECTIONS = ("checkpoints", "checkpoint_writes")
TTL_FIELD = "created_at"
SECONDS_PER_DAY = 86_400


def retention_ttl_seconds(days: int = CHECKPOINT_RETENTION_DAYS) -> Optional[int]:
    """TTL in seconds for ``days`` of retention, ``None`` when disabled (<=0)."""
    if days is None or days <= 0:
        return None
    return int(days) * SECONDS_PER_DAY


def _ttl_index(collection: Any) -> Optional[dict]:
    for index in collection.list_indexes():
        if list(index["key"].items()) == [(TTL_FIELD, 1)]:
            return index
    return None


def ensure_ttl_indexes(database: Any, ttl_seconds: Optional[int]) -> None:
    """Make the ``created_at`` TTL index of both collections match ``ttl_seconds``.

    ``None`` removes the index. The index keeps the driver's default name
    (``created_at_1``) so the saver's own ``create_index`` call matches it.
    """
    for name in CHECKPOINT_COLLECTIONS:
        collection = database[name]
        existing = _ttl_index(collection)
        if existing is not None and existing.get("expireAfterSeconds") == ttl_seconds:
            continue
        if existing is not None:
            collection.drop_index(existing["name"])
            logger.info(
                "Dropped checkpoint TTL index %s.%s (expireAfterSeconds=%s)",
                name,
                existing["name"],
                existing.get("expireAfterSeconds"),
            )
        if ttl_seconds is not None:
            collection.create_index([(TTL_FIELD, 1)], expireAfterSeconds=ttl_seconds)
            logger.info(
                "Checkpoint TTL index on %s.%s, expireAfterSeconds=%s",
                name,
                TTL_FIELD,
                ttl_seconds,
            )


def build_mongo_checkpointer(
    client: Any,
    days: int = CHECKPOINT_RETENTION_DAYS,
    db_name: str = CHECKPOINT_DB_NAME,
) -> Any:
    """Create the ``MongoDBSaver`` with the configured retention.

    Synchronous and possibly slow (an index build on a large collection): call
    it through ``asyncio.to_thread`` from async code.
    """
    from langgraph.checkpoint.mongodb import MongoDBSaver

    ttl_seconds = retention_ttl_seconds(days)
    ensure_ttl_indexes(client[db_name], ttl_seconds)
    return MongoDBSaver(client, db_name=db_name, ttl=ttl_seconds)
