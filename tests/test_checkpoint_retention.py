"""Retention for LangGraph checkpoints: TTL index, saver fields, purge command."""

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from bson import ObjectId
from langgraph.checkpoint.base import empty_checkpoint
from pymongo import MongoClient

from src.commands.purge_checkpoints import purge_checkpoints
from src.services import checkpoint_retention
from src.services.checkpoint_retention import (
    CHECKPOINT_COLLECTIONS,
    build_mongo_checkpointer,
    retention_ttl_seconds,
)
from tests.conftest import _resolve_test_mongo_uri

pytestmark = pytest.mark.no_db

NOW = datetime.now(timezone.utc)


@pytest.fixture
def client():
    mongo = MongoClient(_resolve_test_mongo_uri())
    yield mongo
    mongo.close()


@pytest.fixture
def db_name(client):
    name = f"eve_test_checkpoints_{uuid.uuid4().hex[:12]}"
    yield name
    client.drop_database(name)


def _ttl_indexes(collection):
    return [i for i in collection.list_indexes() if list(i["key"].items()) == [("created_at", 1)]]


def _oid(days_ago: float, offset_s: int = 0) -> ObjectId:
    return ObjectId.from_datetime(NOW - timedelta(days=days_ago) + timedelta(seconds=offset_s))


def _doc(oid: ObjectId, thread_id: str, i: int) -> dict:
    # Unique on the saver's compound keys, no created_at: a pre-retention document.
    return {
        "_id": oid,
        "thread_id": thread_id,
        "checkpoint_ns": "",
        "checkpoint_id": f"{thread_id}-{i}",
        "task_id": "t",
        "idx": 0,
    }


def _seed(db, old: int, recent: int) -> None:
    for name in CHECKPOINT_COLLECTIONS:
        db[name].insert_many(
            [_doc(_oid(45, i), "legacy", i) for i in range(old)]
            + [_doc(_oid(1, i), "recent", i) for i in range(recent)]
        )


def _put_one(saver, thread_id="t1"):
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    saved = saver.put(config, empty_checkpoint(), {}, {})
    saver.put_writes(saved, [("messages", "hello")], "task-1")


def test_ttl_seconds_from_days():
    assert retention_ttl_seconds(30) == 30 * 86_400
    assert retention_ttl_seconds(0) is None
    assert retention_ttl_seconds(-5) is None


def test_ttl_index_on_both_collections_with_expiry(client, db_name):
    saver = build_mongo_checkpointer(client, days=30, db_name=db_name)

    assert saver.ttl == 30 * 86_400
    for name in CHECKPOINT_COLLECTIONS:
        indexes = _ttl_indexes(client[db_name][name])
        assert len(indexes) == 1
        assert indexes[0]["expireAfterSeconds"] == 30 * 86_400
        assert indexes[0]["name"] == "created_at_1"


def test_saver_writes_created_at_as_bson_date(client, db_name):
    saver = build_mongo_checkpointer(client, days=30, db_name=db_name)
    _put_one(saver)

    for name in CHECKPOINT_COLLECTIONS:
        doc = client[db_name][name].find_one({"thread_id": "t1"})
        assert isinstance(doc["created_at"], datetime)
        assert isinstance(doc["_id"], ObjectId)


def test_changed_retention_recreates_the_index(client, db_name):
    build_mongo_checkpointer(client, days=30, db_name=db_name)
    # The saver alone raises an index options conflict here.
    saver = build_mongo_checkpointer(client, days=7, db_name=db_name)

    assert saver.ttl == 7 * 86_400
    for name in CHECKPOINT_COLLECTIONS:
        indexes = _ttl_indexes(client[db_name][name])
        assert [i["expireAfterSeconds"] for i in indexes] == [7 * 86_400]


def test_same_retention_twice_is_a_no_op(client, db_name):
    build_mongo_checkpointer(client, days=30, db_name=db_name)
    build_mongo_checkpointer(client, days=30, db_name=db_name)

    for name in CHECKPOINT_COLLECTIONS:
        assert len(_ttl_indexes(client[db_name][name])) == 1


def test_zero_disables_ttl_and_drops_an_existing_index(client, db_name):
    build_mongo_checkpointer(client, days=30, db_name=db_name)
    saver = build_mongo_checkpointer(client, days=0, db_name=db_name)
    _put_one(saver)

    assert saver.ttl is None
    for name in CHECKPOINT_COLLECTIONS:
        assert _ttl_indexes(client[db_name][name]) == []
        assert "created_at" not in client[db_name][name].find_one({"thread_id": "t1"})


def test_documents_without_created_at_are_left_to_the_purge(client, db_name):
    db = client[db_name]
    _seed(db, old=1, recent=0)
    build_mongo_checkpointer(client, days=30, db_name=db_name)

    for name in CHECKPOINT_COLLECTIONS:
        legacy = db[name].find_one({"thread_id": "legacy"})
        assert legacy is not None and "created_at" not in legacy

    assert purge_checkpoints(db, days=30) == {n: 1 for n in CHECKPOINT_COLLECTIONS}
    assert purge_checkpoints(db, days=30, apply=True) == {n: 1 for n in CHECKPOINT_COLLECTIONS}
    for name in CHECKPOINT_COLLECTIONS:
        assert db[name].count_documents({"thread_id": "legacy"}) == 0


def test_purge_dry_run_counts_and_deletes_nothing(client, db_name):
    db = client[db_name]
    _seed(db, old=3, recent=2)

    assert purge_checkpoints(db, days=30) == {n: 3 for n in CHECKPOINT_COLLECTIONS}
    for name in CHECKPOINT_COLLECTIONS:
        assert db[name].count_documents({}) == 5


def test_purge_apply_deletes_only_older_in_batches_and_is_idempotent(client, db_name):
    db = client[db_name]
    _seed(db, old=5, recent=2)
    saver = build_mongo_checkpointer(client, days=30, db_name=db_name)
    _put_one(saver, thread_id="live")

    deleted = purge_checkpoints(db, days=30, apply=True, batch_size=2)

    assert deleted == {n: 5 for n in CHECKPOINT_COLLECTIONS}
    for name in CHECKPOINT_COLLECTIONS:
        assert db[name].count_documents({"thread_id": "legacy"}) == 0
        assert db[name].count_documents({"thread_id": "recent"}) == 2
        assert db[name].count_documents({"thread_id": "live"}) == 1
    assert purge_checkpoints(db, days=30, apply=True) == {n: 0 for n in CHECKPOINT_COLLECTIONS}


def test_purge_with_zero_days_does_nothing(client, db_name):
    db = client[db_name]
    _seed(db, old=2, recent=0)

    assert purge_checkpoints(db, days=0, apply=True) == {}
    for name in CHECKPOINT_COLLECTIONS:
        assert db[name].count_documents({}) == 2


async def test_agentic_runner_builds_the_saver_through_retention(monkeypatch):
    from src.services.agents.core import runner

    built = []

    def fake_build(mongo_client):
        built.append(mongo_client)
        return "saver"

    monkeypatch.setattr(runner, "build_mongo_checkpointer", fake_build)
    monkeypatch.setattr(runner, "_agentic_checkpointer", None)
    monkeypatch.setattr(runner, "get_mongodb_uri", _resolve_test_mongo_uri)

    assert await runner._get_agentic_checkpointer() == "saver"
    assert len(built) == 1
    built[0].close()


def test_default_days_come_from_config():
    from src import config

    assert checkpoint_retention.build_mongo_checkpointer.__defaults__[0] == config.CHECKPOINT_RETENTION_DAYS
