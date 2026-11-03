"""DELETE /conversations/{id} also deletes the thread's LangGraph checkpoints."""

import asyncio
import logging
import operator
import time
import uuid
from types import SimpleNamespace
from typing import Annotated, TypedDict

import pytest
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.graph import END, START, StateGraph
from pymongo import MongoClient

from src.database.models.conversation import Conversation
from src.database.models.message import Message
from src.services import checkpoint_retention
from src.services.checkpoint_retention import (
    CHECKPOINT_COLLECTIONS,
    build_mongo_checkpointer,
    delete_thread_checkpoints,
)
from tests.conftest import _resolve_test_mongo_uri
from tests.utils.cleaner import cleanup_models
from tests.utils.utils import create_test_user_and_token

USER_TEXT = "my private question about Sentinel-2"
LOGGER = "src.services.checkpoint_retention"


@pytest.fixture
def checkpoints(monkeypatch):
    """A throwaway checkpoint database, seeded through the real saver."""
    client = MongoClient(_resolve_test_mongo_uri())
    name = f"eve_test_checkpoints_{uuid.uuid4().hex[:12]}"
    monkeypatch.setattr(checkpoint_retention, "CHECKPOINT_DB_NAME", name)
    saver = build_mongo_checkpointer(client, db_name=name)

    def seed(thread_id: str) -> None:
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        saved = saver.put(config, empty_checkpoint(), {}, {})
        saver.put_writes(saved, [("messages", USER_TEXT)], "task-1")

    def count(thread_id: str) -> dict:
        return {
            collection: client[name][collection].count_documents(
                {"thread_id": thread_id}
            )
            for collection in CHECKPOINT_COLLECTIONS
        }

    yield SimpleNamespace(seed=seed, count=count, saver=saver, database=client[name])
    client.drop_database(name)
    client.close()


class _Collection:
    def __init__(self, real, fail, hang=False):
        self._real = real
        self._fail = fail
        self._hang = hang

    async def delete_many(self, query):
        if self._hang:
            await asyncio.sleep(60)
        if self._fail:
            # Mongo errors can quote documents: the log must not repeat this.
            raise RuntimeError(f"write failed on {USER_TEXT}")
        return await self._real.delete_many(query)


class _Client:
    """The real Motor client, with delete_many failing on the named collections."""

    def __init__(self, real, failing, hanging=()):
        self._real = real
        self._failing = failing
        self._hanging = hanging

    def __getitem__(self, db_name):
        database = self._real[db_name]

        class _Database:
            def __getitem__(_, name):
                return _Collection(
                    database[name], name in self._failing, name in self._hanging
                )

        return _Database()


def _break_collections(monkeypatch, *failing):
    manager = checkpoint_retention.async_mongo_manager
    monkeypatch.setattr(manager, "client", _Client(manager.client, failing))


async def test_delete_removes_the_thread_from_both_collections(
    async_client, checkpoints
):
    seed, count = checkpoints.seed, checkpoints.count
    user, token = await create_test_user_and_token()
    conversation = await Conversation.create(user_id=user.id, name="Memory")
    other_thread = uuid.uuid4().hex
    try:
        seed(conversation.id)
        seed(other_thread)
        assert all(count(conversation.id).values())

        resp = await async_client.delete(
            f"/conversations/{conversation.id}",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert resp.status_code == 200
        assert not any(count(conversation.id).values())
        assert all(count(other_thread).values())
    finally:
        await cleanup_models([user, conversation])


async def test_failed_checkpoint_delete_still_answers_200_and_warns_once(
    async_client, checkpoints, monkeypatch, caplog
):
    seed, count = checkpoints.seed, checkpoints.count
    user, token = await create_test_user_and_token()
    conversation = await Conversation.create(user_id=user.id, name="Memory")
    try:
        seed(conversation.id)
        _break_collections(monkeypatch, *CHECKPOINT_COLLECTIONS)

        with caplog.at_level(logging.WARNING, logger=LOGGER):
            resp = await async_client.delete(
                f"/conversations/{conversation.id}",
                headers={"Authorization": f"Bearer {token}"},
            )

        assert resp.status_code == 200
        assert await Conversation.find_by_id(conversation.id) is None
        records = [r for r in caplog.records if r.name == LOGGER]
        assert [r.levelno for r in records] == [logging.WARNING]
        message = records[0].getMessage()
        assert message.startswith("checkpoint.delete_failed")
        assert conversation.id in message
        assert USER_TEXT not in caplog.text
        assert records[0].exc_info is None
    finally:
        monkeypatch.undo()
        await cleanup_models([user, conversation])


async def test_one_failing_collection_does_not_stop_the_other(
    checkpoints, monkeypatch, caplog
):
    seed, count = checkpoints.seed, checkpoints.count
    thread = uuid.uuid4().hex
    seed(thread)
    _break_collections(monkeypatch, "checkpoints")

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert await delete_thread_checkpoints(thread) is False

    remaining = count(thread)
    assert remaining["checkpoints"] >= 1
    assert remaining["checkpoint_writes"] == 0
    assert len([r for r in caplog.records if r.name == LOGGER]) == 1


async def test_a_thread_without_checkpoints_is_not_a_failure(checkpoints, caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert await delete_thread_checkpoints(uuid.uuid4().hex) is True
    assert not [r for r in caplog.records if r.name == LOGGER]


class _State(TypedDict):
    messages: Annotated[list, operator.add]


def _graph_with_subgraph(saver):
    def answer(state):
        return {"messages": ["answer"]}

    inner = StateGraph(_State)
    inner.add_node("answer", answer)
    inner.add_edge(START, "answer")
    inner.add_edge("answer", END)

    outer = StateGraph(_State)
    outer.add_node("agent", inner.compile())
    outer.add_edge(START, "agent")
    outer.add_edge("agent", END)
    return outer.compile(checkpointer=saver)


async def test_a_real_graph_run_leaves_nothing_after_the_delete(checkpoints):
    """Documents as the pipelines write them, subgraph namespaces included."""
    thread = uuid.uuid4().hex
    _graph_with_subgraph(checkpoints.saver).invoke(
        {"messages": [USER_TEXT]}, {"configurable": {"thread_id": thread}}
    )
    documents = checkpoints.database["checkpoints"]
    assert documents.count_documents({"thread_id": thread, "checkpoint_ns": ""})
    assert documents.count_documents(
        {"thread_id": thread, "checkpoint_ns": {"$ne": ""}}
    ), "the subgraph wrote no namespaced checkpoint, so this test proves nothing"

    assert await delete_thread_checkpoints(thread) is True

    assert not any(checkpoints.count(thread).values())


async def test_a_hung_delete_is_bounded_and_still_answers_200(
    async_client, checkpoints, monkeypatch, caplog
):
    monkeypatch.setattr(checkpoint_retention, "CHECKPOINT_DELETE_TIMEOUT_S", 0.1)
    user, token = await create_test_user_and_token()
    conversation = await Conversation.create(user_id=user.id, name="Memory")
    try:
        manager = checkpoint_retention.async_mongo_manager
        monkeypatch.setattr(
            manager, "client", _Client(manager.client, (), CHECKPOINT_COLLECTIONS)
        )
        started = time.monotonic()
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            resp = await async_client.delete(
                f"/conversations/{conversation.id}",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 200
        assert time.monotonic() - started < 5
        records = [r for r in caplog.records if r.name == LOGGER]
        assert len(records) == 1
        assert "checkpoint.delete_failed" in records[0].getMessage()
        assert "error=TimeoutError" in records[0].getMessage()
    finally:
        monkeypatch.undo()
        await cleanup_models([user, conversation])


async def test_mongo_not_connected_is_logged_not_raised(monkeypatch, caplog):
    monkeypatch.setattr(checkpoint_retention.async_mongo_manager, "client", None)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert await delete_thread_checkpoints(uuid.uuid4().hex) is False
    assert len([r for r in caplog.records if r.name == LOGGER]) == 1


async def test_refused_delete_leaves_the_checkpoints(async_client, checkpoints):
    owner, owner_token = await create_test_user_and_token()
    other, other_token = await create_test_user_and_token()
    conversation = await Conversation.create(user_id=owner.id, name="Memory")
    try:
        checkpoints.seed(conversation.id)

        resp = await async_client.delete(
            f"/conversations/{conversation.id}",
            headers={"Authorization": f"Bearer {other_token}"},
        )

        assert resp.status_code == 403
        assert all(checkpoints.count(conversation.id).values())
    finally:
        await cleanup_models([owner, other, conversation])


async def test_delete_removes_messages_with_the_checkpoints(async_client, checkpoints):
    user, token = await create_test_user_and_token()
    conversation = await Conversation.create(user_id=user.id, name="Memory")
    try:
        checkpoints.seed(conversation.id)
        await Message.create(
            conversation_id=conversation.id, input=USER_TEXT, output="answer"
        )

        resp = await async_client.delete(
            f"/conversations/{conversation.id}",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert resp.status_code == 200
        assert await Message.count_documents({"conversation_id": conversation.id}) == 0
        assert not any(checkpoints.count(conversation.id).values())
    finally:
        await cleanup_models([user, conversation])
