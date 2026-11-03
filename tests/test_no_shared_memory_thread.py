"""LangGraph memory follows the conversation.

A turn without a conversation (``POST /generate``, the agentic runner without a
conversation id) runs on a graph with no checkpointer: it remembers nothing and
writes nothing to ``checkpointing_db``. A turn with a conversation keeps its own
thread. The streamed agentic helper refuses a turn without a conversation id.

The classic tests drive the real graph and the real ``MongoDBSaver`` on a
throwaway database; only the model is a stand-in that records what it was sent.
"""

import uuid
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage
from pymongo import MongoClient

from src.database.models.conversation import Conversation
from src.schemas.generation_request import GenerationRequest
from src.services import generate_answer as ga
from src.services.checkpoint_retention import (
    CHECKPOINT_COLLECTIONS,
    build_mongo_checkpointer,
)
from tests.conftest import _resolve_test_mongo_uri
from tests.utils.cleaner import cleanup_models
from tests.utils.utils import create_test_user_and_token

FIRST_TEXT = "question about a Sentinel-1 orbit"
SECOND_TEXT = "question about the Copernicus programme"


class _RecordingLLM:
    """Chat model stand-in: records every message list it is sent."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def bind(self, **_kwargs):
        return self

    async def ainvoke(self, messages):
        self.calls.append([str(getattr(m, "content", m)) for m in messages])
        return AIMessage(content=f"answer number {len(self.calls)}")


class _Manager:
    """LLM manager stand-in: one healthy endpoint, the fallback must not run."""

    def __init__(self, llm: _RecordingLLM) -> None:
        self.llm = llm
        self.health = SimpleNamespace(is_open=lambda _name: False)
        self.generate_answer_fallback = AsyncMock(
            side_effect=AssertionError("the fallback model must not answer")
        )

    def resolve_chain(self, _llm_type):
        return ["main"]

    def get_client_for_model(self, _llm_type):
        return self.llm


async def _no_rag(_manager, query, _context, _llm_type=None):
    return (
        ga.ShouldUseRagDecision(use_rag=False, reason="test", requery=query),
        "is-rag prompt",
        False,
    )


@pytest.fixture
def classic(monkeypatch):
    """Fresh graph singletons over a throwaway checkpoint database."""
    client = MongoClient(_resolve_test_mongo_uri())
    name = f"eve_test_ckpt_{uuid.uuid4().hex[:12]}"
    llm = _RecordingLLM()
    manager = _Manager(llm)
    error_logger = SimpleNamespace(log_error=AsyncMock())

    for attr in (
        "_compiled_graph",
        "_compiled_graph_mode",
        "_mongo_checkpointer",
        "_inmemory_compiled_graph",
        "_stateless_compiled_graph",
    ):
        monkeypatch.setattr(ga, attr, None, raising=False)
    monkeypatch.setattr(ga, "get_mongodb_uri", _resolve_test_mongo_uri)
    monkeypatch.setattr(
        ga, "build_mongo_checkpointer", partial(build_mongo_checkpointer, db_name=name)
    )
    monkeypatch.setattr(ga, "get_shared_llm_manager", lambda: manager)
    monkeypatch.setattr(ga, "get_error_logger", lambda: error_logger)
    monkeypatch.setattr(ga, "should_use_rag", _no_rag)

    def count(query: dict) -> int:
        return sum(
            client[name][collection].count_documents(query)
            for collection in CHECKPOINT_COLLECTIONS
        )

    yield SimpleNamespace(
        llm=llm, manager=manager, error_logger=error_logger, count=count
    )
    client.drop_database(name)
    client.close()


async def _post_generate(async_client, token: str, query: str):
    return await async_client.post(
        "/generate",
        json={"query": query, "public_collections": [], "k": 0},
        headers={"Authorization": f"Bearer {token}"},
    )


async def test_generate_without_a_conversation_writes_no_checkpoint(
    async_client, classic
):
    user, token = await create_test_user_and_token()
    try:
        resp = await _post_generate(async_client, token, FIRST_TEXT)

        assert resp.status_code == 200, resp.text
        assert resp.json()["answer"] == "answer number 1"
        # The primary answered through the graph: no error row, no fallback.
        classic.error_logger.log_error.assert_not_awaited()
        classic.manager.generate_answer_fallback.assert_not_awaited()
        # The Mongo saver is live in this process, and the turn wrote nothing.
        assert ga._compiled_graph_mode == "mongo"
        assert classic.count({}) == 0
    finally:
        await cleanup_models([user])


async def test_turns_without_a_conversation_are_independent(async_client, classic):
    first, first_token = await create_test_user_and_token()
    second, second_token = await create_test_user_and_token()
    try:
        assert (await _post_generate(async_client, first_token, FIRST_TEXT)).status_code == 200
        assert (await _post_generate(async_client, second_token, SECOND_TEXT)).status_code == 200

        assert len(classic.llm.calls) == 2
        second_input = "\n".join(classic.llm.calls[1])
        assert SECOND_TEXT in second_input
        assert FIRST_TEXT not in second_input
        assert "answer number 1" not in second_input
        classic.manager.generate_answer_fallback.assert_not_awaited()
        assert classic.count({}) == 0
    finally:
        await cleanup_models([first, second])


async def test_a_conversation_keeps_its_own_thread(classic):
    user, _token = await create_test_user_and_token()
    conversation = Conversation(user_id=user.id, name="thread-test")
    await conversation.save()
    try:
        for text in (FIRST_TEXT, SECOND_TEXT):
            await ga.generate_answer(
                GenerationRequest(query=text, public_collections=[], k=0),
                conversation_id=conversation.id,
                user_id=user.id,
            )

        assert classic.count({"thread_id": conversation.id}) > 0
        assert classic.count({"thread_id": {"$ne": conversation.id}}) == 0
        # Same thread on the second turn: the first turn is remembered.
        assert FIRST_TEXT in "\n".join(classic.llm.calls[1])
        classic.error_logger.log_error.assert_not_awaited()
    finally:
        await cleanup_models([conversation, user])


# ─── agentic runner ───────────────────────────────────────────────────────────

_RUNNER = "src.services.agents.core.runner"


class _ConfigRecordingGraph:
    def __init__(self) -> None:
        self.configs: list = []

    async def astream(self, *_args, **kwargs):
        self.configs.append(kwargs.get("config"))
        yield {"agent": {"messages": [AIMessage(content="agentic answer")]}}


def _patch_runner(monkeypatch, graph):
    from src.services.agents.core import runner

    agent = MagicMock()
    agent.instruction_text.return_value = ""
    agent.prompts = {}
    saver = object()
    build = MagicMock(return_value=graph)
    get_saver = AsyncMock(return_value=saver)
    for name, value in {
        "_langgraph_available": True,
        "_build_tools": AsyncMock(return_value=[]),
        "_get_agentic_checkpointer": get_saver,
        "_fetch_conversation_context": AsyncMock(return_value=([], None)),
        "_resolve_agent_graph_type": MagicMock(return_value="react"),
        "get_agent_graph": MagicMock(return_value=agent),
        "_resolve_agentic_llm_client": AsyncMock(return_value=(MagicMock(), {})),
        "_build_react_graph": build,
        "persist_policy_event": AsyncMock(),
        "persist_message_state": AsyncMock(),
        "maybe_rollup_and_trim_history": AsyncMock(),
    }.items():
        monkeypatch.setattr(runner, name, value)
    return SimpleNamespace(build=build, get_saver=get_saver, saver=saver)


@pytest.mark.no_db
async def test_agentic_without_a_conversation_has_no_checkpointer(monkeypatch):
    from src.services.agents.core.runner import generate_answer_agentic

    graph = _ConfigRecordingGraph()
    patched = _patch_runner(monkeypatch, graph)

    answer, *_rest = await generate_answer_agentic(
        GenerationRequest(query=FIRST_TEXT, agent="react"), user_id="u1"
    )

    assert answer == "agentic answer"
    assert patched.build.call_args.args[2] is None
    patched.get_saver.assert_not_awaited()
    thread = ((graph.configs[0] or {}).get("configurable") or {}).get("thread_id")
    assert thread is None


@pytest.mark.no_db
async def test_agentic_with_a_conversation_keeps_its_thread(monkeypatch):
    from src.services.agents.core.runner import generate_answer_agentic

    graph = _ConfigRecordingGraph()
    patched = _patch_runner(monkeypatch, graph)

    await generate_answer_agentic(
        GenerationRequest(query=FIRST_TEXT, agent="react"),
        user_id="u1",
        conversation_id="c1",
    )

    assert patched.build.call_args.args[2] is patched.saver
    assert graph.configs[0]["configurable"]["thread_id"] == "c1"


@pytest.mark.no_db
@pytest.mark.parametrize("conversation_id", [None, ""])
async def test_agentic_stream_needs_a_conversation(monkeypatch, conversation_id):
    from src.services.agents.core.runner import generate_answer_agentic_stream_helper

    patched = _patch_runner(monkeypatch, _ConfigRecordingGraph())

    with pytest.raises(ValueError, match="conversation id"):
        async for _event in generate_answer_agentic_stream_helper(
            GenerationRequest(query=FIRST_TEXT, agent="react"),
            conversation_id=conversation_id,
            message_id="m1",
            user_id="u1",
        ):
            pass

    patched.get_saver.assert_not_awaited()
    patched.build.assert_not_called()
