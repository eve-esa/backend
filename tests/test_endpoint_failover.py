"""Endpoint failover on the classic streaming path.

A dead endpoint must cost one request, not every request: the chain moves to
the next candidate while nothing has reached the client yet, opens the failed
endpoint's circuit so the following requests skip it, and records the walk in
``metadata.endpoint``. Once tokens are on the wire the turn is committed to the
endpoint that started it.
"""

import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import httpx
import pytest

from src.config import EVE_JSC_MODEL_NAME, FALLBACK_MODEL_NAME, MAIN_MODEL_NAME
from src.core.llm_manager import LLMManager
from src.database.models.conversation import Conversation
from src.database.models.message import Message
from src.schemas.generation_request import GenerationRequest
from src.services.generate_answer import (
    ShouldUseRagDecision,
    generate_answer_stream_generator_helper,
)
from tests.utils.cleaner import cleanup_models
from tests.utils.utils import create_test_user_and_token

_MODULE = "src.services.generate_answer"


class _FakeGraph:
    """Compiled graph whose stream behaviour is picked by the state's llm_type.

    A plan is a list of token strings; an exception in it is raised at that
    point in the stream, so "fails before the first token" and "fails after
    three" are the same fixture.
    """

    def __init__(self, plans: Dict[str, List[Any]]) -> None:
        self.plans = plans
        self.seen: List[str] = []

    def astream(self, state, config=None, stream_mode=None):
        candidate = state.get("llm_type")
        self.seen.append(candidate)
        steps = self.plans.get(candidate, [])

        async def _stream():
            for step in steps:
                if isinstance(step, BaseException):
                    raise step
                yield SimpleNamespace(content=step), {}

        return _stream()

    async def aclose(self) -> None:
        return None


async def _should_use_rag(llm_manager, query, conversation, llm_type=None):
    return (
        ShouldUseRagDecision(use_rag=False, reason="test", requery=""),
        "is-rag prompt",
        False,
    )


def _patch_pipeline(
    monkeypatch,
    graph: _FakeGraph,
    *,
    configured=("eve_jsc", "main", "fallback"),
    fallback_tokens=("fallback answer",),
) -> LLMManager:
    """Wire the streaming helper to a real chain resolver over a fake graph."""
    monkeypatch.setattr(
        "src.core.llm_manager.EVE_ENDPOINT_ORDER", "eve_jsc,main,fallback"
    )
    manager = LLMManager()
    monkeypatch.setattr(manager, "_is_configured", lambda name: name in configured)

    async def _fallback_stream(**kwargs):
        for token in fallback_tokens:
            yield token, "fallback generation prompt"

    monkeypatch.setattr(manager, "generate_answer_fallback_stream", _fallback_stream)
    monkeypatch.setattr(f"{_MODULE}.get_shared_llm_manager", lambda: manager)
    monkeypatch.setattr(
        f"{_MODULE}._get_or_create_compiled_graph",
        AsyncMock(return_value=(graph, "memory")),
    )
    monkeypatch.setattr(f"{_MODULE}.should_use_rag", _should_use_rag)
    monkeypatch.setattr(f"{_MODULE}.maybe_rollup_and_trim_history", AsyncMock())
    return manager


async def _new_turn(user) -> tuple[Conversation, Message]:
    conversation = Conversation(user_id=user.id, name="failover-test")
    await conversation.save()
    message = await Message.create(
        conversation_id=conversation.id,
        input="What is the bounding box of Rome?",
        output="",
        documents=[],
        use_rag=False,
        metadata={},
    )
    return conversation, message


async def _stream(
    conversation: Conversation, message: Message, *, llm_type: Optional[str] = None
) -> List[str]:
    request = GenerationRequest(
        query="What is the bounding box of Rome?", llm_type=llm_type
    )
    return [
        event
        async for event in generate_answer_stream_generator_helper(
            request,
            conversation.id,
            message.id,
            "json",
            None,
            None,
            "test-user",
        )
    ]


def _tokens(events: List[str]) -> List[str]:
    payloads = [json.loads(event.removeprefix("data: ").strip()) for event in events]
    return [p["content"] for p in payloads if p.get("type") == "token"]


def _event_types(events: List[str]) -> List[str]:
    return [
        json.loads(event.removeprefix("data: ").strip()).get("type")
        for event in events
    ]


@pytest.mark.asyncio
async def test_failure_before_the_first_token_moves_to_the_next_endpoint(
    monkeypatch,
):
    user, _token = await create_test_user_and_token()
    conversation = message = None
    try:
        conversation, message = await _new_turn(user)
        graph = _FakeGraph(
            {
                "eve_jsc": [TimeoutError("cold start")],
                "main": ["Rome ", "is here."],
            }
        )
        manager = _patch_pipeline(monkeypatch, graph)

        events = await _stream(conversation, message)

        assert "error" not in _event_types(events)
        assert _tokens(events) == ["Rome ", "is here."]
        assert graph.seen == ["eve_jsc", "main"]

        saved = await Message.find_by_id(message.id)
        assert saved.output == "Rome is here."
        assert saved.metadata["generated_model_name"] == MAIN_MODEL_NAME
        endpoint = saved.metadata["endpoint"]
        assert endpoint["requested"] is None
        assert endpoint["chain"] == ["eve_jsc", "main", "fallback"]
        assert endpoint["answered"] == "main"
        assert endpoint["attempts"] == [
            {"llm_type": "eve_jsc", "outcome": "timeout"}
        ]
        assert endpoint["substituted"] is False
        assert manager.health.is_open("eve_jsc") is True
    finally:
        await cleanup_models(
            [doc for doc in (user, conversation, message) if doc is not None]
        )


@pytest.mark.asyncio
async def test_an_open_circuit_keeps_the_next_request_off_the_dead_endpoint(
    monkeypatch,
):
    user, _token = await create_test_user_and_token()
    conversation = message = second_message = None
    try:
        conversation, message = await _new_turn(user)
        graph = _FakeGraph(
            {
                "eve_jsc": [TimeoutError("cold start")],
                "main": ["Rome ", "is here."],
            }
        )
        _patch_pipeline(monkeypatch, graph)

        await _stream(conversation, message)
        second_message = await Message.create(
            conversation_id=conversation.id,
            input="And Milan?",
            output="",
            documents=[],
            use_rag=False,
            metadata={},
        )
        graph.seen.clear()
        await _stream(conversation, second_message)

        assert graph.seen == ["main"]
        saved = await Message.find_by_id(second_message.id)
        endpoint = saved.metadata["endpoint"]
        assert endpoint["chain"] == ["main", "eve_jsc", "fallback"]
        assert endpoint["circuit_open"] == ["eve_jsc"]
        assert endpoint["answered"] == "main"
        assert endpoint["attempts"] == []
    finally:
        await cleanup_models(
            [
                doc
                for doc in (user, conversation, message, second_message)
                if doc is not None
            ]
        )


@pytest.mark.asyncio
async def test_an_explicit_endpoint_falls_back_and_says_it_was_substituted(
    monkeypatch,
):
    user, _token = await create_test_user_and_token()
    conversation = message = None
    try:
        conversation, message = await _new_turn(user)
        graph = _FakeGraph({"eve_jsc": [httpx.ConnectError("connection refused")]})
        manager = _patch_pipeline(monkeypatch, graph)

        events = await _stream(conversation, message, llm_type="eve_jsc")

        assert "error" not in _event_types(events)
        assert _tokens(events) == ["fallback answer"]
        # An explicit pick is never promoted to another primary endpoint.
        assert graph.seen == ["eve_jsc"]

        saved = await Message.find_by_id(message.id)
        assert saved.metadata["generated_model_name"] == FALLBACK_MODEL_NAME
        endpoint = saved.metadata["endpoint"]
        assert endpoint["requested"] == "eve_jsc"
        assert endpoint["chain"] == ["eve_jsc", "fallback"]
        assert endpoint["answered"] == "fallback"
        assert endpoint["substituted"] is True
        assert endpoint["attempts"] == [
            {"llm_type": "eve_jsc", "outcome": "upstream_error"}
        ]
        assert manager.health.is_open("eve_jsc") is True
    finally:
        await cleanup_models(
            [doc for doc in (user, conversation, message) if doc is not None]
        )


@pytest.mark.asyncio
async def test_a_failure_after_the_first_token_never_fails_over(monkeypatch):
    user, _token = await create_test_user_and_token()
    conversation = message = None
    try:
        conversation, message = await _new_turn(user)
        graph = _FakeGraph(
            {
                "eve_jsc": ["Rome ", TimeoutError("read timeout")],
                "main": ["a second answer nobody asked for"],
            }
        )
        _patch_pipeline(monkeypatch, graph)

        events = await _stream(conversation, message)

        assert _tokens(events) == ["Rome "]
        assert _event_types(events)[-1] == "error"
        # No second candidate, and no fallback re-answer appended to the partial.
        assert graph.seen == ["eve_jsc"]

        saved = await Message.find_by_id(message.id)
        assert saved.output == "Rome "
        assert saved.metadata["error"]["code"] == "timeout"
    finally:
        await cleanup_models(
            [doc for doc in (user, conversation, message) if doc is not None]
        )


@pytest.mark.asyncio
async def test_an_empty_answer_is_not_an_endpoint_failure(monkeypatch):
    user, _token = await create_test_user_and_token()
    conversation = message = None
    try:
        conversation, message = await _new_turn(user)
        graph = _FakeGraph({"eve_jsc": [""]})
        manager = _patch_pipeline(monkeypatch, graph)

        events = await _stream(conversation, message)

        assert "error" not in _event_types(events)
        assert _tokens(events) == []
        assert graph.seen == ["eve_jsc"]

        saved = await Message.find_by_id(message.id)
        assert saved.output == ""
        endpoint = saved.metadata["endpoint"]
        assert endpoint["answered"] == "eve_jsc"
        assert endpoint["attempts"] == []
        assert saved.metadata["generated_model_name"] == EVE_JSC_MODEL_NAME
        assert manager.health.is_open("eve_jsc") is False
    finally:
        await cleanup_models(
            [doc for doc in (user, conversation, message) if doc is not None]
        )
