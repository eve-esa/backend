"""trace_id on the Message and in every SSE ``final`` event.

Four emit sites: the agentic stream (runner), the classic stream
(generate_answer), and the two hallucination outcomes (message router). With
telemetry on the id is the 32 hex trace id of the answer; with telemetry off
the field is present and null, and nothing else changes.
"""

import json
import re
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src import observability
from src.database.models.conversation import Conversation
from src.database.models.message import Message
from src.schemas.generation_request import GenerationRequest
from src.services.agents.core.runner import generate_answer_agentic_stream_helper
from tests.test_agentic_fallback import _FakeStreamGraph, _patched_runner
from tests.test_endpoint_failover import _FakeGraph, _new_turn, _patch_pipeline, _stream
from tests.utils.cleaner import cleanup_models
from tests.utils.utils import create_test_user_and_token

HEX32 = re.compile(r"^[0-9a-f]{32}$")


def _payloads(events: List[str]) -> List[Dict[str, Any]]:
    return [json.loads(e.removeprefix("data: ").strip()) for e in events]


def _final(events: List[str]) -> Dict[str, Any]:
    finals = [p for p in _payloads(events) if p.get("type") == "final"]
    assert len(finals) == 1, events
    return finals[0]


@pytest.fixture
def telemetry(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = observability.build_tracer_provider(exporter, batch=False)
    monkeypatch.setitem(observability._state, "tracer_provider", provider)
    yield exporter
    provider.shutdown()


@pytest.fixture
def no_telemetry(monkeypatch):
    monkeypatch.setitem(observability._state, "tracer_provider", None)


# ─── agentic stream (runner) ──────────────────────────────────────────────────


async def _agentic_events(persist: AsyncMock) -> List[str]:
    graph = _FakeStreamGraph(
        messages=[(AIMessage(content="over there"), {"langgraph_node": "agent"})]
    )
    request = GenerationRequest(query="hi", llm_type="main", agent="react")
    with _patched_runner(
        _build_react_graph=MagicMock(return_value=graph),
        persist_message_state=persist,
    ):
        return [
            event
            async for event in generate_answer_agentic_stream_helper(
                request, conversation_id="c1", message_id="m1", user_id="u1"
            )
        ]


@pytest.mark.no_db
async def test_agentic_final_event_carries_the_trace_id(telemetry):
    persist = AsyncMock()
    final = _final(await _agentic_events(persist))

    assert HEX32.match(final["trace_id"])
    (root,) = [s for s in telemetry.get_finished_spans() if s.name == "invoke_agent"]
    assert final["trace_id"] == f"{root.context.trace_id:032x}"
    assert persist.await_args.kwargs["trace_id"] == final["trace_id"]


@pytest.mark.no_db
async def test_agentic_final_event_trace_id_is_null_without_telemetry(no_telemetry):
    persist = AsyncMock()
    final = _final(await _agentic_events(persist))

    assert "trace_id" in final and final["trace_id"] is None
    assert final["answer"] == "over there"
    assert persist.await_args.kwargs["trace_id"] is None


# ─── classic stream (generate_answer) ─────────────────────────────────────────


async def test_classic_final_event_and_message_carry_the_trace_id(
    monkeypatch, telemetry
):
    user, _token = await create_test_user_and_token()
    conversation = message = None
    try:
        conversation, message = await _new_turn(user)
        graph = _FakeGraph(
            {"eve_jsc": [TimeoutError("cold start")], "main": ["Rome ", "is here."]}
        )
        _patch_pipeline(monkeypatch, graph)

        final = _final(await _stream(conversation, message))

        assert HEX32.match(final["trace_id"])
        saved = await Message.find_by_id(message.id)
        assert saved.trace_id == final["trace_id"]

        spans = telemetry.get_finished_spans()
        (root,) = [s for s in spans if s.name == "invoke_agent"]
        assert final["trace_id"] == f"{root.context.trace_id:032x}"
        assert root.attributes["gen_ai.agent.name"] == "generation_stream"
        assert root.attributes["eve.llm.answered"] == "main"
        assert root.attributes["eve.llm.chain"] == ("eve_jsc", "main", "fallback")

        # One span per candidate of the chain walk, both under the root.
        candidates = [s for s in spans if s.name == "eve.llm.candidate"]
        assert [s.attributes["eve.llm.candidate"] for s in candidates] == [
            "eve_jsc",
            "main",
        ]
        assert all(s.parent.span_id == root.context.span_id for s in candidates)
        failed, answered = candidates
        assert failed.attributes["eve.llm.outcome"] == "timeout"
        assert "eve.llm.answered" not in failed.attributes
        assert answered.attributes["eve.llm.answered"] == "main"
        assert "gen_ai.first_token" in [e.name for e in answered.events]
    finally:
        await cleanup_models(
            [doc for doc in (user, conversation, message) if doc is not None]
        )


async def test_classic_handled_failure_marks_the_root_span(monkeypatch, telemetry):
    user, _token = await create_test_user_and_token()
    conversation = message = None
    try:
        conversation, message = await _new_turn(user)
        # Text reaches the client, then the endpoint dies: the turn ends with
        # an error event, and the root must say so.
        _patch_pipeline(
            monkeypatch,
            _FakeGraph({"eve_jsc": ["Rome ", ConnectionError("reset")]}),
        )

        events = await _stream(conversation, message)

        assert _payloads(events)[-1]["type"] == "error"
        (root,) = [
            s for s in telemetry.get_finished_spans() if s.name == "invoke_agent"
        ]
        assert root.status.status_code.name == "ERROR"
        saved = await Message.find_by_id(message.id)
        assert saved.trace_id == f"{root.context.trace_id:032x}"
    finally:
        await cleanup_models(
            [doc for doc in (user, conversation, message) if doc is not None]
        )


async def test_classic_trace_id_is_null_without_telemetry(monkeypatch, no_telemetry):
    user, _token = await create_test_user_and_token()
    conversation = message = None
    try:
        conversation, message = await _new_turn(user)
        _patch_pipeline(monkeypatch, _FakeGraph({"eve_jsc": ["Rome ", "is here."]}))

        final = _final(await _stream(conversation, message))

        assert "trace_id" in final and final["trace_id"] is None
        assert final["answer"] == "Rome is here."
        saved = await Message.find_by_id(message.id)
        assert saved.trace_id is None
        assert saved.output == "Rome is here."
    finally:
        await cleanup_models(
            [doc for doc in (user, conversation, message) if doc is not None]
        )


# ─── hallucination stream (message router), both final events ───────────────


class _FakeDetector:
    """Stands in for HallucinationDetector: fixed label, scripted rewrite and answer."""

    label = 0

    def __init__(self):
        self.llm_manager = MagicMock()
        self.llm_manager.get_client_for_model.return_value = GenericFakeChatModel(
            messages=iter([AIMessage(content="grounded answer")])
        )

    async def detect(self, **kwargs):
        return self.label, "reason"

    async def rewrite_query(self, **kwargs):
        return kwargs.get("query"), "rewritten question"


async def _hallucination_final(async_client, monkeypatch, label: int) -> Dict[str, Any]:
    detector = type("_Detector", (_FakeDetector,), {"label": label})
    monkeypatch.setattr("src.routers.message.HallucinationDetector", detector)
    monkeypatch.setattr(
        "src.routers.message.setup_rag_and_context",
        AsyncMock(return_value=("", [], {}, [])),
    )
    user, token = await create_test_user_and_token()
    conversation = message = None
    try:
        conversation = Conversation(user_id=user.id, name="trace-id-hallucination")
        await conversation.save()
        message = await Message.create(
            conversation_id=conversation.id,
            input="Is Rome in Italy?",
            output="Rome is in France.",
            documents=[],
            use_rag=False,
            metadata={},
            request_input=GenerationRequest(query="Is Rome in Italy?", llm_type="main"),
        )
        response = await async_client.post(
            f"/conversations/{conversation.id}/messages/{message.id}/stream-hallucination",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        events = [line for line in response.text.split("\n\n") if line.startswith("data: ")]
        return _final(events)
    finally:
        await cleanup_models(
            [doc for doc in (user, conversation, message) if doc is not None]
        )


@pytest.mark.parametrize("label", [0, 1])
async def test_hallucination_final_events_carry_the_request_trace_id(
    async_client, monkeypatch, telemetry, label
):
    tracer = observability._state["tracer_provider"].get_tracer("test")
    # Stands in for the ASGI server span wrap_asgi opens in production.
    with tracer.start_as_current_span("POST stream-hallucination") as server_span:
        final = await _hallucination_final(async_client, monkeypatch, label)

    assert final["label"] == label
    assert final["trace_id"] == f"{server_span.get_span_context().trace_id:032x}"
    if label == 1:
        assert final["answer"] == "grounded answer"


@pytest.mark.parametrize("label", [0, 1])
async def test_hallucination_trace_id_is_null_without_telemetry(
    async_client, monkeypatch, no_telemetry, label
):
    final = await _hallucination_final(async_client, monkeypatch, label)
    assert final["label"] == label
    assert "trace_id" in final and final["trace_id"] is None
