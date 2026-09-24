"""Request ids on every span, the agent root span, trace id helpers, GenAI init.

``ContextAttributesSpanProcessor`` runs inside the production provider from
``build_tracer_provider``, so these spans go through the same processors and
redacting exporter the server uses.
"""

import asyncio
import os
import re

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src import observability
from src.observability.context import (
    ContextAttributesSpanProcessor,
    agent_span,
    child_span,
    current_trace_id,
    set_llm_attributes,
    span_trace_id,
)
from src.services.generate_answer import build_endpoint_metadata
from src.utils.error_logger import (
    conversation_id_context,
    message_id_context,
    set_conversation_context,
    set_message_context,
    set_user_context,
    user_id_context,
)

pytestmark = pytest.mark.no_db

HEX32 = re.compile(r"^[0-9a-f]{32}$")
CONVERSATION = "65f0c0ffee00000000000001"
MESSAGE = "65f0c0ffee00000000000002"
USER = "65f0c0ffee00000000000003"
EMAIL = "someone.real@example.org"


@pytest.fixture
def exporter(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = observability.build_tracer_provider(exporter, batch=False)
    monkeypatch.setitem(observability._state, "tracer_provider", provider)
    yield exporter
    provider.shutdown()


@pytest.fixture(autouse=True)
def _clean_request_context():
    tokens = [
        (conversation_id_context, conversation_id_context.set(None)),
        (message_id_context, message_id_context.set(None)),
        (user_id_context, user_id_context.set(None)),
    ]
    yield
    for var, token in tokens:
        var.reset(token)


def test_every_span_of_a_request_carries_session_and_user(exporter):
    # What the message routers do before generation starts.
    set_user_context(USER)
    set_conversation_context(CONVERSATION)
    set_message_context(MESSAGE)
    tracer = observability._state["tracer_provider"].get_tracer("test")

    with agent_span("agentic_generation_stream"):
        with tracer.start_as_current_span("LangGraph.workflow"):
            with tracer.start_as_current_span("chat model"):
                pass
        with child_span("mcp.call_tool"):
            pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 4
    for span in spans:
        assert span.attributes["session.id"] == CONVERSATION, span.name
        assert span.attributes["gen_ai.conversation.id"] == CONVERSATION, span.name
        assert span.attributes["user.id"] == USER, span.name
        assert span.attributes["eve.message_id"] == MESSAGE, span.name
        assert EMAIL not in repr(dict(span.attributes))


def test_user_id_is_never_an_email(exporter):
    set_user_context(EMAIL)
    set_conversation_context(CONVERSATION)
    tracer = observability._state["tracer_provider"].get_tracer("test")

    with tracer.start_as_current_span("request"):
        pass
    with agent_span("generation", user_id=EMAIL):
        pass

    for span in exporter.get_finished_spans():
        assert "user.id" not in span.attributes, span.name
        assert span.attributes["session.id"] == CONVERSATION
        assert EMAIL not in repr(dict(span.attributes))


def test_no_request_context_adds_nothing(exporter):
    tracer = observability._state["tracer_provider"].get_tracer("test")
    with tracer.start_as_current_span("background job"):
        pass
    (span,) = exporter.get_finished_spans()
    for key in ("session.id", "user.id", "gen_ai.conversation.id", "eve.message_id"):
        assert key not in span.attributes


def test_ids_follow_the_task_that_set_them(exporter):
    """Two concurrent turns: each span gets its own conversation, not the other's."""
    tracer = observability._state["tracer_provider"].get_tracer("test")

    async def turn(conversation, user):
        set_conversation_context(conversation)
        set_user_context(user)
        await asyncio.sleep(0)
        with tracer.start_as_current_span(f"turn {conversation}"):
            await asyncio.sleep(0)

    async def both():
        await asyncio.gather(turn("conv-a", "user-a"), turn("conv-b", "user-b"))

    asyncio.run(both())
    by_name = {s.name: s for s in exporter.get_finished_spans()}
    assert by_name["turn conv-a"].attributes["user.id"] == "user-a"
    assert by_name["turn conv-b"].attributes["session.id"] == "conv-b"
    assert by_name["turn conv-b"].attributes["user.id"] == "user-b"


def test_processor_never_breaks_a_span():
    class _Span:
        def set_attributes(self, _):
            raise RuntimeError("boom")

    set_conversation_context(CONVERSATION)
    ContextAttributesSpanProcessor().on_start(_Span())


def test_agent_span_is_the_root_with_a_32_hex_trace_id(exporter):
    with agent_span(
        "agentic_generation_stream",
        conversation_id=CONVERSATION,
        user_id=USER,
        message_id=MESSAGE,
        attributes={"eve.pipeline": "agentic", "eve.stream": True, "skip": None},
    ) as root:
        trace_id = span_trace_id(root)
        assert trace_id == current_trace_id()
    assert HEX32.match(trace_id)

    (span,) = exporter.get_finished_spans()
    assert span.name == "invoke_agent"
    assert span.parent is None
    assert f"{span.context.trace_id:032x}" == trace_id
    assert span.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert span.attributes["gen_ai.agent.name"] == "agentic_generation_stream"
    assert span.attributes["session.id"] == CONVERSATION
    assert span.attributes["user.id"] == USER
    assert "skip" not in span.attributes


def test_agent_span_records_errors_and_reraises(exporter):
    with pytest.raises(ValueError):
        with agent_span("generation"):
            raise ValueError("llm down")
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code == trace.StatusCode.ERROR
    assert [e.name for e in span.events] == ["exception"]


def test_cancellation_is_not_an_error(exporter):
    async def cancelled_turn():
        with agent_span("agentic_generation_stream"):
            raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(cancelled_turn())
    (span,) = exporter.get_finished_spans()
    assert span.attributes["eve.cancelled"] is True
    assert span.status.status_code != trace.StatusCode.ERROR


def test_llm_attributes_from_endpoint_metadata(exporter):
    endpoint = build_endpoint_metadata(
        requested="eve_jsc", chain=["eve_jsc", "main", "fallback"], answered="main"
    )
    with agent_span("generation_stream") as root:
        set_llm_attributes(root, endpoint)
    with agent_span("agentic_generation_stream") as early:
        set_llm_attributes(early, endpoint, include_answered=False)

    done, started = exporter.get_finished_spans()
    assert done.attributes["eve.llm.requested"] == "eve_jsc"
    assert done.attributes["eve.llm.chain"] == ("eve_jsc", "main", "fallback")
    assert done.attributes["eve.llm.answered"] == "main"
    assert done.attributes["eve.fallback_used"] is True
    assert "eve.llm.answered" not in started.attributes
    assert "eve.fallback_used" not in started.attributes


def test_without_telemetry_trace_ids_are_none_and_nothing_breaks(monkeypatch):
    """No provider: the API no-op tracer, invalid span contexts, null trace ids."""
    monkeypatch.setitem(observability._state, "tracer_provider", None)
    with agent_span("agentic_generation_stream", conversation_id=CONVERSATION) as root:
        assert root.is_recording() is False
        assert span_trace_id(root) is None
        assert current_trace_id() is None
        with child_span("mcp.call_tool") as child:
            assert span_trace_id(child) is None
    assert span_trace_id(None) is None


def test_instrument_genai_sets_the_content_switch(monkeypatch):
    """TRACELOOP_TRACE_CONTENT follows EVE_OTEL_CAPTURE_CONTENT, default false.

    The instrumentor itself is stubbed: tests/test_trace_tree.py owns the real
    instrumentation (OpenLLMetry does not survive being instrumented twice in
    one process).
    """
    calls = []

    class _Stub:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def instrument(self, **kwargs):
            calls.append(("instrument", os.environ["TRACELOOP_TRACE_CONTENT"]))

    monkeypatch.setattr(
        "opentelemetry.instrumentation.langchain.LangchainInstrumentor", _Stub
    )
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "unset-by-test")

    monkeypatch.delenv("EVE_OTEL_CAPTURE_CONTENT", raising=False)
    assert observability.instrument_genai(None) is True
    assert calls[-1] == ("instrument", "false")
    assert calls[-2] == {"disable_trace_context_propagation": True}

    monkeypatch.setenv("EVE_OTEL_CAPTURE_CONTENT", "true")
    observability.instrument_genai(None)
    assert calls[-1] == ("instrument", "true")

    monkeypatch.setenv("EVE_OTEL_CAPTURE_CONTENT", "no")
    observability.instrument_genai(None)
    assert calls[-1] == ("instrument", "false")


def test_instrument_genai_never_raises(monkeypatch, caplog):
    class _Broken:
        def __init__(self, **kwargs):
            raise RuntimeError("incompatible langchain")

    monkeypatch.setattr(
        "opentelemetry.instrumentation.langchain.LangchainInstrumentor", _Broken
    )
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "unset-by-test")
    assert observability.instrument_genai(None) is False
    assert "LangChain instrumentation skipped" in caplog.text
