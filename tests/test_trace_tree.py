"""OpenLLMetry canary: the span tree of one agentic answer, fully in process.

The real runner, the real react graph and the real MCP adapter run against
the dummy FastMCP server of tests/e2e/dummy_mcp_server, served in process
through httpx.ASGITransport. The model is a scripted GenericFakeChatModel that
asks for one tool call, then answers. No compose service, no LLM, no Mongo.

Asserted: the invoke_agent root, the LangGraph agent span under it, an LLM
span carrying gen_ai.request.model, the LangChain tool span, the mcp.call_tool
client span under the tool span with eve.mcp.server and gen_ai.tool.name, the
traceparent of that span on the request the MCP server received, the
conversation and user ids on every span, the trace id in the final event, and
prompt, completion and tool argument content only when capture is on.

Run it whenever OpenLLMetry, the OpenTelemetry SDK or the MCP adapter moves.
"""

import functools
import json
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langgraph.checkpoint.memory import InMemorySaver
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src import observability
from src.schemas.generation_request import GenerationRequest
from src.services.agents.core import runner
from src.services.generate_answer import build_endpoint_metadata
from src.services.mcp import tool_loader
from src.utils.error_logger import (
    conversation_id_context,
    message_id_context,
    user_id_context,
)
from tests.e2e.dummy_mcp_server import server as dummy_server

pytestmark = pytest.mark.no_db

SERVER = "dummy"
TOOL = "trace_probe"
CONVERSATION_ID = "conv-trace-tree"
MESSAGE_ID = "msg-trace-tree"
USER_ID = "user-trace-tree"
# Markers that must only ever reach span attributes with capture on.
QUERY_MARK = "zebraquery7f3c"
ARG_MARK = "orbitarg91d2"
ANSWER_MARK = "answermarkc0ffee"
MODEL_NAME = "eve-fake-model"
# Attributes OpenLLMetry uses for prompts, completions and tool input/output.
CONTENT_KEYS = {
    "gen_ai.input.messages",
    "gen_ai.output.messages",
    "gen_ai.system_instructions",
    "gen_ai.tool.call.arguments",
    "gen_ai.tool.call.result",
    "traceloop.entity.input",
    "traceloop.entity.output",
}


class _ScriptedChatModel(GenericFakeChatModel):
    """GenericFakeChatModel that keeps tool calls when streamed.

    The stock ``_stream`` only splits content, so a tool call would vanish
    under LangGraph's messages stream mode; this emits each scripted message
    as one chunk, tool call chunks and usage included.
    """

    model_name: str = MODEL_NAME

    @classmethod
    def is_lc_serializable(cls) -> bool:
        # OpenLLMetry reads gen_ai.request.model from the serialized constructor
        # kwargs, as it does for ChatOpenAI; a non serializable model is "unknown".
        return True

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ScriptedChatModel":
        return self

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        message = self._generate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        ).generations[0].message
        chunk = ChatGenerationChunk(
            message=AIMessageChunk(
                content=message.content,
                id=message.id,
                tool_call_chunks=[
                    {
                        "name": call["name"],
                        "args": json.dumps(call["args"]),
                        "id": call["id"],
                        "index": index,
                    }
                    for index, call in enumerate(message.tool_calls)
                ],
                usage_metadata={
                    "input_tokens": 11,
                    "output_tokens": 7,
                    "total_tokens": 18,
                },
            )
        )
        if run_manager is not None:
            run_manager.on_llm_new_token(str(message.content or ""), chunk=chunk)
        yield chunk


def _scripted_model() -> _ScriptedChatModel:
    return _ScriptedChatModel(
        model_name=MODEL_NAME,
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": f"{SERVER}_{TOOL}",
                            "args": {"text": ARG_MARK},
                            "id": "call-probe-1",
                        }
                    ],
                ),
                AIMessage(content=f"The probe answered {ANSWER_MARK}."),
            ]
        )
    )


def trace_probe(text: str) -> str:
    """Report how long ``text`` is (test tool, never echoes its input)."""
    return f"probe saw {len(text)} characters"


@pytest.fixture
def dummy_mcp():
    """The dummy FastMCP app plus the trace probe tool, recording request headers."""
    mcp = dummy_server.mcp
    mcp.add_tool(trace_probe, name=TOOL)
    # A session manager runs once per instance: give every test a fresh one.
    mcp._session_manager = None
    app = mcp.streamable_http_app()
    received: List[Dict[str, str]] = []

    async def recording_app(scope, receive, send):
        if scope["type"] == "http":
            received.append(
                {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
            )
        await app(scope, receive, send)

    try:
        yield SimpleNamespace(mcp=mcp, app=recording_app, received=received)
    finally:
        mcp.remove_tool(TOOL)
        mcp._session_manager = None


def _connections(asgi_app) -> Dict[str, Any]:
    def factory(headers=None, timeout=None, auth=None):
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=asgi_app),
            headers=headers,
            timeout=timeout,
            auth=auth,
        )

    return {
        SERVER: {
            "transport": "streamable_http",
            "url": "http://dummy-mcp.test/mcp",
            "headers": {},
            "httpx_client_factory": factory,
        }
    }


@pytest.fixture(scope="module")
def otel():
    """One provider and one OpenLLMetry instrumentation for the whole module.

    Production instruments once per worker. OpenLLMetry 0.62.3 does not
    survive uninstrument then instrument in one process: callbacks keep going
    to the first handler and its (by then shut down) provider. So this module
    instruments once and clears the exporter per test. Nothing global is
    registered: the provider reaches the context helpers through
    ``observability._state`` and OpenLLMetry directly.
    """
    exporter = InMemorySpanExporter()
    provider = observability.build_tracer_provider(exporter, batch=False)
    previous = observability._state.get("tracer_provider")
    observability._state["tracer_provider"] = provider
    try:
        yield SimpleNamespace(exporter=exporter, provider=provider)
    finally:
        observability.uninstrument_genai()
        observability._state["tracer_provider"] = previous
        provider.shutdown()


@pytest.fixture
def traced(otel, monkeypatch):
    """Set the capture switch the way init does, then hand out a clean exporter."""

    def _install(capture: Optional[bool]):
        if capture is None:
            monkeypatch.delenv("EVE_OTEL_CAPTURE_CONTENT", raising=False)
        else:
            monkeypatch.setenv("EVE_OTEL_CAPTURE_CONTENT", "true" if capture else "false")
        # instrument_genai writes this; monkeypatch restores it afterwards.
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "unset-by-test")
        assert observability.instrument_genai(otel.provider) is True
        otel.exporter.clear()
        return otel.exporter

    return _install


async def _run_turn(dummy_mcp) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    request = GenerationRequest(
        query=f"Ask the probe about {QUERY_MARK}", llm_type="main", agent="react"
    )
    request.mcp_server_configs = [SimpleNamespace(name=SERVER)]
    model = _scripted_model()
    endpoint = build_endpoint_metadata(requested="main", chain=["main"], answered="main")

    tokens = [
        conversation_id_context.set(CONVERSATION_ID),
        message_id_context.set(MESSAGE_ID),
        user_id_context.set(USER_ID),
    ]
    try:
        with (
            patch.object(
                tool_loader,
                "_build_mcp_connections",
                return_value=(_connections(dummy_mcp.app), False, []),
            ),
            patch.object(tool_loader, "get_cognito_token_provider", return_value=None),
            patch.object(
                runner,
                "_get_agentic_checkpointer",
                AsyncMock(return_value=InMemorySaver()),
            ),
            patch.object(
                runner, "_fetch_conversation_context", AsyncMock(return_value=([], None))
            ),
            patch.object(
                runner,
                "_resolve_agentic_llm_client",
                AsyncMock(
                    return_value=(
                        model,
                        {"agentic_llm_resolved": "main", "endpoint": endpoint},
                    )
                ),
            ),
            patch.object(
                runner,
                "_build_react_graph",
                functools.partial(runner._build_react_graph, fallback_llm=model),
            ),
            patch.object(runner, "persist_message_state", AsyncMock()) as persist,
            patch.object(runner, "persist_policy_event", AsyncMock()),
            patch.object(runner, "maybe_rollup_and_trim_history", AsyncMock()),
        ):
            async with dummy_mcp.mcp.session_manager.run():
                raw = [
                    event
                    async for event in runner.generate_answer_agentic_stream_helper(
                        request,
                        conversation_id=CONVERSATION_ID,
                        message_id=MESSAGE_ID,
                        user_id=USER_ID,
                    )
                ]
            persisted = persist.await_args.kwargs if persist.await_args else {}
    finally:
        for var, token in zip(
            (conversation_id_context, message_id_context, user_id_context), tokens
        ):
            var.reset(token)
    events = [json.loads(e.removeprefix("data: ").strip()) for e in raw]
    return events, persisted


def _by_id(spans: List[ReadableSpan]) -> Dict[int, ReadableSpan]:
    return {span.context.span_id: span for span in spans}


def _parent(span: ReadableSpan, index: Dict[int, ReadableSpan]) -> Optional[ReadableSpan]:
    return index.get(span.parent.span_id) if span.parent is not None else None


def _ancestors(span: ReadableSpan, index: Dict[int, ReadableSpan]) -> List[ReadableSpan]:
    chain = []
    parent = _parent(span, index)
    while parent is not None:
        chain.append(parent)
        parent = _parent(parent, index)
    return chain


def _dump(spans: List[ReadableSpan]) -> str:
    index = _by_id(spans)
    lines = []
    for span in spans:
        parent = _parent(span, index)
        lines.append(
            f"{span.name!r} parent={parent.name if parent else None!r} "
            f"attrs={dict(span.attributes or {})}"
        )
    return "\n".join(lines)


def _text_of(spans: List[ReadableSpan]) -> str:
    parts = []
    for span in spans:
        parts.append(span.name)
        for attrs in [span.attributes] + [e.attributes for e in span.events]:
            parts.extend(f"{k}={v!r}" for k, v in (attrs or {}).items())
    return "\n".join(parts)


def _assert_genai_spans(spans: List[ReadableSpan]) -> None:
    """The negative content checks mean nothing without the OpenLLMetry spans."""
    kinds = {(s.attributes or {}).get("gen_ai.operation.name") for s in spans}
    assert {"chat", "execute_tool", "invoke_agent"} <= kinds, _dump(spans)


@pytest.mark.asyncio
async def test_agentic_turn_span_tree(traced, dummy_mcp):
    exporter = traced(capture=None)

    events, persisted = await _run_turn(dummy_mcp)
    spans = list(exporter.get_finished_spans())
    index = _by_id(spans)

    final = events[-1]
    assert final["type"] == "final", events
    assert ANSWER_MARK in final["answer"]

    roots = [s for s in spans if s.name == "invoke_agent"]
    assert len(roots) == 1, _dump(spans)
    root = roots[0]
    assert root.parent is None
    assert root.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert root.attributes["gen_ai.agent.name"] == "agentic_generation_stream"
    trace_id = root.context.trace_id
    assert all(s.context.trace_id == trace_id for s in spans)

    # The trace id reaches the client and the Message.
    assert final["trace_id"] == f"{trace_id:032x}"
    assert persisted["trace_id"] == final["trace_id"]

    # First streamed answer token, as an event on the root.
    assert "gen_ai.first_token" in [e.name for e in root.events]

    # The LangGraph agent span directly under the root, then the agent node.
    graph_spans = [
        s
        for s in spans
        if _parent(s, index) is root
        and (s.attributes or {}).get("gen_ai.operation.name") == "invoke_agent"
        and (s.attributes or {}).get("gen_ai.provider.name") == "langgraph"
    ]
    assert len(graph_spans) == 1, _dump(spans)
    graph_span = graph_spans[0]
    agent_nodes = [
        s
        for s in spans
        if (s.attributes or {}).get("gen_ai.task.name") == "agent"
        and graph_span in _ancestors(s, index)
    ]
    assert agent_nodes, _dump(spans)

    # LLM spans with the model and token usage, inside an agent node.
    llm_spans = [s for s in spans if "gen_ai.request.model" in (s.attributes or {})]
    assert len(llm_spans) == 2, _dump(spans)
    for llm_span in llm_spans:
        assert llm_span.attributes["gen_ai.request.model"] == MODEL_NAME
        assert llm_span.attributes["gen_ai.usage.input_tokens"] == 11
        assert _parent(llm_span, index) in agent_nodes

    # The LangChain tool span, under the root.
    tool_spans = [
        s
        for s in spans
        if (s.attributes or {}).get("traceloop.span.kind") == "tool"
    ]
    assert len(tool_spans) == 1, _dump(spans)
    tool_span = tool_spans[0]
    assert TOOL in tool_span.name
    assert graph_span in _ancestors(tool_span, index)

    # The MCP client span, child of the tool span.
    mcp_spans = [s for s in spans if s.name == "mcp.call_tool"]
    assert len(mcp_spans) == 1, _dump(spans)
    mcp_span = mcp_spans[0]
    assert mcp_span.attributes["eve.mcp.server"] == SERVER
    assert mcp_span.attributes["gen_ai.tool.name"] == TOOL
    assert _parent(mcp_span, index) is tool_span

    # The MCP server received the traceparent of the mcp.call_tool span.
    # W3C traceparent: version, trace id, parent span id, flags (sampled bit).
    expected = f"00-{trace_id:032x}-{mcp_span.context.span_id:016x}-"
    traceparents = [h.get("traceparent") or "" for h in dummy_mcp.received]
    carrying = [t for t in traceparents if t.startswith(expected)]
    assert carrying, traceparents
    assert all(int(t.rsplit("-", 1)[1], 16) & 0x01 for t in carrying)

    # Request ids on every span, the user id never an email.
    for span in spans:
        attrs = span.attributes or {}
        assert attrs.get("session.id") == CONVERSATION_ID, span.name
        assert attrs.get("gen_ai.conversation.id") == CONVERSATION_ID, span.name
        assert attrs.get("user.id") == USER_ID, span.name
        assert attrs.get("eve.message_id") == MESSAGE_ID, span.name


@pytest.mark.asyncio
@pytest.mark.parametrize("capture", [None, False])
async def test_content_is_absent_with_capture_off(traced, dummy_mcp, capture):
    exporter = traced(capture=capture)

    events, _ = await _run_turn(dummy_mcp)
    assert events[-1]["type"] == "final"
    spans = list(exporter.get_finished_spans())
    _assert_genai_spans(spans)
    text = _text_of(spans)

    assert os.environ["TRACELOOP_TRACE_CONTENT"] == "false"
    keys = {k for span in spans for k in (span.attributes or {})}
    assert not keys & CONTENT_KEYS, keys & CONTENT_KEYS
    assert QUERY_MARK not in text
    assert ARG_MARK not in text
    assert ANSWER_MARK not in text
    assert "probe saw" not in text


@pytest.mark.asyncio
async def test_content_is_present_with_capture_on(traced, dummy_mcp):
    exporter = traced(capture=True)

    events, _ = await _run_turn(dummy_mcp)
    assert events[-1]["type"] == "final"
    spans = list(exporter.get_finished_spans())
    _assert_genai_spans(spans)
    text = _text_of(spans)

    assert os.environ["TRACELOOP_TRACE_CONTENT"] == "true"
    keys = {k for span in spans for k in (span.attributes or {})}
    assert {"gen_ai.input.messages", "gen_ai.output.messages"} <= keys
    assert QUERY_MARK in text
    assert ARG_MARK in text
    assert ANSWER_MARK in text
