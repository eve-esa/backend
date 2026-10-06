"""A user Stop leaves no error span; a model failure still does.

Classic and agentic turns run over a real LangGraph graph instrumented by OpenLLMetry,
which marks every ``BaseException`` through the stream as ERROR: the Stop's
``CancelledError`` when the Stop route cancels the task, and the
``GeneratorExit`` of the closed stream when the Stop arrives through the
cancel channel of another worker.
"""

import asyncio
import contextlib
from typing import TypedDict
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from opentelemetry import trace

from src import observability
from src.database.models.message import Message
from src.observability.context import ROOT_SPAN_NAME, TRACER_NAME
from src.services.generate_answer import persist_message_state
from tests.test_agentic_fallback import _patched_runner
from tests.test_endpoint_failover import _patch_pipeline
from tests.test_stop_before_first_token import (  # noqa: F401 - fixture
    _BlocksOnBus,
    _TokenSeenBus,
    _is_type,
    _patch_bus,
    _start,
    turn,
)


class _State(TypedDict):
    q: str
    a: str


class _Model(GenericFakeChatModel):
    """Streams its message word by word, then waits for ``after``."""

    after: str = "hang"

    async def _astream(self, *args, **kwargs):
        async for chunk in super()._astream(*args, **kwargs):
            yield chunk
            if self.after == "hang":
                await asyncio.Event().wait()
            if self.after == "fail":
                raise RuntimeError("model down")
            await asyncio.sleep(0.02)


class _Graph:
    """The pipelines' graph interface over a compiled LangGraph."""

    def __init__(self, after):
        model = _Model(messages=iter([AIMessage(content="Rome is here and there")]), after=after)

        async def generate(state):
            text = ""
            async for chunk in model.astream(state["q"]):
                text += chunk.content
            return {"a": text}

        graph = StateGraph(_State)
        graph.add_node("generate", generate)
        graph.add_edge(START, "generate")
        graph.add_edge("generate", END)
        self._graph = graph.compile()

    def astream(self, state, config=None, stream_mode=None):
        return self._graph.astream({"q": "where is Rome?"}, stream_mode=stream_mode)

    async def aclose(self):
        return None


@pytest.fixture
def exporter(genai_otel, monkeypatch):
    monkeypatch.setitem(observability._state, "tracer_provider", genai_otel.provider)
    genai_otel.exporter.clear()
    return genai_otel.exporter


@contextlib.contextmanager
def _pipeline(kind, monkeypatch, after):
    graph = _Graph(after)
    if kind == "agentic":
        with _patched_runner(
            _build_react_graph=MagicMock(return_value=graph),
            persist_message_state=persist_message_state,
        ):
            yield
    else:
        _patch_pipeline(monkeypatch, graph, configured=("eve_jsc",))
        yield


async def _run(kind, turn, monkeypatch, after, stop, bus=None):
    """One turn; ``stop`` lands once ``bus`` has seen a token (or blocked)."""
    user, conversation, message = turn
    bus = bus or _TokenSeenBus()
    p1, p2 = _patch_bus(bus)
    with p1, p2, _pipeline(kind, monkeypatch, after):
        cm, task = _start(kind, user, conversation, message, None)
        if stop is not None:
            seen = bus.entered if isinstance(bus, _BlocksOnBus) else bus.token_seen
            await asyncio.wait_for(seen.wait(), timeout=5)
            await asyncio.sleep(0.05)
            stop(cm, message.id)
        await asyncio.wait_for(task, timeout=5)
    return await Message.find_by_id_on_primary(message.id)


def _route_stop(cm, message_id):
    cm.cancel(message_id)


def _channel_stop(cm, message_id):
    # What the cancel channel does on the worker that owns the turn: the event
    # only, the generator notices it between two chunks and closes the stream.
    cm._events[message_id].set()


def _root(spans):
    (root,) = [
        s
        for s in spans
        if s.name == ROOT_SPAN_NAME and s.instrumentation_scope.name == TRACER_NAME
    ]
    return root


def _error_spans(spans):
    return [s.name for s in spans if s.status.status_code == trace.StatusCode.ERROR]


@pytest.mark.parametrize("kind", ["classic", "agentic"])
@pytest.mark.parametrize("stop", [_route_stop, _channel_stop], ids=["route", "channel"])
async def test_a_stop_leaves_no_error_span(turn, monkeypatch, exporter, kind, stop):
    after = "hang" if stop is _route_stop else "next"
    row = await _run(kind, turn, monkeypatch, after, stop)

    spans = exporter.get_finished_spans()
    assert row.stopped is True
    assert any(s.name == "invoke_agent LangGraph" for s in spans), "OpenLLMetry did not run"
    assert _error_spans(spans) == []
    assert [
        (s.name, e.attributes.get("exception.type"))
        for s in spans
        for e in s.events
        if e.name == "exception"
    ] == []
    assert _root(spans).attributes["eve.stopped"] is True


@pytest.mark.parametrize("kind", ["classic", "agentic"])
async def test_a_stop_on_the_final_event_is_not_a_stopped_turn(
    turn, monkeypatch, exporter, kind
):
    """The answer is saved complete, so its root is not marked stopped."""
    bus = _BlocksOnBus(_is_type("final"))
    row = await _run(kind, turn, monkeypatch, "next", _route_stop, bus=bus)

    spans = exporter.get_finished_spans()
    assert row.stopped is False
    assert _error_spans(spans) == []
    assert "eve.stopped" not in _root(spans).attributes


async def test_a_model_failure_is_still_an_error(turn, monkeypatch, exporter):
    row = await _run("classic", turn, monkeypatch, "fail", None)

    spans = exporter.get_finished_spans()
    root = _root(spans)
    assert row.stopped is False
    assert root.status.status_code == trace.StatusCode.ERROR
    assert [e.attributes["exception.type"] for e in root.events if e.name == "exception"] == [
        "RuntimeError"
    ]
    assert "eve.stopped" not in root.attributes
