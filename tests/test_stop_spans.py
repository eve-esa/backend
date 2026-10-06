"""A user Stop leaves no error span; a model failure still does.

The classic turn runs over a real LangGraph graph instrumented by OpenLLMetry,
which marks every ``BaseException`` through the stream as ERROR: the Stop's
``CancelledError`` when the Stop route cancels the task, and the
``GeneratorExit`` of the closed stream when the Stop arrives through the
cancel channel of another worker.
"""

import asyncio
from typing import TypedDict

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from opentelemetry import trace

from src import observability
from src.database.models.message import Message
from src.observability.context import ROOT_SPAN_NAME, TRACER_NAME
from tests.test_endpoint_failover import _patch_pipeline
from tests.test_stop_before_first_token import (  # noqa: F401 - fixture
    _TokenSeenBus,
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
    """The classic pipeline's graph interface over a compiled LangGraph."""

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
        return self._graph.astream({"q": "where is Rome?"}, stream_mode="messages")

    async def aclose(self):
        return None


@pytest.fixture
def exporter(genai_otel, monkeypatch):
    monkeypatch.setitem(observability._state, "tracer_provider", genai_otel.provider)
    genai_otel.exporter.clear()
    return genai_otel.exporter


async def _run(turn, monkeypatch, after, stop):
    user, conversation, message = turn
    _patch_pipeline(monkeypatch, _Graph(after), configured=("eve_jsc",))
    bus = _TokenSeenBus()
    p1, p2 = _patch_bus(bus)
    with p1, p2:
        cm, task = _start("classic", user, conversation, message, None)
        if stop is not None:
            await asyncio.wait_for(bus.token_seen.wait(), timeout=5)
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


@pytest.mark.parametrize("stop", [_route_stop, _channel_stop], ids=["route", "channel"])
async def test_a_stop_leaves_no_error_span(turn, monkeypatch, exporter, stop):
    row = await _run(turn, monkeypatch, "hang" if stop is _route_stop else "next", stop)

    spans = exporter.get_finished_spans()
    assert row.stopped is True
    assert any(s.name == "invoke_agent LangGraph" for s in spans), "OpenLLMetry did not run"
    assert [s.name for s in spans if s.status.status_code == trace.StatusCode.ERROR] == []
    assert [
        (s.name, e.attributes.get("exception.type"))
        for s in spans
        for e in s.events
        if e.name == "exception"
    ] == []
    assert _root(spans).attributes["eve.stopped"] is True


async def test_a_model_failure_is_still_an_error(turn, monkeypatch, exporter):
    row = await _run(turn, monkeypatch, "fail", None)

    spans = exporter.get_finished_spans()
    root = _root(spans)
    assert row.stopped is False
    assert root.status.status_code == trace.StatusCode.ERROR
    assert [e.attributes["exception.type"] for e in root.events if e.name == "exception"] == [
        "RuntimeError"
    ]
    assert "eve.stopped" not in root.attributes
