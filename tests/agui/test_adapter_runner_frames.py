"""The adapter over frames the real runner produces (not hand-built ones).

Each case runs ``generate_answer_agentic_stream_helper`` with a scripted graph
(the harness of ``tests/test_agent_step_frames.py``), feeds every frame it
published to ``AguiAdapter`` as the route does, and checks the transcript with
the shared validator: the enriched frames of B-S2 and the AG-UI mapping agree.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from src.services.agui.adapter import AguiAdapter, RunContext, encode_sse
from tests.agui.validator import validate_transcript
from tests.test_agent_step_frames import (
    RETRIEVE,
    _answer,
    _call,
    _Graph,
    _react_failure,
    _result,
    _tool,
)
from tests.test_agent_step_frames import _run as run_runner

pytestmark = pytest.mark.no_db

T = "6a3f0c2e9b1d4e7f8a2b3c4d"
M = "m1"  # the message id the runner harness uses
R = "0f8e6c1a-3b2d-4e5f-9a7b-1c2d3e4f5a6b"
ENDPOINT = {"requested": "main", "chain": ["main"], "answered": "main",
            "attempts": [], "circuit_open": [], "substituted": False}


def _resolver():
    return AsyncMock(return_value=(MagicMock(), {"endpoint": ENDPOINT}))


async def _transcript(graph, **runner_kwargs):
    frames, _ = await run_runner(graph, _resolve_agentic_llm_client=_resolver(), **runner_kwargs)
    adapter = AguiAdapter(RunContext(thread_id=T, run_id=R, message_id=M, trace_id="a" * 32, graph="react"))
    events: List[Any] = list(adapter.start())
    for frame in frames:
        events.extend(adapter.adapt(frame))
    events.extend(adapter.finish())
    body = encode_sse(events)
    violations = validate_transcript(body, thread_id=T, run_id=R)
    assert violations == [], violations
    parsed = [json.loads(line[len("data: "):]) for line in body.splitlines() if line.startswith("data: ")]
    return frames, parsed


def _of(events: List[Dict[str, Any]], kind: str) -> List[Dict[str, Any]]:
    return [e for e in events if e["type"] == kind]


async def test_tool_turn_from_the_runner_is_a_valid_success():
    step = AIMessage(
        content="",
        tool_calls=[{"name": RETRIEVE, "args": {"query": "SAR"}, "id": "call-1"}],
        usage_metadata={"input_tokens": 100, "output_tokens": 8, "total_tokens": 108},
    )
    frames, events = await _transcript(
        _Graph([("updates", {"agent": {"messages": [step]}}), _result(RETRIEVE, "call-1"), _answer("SAR is radar.")]),
        tools=[_tool(RETRIEVE, "eve_retrieval")],
    )

    assert events[-1]["type"] == "RUN_FINISHED"
    assert events[-1]["outcome"] == {"type": "success"}
    assert events[-1]["usage"][0]["inputTokens"] == 100
    [start] = _of(events, "TOOL_CALL_START")
    assert start["toolCallId"] == "call-1"
    assert start["metadata"]["eve.server"] == "eve_retrieval"
    [result] = _of(events, "TOOL_CALL_RESULT")
    assert result["metadata"]["eve.status"] == "ok"
    assert isinstance(result["metadata"]["eve.durationMs"], int)
    assert _of(events, "TEXT_MESSAGE_END")[-1]["metadata"]["eve.endpoint"]["answered"] == "main"


async def test_failure_after_text_from_the_runner_is_run_error():
    frames, events = await _transcript(
        _Graph([_answer("partial")], raise_exc=RuntimeError("boom")),
    )

    assert [f["type"] for f in frames][-1] == "final"
    assert events[-1]["type"] == "RUN_ERROR"
    assert events[-1]["code"] == "upstream_error"
    assert _of(events, "RUN_FINISHED") == []
    assert "".join(e["delta"] for e in _of(events, "TEXT_MESSAGE_CONTENT")) == "partial"


async def test_empty_answer_from_the_runner_is_run_error():
    _, events = await _transcript(_Graph([]))

    assert events[-1]["type"] == "RUN_ERROR"
    assert events[-1]["code"] == "empty_answer"


async def test_react_tool_failure_and_retry_keep_distinct_calls():
    _, events = await _transcript(
        _Graph(
            [
                _call(RETRIEVE, {"query": "q"}, "call-1"),
                _react_failure(RETRIEVE, "call-1", "read timed out"),
                _call(RETRIEVE, {"query": "q"}, "call-1"),
                _result(RETRIEVE, "call-1"),
                _answer("done"),
            ]
        ),
    )

    starts = _of(events, "TOOL_CALL_START")
    results = _of(events, "TOOL_CALL_RESULT")
    assert len({s["toolCallId"] for s in starts}) == 2
    assert [r["toolCallId"] for r in results] == [s["toolCallId"] for s in starts]
    assert [r["metadata"]["eve.status"] for r in results] == ["timeout", "ok"]
    assert len({r["messageId"] for r in results}) == 2
    assert events[-1]["outcome"] == {"type": "success"}
