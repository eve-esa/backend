"""Tool images on the streamed answer path.

A normal completion, and a fallback answer kept after the primary node
failed, carry the drawn image in the ``final`` frame and in the stored
message. Stop, timeout and error keep the answer as streamed: nothing is
appended there.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from src.schemas.generation_request import GenerationRequest
from src.services.agents.core.runner import generate_answer_agentic_stream_helper
from src.services.mcp.artifact_context import get_artifact_context
from tests.test_agentic_fallback import _patched_runner

pytestmark = pytest.mark.no_db

_IMAGE_ID = "6afea7ccac79eb8afdbaf804"
_IMAGE_LINE = f"![get_sample_image-1.png](/artifacts/{_IMAGE_ID})"
_ANSWER = "Here is the sample image:"


class _ImageTurnGraph:
    """Records a tool image on the turn, streams the answer, then ends as told."""

    def __init__(
        self,
        *,
        node: str = "agent",
        updates: Optional[List[Dict[str, Any]]] = None,
        raise_exc: Optional[BaseException] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ):
        self._node = node
        self._updates = updates or []
        self._raise = raise_exc
        self._cancel_event = cancel_event

    async def astream(self, *args, **kwargs):
        # What the interceptor records for a tool that returned an image stub.
        ctx = get_artifact_context()
        ctx.collected_artifact_ids.append(_IMAGE_ID)
        ctx.collected_artifact_files["get_sample_image-1.png"] = _IMAGE_ID
        ctx.collected_artifact_types[_IMAGE_ID] = "image/png"
        ctx.shown_artifact_ids.append(_IMAGE_ID)
        for update in self._updates:
            yield "updates", update
        yield "messages", (AIMessage(content=_ANSWER), {"langgraph_node": self._node})
        if self._cancel_event is not None:
            self._cancel_event.set()
            raise asyncio.CancelledError()
        if self._raise is not None:
            raise self._raise


async def _run(graph, cancel_event=None):
    persist = AsyncMock()
    request = GenerationRequest(query="hi", llm_type="main", agent="react")
    with _patched_runner(
        _build_react_graph=MagicMock(return_value=graph),
        persist_message_state=persist,
    ):
        events = [
            json.loads(event.removeprefix("data: ").strip())
            async for event in generate_answer_agentic_stream_helper(
                request,
                conversation_id="c1",
                message_id="m1",
                user_id="u1",
                cancel_event=cancel_event,
            )
        ]
    return events, persist.await_args.kwargs


def _final_answer(events) -> Optional[str]:
    finals = [e for e in events if e.get("type") == "final"]
    return finals[-1]["answer"] if finals else None


async def test_a_normal_answer_carries_the_tool_image():
    events, stored = await _run(_ImageTurnGraph())

    expected = f"{_ANSWER}\n\n{_IMAGE_LINE}"
    assert _final_answer(events) == expected
    assert stored["output"] == expected
    assert stored["artifact_ids"] == [_IMAGE_ID]


async def test_a_recovered_fallback_answer_carries_the_tool_image():
    graph = _ImageTurnGraph(
        node="agent_fallback",
        updates=[{"agent_fallback": {"messages": [AIMessage(content="")]}}],
        raise_exc=TimeoutError("Node 'agent' exceeded its idle timeout"),
    )

    events, stored = await _run(graph)

    expected = f"{_ANSWER}\n\n{_IMAGE_LINE}"
    assert _final_answer(events) == expected
    assert stored["output"] == expected


async def test_a_stop_keeps_the_answer_as_streamed():
    cancel_event = asyncio.Event()

    events, stored = await _run(_ImageTurnGraph(cancel_event=cancel_event), cancel_event)

    assert stored["stopped"] is True
    assert stored["output"] == _ANSWER
    assert _final_answer(events) is None


@pytest.mark.parametrize(
    "exc",
    [TimeoutError("run timeout"), RuntimeError("boom")],
    ids=["timeout", "error"],
)
async def test_a_failed_turn_keeps_the_answer_as_streamed(exc):
    events, stored = await _run(_ImageTurnGraph(raise_exc=exc))

    assert stored["error"]
    assert stored["output"] == _ANSWER
    assert _final_answer(events) == _ANSWER
