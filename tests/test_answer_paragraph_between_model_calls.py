"""A paragraph break between the answer text of two model calls in one agentic turn.

The agent can write text, call a tool, then write more text in a later model
call. Without a separator the stream, the AG-UI transcript and the stored
answer glued the two ("...the flooded area.I am sorry, but..."). The break is
part of the token stream, so every consumer sees the same text.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, List

import pytest

from src.services.agui.adapter import AguiAdapter, RunContext, encode_sse
from src.services.sources.citations import marker_spans
from tests.agui.validator import validate_transcript
from tests.test_agent_step_frames import RETRIEVE, _call, _Graph, _of, _result, _tool
from tests.test_agent_step_frames import _run as run_runner
from tests.test_inline_citations import (  # noqa: F401  (fixtures)
    R,
    T,
    _cited_run,
    _persist_kwargs,
    _resolver,
    _tokens,
    flag_off,
    flag_on,
)

pytestmark = pytest.mark.no_db

FIRST = "I will analyze the flooded area."
SECOND = "I am sorry, but the tool found nothing."


def _answer(events) -> str:
    return "".join(t["content"] for t in _of(events, "token"))


def _two_calls(first, second, result=None):
    """Text from one model call, a tool call and its result, text from the next call."""
    return [
        *_tokens(*first),
        _call("ls", {}, "c1"),
        result or _result("ls", "c1"),
        *_tokens(*second),
    ]


async def _run(events, **kw):
    return await run_runner(_Graph(events), tools=[_tool("ls")], _resolve_agentic_llm_client=_resolver(), **kw)


class TestParagraphBreak:
    async def test_two_model_calls_with_a_tool_between(self, flag_off):
        events, applied = await _run(_two_calls(["I will analyze ", "the flooded area."], ["I am sorry, ", "but the tool found nothing."]))
        expected = f"{FIRST}\n\n{SECOND}"
        assert _answer(events) == expected
        assert _answer(events).count("\n\n") == 1
        assert _of(events, "final")[0]["answer"] == expected
        assert _persist_kwargs(applied)["output"] == expected

    async def test_previous_text_ending_in_a_newline_gets_no_extra_break(self, flag_off):
        events, applied = await _run(_two_calls([FIRST + "\n"], [SECOND]))
        assert _answer(events) == f"{FIRST}\n{SECOND}"
        assert _persist_kwargs(applied)["output"] == f"{FIRST}\n{SECOND}"

    async def test_next_text_starting_with_a_newline_gets_no_extra_break(self, flag_off):
        events, _ = await _run(_two_calls([FIRST], ["\n\n" + SECOND]))
        assert _answer(events) == f"{FIRST}\n\n{SECOND}"

    async def test_one_model_call_in_many_chunks_has_no_break(self, flag_off):
        parts = ["The ", "flooded ", "area ", "covers ", "12", " km2", "."]
        events, applied = await _run(_tokens(*parts))
        assert _answer(events) == "".join(parts)
        assert "\n" not in _persist_kwargs(applied)["output"]

    async def test_first_text_of_the_turn_has_no_break(self, flag_off):
        events, applied = await _run([_call("ls", {}, "c1"), _result("ls", "c1"), *_tokens(SECOND)])
        assert _answer(events) == SECOND
        assert _persist_kwargs(applied)["output"] == SECOND

    async def test_tool_error_then_text_gets_the_break(self, flag_off):
        error = _result("ls", "c1", content="Tool error: boom", status="error")
        events, applied = await _run(_two_calls([FIRST], [SECOND], result=error))
        assert _answer(events) == f"{FIRST}\n\n{SECOND}"
        assert _persist_kwargs(applied)["output"] == f"{FIRST}\n\n{SECOND}"

    async def test_stop_between_the_two_calls_stores_no_dangling_break(self, flag_off):
        cancel = asyncio.Event()
        graph_events = _two_calls([FIRST], [SECOND])
        stop_at = len(graph_events) - 1  # the first text of the second call

        def hook(index):
            if index == stop_at:
                cancel.set()

        graph = _Graph(graph_events, before_each=hook)
        events, applied = await run_runner(
            graph, tools=[_tool("ls")], cancel_event=cancel, _resolve_agentic_llm_client=_resolver()
        )
        assert events[-1]["type"] == "stopped"
        assert _answer(events) == FIRST
        kwargs = _persist_kwargs(applied)
        assert kwargs["stopped"] is True
        assert kwargs["output"] == FIRST

    async def test_text_tool_call_between_calls_is_not_split(self, flag_off):
        # The second model call writes its tool call as text: no break before
        # it, and none leaks into the answer; the third call's text gets one.
        events, applied = await _run(
            [
                *_tokens(FIRST),
                _call("ls", {}, "c1"),
                _result("ls", "c1"),
                *_tokens("[TOOL_", "CALLS]ls{}"),
                _result("ls", "call_0_ls"),
                *_tokens(SECOND),
            ]
        )
        assert _answer(events) == f"{FIRST}\n\n{SECOND}"
        assert "TOOL" not in _persist_kwargs(applied)["output"]
        assert [c["id"] for c in _of(events, "tool_call")] == ["c1", "call_0_ls"]

    async def test_agui_transcript_carries_the_break(self, flag_off):
        events, _ = await _run(_two_calls([FIRST], [SECOND]))
        adapter = AguiAdapter(RunContext(thread_id=T, run_id=R, message_id="m1", trace_id="a" * 32, graph="react"))
        out: List[Any] = list(adapter.start())
        for frame in events:
            out.extend(adapter.adapt(frame))
        out.extend(adapter.finish())
        body = encode_sse(out)
        assert validate_transcript(body, thread_id=T, run_id=R) == []
        parsed = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]
        text = "".join(e["delta"] for e in parsed if e["type"] == "TEXT_MESSAGE_CONTENT")
        assert text == f"{FIRST}\n\n{SECOND}"


class TestParagraphBreakWithCitations:
    async def test_markers_on_both_sides_keep_their_offsets(self, flag_on):
        events, applied = await _cited_run(
            ["SAR is radar [S1]."],
            tail=[_call("ls", {}, "c2"), _result("ls", "c2"), *_tokens("Files agree [S2].")],
            tools=[_tool(RETRIEVE, "eve_retrieval"), _tool("ls")],
        )
        answer = _answer(events)
        assert answer == "SAR is radar [1].\n\nFiles agree [2]."
        final = _of(events, "final")[0]
        assert marker_spans(answer, final["citations"]) == ["[1]", "[2]"]
        assert _persist_kwargs(applied)["output"] == answer

    async def test_a_held_handle_across_the_boundary_is_still_rewritten(self, flag_on):
        events, applied = await _cited_run(
            ["SAR is radar [S"],
            tail=[_call("ls", {}, "c2"), _result("ls", "c2"), *_tokens("1", "] and more.")],
            tools=[_tool(RETRIEVE, "eve_retrieval"), _tool("ls")],
        )
        answer = _answer(events)
        assert "[S" not in answer
        assert answer == "SAR is radar [1]\n\n and more."
        assert answer.count("\n\n") == 1
        final = _of(events, "final")[0]
        assert marker_spans(answer, final["citations"]) == ["[1]"]
        assert _persist_kwargs(applied)["output"] == answer

    async def test_a_held_bracket_that_is_no_marker_comes_before_the_break(self, flag_on):
        events, _ = await _cited_run(
            ["SAR is radar ["],
            tail=[_call("ls", {}, "c2"), _result("ls", "c2"), *_tokens("More text.")],
            tools=[_tool(RETRIEVE, "eve_retrieval"), _tool("ls")],
        )
        assert _answer(events) == "SAR is radar [\n\nMore text."
