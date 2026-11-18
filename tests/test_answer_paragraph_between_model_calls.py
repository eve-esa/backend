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
from langchain_core.messages import AIMessage

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


def _node_tokens(node, *parts, run=None):
    """Answer chunks of one model call in ``node``; ``run`` is the chunk id of that call."""
    from langchain_core.messages import AIMessageChunk

    return [("messages", (AIMessageChunk(content=p, id=run), {"langgraph_node": node})) for p in parts]


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

    async def test_previous_text_ending_in_a_newline_is_topped_up_to_one_break(self, flag_off):
        events, applied = await _run(_two_calls([FIRST + "\n"], [SECOND]))
        assert _answer(events) == f"{FIRST}\n\n{SECOND}"
        assert _persist_kwargs(applied)["output"] == f"{FIRST}\n\n{SECOND}"

    async def test_previous_text_ending_in_a_break_gets_no_extra_break(self, flag_off):
        events, _ = await _run(_two_calls([FIRST + "\n\n"], [SECOND]))
        assert _answer(events) == f"{FIRST}\n\n{SECOND}"

    async def test_a_list_item_above_does_not_swallow_the_next_part(self, flag_off):
        # In Markdown "- item\ntext" makes the text part of the list item.
        events, applied = await _run(_two_calls(["- item one\n", "- item two\n"], [SECOND]))
        expected = f"- item one\n- item two\n\n{SECOND}"
        assert _answer(events) == expected
        assert _persist_kwargs(applied)["output"] == expected

    async def test_next_text_starting_with_one_newline_is_topped_up(self, flag_off):
        events, _ = await _run(_two_calls(["| a | b |\n|---|---|\n| 1 | 2 |"], ["\n" + SECOND]))
        assert _answer(events) == f"| a | b |\n|---|---|\n| 1 | 2 |\n\n{SECOND}"

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

    async def test_leading_spaces_of_the_next_part_are_dropped(self, flag_off):
        events, applied = await _run(_two_calls([FIRST], [" ", "  I am sorry, ", "but the tool found nothing."]))
        assert _answer(events) == f"{FIRST}\n\n{SECOND}"
        assert _persist_kwargs(applied)["output"] == f"{FIRST}\n\n{SECOND}"

    async def test_a_whitespace_only_call_keeps_the_break_for_the_next(self, flag_off):
        events, applied = await _run(
            [
                *_tokens(FIRST),
                _call("ls", {}, "c1"),
                _result("ls", "c1"),
                *_tokens(" ", "\n\n", " "),
                _call("ls", {}, "c2"),
                _result("ls", "c2"),
                *_tokens(SECOND),
            ]
        )
        assert _answer(events) == f"{FIRST}\n\n{SECOND}"
        assert _persist_kwargs(applied)["output"] == f"{FIRST}\n\n{SECOND}"

    async def test_a_whitespace_only_last_call_adds_nothing(self, flag_off):
        events, applied = await _run(_two_calls([FIRST], ["\n\n", " "]))
        assert _answer(events) == FIRST
        assert _of(events, "final")[0]["answer"] == FIRST
        assert _persist_kwargs(applied)["output"] == FIRST

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

    async def test_agent_fallback_after_a_partial_answer_starts_a_paragraph(self, flag_off):
        # The primary agent node streamed part of an answer and failed; the
        # agent_fallback node answers in full. Only the node changes.
        events, applied = await _run(
            [
                *_tokens("SAR sees through"),
                ("updates", {"agent_fallback": {"messages": [AIMessage(content="")]}}),
                *_node_tokens("agent_fallback", "SAR sees ", "through clouds."),
            ]
        )
        expected = "SAR sees through\n\nSAR sees through clouds."
        assert _answer(events) == expected
        assert _persist_kwargs(applied)["output"] == expected

    @pytest.mark.parametrize("node", ["agent", "model"])
    async def test_a_retry_inside_one_node_starts_a_paragraph(self, flag_off, node):
        # LangGraph retry on the react agent node, or retry and fallback
        # middleware in the deep model node: one node, two model calls, one
        # chunk id each.
        events, applied = await _run(
            [
                *_node_tokens(node, "SAR sees ", "through", run="run-1"),
                *_node_tokens(node, "SAR sees ", "through clouds.", run="run-2"),
            ]
        )
        expected = "SAR sees through\n\nSAR sees through clouds."
        assert _answer(events) == expected
        assert _persist_kwargs(applied)["output"] == expected

    async def test_one_call_with_one_chunk_id_has_no_break(self, flag_off):
        events, _ = await _run(_node_tokens("model", "The ", "flooded ", "area.", run="run-1"))
        assert _answer(events) == "The flooded area."

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

    async def test_a_held_handle_is_released_before_the_break_not_joined(self, flag_on):
        # A marker is written inside one model call: "[S" from one call and
        # "1]" from the next are two texts, not the handle S1.
        events, applied = await _cited_run(
            ["SAR is radar [S"],
            tail=[_call("ls", {}, "c2"), _result("ls", "c2"), *_tokens("1", "] and more.")],
            tools=[_tool(RETRIEVE, "eve_retrieval"), _tool("ls")],
        )
        answer = _answer(events)
        assert answer == "SAR is radar [S\n\n1] and more."
        final = _of(events, "final")[0]
        assert final["citations"] == []
        assert _persist_kwargs(applied)["output"] == answer

    async def test_the_next_call_text_never_lands_before_the_break(self, flag_on):
        # "1" alone kept "[S1" a possible handle and glued it to the old part.
        events, applied = await _cited_run(
            ["SAR is radar [S"],
            tail=[_call("ls", {}, "c2"), _result("ls", "c2"), *_tokens("1", " km wide.")],
            tools=[_tool(RETRIEVE, "eve_retrieval"), _tool("ls")],
        )
        answer = _answer(events)
        assert "[S1" not in answer
        assert answer == "SAR is radar [S\n\n1 km wide."
        assert _persist_kwargs(applied)["output"] == answer

    async def test_a_marker_opening_the_next_call_is_rewritten(self, flag_on):
        events, _ = await _cited_run(
            ["SAR is radar [S"],
            tail=[_call("ls", {}, "c2"), _result("ls", "c2"), *_tokens("[S1] agrees.")],
            tools=[_tool(RETRIEVE, "eve_retrieval"), _tool("ls")],
        )
        answer = _answer(events)
        assert answer == "SAR is radar [S\n\n[1] agrees."
        assert marker_spans(answer, _of(events, "final")[0]["citations"]) == ["[1]"]

    async def test_a_held_bracket_that_is_no_marker_comes_before_the_break(self, flag_on):
        events, _ = await _cited_run(
            ["SAR is radar ["],
            tail=[_call("ls", {}, "c2"), _result("ls", "c2"), *_tokens("More text.")],
            tools=[_tool(RETRIEVE, "eve_retrieval"), _tool("ls")],
        )
        assert _answer(events) == "SAR is radar [\n\nMore text."
