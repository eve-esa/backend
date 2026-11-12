"""Enriched agentic stream frames (additive fields on the legacy frames).

The legacy client reads ``type``, ``content``, ``tool``, ``label``, ``query``,
``id``, ``args`` and ``status`` and ignores everything else, so every field
here is new and every legacy field keeps its old value.
"""

import asyncio
import contextlib
import json
import logging
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from src.schemas.generation_request import GenerationRequest
from src.services.agents.core.runner import (
    STREAM_TOOL_ARG_MAX_CHARS,
    generate_answer_agentic_stream_helper,
)

_RUNNER = "src.services.agents.core.runner"

pytestmark = pytest.mark.no_db


class _Graph:
    """Yields ``(mode, payload)`` pairs in order, then raises ``raise_exc``."""

    def __init__(self, events, raise_exc=None, before_each=None):
        self._events = events
        self._raise = raise_exc
        self._before_each = before_each

    async def astream(self, *args, **kwargs):
        for index, event in enumerate(self._events):
            if self._before_each is not None:
                self._before_each(index)
            yield event
        if self._raise is not None:
            raise self._raise


def _agent():
    agent = MagicMock()
    agent.instruction_text.return_value = ""
    agent.prompts = {}
    return agent


def _tool(name, server=None):
    return SimpleNamespace(
        name=name, metadata={"eve_mcp_server": server} if server else None
    )


@contextlib.contextmanager
def _runner(graph, tools=(), **overrides):
    defaults = {
        "_langgraph_available": True,
        "_build_tools": AsyncMock(return_value=list(tools)),
        "_get_agentic_checkpointer": AsyncMock(return_value=None),
        "_fetch_conversation_context": AsyncMock(return_value=([], None)),
        "_resolve_agent_graph_type": MagicMock(return_value="react"),
        "get_agent_graph": MagicMock(return_value=_agent()),
        "persist_policy_event": AsyncMock(),
        "persist_message_state": AsyncMock(),
        "maybe_rollup_and_trim_history": AsyncMock(),
        "_build_react_graph": MagicMock(return_value=graph),
    }
    defaults.update(overrides)
    with contextlib.ExitStack() as stack:
        applied = {
            name: stack.enter_context(patch(f"{_RUNNER}.{name}", value))
            for name, value in defaults.items()
        }
        yield applied


async def _run(graph, tools=(), cancel_event=None, query="hi", **overrides):
    request = GenerationRequest(query=query, llm_type="main", agent="react")
    with _runner(graph, tools, **overrides) as applied:
        events = [
            json.loads(event.removeprefix("data: ").strip())
            async for event in generate_answer_agentic_stream_helper(
                request,
                conversation_id="c1",
                message_id="m1",
                user_id="test-user",
                cancel_event=cancel_event,
            )
        ]
    return events, applied


def _of(events, kind):
    return [e for e in events if e["type"] == kind]


def _call(name, args, call_id):
    call = {"name": name, "args": args, "id": call_id}
    return ("updates", {"agent": {"messages": [AIMessage(content="", tool_calls=[call])]}})


def _result(name, call_id, content="ok", status="success"):
    return (
        "messages",
        (
            ToolMessage(content=content, name=name, tool_call_id=call_id, status=status),
            {"langgraph_node": "tools"},
        ),
    )


def _answer(text="done"):
    return ("messages", (AIMessage(content=text), {"langgraph_node": "agent"}))


RETRIEVE = "eve_retrieval_retrieve"


class TestToolCallFields:
    async def test_mcp_call_carries_id_kind_server_and_start(self):
        events, _ = await _run(
            _Graph([_call(RETRIEVE, {"query": "sea ice"}, "call-1"), _answer()]),
            tools=[_tool(RETRIEVE, server="eve_retrieval")],
        )

        [call] = _of(events, "tool_call")
        assert call["tool_call_id"] == "call-1"
        assert call["kind"] == "mcp"
        assert call["server"] == "eve_retrieval"
        assert datetime.fromisoformat(call["started_at"]).tzinfo is not None
        assert "args_truncated" not in call
        # Legacy fields untouched.
        assert call["id"] == "call-1"
        assert call["tool"] == RETRIEVE

    async def test_builtin_tool_has_no_server(self):
        events, _ = await _run(
            _Graph([_call("ls", {"path": "/"}, "call-1"), _answer()]),
        )

        [call] = _of(events, "tool_call")
        assert call["kind"] == "builtin"
        assert call["server"] is None

    async def test_call_without_id_gets_one_and_its_result_pairs(self):
        events, _ = await _run(
            _Graph([_call("ls", {"path": "/"}, None), _result("ls", ""), _answer()]),
        )

        [call] = _of(events, "tool_call")
        [result] = _of(events, "tool_result")
        assert call["id"] is None
        assert call["tool_call_id"]
        assert result["tool_call_id"] == call["tool_call_id"]

    async def test_raw_id_reused_for_a_new_call_is_made_unique(self):
        events, _ = await _run(
            _Graph(
                [
                    _call("ls", {"path": "/a"}, "call_1_x"),
                    _result("ls", "call_1_x"),
                    _call("ls", {"path": "/b"}, "call_1_x"),
                    _result("ls", "call_1_x"),
                    _answer(),
                ]
            ),
        )

        ids = [c["tool_call_id"] for c in _of(events, "tool_call")]
        assert ids == ["call_1_x", "call_1_x~2"]
        assert [r["tool_call_id"] for r in _of(events, "tool_result")] == ids

    async def test_cut_arguments_are_flagged(self):
        events, _ = await _run(
            _Graph(
                [
                    _call(
                        "write_file",
                        {"file_path": "a.py", "content": "x" * (STREAM_TOOL_ARG_MAX_CHARS + 1)},
                        "call-1",
                    ),
                    _answer(),
                ]
            ),
        )

        [call] = _of(events, "tool_call")
        assert call["args_truncated"] is True


class TestStepModel:
    async def test_call_names_the_model_from_the_message(self):
        message = AIMessage(
            content="",
            tool_calls=[{"name": "ls", "args": {}, "id": "c1"}],
            response_metadata={"model_name": "eve-instruct"},
        )
        events, _ = await _run(
            _Graph([("updates", {"agent": {"messages": [message]}}), _answer()]),
        )

        [call] = _of(events, "tool_call")
        assert call["model"] == "eve-instruct"

    async def test_call_falls_back_to_the_answering_endpoint(self):
        endpoint = {"requested": "main", "chain": ["main"], "answered": "main",
                    "attempts": [], "circuit_open": [], "substituted": False}
        events, _ = await _run(
            _Graph([_call("ls", {}, "c1"), _answer()]),
            _resolve_agentic_llm_client=AsyncMock(
                return_value=(MagicMock(), {"endpoint": endpoint})
            ),
        )

        [call] = _of(events, "tool_call")
        assert call["model"] == "main"


class TestToolResultFields:
    async def test_ok_result_has_status_duration_and_attempt(self):
        events, _ = await _run(
            _Graph(
                [
                    _call(RETRIEVE, {"query": "q"}, "call-1"),
                    _result(RETRIEVE, "call-1"),
                    _answer(),
                ]
            ),
            tools=[_tool(RETRIEVE, server="eve_retrieval")],
        )

        [result] = _of(events, "tool_result")
        assert result["status"] == "ok"
        assert result["tool_call_id"] == "call-1"
        assert isinstance(result["duration_ms"], int) and result["duration_ms"] >= 0
        assert result["attempt"] == 1
        assert result["server"] == "eve_retrieval"
        assert result["kind"] == "mcp"
        assert "error_code" not in result

    async def test_error_result_reports_error(self):
        events, _ = await _run(
            _Graph(
                [
                    _call(RETRIEVE, {"query": "q"}, "call-1"),
                    _result(RETRIEVE, "call-1", content="Error: boom", status="error"),
                    _answer(),
                ]
            ),
        )

        [result] = _of(events, "tool_result")
        assert result["status"] == "error"
        assert result["error_code"] == "tool_error"

    async def test_timed_out_tool_reports_timeout(self):
        events, _ = await _run(
            _Graph(
                [
                    _call(RETRIEVE, {"query": "q"}, "call-1"),
                    _result(
                        RETRIEVE,
                        "call-1",
                        content="Error: TimeoutError('read timed out')",
                        status="error",
                    ),
                    _answer(),
                ]
            ),
        )

        [result] = _of(events, "tool_result")
        assert result["status"] == "timeout"
        assert result["error_code"] == "timeout"

    async def test_retried_call_keeps_its_id_with_attempt_two(self):
        events, _ = await _run(
            _Graph(
                [
                    _call(RETRIEVE, {"query": "q"}, "call-1"),
                    _result(RETRIEVE, "call-1", content="Error: boom", status="error"),
                    _call(RETRIEVE, {"query": "q"}, "call-2"),
                    _result(RETRIEVE, "call-2"),
                    _answer(),
                ]
            ),
        )

        calls = _of(events, "tool_call")
        results = _of(events, "tool_result")
        assert [c["tool_call_id"] for c in calls] == ["call-1", "call-1"]
        assert [c["attempt"] for c in calls] == [1, 2]
        assert [c["id"] for c in calls] == ["call-1", "call-2"]
        assert [(r["tool_call_id"], r["attempt"], r["status"]) for r in results] == [
            ("call-1", 1, "error"),
            ("call-1", 2, "ok"),
        ]
        assert "retry_of" not in calls[0] and "retry_of" not in results[0]
        assert calls[1]["retry_of"] == "call-1"
        assert results[1]["retry_of"] == "call-1"

    async def test_same_call_after_a_success_is_a_new_call(self):
        events, _ = await _run(
            _Graph(
                [
                    _call(RETRIEVE, {"query": "q"}, "call-1"),
                    _result(RETRIEVE, "call-1"),
                    _call(RETRIEVE, {"query": "q"}, "call-2"),
                    _result(RETRIEVE, "call-2"),
                    _answer(),
                ]
            ),
        )

        calls = _of(events, "tool_call")
        assert [(c["tool_call_id"], c["attempt"]) for c in calls] == [
            ("call-1", 1),
            ("call-2", 1),
        ]


class TestSkillAndSubagent:
    async def test_reading_a_skill_file_is_marked(self):
        events, _ = await _run(
            _Graph(
                [
                    _call("read_file", {"file_path": "/skills/ndvi-map/SKILL.md"}, "c1"),
                    _call("read_file", {"file_path": "/workspace/notes.md"}, "c2"),
                    _answer(),
                ]
            ),
        )

        skill, plain = _of(events, "tool_call")
        assert skill["kind"] == "skill"
        assert skill["skill_name"] == "ndvi-map"
        assert plain["kind"] == "builtin"
        assert "skill_name" not in plain

    async def test_task_call_is_a_subagent_with_a_run_id(self):
        events, _ = await _run(
            _Graph(
                [
                    _call(
                        "task",
                        {"description": "review", "subagent_type": "literature-review"},
                        "call-7",
                    ),
                    _answer(),
                ]
            ),
        )

        [call] = _of(events, "tool_call")
        assert call["kind"] == "subagent"
        assert call["subagent_run_id"] == "sa-call-7"
        assert call["subagent_name"] == "literature-review"


class TestModelFallbackFrame:
    async def test_in_graph_fallback_emits_one_frame(self):
        fallback_update = {
            "agent_fallback": {"messages": [AIMessage(content="")]}
        }
        events, _ = await _run(
            _Graph(
                [
                    ("updates", fallback_update),
                    ("updates", fallback_update),
                    _answer(),
                ]
            ),
        )

        frames = _of(events, "model_fallback")
        assert len(frames) == 1
        assert frames[0]["to"] == "fallback"
        assert frames[0]["reason"] == "agent_error"
        assert "from" in frames[0]

    async def test_open_circuit_at_resolution_emits_a_frame(self):
        endpoint = {
            "requested": "main",
            "chain": ["fallback", "main"],
            "answered": "fallback",
            "attempts": [],
            "circuit_open": ["main"],
            "substituted": True,
        }
        events, _ = await _run(
            _Graph([_answer()]),
            _resolve_agentic_llm_client=AsyncMock(
                return_value=(MagicMock(), {"endpoint": endpoint})
            ),
        )

        assert _of(events, "model_fallback") == [
            {"type": "model_fallback", "from": "main", "to": "fallback", "reason": "circuit_open"}
        ]

    async def test_no_frame_on_a_plain_turn(self):
        events, _ = await _run(_Graph([_answer()]))
        assert _of(events, "model_fallback") == []


class TestFinalError:
    async def test_error_after_streamed_text_is_on_final(self):
        events, _ = await _run(
            _Graph([_answer("partial")], raise_exc=RuntimeError("boom")),
        )

        [final] = _of(events, "final")
        assert final["answer"] == "partial"
        assert final["error"] == {"code": "upstream_error"}
        assert _of(events, "error") == []

    async def test_clean_final_has_no_error(self):
        events, _ = await _run(_Graph([_answer()]))
        [final] = _of(events, "final")
        assert "error" not in final


class TestPersistedTrace:
    async def test_trace_steps_carry_the_frame_fields(self):
        events, applied = await _run(
            _Graph(
                [
                    _call(RETRIEVE, {"query": "q"}, "call-1"),
                    _result(RETRIEVE, "call-1", content="Error: boom", status="error"),
                    _answer(),
                ]
            ),
            tools=[_tool(RETRIEVE, server="eve_retrieval")],
        )

        trace = applied["persist_message_state"].await_args.kwargs["trace"]
        [agent_entry] = [e for e in trace if e.get("tool_calls")]
        [tool_entry] = [e for e in trace if e.get("role") == "tool"]
        call_step = agent_entry["tool_calls"][0]["step"]
        [call] = _of(events, "tool_call")
        [result] = _of(events, "tool_result")
        for key in ("tool_call_id", "kind", "server", "started_at", "attempt"):
            assert call_step[key] == call[key]
        for key in ("tool_call_id", "status", "duration_ms", "error_code", "attempt"):
            assert tool_entry["step"][key] == result[key]
        # Raw LangChain fields keep their meaning for old readers.
        assert agent_entry["tool_calls"][0]["id"] == "call-1"
        assert tool_entry["status"] == "error"


class TestStopWithPendingCall:
    async def test_stopped_frame_lists_cancelled_calls(self):
        cancel = asyncio.Event()

        def stop_before_answer(index):
            if index == 1:
                cancel.set()

        events, _ = await _run(
            _Graph(
                [_call(RETRIEVE, {"query": "q"}, "call-1"), _answer()],
                before_each=stop_before_answer,
            ),
            cancel_event=cancel,
        )

        [stopped] = _of(events, "stopped")
        [cancelled] = stopped["cancelled_tool_calls"]
        assert cancelled["tool_call_id"] == "call-1"
        assert cancelled["status"] == "cancelled"
        # No tool_result: the legacy client would mark the chip done.
        assert _of(events, "tool_result") == []


class TestNoUserTextInLogs:
    async def test_args_and_query_never_reach_the_log(self, caplog):
        caplog.set_level(logging.DEBUG)
        await _run(
            _Graph(
                [
                    _call(RETRIEVE, {"query": "private-query-text", "token": "sk-secret-123"}, "c1"),
                    _result(RETRIEVE, "c1", content="Error: sk-secret-123", status="error"),
                    _answer("answer"),
                ],
                raise_exc=RuntimeError("late"),
            ),
            query="private-user-question",
        )

        for needle in ("private-query-text", "sk-secret-123", "private-user-question"):
            assert needle not in caplog.text
