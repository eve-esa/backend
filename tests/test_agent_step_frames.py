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

from src.routers import message as message_router
from src.schemas.generation_request import GenerationRequest
from src.services.agents.core import runner as runner_module
from src.services.agents.core.runner import (
    STREAM_TOOL_ARG_MAX_CHARS,
    generate_answer_agentic_stream_helper,
    run_agentic_generation_to_bus,
)
from src.services.cancel_manager import CancelManager
from src.services.stream_bus import StreamBus

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


def _calls(*calls):
    """One agent step that asks several tools at once."""
    tool_calls = [{"name": n, "args": a, "id": i} for n, a, i in calls]
    return ("updates", {"agent": {"messages": [AIMessage(content="", tool_calls=tool_calls)]}})


def _result(name, call_id, content="ok", status="success", duration_ms=None):
    extra = {"response_metadata": {"duration_ms": duration_ms}} if duration_ms is not None else {}
    return (
        "messages",
        (
            ToolMessage(
                content=content, name=name, tool_call_id=call_id, status=status, **extra
            ),
            {"langgraph_node": "tools"},
        ),
    )


def _react_failure(name, call_id, exc):
    """What the agents react tools node (pin v0.5.5) returns for a raising tool:
    the error text as content and the default status."""
    return _result(name, call_id, content=f"Tool error: {exc}")


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
    async def test_model_is_the_endpoint_key_and_the_provider_id_is_separate(self):
        endpoint = {"requested": "main", "chain": ["main"], "answered": "main",
                    "attempts": [], "circuit_open": [], "substituted": False}
        message = AIMessage(
            content="",
            tool_calls=[{"name": "ls", "args": {}, "id": "c1"}],
            response_metadata={"model_name": "eve-instruct"},
        )
        events, _ = await _run(
            _Graph([("updates", {"agent": {"messages": [message]}}), _answer()]),
            _resolve_agentic_llm_client=AsyncMock(
                return_value=(MagicMock(), {"endpoint": endpoint})
            ),
        )

        [call] = _of(events, "tool_call")
        assert call["model"] == "main"
        assert call["model_name"] == "eve-instruct"

    async def test_a_fallback_node_step_names_the_fallback_endpoint(self):
        endpoint = {"requested": "main", "chain": ["main"], "answered": "main",
                    "attempts": [], "circuit_open": [], "substituted": False}
        message = AIMessage(
            content="",
            tool_calls=[{"name": "ls", "args": {}, "id": "c1"}],
            response_metadata={"model_name": "mistral-small-latest"},
        )
        events, _ = await _run(
            _Graph([("updates", {"agent_fallback": {"messages": [message]}}), _answer()]),
            _resolve_agentic_llm_client=AsyncMock(
                return_value=(MagicMock(), {"endpoint": endpoint})
            ),
        )

        [call] = _of(events, "tool_call")
        assert call["model"] == "fallback"
        assert call["model_name"] == "mistral-small-latest"

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
        assert "model_name" not in call


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

    async def test_react_tool_error_without_status_is_an_error(self):
        events, _ = await _run(
            _Graph(
                [
                    _call(RETRIEVE, {"query": "q"}, "call-1"),
                    _react_failure(RETRIEVE, "call-1", "HTTP 500 from upstream"),
                    _answer(),
                ]
            ),
        )

        [result] = _of(events, "tool_result")
        assert result["status"] == "error"
        assert result["error_code"] == "tool_error"

    async def test_react_mcp_timeout_without_status_is_a_timeout(self):
        events, _ = await _run(
            _Graph(
                [
                    _call(RETRIEVE, {"query": "q"}, "call-1"),
                    _react_failure(
                        RETRIEVE,
                        "call-1",
                        "Timed out while waiting for response to ClientRequest. "
                        "Waited 30.0 seconds.",
                    ),
                    _answer(),
                ]
            ),
        )

        [result] = _of(events, "tool_result")
        assert result["status"] == "timeout"
        assert result["error_code"] == "timeout"

    async def test_unknown_tool_is_an_error(self):
        events, _ = await _run(
            _Graph(
                [
                    _call("nope", {}, "call-1"),
                    _result("nope", "call-1", content="Unknown tool: nope"),
                    _answer(),
                ]
            ),
        )

        [result] = _of(events, "tool_result")
        assert result["status"] == "error"

    async def test_a_query_that_says_timeout_does_not_make_a_timeout(self):
        events, _ = await _run(
            _Graph(
                [
                    _call(RETRIEVE, {"query": "timeout"}, "call-1"),
                    _react_failure(
                        RETRIEVE, "call-1", "no index for query 'timeout handling'"
                    ),
                    _answer(),
                ]
            ),
        )

        [result] = _of(events, "tool_result")
        assert result["status"] == "error"

    async def test_a_result_that_mentions_timeout_is_ok(self):
        events, _ = await _run(
            _Graph(
                [
                    _call(RETRIEVE, {"query": "q"}, "call-1"),
                    _result(RETRIEVE, "call-1", content="Docs on timeout handling"),
                    _answer(),
                ]
            ),
        )

        [result] = _of(events, "tool_result")
        assert result["status"] == "ok"

    async def test_each_call_of_a_step_reports_its_own_duration(self):
        events, _ = await _run(
            _Graph(
                [
                    _calls(("ls", {"path": "/a"}, "c1"), ("ls", {"path": "/b"}, "c2")),
                    _result("ls", "c1", duration_ms=5),
                    _result("ls", "c2", duration_ms=1200),
                    _answer(),
                ]
            ),
        )

        assert [(r["tool_call_id"], r["duration_ms"]) for r in _of(events, "tool_result")] == [
            ("c1", 5),
            ("c2", 1200),
        ]

    async def test_retry_gets_a_new_id_and_points_at_the_failed_attempt(self):
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
        assert [c["tool_call_id"] for c in calls] == ["call-1", "call-2"]
        assert [c["attempt"] for c in calls] == [1, 2]
        assert [c["id"] for c in calls] == ["call-1", "call-2"]
        assert [(r["tool_call_id"], r["attempt"], r["status"]) for r in results] == [
            ("call-1", 1, "error"),
            ("call-2", 2, "ok"),
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

    async def test_final_carries_the_endpoint_and_one_usage_row_per_model_call(self):
        endpoint = {"requested": "main", "chain": ["main"], "answered": "main",
                    "attempts": [], "circuit_open": [], "substituted": False}
        step = AIMessage(
            content="",
            tool_calls=[{"name": "ls", "args": {}, "id": "c1"}],
            response_metadata={"model_name": "eve-instruct"},
            usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        )
        last = AIMessage(
            content="done",
            usage_metadata={"input_tokens": 20, "output_tokens": 5, "total_tokens": 25},
        )
        events, _ = await _run(
            _Graph(
                [
                    ("updates", {"agent": {"messages": [step]}}),
                    _result("ls", "c1"),
                    _answer("done"),
                    ("updates", {"agent": {"messages": [last]}}),
                ]
            ),
            _resolve_agentic_llm_client=AsyncMock(
                return_value=(MagicMock(), {"endpoint": endpoint})
            ),
        )

        [final] = _of(events, "final")
        assert final["endpoint"]["answered"] == "main"
        assert final["usage"] == [
            {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12,
             "provider": "main", "model": "eve-instruct"},
            {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25,
             "provider": "main", "model": "main"},
        ]

    async def test_late_failure_final_keeps_endpoint_and_error(self):
        endpoint = {"requested": "main", "chain": ["main"], "answered": "main",
                    "attempts": [], "circuit_open": [], "substituted": False}
        events, _ = await _run(
            _Graph([_answer("partial")], raise_exc=RuntimeError("boom")),
            _resolve_agentic_llm_client=AsyncMock(
                return_value=(MagicMock(), {"endpoint": endpoint})
            ),
        )

        [final] = _of(events, "final")
        assert final["error"] == {"code": "upstream_error"}
        assert final["endpoint"]["requested"] == "main"
        assert "usage" not in final

    async def test_empty_answer_ends_with_the_empty_answer_error_frame(self):
        events, _ = await _run(_Graph([]))

        assert _of(events, "final") == []
        [error] = _of(events, "error")
        assert error["code"] == "empty_answer"
        assert events[-1]["type"] == "error"

    async def test_error_before_text_with_a_pending_call_is_an_error_frame(self):
        events, _ = await _run(
            _Graph(
                [_call(RETRIEVE, {"query": "q"}, "call-1")],
                raise_exc=RuntimeError("boom"),
            ),
        )

        [error] = _of(events, "error")
        assert error["code"] == "upstream_error"
        assert _of(events, "final") == []
        assert _of(events, "tool_result") == []
        assert events[-1]["type"] == "error"


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


class _BlockingGraph(_Graph):
    """Like ``_Graph``, but waits on ``gate`` before the event at ``block_at``:
    a tool still running when the Stop lands."""

    def __init__(self, events, block_at, gate=None):
        super().__init__(events)
        self._block_at = block_at
        self.gate = gate or asyncio.Event()
        self.blocked = asyncio.Event()

    async def astream(self, *args, **kwargs):
        for index, event in enumerate(self._events):
            if index == self._block_at:
                self.blocked.set()
                await self.gate.wait()
            yield event


def _local_cancel_manager():
    cm = CancelManager()
    cm._is_redis_enabled = lambda: False
    return cm


@contextlib.asynccontextmanager
async def _route_turn(graph, tools=()):
    """What the agentic stream route runs, on the in-process bus and cancel
    manager: message registered, bus runner task, one subscriber."""
    bus = StreamBus()
    cm = _local_cancel_manager()
    frames = []
    ready = asyncio.Event()

    async def _subscribe():
        async for data in bus.subscribe("m1", ready=ready):
            frame = json.loads(data.removeprefix("data: ").strip())
            # The opening thinking notice is not under test here.
            if frame.get("type") != "status":
                frames.append(frame)

    with _runner(graph, tools) as applied, contextlib.ExitStack() as stack:
        for target in (
            "src.routers.message.get_cancel_manager",
            "src.services.cancel_manager.get_cancel_manager",
            "src.services.generate_answer.get_cancel_manager",
        ):
            stack.enter_context(patch(target, MagicMock(return_value=cm)))
        stack.enter_context(patch("src.routers.message.get_stream_bus", MagicMock(return_value=bus)))
        stack.enter_context(patch(f"{_RUNNER}.get_stream_bus", MagicMock(return_value=bus)))
        stack.enter_context(patch.object(runner_module.User, "find_by_id", AsyncMock(return_value=None)))
        stack.enter_context(patch.object(runner_module.Message, "find_by_id", AsyncMock(return_value=None)))
        stack.enter_context(patch(f"{_RUNNER}.persist_runner_cancel", AsyncMock()))
        cancel_event = cm.create("m1")
        cm.link_conversation("c1", "m1")
        subscriber = asyncio.create_task(_subscribe())
        task = asyncio.create_task(
            run_agentic_generation_to_bus(
                GenerationRequest(query="hi", llm_type="main", agent="react"),
                conversation_id="c1",
                message_id="m1",
                user_id="test-user",
                cancel_event=cancel_event,
                subscriber_ready=ready,
            )
        )
        cm.set_task("m1", task)
        turn = SimpleNamespace(
            bus=bus, cm=cm, frames=frames, task=task, subscriber=subscriber,
            cancel_event=cancel_event, applied=applied,
        )
        try:
            yield turn
        finally:
            for pending in (task, subscriber):
                if not pending.done():
                    pending.cancel()
                with contextlib.suppress(BaseException):
                    await pending


def _stopped_persist(applied):
    calls = [
        c for c in applied["persist_message_state"].await_args_list
        if c.kwargs.get("stopped") is True
    ]
    assert calls, "the turn was never persisted as stopped"
    return calls[-1].kwargs


def _trace_steps(trace):
    return {
        call["step"]["tool_call_id"]: call["step"]
        for entry in trace or []
        for call in entry.get("tool_calls") or []
        if "step" in call
    }


class TestStopThroughTheRoute:
    async def test_same_worker_stop_mid_tool_persists_the_cut_call(self):
        graph = _BlockingGraph(
            [_call(RETRIEVE, {"query": "q"}, "call-1"), _result(RETRIEVE, "call-1"), _answer()],
            block_at=1,
        )
        async with _route_turn(graph, tools=[_tool(RETRIEVE, "eve_retrieval")]) as turn:
            await asyncio.wait_for(graph.blocked.wait(), 5)
            assert await message_router._signal_stop("c1") == "m1"
            await asyncio.wait_for(turn.task, 5)
            await asyncio.wait_for(turn.subscriber, 5)

        persisted = _stopped_persist(turn.applied)
        step = _trace_steps(persisted["trace"])["call-1"]
        assert step["status"] == "cancelled"
        assert isinstance(step["duration_ms"], int)
        # The client got the route's plain stopped frame, no result for the chip.
        assert [f["type"] for f in turn.frames] == ["tool_call", "stopped"]
        assert turn.frames[-1] == {"type": "stopped"}

    async def test_cross_worker_stop_cuts_the_unfinished_call_only(self):
        # The Stop lands on another worker: it sets the cancel flag (here the
        # event, as the Redis subscription does) and ends the stream; this
        # worker's task is not cancelled and sees the flag on its next event,
        # the finished result of the first call.
        graph = _BlockingGraph(
            [
                _calls((RETRIEVE, {"query": "a"}, "call-1"), (RETRIEVE, {"query": "b"}, "call-2")),
                _result(RETRIEVE, "call-1"),
                _result(RETRIEVE, "call-2"),
                _answer(),
            ],
            block_at=1,
        )
        async with _route_turn(graph, tools=[_tool(RETRIEVE, "eve_retrieval")]) as turn:
            await asyncio.wait_for(graph.blocked.wait(), 5)
            turn.cancel_event.set()
            await turn.bus.stop("m1", f"data: {json.dumps({'type': 'stopped'})}\n\n")
            await asyncio.wait_for(turn.subscriber, 5)
            graph.gate.set()
            await asyncio.wait_for(turn.task, 5)

        persisted = _stopped_persist(turn.applied)
        steps = _trace_steps(persisted["trace"])
        assert steps["call-2"]["status"] == "cancelled"
        assert "status" not in steps["call-1"]
        [tool_entry] = [e for e in persisted["trace"] if e.get("role") == "tool"]
        assert tool_entry["step"]["tool_call_id"] == "call-1"
        assert tool_entry["step"]["status"] == "ok"
        assert [f["type"] for f in turn.frames] == ["tool_call", "tool_call", "stopped"]
        assert all("cancelled_tool_calls" not in f for f in turn.frames)

    async def test_stop_before_the_first_token(self):
        graph = _BlockingGraph([_answer("never")], block_at=0)
        async with _route_turn(graph) as turn:
            await asyncio.wait_for(graph.blocked.wait(), 5)
            await message_router._signal_stop("c1")
            await asyncio.wait_for(turn.task, 5)
            await asyncio.wait_for(turn.subscriber, 5)

        persisted = _stopped_persist(turn.applied)
        assert persisted["output"] == ""
        assert persisted["trace"] is None
        assert turn.frames == [{"type": "stopped"}]

    async def test_a_ghost_call_is_neither_cancelled_nor_finished(self):
        # call-ghost is announced with call-1 but no tool answers it (the
        # text-format repeat): the next agent step leaves it behind.
        graph = _BlockingGraph(
            [
                _calls(("ls", {"path": "/a"}, "call-1"), ("ls", {"path": "/g"}, "call-ghost")),
                _result("ls", "call-1"),
                _call("ls", {"path": "/b"}, "call-2"),
                _result("ls", "call-2"),
            ],
            block_at=3,
        )
        async with _route_turn(graph) as turn:
            await asyncio.wait_for(graph.blocked.wait(), 5)
            await message_router._signal_stop("c1")
            await asyncio.wait_for(turn.task, 5)

        steps = _trace_steps(_stopped_persist(turn.applied)["trace"])
        assert steps["call-ghost"]["status"] == "not_run"
        assert steps["call-2"]["status"] == "cancelled"
        assert "status" not in steps["call-1"]


class TestNoUserTextInLogs:
    async def test_guard_args_and_query_never_reach_the_log(self, caplog):
        """Guard: this diff adds no log line. It keeps a later one from logging
        args, results or the question. The agents tools node logs its own
        exception text, outside this runner."""
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
