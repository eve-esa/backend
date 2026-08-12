"""Unit tests for the agentic LangGraph error-handling layer.

Covers:
- _resolve_agentic_llm_type: request llm_type precedence over AGENTIC_LLM_TYPE
- _build_react_graph: delegates llm_type resolution + in-graph fallback_llm wiring
- _resolve_agentic_llm_client: endpoint chain resolution and honest attribution
- generate_answer_agentic: records in-graph fallback via agent_fallback node
- generate_answer_agentic_stream_helper: no UnboundLocalError on early setup failure
- generate_answer_agentic_stream_helper: pre-answer status event
- node budget semantics: endpoint budget is the idle cap, not the wall clock
- error paths: attribution survives a failed turn, and never contradict a final
"""

import asyncio
import contextlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from src.config import AGENTIC_TIMEOUT, MODEL_TIMEOUT
from src.core.llm_manager import LLMManager
from src.services.agents.core.runner import (
    _build_react_graph,
    _resolve_agentic_llm_client,
    _resolve_agentic_llm_type,
    _serialise_trace_entry,
    generate_answer_agentic,
    generate_answer_agentic_stream_helper,
)
from src.schemas.generation_request import GenerationRequest

_RUNNER = "src.services.agents.core.runner"


def _chain_manager(configured=("eve_jsc", "main", "fallback")) -> LLMManager:
    """A manager with a real chain resolver and no real endpoint behind it."""
    manager = LLMManager()
    manager._is_configured = lambda name: name in configured
    for getter in ("_get_eve_jsc_llm", "_get_main_llm", "_get_fallback_llm"):
        setattr(manager, getter, MagicMock(return_value=MagicMock(name=getter)))
    return manager


class _FakeStreamGraph:
    """Graph that streams fallback output then raises the primary-node error."""

    def __init__(self, updates=None, messages=None, raise_exc=None):
        self._updates = updates or []
        self._messages = messages or []
        self._raise = raise_exc

    async def astream(self, *args, **kwargs):
        for update in self._updates:
            yield "updates", update
        for message in self._messages:
            yield "messages", message
        if self._raise is not None:
            raise self._raise


class _FakeUpdatesGraph:
    """Minimal stand-in for a compiled graph driving ``astream(stream_mode=...)``."""

    def __init__(self, updates=None, raise_exc=None):
        self._updates = updates or []
        self._raise = raise_exc

    async def astream(self, *args, **kwargs):
        for update in self._updates:
            yield update
        if self._raise is not None:
            raise self._raise


def _fake_agent():
    agent = MagicMock()
    agent.instruction_text.return_value = ""
    agent.prompts = {}
    return agent


@contextlib.contextmanager
def _patched_runner(**overrides):
    """Patch the runner's external boundaries with sensible async defaults."""
    error_logger = MagicMock()
    error_logger.log_error = AsyncMock()

    defaults = {
        "_langgraph_available": True,
        "_build_tools": AsyncMock(return_value=[]),
        "_get_agentic_checkpointer": AsyncMock(return_value=None),
        "_fetch_conversation_context": AsyncMock(return_value=([], None)),
        "_resolve_agent_graph_type": MagicMock(return_value="react"),
        "get_agent_graph": MagicMock(return_value=_fake_agent()),
        "get_callbacks": MagicMock(return_value=[]),
        "get_error_logger": MagicMock(return_value=error_logger),
        "langfuse_context": lambda **kwargs: contextlib.nullcontext(),
        "persist_message_state": AsyncMock(),
        "maybe_rollup_and_trim_history": AsyncMock(),
    }
    defaults.update(overrides)

    with contextlib.ExitStack() as stack:
        applied = {}
        for name, value in defaults.items():
            applied[name] = stack.enter_context(patch(f"{_RUNNER}.{name}", value))
        applied["error_logger"] = error_logger
        yield applied


# ─── trace serialisation ──────────────────────────────────────────────────────


class TestSerialiseTraceEntry:
    def test_includes_response_and_usage_metadata(self):
        msg = AIMessage(
            content="answer",
            id="msg-1",
            response_metadata={
                "model_name": "gpt-4",
                "finish_reason": "stop",
                "token_usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
            tool_calls=[{"name": "search", "args": {"q": "x"}, "id": "call-1"}],
        )

        entry = _serialise_trace_entry(msg, node="agent", latency_s=1.2)

        assert entry["response_metadata"]["model_name"] == "gpt-4"
        assert entry["usage_metadata"]["total_tokens"] == 15
        assert entry["id"] == "msg-1"
        assert entry["tool_calls"] == [
            {"name": "search", "args": {"q": "x"}, "id": "call-1"}
        ]

    def test_includes_tool_message_correlation_fields(self):
        msg = ToolMessage(
            content="geocode result",
            name="geocode",
            tool_call_id="call-1",
            id="tool-1",
            status="success",
        )

        entry = _serialise_trace_entry(msg, node="tools")

        assert entry["tool_call_id"] == "call-1"
        assert entry["status"] == "success"
        assert entry["id"] == "tool-1"


# ─── _build_react_graph ───────────────────────────────────────────────────────


class TestBuildReactGraph:
    def _make_agent(self):
        agent = MagicMock()
        agent.compile.return_value = MagicMock(name="compiled_graph")
        return agent

    def test_compiles_with_resolved_llm(self):
        agent = self._make_agent()
        fake_llm = MagicMock(name="llm_client")
        fake_llm_manager = MagicMock()
        fake_llm_manager.get_client_for_model.return_value = fake_llm

        with patch(f"{_RUNNER}.AGENTIC_LLM_TYPE", None), patch(
            "src.services.agents.core.runner.get_shared_llm_manager",
            return_value=fake_llm_manager,
        ):
            graph = _build_react_graph(
                "main",
                tools=[],
                checkpointer=None,
                agent=agent,
                history=[],
                summary=None,
            )

        fake_llm_manager.get_client_for_model.assert_any_call("main")
        fake_llm_manager.get_client_for_model.assert_any_call("fallback")
        agent.compile.assert_called_once()
        compile_kwargs = agent.compile.call_args.kwargs
        assert compile_kwargs["llm"] is fake_llm
        assert graph is agent.compile.return_value

    def test_llm_type_override_takes_precedence(self):
        agent = self._make_agent()
        fake_llm_manager = MagicMock()
        fake_llm_manager.get_client_for_model.return_value = MagicMock()

        with patch(
            "src.services.agents.core.runner.get_shared_llm_manager",
            return_value=fake_llm_manager,
        ):
            _build_react_graph(
                "main",
                tools=[],
                checkpointer=None,
                agent=agent,
                llm_type_override="fallback",
            )

        fake_llm_manager.get_client_for_model.assert_called_with("fallback")
        assert fake_llm_manager.get_client_for_model.call_count == 2

    def test_fallback_llm_forwarded_to_compile(self):
        agent = self._make_agent()
        fallback_llm = MagicMock(name="fallback_llm")
        fake_llm_manager = MagicMock()
        fake_llm_manager.get_client_for_model.return_value = MagicMock()

        with patch(
            "src.services.agents.core.runner.get_shared_llm_manager",
            return_value=fake_llm_manager,
        ):
            _build_react_graph(
                "main",
                tools=[],
                checkpointer=None,
                agent=agent,
                fallback_llm=fallback_llm,
            )

        fake_llm_manager.get_client_for_model.assert_called_once_with("main")
        compile_kwargs = agent.compile.call_args.kwargs
        assert compile_kwargs["fallback_llm"] is fallback_llm

    def test_default_in_graph_fallback_llm_wired(self):
        agent = self._make_agent()
        primary_llm = MagicMock(name="primary_llm")
        fallback_llm = MagicMock(name="fallback_llm")
        fake_mgr = MagicMock()

        def get_client(llm_type):
            if llm_type == "fallback":
                return fallback_llm
            return primary_llm

        fake_mgr.get_client_for_model.side_effect = get_client

        with patch(f"{_RUNNER}.AGENTIC_LLM_TYPE", None), patch(
            "src.services.agents.core.runner.get_shared_llm_manager",
            return_value=fake_mgr,
        ):
            _build_react_graph("main", [], None, agent=agent)

        compile_kwargs = agent.compile.call_args.kwargs
        assert compile_kwargs["llm"] is primary_llm
        assert compile_kwargs["fallback_llm"] is fallback_llm

    def test_llm_run_timeout_forwarded_to_compile(self):
        agent = self._make_agent()
        fake_mgr = MagicMock()
        fake_mgr.get_client_for_model.return_value = MagicMock()

        with patch(f"{_RUNNER}.get_shared_llm_manager", return_value=fake_mgr):
            _build_react_graph("main", [], None, agent=agent, llm_run_timeout=120)

        assert agent.compile.call_args.kwargs["llm_run_timeout"] == 120

    def test_llm_idle_timeout_forwarded_to_compile(self):
        agent = self._make_agent()
        fake_mgr = MagicMock()
        fake_mgr.get_client_for_model.return_value = MagicMock()

        with patch(f"{_RUNNER}.get_shared_llm_manager", return_value=fake_mgr):
            _build_react_graph("main", [], None, agent=agent, llm_idle_timeout=15)

        assert agent.compile.call_args.kwargs["llm_idle_timeout"] == 15

    def test_default_budgets_keep_the_wall_clock_above_the_stall_cap(self):
        """A first-token budget as the wall clock would cap answer *length*."""
        agent = self._make_agent()
        fake_mgr = MagicMock()
        fake_mgr.get_client_for_model.return_value = MagicMock()

        with patch(f"{_RUNNER}.get_shared_llm_manager", return_value=fake_mgr):
            _build_react_graph("main", [], None, agent=agent)

        kwargs = agent.compile.call_args.kwargs
        assert kwargs["llm_run_timeout"] == AGENTIC_TIMEOUT
        assert kwargs["llm_idle_timeout"] == MODEL_TIMEOUT

    def test_custom_llm_still_wires_platform_fallback(self):
        agent = self._make_agent()
        custom_llm = MagicMock(name="custom_llm")
        fallback_llm = MagicMock(name="fallback_llm")
        fake_mgr = MagicMock()
        fake_mgr.get_client_for_model.return_value = fallback_llm

        with patch(f"{_RUNNER}.AGENTIC_LLM_TYPE", None), patch(
            "src.services.agents.core.runner.get_shared_llm_manager",
            return_value=fake_mgr,
        ):
            _build_react_graph(
                "main",
                tools=[],
                checkpointer=None,
                agent=agent,
                llm=custom_llm,
            )

        fake_mgr.get_client_for_model.assert_called_once_with("fallback")
        compile_kwargs = agent.compile.call_args.kwargs
        assert compile_kwargs["llm"] is custom_llm
        assert compile_kwargs["fallback_llm"] is fallback_llm


# ─── _resolve_agentic_llm_type ────────────────────────────────────────────────


class TestResolveAgenticLlmType:
    def test_explicit_override_wins(self):
        with patch(f"{_RUNNER}.AGENTIC_LLM_TYPE", "fallback"):
            assert (
                _resolve_agentic_llm_type("main", override="satcom_large")
                == "satcom_large"
            )

    def test_request_overrides_env(self):
        with patch(f"{_RUNNER}.AGENTIC_LLM_TYPE", "fallback"):
            assert _resolve_agentic_llm_type("main") == "main"

    def test_env_used_when_request_unset(self):
        with patch(f"{_RUNNER}.AGENTIC_LLM_TYPE", "fallback"):
            assert _resolve_agentic_llm_type(None) == "fallback"

    def test_request_used_when_env_unset(self):
        with patch(f"{_RUNNER}.AGENTIC_LLM_TYPE", None):
            assert _resolve_agentic_llm_type("main") == "main"

    def test_none_when_neither_set(self):
        with patch(f"{_RUNNER}.AGENTIC_LLM_TYPE", None):
            assert _resolve_agentic_llm_type(None) is None


class TestBuildReactGraphHonoursRequestLlmType:
    def test_request_llm_type_passed_to_llm_manager(self):
        agent = MagicMock()
        agent.compile.return_value = MagicMock(name="graph")
        fake_mgr = MagicMock()
        fake_mgr.get_client_for_model.return_value = MagicMock()

        with patch(f"{_RUNNER}.AGENTIC_LLM_TYPE", "fallback"), patch(
            f"{_RUNNER}.get_shared_llm_manager", return_value=fake_mgr
        ):
            _build_react_graph("main", [], None, agent=agent)

        fake_mgr.get_client_for_model.assert_any_call("main")
        fake_mgr.get_client_for_model.assert_any_call("fallback")


# ─── _resolve_agentic_llm_client ──────────────────────────────────────────────


class TestResolveAgenticLlmClient:
    async def test_unnamed_request_takes_the_head_of_the_chain(self):
        manager = _chain_manager()
        request = GenerationRequest(query="hi", agent="react")

        with patch(f"{_RUNNER}.AGENTIC_LLM_TYPE", None), patch(
            f"{_RUNNER}.get_shared_llm_manager", return_value=manager
        ), patch("src.core.llm_manager.EVE_ENDPOINT_ORDER", "eve_jsc,main,fallback"):
            _llm, metadata = await _resolve_agentic_llm_client(
                request, user_id="test-user"
            )

        assert metadata["agentic_llm_resolved"] == "eve_jsc"
        assert metadata["endpoint"]["chain"] == ["eve_jsc", "main", "fallback"]
        assert metadata["endpoint"]["substituted"] is False

    async def test_open_circuit_moves_the_pick_to_the_next_candidate(self):
        """The resolved name must be the endpoint that will really be called."""
        manager = _chain_manager()
        manager.health.record_failure("eve_jsc", TimeoutError("cold start"))
        request = GenerationRequest(query="hi", agent="react")

        with patch(f"{_RUNNER}.AGENTIC_LLM_TYPE", None), patch(
            f"{_RUNNER}.get_shared_llm_manager", return_value=manager
        ), patch("src.core.llm_manager.EVE_ENDPOINT_ORDER", "eve_jsc,main,fallback"):
            llm, metadata = await _resolve_agentic_llm_client(
                request, user_id="test-user"
            )

        assert metadata["agentic_llm_resolved"] == "main"
        assert metadata["endpoint"]["answered"] == "main"
        assert metadata["endpoint"]["circuit_open"] == ["eve_jsc"]
        assert llm is manager._get_main_llm.return_value

    async def test_an_unconfigured_explicit_pick_reports_the_substitution(self):
        manager = _chain_manager(configured=("main", "fallback"))
        request = GenerationRequest(query="hi", llm_type="eve_jsc", agent="react")

        with patch(f"{_RUNNER}.AGENTIC_LLM_TYPE", None), patch(
            f"{_RUNNER}.get_shared_llm_manager", return_value=manager
        ):
            _llm, metadata = await _resolve_agentic_llm_client(
                request, user_id="test-user"
            )

        assert metadata["agentic_llm_resolved"] == "fallback"
        assert metadata["endpoint"]["substituted"] is True


# ─── generate_answer_agentic (non-streaming) ──────────────────────────────────


class TestGenerateAnswerAgenticInGraphFallback:
    async def test_in_graph_fallback_recorded_in_metadata(self):
        """A mid-run ``agent_fallback`` update must surface in persisted prompts."""
        updates = [
            {"agent_fallback": {"messages": [AIMessage(content="")]}},
            {"agent": {"messages": [AIMessage(content="final answer")]}},
        ]
        graph = _FakeUpdatesGraph(updates)
        request = GenerationRequest(query="hi", llm_type="main", agent="react")

        with _patched_runner(
            _build_react_graph=MagicMock(return_value=graph),
        ):
            (
                final_answer,
                _tool_results,
                _use_rag,
                _latencies,
                prompts,
                _trace,
                _artifact_ids,
            ) = await generate_answer_agentic(
                request, user_id="test-user", conversation_id="c1"
            )

        assert final_answer == "final answer"
        assert prompts["used_fallback_llm"] is True

    async def test_in_graph_fallback_survives_graph_raise(self):
        """Non-streaming must return fallback output when the graph raises afterward."""
        updates = [
            {"agent_fallback": {"messages": [AIMessage(content="")]}},
            {"agent_fallback": {"messages": [AIMessage(content="fallback answer")]}},
        ]
        graph = _FakeUpdatesGraph(updates, raise_exc=TimeoutError("agent timeout"))
        request = GenerationRequest(query="hi", llm_type="main", agent="react")

        with _patched_runner(
            _build_react_graph=MagicMock(return_value=graph),
        ):
            (
                final_answer,
                _tool_results,
                _use_rag,
                _latencies,
                prompts,
                trace,
                _artifact_ids,
            ) = await generate_answer_agentic(
                request, user_id="test-user", conversation_id="c1"
            )

        assert final_answer == "fallback answer"
        assert prompts["used_fallback_llm"] is True
        assert trace


# ─── generate_answer_agentic_stream_helper (streaming) ────────────────────────


class TestStreamingEarlyFailure:
    async def test_setup_failure_yields_error_event(self):
        """Setup failures must degrade to an error SSE event, not crash the stream."""
        request = GenerationRequest(query="hi", llm_type="main", agent="react")

        with _patched_runner(
            _build_tools=AsyncMock(side_effect=ConnectionError("net")),
        ), patch(f"{_RUNNER}.logger"):
            events = []
            async for event in generate_answer_agentic_stream_helper(
                request,
                conversation_id="c1",
                message_id="m1",
                user_id="test-user",
            ):
                events.append(event)

        assert events, "expected at least one SSE event"
        assert any('"type": "error"' in e for e in events)

    async def test_agent_fallback_success_persists_metadata_and_trace(self):
        """Fallback answers must persist metadata/trace even if the graph later raises."""
        request = GenerationRequest(query="hi", llm_type="main", agent="react")
        graph = _FakeStreamGraph(
            updates=[
                {"agent_fallback": {"messages": [AIMessage(content="")]}},
            ],
            messages=[
                (
                    AIMessage(content="fallback answer"),
                    {"langgraph_node": "agent_fallback"},
                ),
            ],
            raise_exc=TimeoutError("Node 'agent' exceeded its idle timeout"),
        )
        persist = AsyncMock()

        with _patched_runner(
            _build_react_graph=MagicMock(return_value=graph),
            persist_message_state=persist,
        ):
            events = []
            async for event in generate_answer_agentic_stream_helper(
                request,
                conversation_id="c1",
                message_id="m1",
                user_id="test-user",
            ):
                events.append(event)

        assert any('"type": "final"' in e for e in events)
        assert not any('"type": "error"' in e for e in events)
        persist.assert_awaited_once()
        kwargs = persist.await_args.kwargs
        assert kwargs["output"] == "fallback answer"
        assert kwargs["latencies"]["generation_latency"] is not None
        assert kwargs["prompts"]["used_fallback_llm"] is True
        assert kwargs["trace"]
        assert kwargs["trace"][-1]["node"] == "agent_fallback"

    async def test_in_graph_fallback_reattributes_without_opening_the_circuit(self):
        """The graph swallows the primary's error, so it cannot be classified:
        re-attribute the answer, but never park the endpoint on evidence that
        could be a prompt-level 400 reproducing on every send."""
        request = GenerationRequest(query="hi", llm_type="main", agent="react")
        graph = _FakeStreamGraph(
            updates=[{"agent_fallback": {"messages": [AIMessage(content="")]}}],
            messages=[
                (
                    AIMessage(content="fallback answer"),
                    {"langgraph_node": "agent_fallback"},
                ),
            ],
        )
        persist = AsyncMock()
        manager = _chain_manager()

        with _patched_runner(
            _build_react_graph=MagicMock(return_value=graph),
            persist_message_state=persist,
            get_shared_llm_manager=MagicMock(return_value=manager),
        ):
            async for _event in generate_answer_agentic_stream_helper(
                request,
                conversation_id="c1",
                message_id="m1",
                user_id="test-user",
            ):
                pass

        endpoint = persist.await_args.kwargs["endpoint"]
        assert endpoint["requested"] == "main"
        assert endpoint["answered"] == "fallback"
        assert endpoint["substituted"] is True
        assert manager.health.is_open("main") is False

    async def test_a_streaming_failure_opens_the_resolved_endpoint(self):
        request = GenerationRequest(query="hi", llm_type="main", agent="react")
        graph = _FakeStreamGraph(raise_exc=TimeoutError("no first token"))
        manager = _chain_manager()

        with _patched_runner(
            _build_react_graph=MagicMock(return_value=graph),
            get_shared_llm_manager=MagicMock(return_value=manager),
        ), patch(f"{_RUNNER}.logger"):
            events = [
                event
                async for event in generate_answer_agentic_stream_helper(
                    request,
                    conversation_id="c1",
                    message_id="m1",
                    user_id="test-user",
                )
            ]

        assert any('"type": "error"' in event for event in events)
        assert manager.health.is_open("main") is True


def _event_payload(event: str) -> dict:
    return json.loads(event.removeprefix("data: ").strip())


class TestStreamingStatusEvent:
    async def test_status_is_the_first_event(self):
        """Tool building blocks for seconds: the UI needs a notice before it."""
        request = GenerationRequest(query="hi", llm_type="main", agent="react")
        graph = _FakeStreamGraph(
            messages=[(AIMessage(content="answer"), {"langgraph_node": "agent"})],
        )

        with _patched_runner(_build_react_graph=MagicMock(return_value=graph)):
            events = [
                event
                async for event in generate_answer_agentic_stream_helper(
                    request,
                    conversation_id="c1",
                    message_id="m1",
                    user_id="test-user",
                )
            ]

        assert events, "expected at least one SSE event"
        assert _event_payload(events[0]) == {"type": "status", "content": "Thinking…"}

    async def test_cancelled_turn_emits_only_stopped(self):
        """A turn cancelled before setup must not announce work it never starts."""
        request = GenerationRequest(query="hi", llm_type="main", agent="react")
        cancel_event = asyncio.Event()
        cancel_event.set()

        with _patched_runner() as patched:
            events = [
                event
                async for event in generate_answer_agentic_stream_helper(
                    request,
                    conversation_id="c1",
                    message_id="m1",
                    user_id="test-user",
                    cancel_event=cancel_event,
                )
            ]

        assert [_event_payload(event)["type"] for event in events] == ["stopped"]
        patched["_build_tools"].assert_not_awaited()


async def _stream_events(graph) -> list:
    request = GenerationRequest(query="hi", llm_type="main", agent="react")
    with _patched_runner(_build_react_graph=MagicMock(return_value=graph)):
        return [
            _event_payload(event)
            async for event in generate_answer_agentic_stream_helper(
                request,
                conversation_id="c1",
                message_id="m1",
                user_id="test-user",
            )
        ]


class TestStructuredToolEvents:
    """The tool events carry machine-readable fields for clients that render
    tool activity themselves. ``content`` keeps the ready-made display string,
    so a client that only knows about it is unaffected.
    """

    async def test_ai_message_tool_call_carries_structured_fields(self):
        graph = _FakeStreamGraph(
            messages=[
                (
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "dummy_search",
                                "args": {"query": "sea ice"},
                                "id": "call-1",
                            }
                        ],
                    ),
                    {"langgraph_node": "agent"},
                ),
                (AIMessage(content="done"), {"langgraph_node": "agent"}),
            ],
        )

        events = await _stream_events(graph)

        assert [e for e in events if e["type"] == "tool_call"] == [
            {
                "type": "tool_call",
                "content": "Calling dummy search: sea ice",
                "tool": "dummy_search",
                "label": "Calling dummy search",
                "query": "sea ice",
            }
        ]

    async def test_tool_result_carries_tool_name_and_status(self):
        graph = _FakeStreamGraph(
            messages=[
                (
                    ToolMessage(
                        content="x" * 300, name="dummy_search", tool_call_id="call-1"
                    ),
                    {"langgraph_node": "tools"},
                ),
                (AIMessage(content="done"), {"langgraph_node": "agent"}),
            ],
        )

        events = await _stream_events(graph)

        assert [e for e in events if e["type"] == "tool_result"] == [
            {
                "type": "tool_result",
                "content": "x" * 200,
                "tool": "dummy_search",
                "status": "ok",
            }
        ]

    async def test_text_marker_tool_calls_carry_structured_fields(self):
        """The `[TOOL_CALLS]` path flushes through _flush_turn_buffer_to_events,
        and a call with no query argument reports ``query`` as null rather than
        as an empty string.
        """
        graph = _FakeStreamGraph(
            messages=[
                (
                    AIMessage(
                        content=(
                            '[TOOL_CALLS]dummy_search{"query": "sea ice"}'
                            "[TOOL_CALLS]dummy_get_sample_report{}"
                        )
                    ),
                    {"langgraph_node": "agent"},
                ),
            ],
        )

        events = await _stream_events(graph)

        assert [e for e in events if e["type"] == "tool_call"] == [
            {
                "type": "tool_call",
                "content": "Calling dummy search: sea ice",
                "tool": "dummy_search",
                "label": "Calling dummy search",
                "query": "sea ice",
            },
            {
                "type": "tool_call",
                "content": "Calling dummy get sample report…",
                "tool": "dummy_get_sample_report",
                "label": "Calling dummy get sample report",
                "query": None,
            },
        ]

    async def test_content_stays_the_display_string_for_older_clients(self):
        graph = _FakeStreamGraph(
            messages=[
                (
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "dummy_search", "args": {}, "id": "call-1"}
                        ],
                    ),
                    {"langgraph_node": "agent"},
                ),
                (
                    ToolMessage(
                        content="result", name="dummy_search", tool_call_id="call-1"
                    ),
                    {"langgraph_node": "tools"},
                ),
                (AIMessage(content="done"), {"langgraph_node": "agent"}),
            ],
        )

        events = await _stream_events(graph)

        contents = {e["type"]: e["content"] for e in events if "tool" in e["type"]}
        assert contents == {
            "tool_call": "Calling dummy search…",
            "tool_result": "result",
        }


class TestNodeBudgetSemantics:
    """The endpoint budget is a first-token/stall cap, so it belongs on the
    node's idle timeout (whose clock resets on every streamed chunk). Used as
    the run timeout it is a hard wall clock that caps how long an answer may
    be, which is what killed long turns mid-stream.
    """

    def _builder(self):
        return MagicMock(
            return_value=_FakeStreamGraph(
                messages=[(AIMessage(content="answer"), {"langgraph_node": "agent"})]
            )
        )

    # A sentinel, not the configured value: MAIN_MODEL_TIMEOUT happens to equal
    # AGENTIC_TIMEOUT in some environments, which would make the assertions pass
    # whichever budget the runner put where.
    _ENDPOINT_BUDGET = 7

    async def test_stream_path_splits_the_two_budgets(self):
        request = GenerationRequest(query="hi", llm_type="main", agent="react")
        builder = self._builder()
        budget = MagicMock(return_value=self._ENDPOINT_BUDGET)

        with _patched_runner(
            _build_react_graph=builder,
            endpoint_timeout=budget,
            get_shared_llm_manager=MagicMock(return_value=_chain_manager()),
        ):
            async for _event in generate_answer_agentic_stream_helper(
                request,
                conversation_id="c1",
                message_id="m1",
                user_id="test-user",
            ):
                pass

        budget.assert_called_once_with("main")
        kwargs = builder.call_args.kwargs
        assert kwargs["llm_idle_timeout"] == self._ENDPOINT_BUDGET
        assert kwargs["llm_run_timeout"] == AGENTIC_TIMEOUT

    async def test_sync_path_splits_the_two_budgets(self):
        request = GenerationRequest(query="hi", llm_type="main", agent="react")
        builder = MagicMock(
            return_value=_FakeUpdatesGraph(
                [{"agent": {"messages": [AIMessage(content="answer")]}}]
            )
        )
        budget = MagicMock(return_value=self._ENDPOINT_BUDGET)

        with _patched_runner(
            _build_react_graph=builder,
            endpoint_timeout=budget,
            get_shared_llm_manager=MagicMock(return_value=_chain_manager()),
        ):
            await generate_answer_agentic(
                request, user_id="test-user", conversation_id="c1"
            )

        budget.assert_called_once_with("main")
        kwargs = builder.call_args.kwargs
        assert kwargs["llm_idle_timeout"] == self._ENDPOINT_BUDGET
        assert kwargs["llm_run_timeout"] == AGENTIC_TIMEOUT

    def test_the_endpoint_budget_is_not_the_wall_clock(self):
        """Guard the category error itself: the two budgets are never the same
        quantity, so the endpoint's first-token cap must never land on run.
        """
        agent = MagicMock()
        agent.compile.return_value = MagicMock(name="graph")
        fake_mgr = MagicMock()
        fake_mgr.get_client_for_model.return_value = MagicMock()

        with patch(f"{_RUNNER}.get_shared_llm_manager", return_value=fake_mgr):
            _build_react_graph(
                "main", [], None, agent=agent, llm_idle_timeout=self._ENDPOINT_BUDGET
            )

        kwargs = agent.compile.call_args.kwargs
        assert kwargs["llm_idle_timeout"] == self._ENDPOINT_BUDGET
        assert kwargs["llm_run_timeout"] == AGENTIC_TIMEOUT


class TestErrorPathAttribution:
    async def test_failed_turn_still_persists_endpoint_and_attribution(self):
        """A turn that dies must still say which endpoint answered it: the
        incident doc had endpoint null and agentic_llm_resolved null, so the
        silent substitution behind it was invisible in the database.
        """
        request = GenerationRequest(query="hi", llm_type="main", agent="react")
        graph = _FakeStreamGraph(raise_exc=RuntimeError("boom"))
        persist = AsyncMock()

        with _patched_runner(
            _build_react_graph=MagicMock(return_value=graph),
            persist_message_state=persist,
            get_shared_llm_manager=MagicMock(return_value=_chain_manager()),
            resolve_generated_model_name=MagicMock(return_value="model-x"),
        ), patch(f"{_RUNNER}.logger"):
            events = [
                _event_payload(event)
                async for event in generate_answer_agentic_stream_helper(
                    request,
                    conversation_id="c1",
                    message_id="m1",
                    user_id="test-user",
                )
            ]

        assert events[-1]["type"] == "error"
        kwargs = persist.await_args.kwargs
        assert kwargs["error"]["code"]
        assert kwargs["endpoint"]["answered"] == "main"
        assert kwargs["prompts"]["agentic_llm_resolved"] == "main"
        assert kwargs["generated_model_name"] == "model-x"

    async def test_a_streamed_answer_ends_with_final_not_error(self):
        """The incident client got 28999 characters of answer and then a
        timeout error on top of it. Whatever failed afterwards belongs in
        metadata.error, not in a second, contradictory terminal event.
        """
        request = GenerationRequest(query="hi", llm_type="main", agent="react")
        graph = _FakeStreamGraph(
            messages=[
                (AIMessage(content="a long answer"), {"langgraph_node": "agent"}),
            ],
            raise_exc=TimeoutError("Node 'agent_fallback' exceeded its run timeout"),
        )
        persist = AsyncMock()

        with _patched_runner(
            _build_react_graph=MagicMock(return_value=graph),
            persist_message_state=persist,
            get_shared_llm_manager=MagicMock(return_value=_chain_manager()),
        ), patch(f"{_RUNNER}.logger"):
            events = [
                _event_payload(event)
                async for event in generate_answer_agentic_stream_helper(
                    request,
                    conversation_id="c1",
                    message_id="m1",
                    user_id="test-user",
                )
            ]

        types = [e["type"] for e in events]
        assert "error" not in types
        assert types[-1] == "final"
        assert events[-1]["answer"] == "a long answer"
        # The failure is not swallowed: it is still on the persisted document.
        kwargs = persist.await_args.kwargs
        assert kwargs["error"]["code"]
        assert kwargs["output"] == "a long answer"
        assert kwargs["endpoint"]["answered"] == "main"

    async def test_a_turn_with_nothing_streamed_still_ends_with_error(self):
        request = GenerationRequest(query="hi", llm_type="main", agent="react")
        graph = _FakeStreamGraph(raise_exc=TimeoutError("no first token"))

        with _patched_runner(
            _build_react_graph=MagicMock(return_value=graph),
            get_shared_llm_manager=MagicMock(return_value=_chain_manager()),
        ), patch(f"{_RUNNER}.logger"):
            events = [
                _event_payload(event)
                async for event in generate_answer_agentic_stream_helper(
                    request,
                    conversation_id="c1",
                    message_id="m1",
                    user_id="test-user",
                )
            ]

        assert events[-1]["type"] == "error"
