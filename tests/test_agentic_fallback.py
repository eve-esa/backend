"""Unit tests for the agentic LangGraph error-handling layer.

Covers:
- _resolve_agentic_llm_type: request llm_type precedence over AGENTIC_LLM_TYPE
- _build_react_graph: delegates llm_type resolution + in-graph fallback_llm wiring
- generate_answer_agentic: records in-graph fallback via agent_fallback node
- generate_answer_agentic_stream_helper: no UnboundLocalError on early setup failure
"""

import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from src.services.agents.core.runner import (
    _build_react_graph,
    _resolve_agentic_llm_type,
    _serialise_trace_entry,
    generate_answer_agentic,
    generate_answer_agentic_stream_helper,
)
from src.schemas.generation_request import GenerationRequest

_RUNNER = "src.services.agents.core.runner"


class _FakeUpdatesGraph:
    """Minimal stand-in for a compiled graph driving ``astream(stream_mode=...)``."""

    def __init__(self, updates=None, raise_exc=None):
        self._updates = updates or []
        self._raise = raise_exc

    async def astream(self, *args, **kwargs):
        if self._raise is not None:
            raise self._raise
        for update in self._updates:
            yield update


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
            ) = await generate_answer_agentic(request, conversation_id="c1")

        assert final_answer == "final answer"
        assert prompts["used_fallback_llm"] is True


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
                request, conversation_id="c1", message_id="m1"
            ):
                events.append(event)

        assert events, "expected at least one SSE event"
        assert any('"type": "error"' in e for e in events)
