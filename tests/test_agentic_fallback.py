"""Unit tests for the agentic LangGraph fallback and error-handling layer.

Covers:
- _is_retryable_agentic_error: error-classification contract
- _resolve_agentic_llm_type: request llm_type precedence over AGENTIC_LLM_TYPE
- _build_react_graph: delegates llm_type resolution + compile call
- _build_react_graph_with_fallback: retries compile with Fallback LLM on init failure
- generate_answer_agentic: records in-graph fallback + runner last-resort retry
- generate_answer_agentic_stream_helper: no UnboundLocalError on early setup failure
"""

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from src.services.agents.core.runner import (
    _build_react_graph,
    _build_react_graph_with_fallback,
    _is_retryable_agentic_error,
    _resolve_agentic_llm_type,
    generate_answer_agentic,
    generate_answer_agentic_stream_helper,
)
from src.schemas.generation_request import GenerationRequest

_RUNNER = "src.services.agents.core.runner"


class _FakeUpdatesGraph:
    """Minimal stand-in for a compiled graph driving ``astream(stream_mode=...)``.

    Yields the scripted ``updates`` payloads, or raises *raise_exc* on first use
    to simulate a run-time provider failure.
    """

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
    """Patch the runner's external boundaries with sensible async defaults.

    Individual values can be overridden via *overrides* (keyword = attr name).
    """
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


# ─── _is_retryable_agentic_error ──────────────────────────────────────────────


class TestIsRetryableAgenticError:
    def test_timeout_is_retryable(self):
        assert _is_retryable_agentic_error(TimeoutError("deadline"))

    def test_connection_error_is_retryable(self):
        assert _is_retryable_agentic_error(ConnectionError("refused"))

    def test_asyncio_timeout_is_retryable(self):
        assert _is_retryable_agentic_error(asyncio.TimeoutError())

    def test_type_error_is_not_retryable(self):
        assert not _is_retryable_agentic_error(TypeError("bad arg"))

    def test_value_error_is_not_retryable(self):
        assert not _is_retryable_agentic_error(ValueError("invalid"))

    def test_key_error_is_not_retryable(self):
        assert not _is_retryable_agentic_error(KeyError("missing"))

    def test_attribute_error_is_not_retryable(self):
        assert not _is_retryable_agentic_error(AttributeError("no attr"))

    def test_import_error_is_not_retryable(self):
        assert not _is_retryable_agentic_error(ImportError("no module"))

    def test_openai_rate_limit_is_retryable(self):
        class RateLimitError(Exception):
            pass

        assert _is_retryable_agentic_error(RateLimitError("429"))

    def test_openai_api_connection_error_is_retryable(self):
        class APIConnectionError(Exception):
            pass

        assert _is_retryable_agentic_error(APIConnectionError("conn"))

    def test_openai_api_timeout_error_is_retryable(self):
        class APITimeoutError(Exception):
            pass

        assert _is_retryable_agentic_error(APITimeoutError("timeout"))

    def test_generic_exception_is_not_retryable(self):
        # Unknown exceptions should not be blindly retried at the runner level.
        assert not _is_retryable_agentic_error(Exception("something unknown"))

    def test_httpx_429_is_retryable(self):
        try:
            import httpx

            response = MagicMock()
            response.status_code = 429
            exc = httpx.HTTPStatusError("rate limited", request=MagicMock(), response=response)
            assert _is_retryable_agentic_error(exc)
        except ImportError:
            pytest.skip("httpx not installed")

    def test_httpx_500_is_not_retryable(self):
        try:
            import httpx

            response = MagicMock()
            response.status_code = 500
            exc = httpx.HTTPStatusError("internal error", request=MagicMock(), response=response)
            assert not _is_retryable_agentic_error(exc)
        except ImportError:
            pytest.skip("httpx not installed")

    def test_httpx_503_is_retryable(self):
        try:
            import httpx

            response = MagicMock()
            response.status_code = 503
            exc = httpx.HTTPStatusError("unavailable", request=MagicMock(), response=response)
            assert _is_retryable_agentic_error(exc)
        except ImportError:
            pytest.skip("httpx not installed")


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

        fake_llm_manager.get_client_for_model.assert_called_once_with("main")
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

        fake_llm_manager.get_client_for_model.assert_called_once_with("fallback")

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

        compile_kwargs = agent.compile.call_args.kwargs
        assert compile_kwargs["fallback_llm"] is fallback_llm

    def test_streaming_flag_forwarded_to_compile(self):
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
                streaming=False,
            )

        compile_kwargs = agent.compile.call_args.kwargs
        assert compile_kwargs["streaming"] is False

# ─── _build_react_graph_with_fallback ────────────────────────────────────────


class TestBuildReactGraphWithFallback:
    def _make_agent(self, *, fail_first=False):
        agent = MagicMock()
        if fail_first:
            primary_graph = MagicMock(name="primary_graph")
            calls = {"count": 0}

            def compile_side_effect(**kwargs):
                calls["count"] += 1
                # Fail on first compile (primary), succeed on second (fallback).
                if calls["count"] == 1:
                    raise RuntimeError("primary LLM unavailable")
                return primary_graph

            agent.compile.side_effect = compile_side_effect
        else:
            agent.compile.return_value = MagicMock(name="compiled_graph")
        return agent

    def _fake_llm_manager(self):
        mgr = MagicMock()
        mgr.get_client_for_model.return_value = MagicMock(name="llm_client")
        return mgr

    def test_returns_primary_graph_when_init_succeeds(self):
        agent = self._make_agent()
        fake_mgr = self._fake_llm_manager()

        with patch(
            "src.services.agents.core.runner.get_shared_llm_manager",
            return_value=fake_mgr,
        ):
            graph, used_fallback = _build_react_graph_with_fallback(
                "main", [], None, agent=agent
            )

        assert not used_fallback
        assert graph is agent.compile.return_value

    def test_falls_back_to_fallback_llm_on_init_failure(self):
        agent = self._make_agent(fail_first=True)
        fake_mgr = self._fake_llm_manager()

        with patch(
            "src.services.agents.core.runner.get_shared_llm_manager",
            return_value=fake_mgr,
        ):
            graph, used_fallback = _build_react_graph_with_fallback(
                "main", [], None, agent=agent
            )

        assert used_fallback
        # compile was called twice: once for primary (raises), once for fallback.
        assert agent.compile.call_count == 2
        # Second call must have used fallback llm_type_override.
        second_call_kwargs = agent.compile.call_args_list[1].kwargs
        assert second_call_kwargs.get("fallback_llm") is None  # no in-graph fallback on retry

    def test_in_graph_fallback_llm_wired_when_already_on_fallback_model(self):
        """In-graph fallback_llm is always wired, even when primary is fallback."""
        agent = self._make_agent()
        fake_mgr = self._fake_llm_manager()

        with patch(
            "src.services.agents.core.runner.get_shared_llm_manager",
            return_value=fake_mgr,
        ):
            graph, used_fallback = _build_react_graph_with_fallback(
                "fallback", [], None, agent=agent
            )

        assert not used_fallback
        compile_kwargs = agent.compile.call_args.kwargs
        assert compile_kwargs.get("fallback_llm") is not None

    def test_in_graph_fallback_llm_wired_for_primary_model(self):
        """When using primary LLM, fallback_llm is passed into compile()."""
        agent = self._make_agent()
        fake_mgr = self._fake_llm_manager()

        with patch(f"{_RUNNER}.AGENTIC_LLM_TYPE", None), patch(
            "src.services.agents.core.runner.get_shared_llm_manager",
            return_value=fake_mgr,
        ):
            _build_react_graph_with_fallback("main", [], None, agent=agent)

        compile_kwargs = agent.compile.call_args.kwargs
        # fallback_llm must be provided so agent_fn can switch mid-node.
        assert compile_kwargs.get("fallback_llm") is not None

    def test_streaming_flag_forwarded_to_compile(self):
        agent = self._make_agent()
        fake_mgr = self._fake_llm_manager()

        with patch(f"{_RUNNER}.AGENTIC_LLM_TYPE", None), patch(
            "src.services.agents.core.runner.get_shared_llm_manager",
            return_value=fake_mgr,
        ):
            _build_react_graph_with_fallback(
                "main", [], None, agent=agent, streaming=False
            )

        compile_kwargs = agent.compile.call_args.kwargs
        assert compile_kwargs["streaming"] is False


# ─── _resolve_agentic_llm_type ────────────────────────────────────────────────


class TestResolveAgenticLlmType:
    def test_explicit_override_wins(self):
        with patch(f"{_RUNNER}.AGENTIC_LLM_TYPE", "fallback"):
            assert (
                _resolve_agentic_llm_type("main", override="satcom_large")
                == "satcom_large"
            )

    def test_request_overrides_env(self):
        # Request llm_type must take precedence over AGENTIC_LLM_TYPE.
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
        """``_build_react_graph`` must resolve the request llm_type over env default."""
        agent = MagicMock()
        agent.compile.return_value = MagicMock(name="graph")
        fake_mgr = MagicMock()
        fake_mgr.get_client_for_model.return_value = MagicMock()

        with patch(f"{_RUNNER}.AGENTIC_LLM_TYPE", "fallback"), patch(
            f"{_RUNNER}.get_shared_llm_manager", return_value=fake_mgr
        ):
            _build_react_graph("main", [], None, agent=agent)

        fake_mgr.get_client_for_model.assert_called_once_with("main")


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
            _build_react_graph_with_fallback=MagicMock(return_value=(graph, False)),
        ):
            (
                final_answer,
                _tool_results,
                _use_rag,
                _latencies,
                prompts,
                _trace,
            ) = await generate_answer_agentic(request, conversation_id="c1")

        assert final_answer == "final answer"
        assert prompts["used_fallback_llm"] is True


class TestGenerateAnswerAgenticLastResort:
    async def test_connection_error_triggers_fallback_primary_graph(self):
        primary = _FakeUpdatesGraph(raise_exc=ConnectionError("provider down"))
        fallback = _FakeUpdatesGraph(
            [{"agent": {"messages": [AIMessage(content="recovered")]}}]
        )
        build_fallback = MagicMock(return_value=fallback)
        request = GenerationRequest(query="hi", llm_type="main", agent="react")

        with _patched_runner(
            _build_react_graph_with_fallback=MagicMock(return_value=(primary, False)),
            _build_react_graph=build_fallback,
        ) as patched:
            (
                final_answer,
                _tool_results,
                _use_rag,
                _latencies,
                prompts,
                _trace,
            ) = await generate_answer_agentic(request, conversation_id="c1")

        assert final_answer == "recovered"
        assert prompts["used_fallback_llm"] is True
        build_fallback.assert_called_once()
        patched["error_logger"].log_error.assert_awaited()


# ─── generate_answer_agentic_stream_helper (streaming) ────────────────────────


class TestStreamingEarlyFailure:
    async def test_retryable_setup_failure_does_not_raise_unbound(self):
        """A retryable failure before graph build must not ``UnboundLocalError``.

        ``used_fallback_llm`` is referenced in the broad ``except`` and must be
        initialised before the ``try`` so early setup failures degrade to an
        error event instead of crashing the stream.
        """
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
        assert not any("tools" in e and "not associated with a value" in e for e in events)
