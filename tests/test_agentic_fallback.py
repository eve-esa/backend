"""Unit tests for the agentic LangGraph fallback and error-handling layer.

Covers:
- _is_retryable_agentic_error: error-classification contract
- _build_react_graph: delegates llm_type resolution + compile call
- _build_react_graph_with_fallback: retries compile with Fallback LLM on init failure
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.agents.core.runner import (
    _build_react_graph,
    _build_react_graph_with_fallback,
    _is_retryable_agentic_error,
)


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

        with patch(
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

    def test_skip_fallback_when_already_on_fallback_model(self):
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
        # in-graph fallback_llm should be None when already on fallback
        compile_kwargs = agent.compile.call_args.kwargs
        assert compile_kwargs.get("fallback_llm") is None

    def test_in_graph_fallback_llm_wired_for_primary_model(self):
        """When using primary LLM, fallback_llm is passed into compile()."""
        agent = self._make_agent()
        fake_mgr = self._fake_llm_manager()

        with patch(
            "src.services.agents.core.runner.get_shared_llm_manager",
            return_value=fake_mgr,
        ):
            _build_react_graph_with_fallback("main", [], None, agent=agent)

        compile_kwargs = agent.compile.call_args.kwargs
        # fallback_llm must be provided so agent_fn can switch mid-node.
        assert compile_kwargs.get("fallback_llm") is not None
