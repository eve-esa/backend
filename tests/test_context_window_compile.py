"""The turn passes the smaller window and the tokenizer counter to compile."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.config import AGENT_CONTEXT_WINDOW_TOKENS
from src.services.agents.core.runner import (
    _build_react_graph,
    _context_compile_args,
)

pytestmark = pytest.mark.no_db


def _client(base_url: str, model_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        openai_api_base=base_url,
        model_name=model_name,
        openai_api_key="key",
        profile=None,
    )


@pytest.mark.asyncio
async def test_compile_args_use_the_smaller_window(monkeypatch):
    monkeypatch.setenv("MODEL_LIMITS_LIVE", "1")
    windows = {
        ("https://eve.test/v1", "eve"): 131072,
        ("https://fallback.test/v1", "mistral"): 262144,
    }

    async def window_for(base_url, model_name, api_key):
        assert api_key == "key"
        return windows[(base_url, model_name)]

    monkeypatch.setattr("src.core.model_limits.window_for", window_for)
    monkeypatch.setattr(
        "src.core.token_counting.counter_for", lambda llm_type: f"counter:{llm_type}"
    )
    fallback = _client("https://fallback.test/v1", "mistral")
    monkeypatch.setattr(
        "src.services.agents.core.runner.get_shared_llm_manager",
        lambda: SimpleNamespace(get_client_for_model=lambda _name: fallback),
    )
    primary = _client("https://eve.test/v1", "eve")
    args = await _context_compile_args(primary, "eve_jsc")
    assert args["context_window"] == 131072
    assert args["token_counter"] == "counter:eve_jsc"
    assert args["fallback_llm"] is fallback
    assert primary.profile == {"max_input_tokens": 131072}
    assert fallback.profile == {"max_input_tokens": 262144}


def test_compile_receives_the_window_and_counter():
    seen = {}

    def compile(
        *,
        llm,
        tools,
        checkpointer,
        history,
        summary,
        fallback_llm,
        llm_run_timeout,
        llm_idle_timeout,
        on_policy,
        context_window,
        token_counter=None,
    ):
        seen["context_window"] = context_window
        seen["token_counter"] = token_counter
        return "graph"

    agent = SimpleNamespace(compile=compile, name="deep")
    counter = lambda messages, tools=None: 1  # noqa: E731
    graph = _build_react_graph(
        "main",
        tools=[],
        checkpointer=None,
        agent=agent,
        llm=MagicMock(),
        fallback_llm=MagicMock(),
        context_window=131072,
        token_counter=counter,
    )
    assert graph == "graph"
    assert seen["context_window"] == 131072
    assert seen["token_counter"] is counter


def test_an_older_graph_compiles_without_the_new_parameters():
    def compile(
        *,
        llm,
        tools,
        checkpointer,
        history,
        summary,
        fallback_llm,
        llm_run_timeout,
        llm_idle_timeout,
        on_policy,
    ):
        return "old"

    agent = SimpleNamespace(compile=compile, name="react")
    graph = _build_react_graph(
        "main",
        tools=[],
        checkpointer=None,
        agent=agent,
        llm=MagicMock(),
        fallback_llm=MagicMock(),
        context_window=1000,
        token_counter=lambda messages, tools=None: 1,
    )
    assert graph == "old"


def test_without_a_resolved_window_the_default_is_kept():
    seen = {}

    def compile(
        *,
        llm,
        tools,
        checkpointer,
        history,
        summary,
        fallback_llm,
        llm_run_timeout,
        llm_idle_timeout,
        on_policy,
        context_window,
    ):
        seen["context_window"] = context_window
        return "graph"

    agent = SimpleNamespace(compile=compile, name="deep")
    _build_react_graph(
        "main",
        tools=[],
        checkpointer=None,
        agent=agent,
        llm=MagicMock(),
        fallback_llm=MagicMock(),
    )
    assert seen["context_window"] == AGENT_CONTEXT_WINDOW_TOKENS
    assert "token_counter" not in seen
