"""Deep-graph summarization wiring. The graph itself is tested in eve-esa-agents."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage

from src.services.agents.core import runner
from src.services.agents.core.history_search import SEARCH_NOTE
from src.services.agents.core.runner import (
    _build_initial_messages,
    _build_react_graph,
    _prepare_agent_context,
)

pytestmark = pytest.mark.no_db


class _DeclaringAgent:
    """compile declares the summarization parameters."""

    name = "deep"
    prompts = {"system": "You are EVE"}

    def compile(
        self,
        *,
        llm,
        tools,
        checkpointer,
        context_window,
        summary_trigger_fraction,
        summary_keep_fraction,
        summary_max_tokens,
        truncate_args_max_chars,
        recall_note,
        **kwargs,
    ):
        self.seen = {
            "context_window": context_window,
            "summary_trigger_fraction": summary_trigger_fraction,
            "summary_keep_fraction": summary_keep_fraction,
            "summary_max_tokens": summary_max_tokens,
            "truncate_args_max_chars": truncate_args_max_chars,
            "recall_note": recall_note,
        }
        return MagicMock()


class _SilentAgent:
    """compile accepts anything and declares none of the new parameters."""

    name = "react"
    prompts = {"system": "You are EVE"}

    def compile(self, *args, **kwargs):
        self.seen = kwargs
        return MagicMock()


def _request(query="now"):
    request = MagicMock()
    request.query = query
    return request


def test_summary_kwargs_are_forwarded_only_when_declared():
    declaring = _DeclaringAgent()
    _build_react_graph(
        "main",
        [],
        None,
        agent=declaring,
        llm=MagicMock(),
        fallback_llm=MagicMock(),
    )
    assert declaring.seen["recall_note"] == SEARCH_NOTE
    assert declaring.seen["context_window"] == runner.AGENT_CONTEXT_WINDOW_TOKENS
    assert declaring.seen["summary_trigger_fraction"] == runner.SUMMARY_TRIGGER_FRACTION
    assert declaring.seen["summary_keep_fraction"] == runner.SUMMARY_KEEP_FRACTION
    assert declaring.seen["summary_max_tokens"] == runner.SUMMARY_MAX_TOKENS
    assert declaring.seen["truncate_args_max_chars"] == runner.TRUNCATE_ARGS_MAX_CHARS

    silent = _SilentAgent()
    _build_react_graph(
        "main",
        [],
        None,
        agent=silent,
        llm=MagicMock(),
        fallback_llm=MagicMock(),
    )
    assert "context_window" not in silent.seen
    assert "recall_note" not in silent.seen


def test_the_new_human_message_carries_its_id():
    messages = _build_initial_messages(
        _request("now"), [], history=[], message_id="msg-9", new_checkpoint=False
    )
    assert messages[-1].content == "now"
    assert messages[-1].additional_kwargs["eve_message_id"] == "msg-9"


class _Cursor:
    def __init__(self, documents):
        self.documents = documents

    def sort(self, *_args, **_kwargs):
        return self

    async def to_list(self, length=None):
        return self.documents


def _collection(documents):
    collection = MagicMock()
    collection.find.return_value = _Cursor(documents)
    return collection


@pytest.mark.asyncio
async def test_a_deep_turn_with_a_checkpoint_drops_the_rollup(monkeypatch):
    agent = _DeclaringAgent()
    monkeypatch.setattr(
        runner,
        "_fetch_conversation_context",
        AsyncMock(return_value=([HumanMessage("old turn")], "rolled up")),
    )
    monkeypatch.setattr(runner, "get_agent_graph", lambda _name: agent)
    monkeypatch.setattr(runner, "PINNED_CONTEXT", "Stay factual.")
    checkpointer = MagicMock()
    checkpointer.aget_tuple = AsyncMock(return_value=object())

    history, summary, prior, new_checkpoint, prepared = await _prepare_agent_context(
        "deep", "conv-1", checkpointer, "msg-now"
    )

    assert history == []
    assert summary is None
    assert prior == []
    assert new_checkpoint is False
    assert prepared is not agent
    assert prepared.prompts["system"].startswith("Stay factual.")
    assert agent.prompts["system"] == "You are EVE"


@pytest.mark.asyncio
async def test_a_deep_turn_without_a_checkpoint_is_seeded(monkeypatch):
    agent = _DeclaringAgent()
    monkeypatch.setattr(
        runner, "_fetch_conversation_context", AsyncMock(return_value=([], "rolled up"))
    )
    monkeypatch.setattr(runner, "get_agent_graph", lambda _name: agent)
    monkeypatch.setattr(runner, "PINNED_CONTEXT", "")
    checkpointer = MagicMock()
    checkpointer.aget_tuple = AsyncMock(return_value=None)
    documents = [
        {"_id": "old-1", "input": "earlier question", "output": "earlier answer"},
        {"_id": "msg-now", "input": "now", "output": ""},
    ]
    monkeypatch.setattr(
        runner.Message, "get_collection", lambda: _collection(documents)
    )

    history, summary, prior, new_checkpoint, _prepared = await _prepare_agent_context(
        "deep", "conv-1", checkpointer, "msg-now"
    )

    assert history == []
    assert summary is None
    assert new_checkpoint is True
    assert [message.content for message in prior] == [
        "earlier question",
        "earlier answer",
    ]
    assert prior[0].additional_kwargs["eve_message_id"] == "old-1"


@pytest.mark.asyncio
async def test_a_react_turn_keeps_the_rollup(monkeypatch):
    agent = _SilentAgent()
    monkeypatch.setattr(
        runner,
        "_fetch_conversation_context",
        AsyncMock(return_value=([HumanMessage("old turn")], "rolled up")),
    )
    monkeypatch.setattr(runner, "get_agent_graph", lambda _name: agent)

    history, summary, prior, new_checkpoint, prepared = await _prepare_agent_context(
        "react", "conv-1", MagicMock(), "msg-now"
    )

    assert [message.content for message in history] == ["old turn"]
    assert "rolled up" in summary
    assert "search_history" in summary
    assert prior == []
    assert new_checkpoint is False
    assert prepared is agent
