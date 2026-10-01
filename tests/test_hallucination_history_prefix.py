"""Hallucination detection receives the same history prefix a new agentic turn gets."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from bson import ObjectId

from src.database.models.conversation import Conversation
from src.database.models.message import Message
from src.services.agents.core.registry import get_agent_graph
from src.services.agents.core.runner import _resolve_agent_graph_type
from src.services.generate_answer import _get_conversation_history_from_db
from src.services.hallucination_detector import (
    HallucinationDetector,
    HallucinationResult,
    RewriteResult,
    load_new_turn_history_prefix,
)


class _CapturingLLM:
    def __init__(self, result):
        self.result = result
        self.prompts = []

    def with_structured_output(self, _schema):
        return self

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        return self.result


def _message(when: datetime, user_text: str, answer: str):
    return SimpleNamespace(
        id=str(ObjectId()),
        timestamp=when,
        input=user_text,
        output=answer,
        conversation_id="conv-1",
    )


def _clause_matches(message, clause) -> bool:
    timestamp = clause.get("timestamp")
    if isinstance(timestamp, dict) and "$lt" in timestamp:
        return message.timestamp < timestamp["$lt"]
    if timestamp == message.timestamp and "$lt" in clause.get("_id", {}):
        return ObjectId(message.id) < clause["_id"]["$lt"]
    return False


def _install_history_store(monkeypatch, messages, summary):
    """Serve Message/Conversation reads, honoring the helper's sort, skip, and limit."""

    async def find_all(
        _cls, filter_dict=None, sort=None, limit=None, skip=None, **_kwargs
    ):
        filt = filter_dict or {}
        rows = [
            message
            for message in messages
            if message.conversation_id == filt.get("conversation_id", message.conversation_id)
        ]
        if "$or" in filt:
            rows = [
                message
                for message in rows
                if any(_clause_matches(message, clause) for clause in filt["$or"])
            ]
        elif isinstance(filt.get("timestamp"), dict) and "$lt" in filt["timestamp"]:
            rows = [
                message
                for message in rows
                if message.timestamp < filt["timestamp"]["$lt"]
            ]
        for key, direction in reversed(sort or []):
            rows.sort(key=lambda message: getattr(message, key), reverse=direction < 0)
        if skip:
            rows = rows[skip:]
        if limit:
            rows = rows[:limit]
        return rows

    async def find_by_id(_cls, _conversation_id):
        return SimpleNamespace(summary=summary)

    monkeypatch.setattr(Message, "find_all", classmethod(find_all))
    monkeypatch.setattr(Conversation, "find_by_id", classmethod(find_by_id))


@pytest.mark.no_db
async def test_prefix_matches_a_new_turn_and_keeps_only_the_history_window(monkeypatch):
    base = datetime(2024, 6, 1, tzinfo=timezone.utc)
    oldest = _message(base, "oldest-user-landsat", "oldest-assistant-landsat")
    window = _message(
        base + timedelta(minutes=1),
        "window-user-sentinel",
        "window-assistant-resolution",
    )
    current = _message(
        base + timedelta(minutes=2),
        "current-user-follow-up",
        "current-answer-being-judged",
    )
    _install_history_store(
        monkeypatch,
        [oldest, window, current],
        summary="Sentinel-1 is a radar imaging mission.",
    )

    prefix = await load_new_turn_history_prefix("conv-1", before_message=current)
    history, summary = await _get_conversation_history_from_db(
        "conv-1", before_message=current
    )
    expected = get_agent_graph(_resolve_agent_graph_type(None)).format_history(
        history or [], summary
    )

    assert prefix == expected
    assert "Sentinel-1 is a radar imaging mission." in prefix
    assert "window-user-sentinel" in prefix
    assert "window-assistant-resolution" in prefix
    assert "oldest-user-landsat" not in prefix
    assert "oldest-assistant-landsat" not in prefix
    assert "current-answer-being-judged" not in prefix
    assert "current-user-follow-up" not in prefix


@pytest.mark.no_db
async def test_older_message_does_not_include_later_turns(monkeypatch):
    base = datetime(2024, 6, 1, tzinfo=timezone.utc)
    oldest = _message(base, "oldest-user-landsat", "oldest-assistant-landsat")
    judged = _message(
        base + timedelta(minutes=1),
        "judged-user-sentinel",
        "judged-answer-being-checked",
    )
    later = _message(
        base + timedelta(minutes=2),
        "later-user-follow-up",
        "later-assistant-answer",
    )
    _install_history_store(
        monkeypatch,
        [oldest, judged, later],
        summary="Sentinel-1 is a radar imaging mission.",
    )

    prefix = await load_new_turn_history_prefix("conv-1", before_message=judged)

    assert "oldest-user-landsat" in prefix
    assert "oldest-assistant-landsat" in prefix
    assert "judged-answer-being-checked" not in prefix
    assert "later-user-follow-up" not in prefix
    assert "later-assistant-answer" not in prefix


@pytest.mark.no_db
async def test_first_message_without_summary_has_an_empty_prefix(monkeypatch):
    only = _message(
        datetime(2024, 6, 1, tzinfo=timezone.utc),
        "only-user-question",
        "only-assistant-answer",
    )
    _install_history_store(monkeypatch, [only], summary=None)
    assert await load_new_turn_history_prefix("conv-1", before_message=only) == ""


@pytest.mark.no_db
async def test_detect_and_rewrite_include_the_prefix_and_leave_a_first_turn_bare():
    detector = HallucinationDetector()
    factual = _CapturingLLM(HallucinationResult(label=0, reason="grounded"))
    rewrite = _CapturingLLM(RewriteResult(question="q", rewritten_question="rewritten"))
    detector.llm_manager.get_client_for_model = lambda *_args, **_kwargs: factual

    await detector.detect(
        query="what about its resolution?",
        model_response="10 metres",
        docs="doc",
        conversation=(
            "Previous conversation summary:\nSentinel-1\n"
            "User: Tell me about Sentinel-1"
        ),
    )
    assert "<<<CONVERSATION>>>\nPrevious conversation summary:\nSentinel-1" in factual.prompts[0]
    assert "User: Tell me about Sentinel-1\n<<<END CONVERSATION>>>" in factual.prompts[0]
    assert "rolling summary of older turns" in factual.prompts[0]
    assert "Previous conversation summary:\nSentinel-1" in factual.prompts[0]
    assert "User: Tell me about Sentinel-1" in factual.prompts[0]
    assert "Question: what about its resolution?" in factual.prompts[0]

    factual.prompts.clear()
    await detector.detect(query="What is NDVI?", model_response="an index", docs="")
    assert "<<<CONVERSATION>>>\n" not in factual.prompts[0]
    assert "Previous conversation summary:\n" not in factual.prompts[0]
    assert "Question: What is NDVI?" in factual.prompts[0]

    detector.llm_manager.get_client_for_model = lambda *_args, **_kwargs: rewrite
    await detector.rewrite_query(
        query="what about its resolution? {not_a_field}",
        answer="10 metres",
        reason="unspecified satellite",
        conversation="Previous conversation summary:\nSentinel-1",
    )
    assert "<<<CONVERSATION>>>\nPrevious conversation summary:\nSentinel-1" in rewrite.prompts[0]
    assert "<<<END CONVERSATION>>>" in rewrite.prompts[0]
    assert "rolling summary of older turns" in rewrite.prompts[0]
    assert "{not_a_field}" in rewrite.prompts[0]
    assert "Original question:" in rewrite.prompts[0]
