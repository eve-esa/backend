"""Thumbs feedback as Langfuse scores (src/services/langfuse_scores.py).

The HTTP side runs against an httpx.MockTransport, so nothing leaves the
process. The first half is the client alone; the second half goes through
PATCH /conversations/{id}/messages/{id} to prove update_message wires it.
"""

import asyncio
import base64
import json
import logging
from types import SimpleNamespace
from typing import List

import httpx
import pytest

from src import config
from src.database.models.conversation import Conversation
from src.database.models.message import Message
from src.services import langfuse_scores
from tests.utils.cleaner import cleanup_models
from tests.utils.utils import create_test_user_and_token

TRACE_ID = "0af7651916cd43dd8448eb211c80319c"
HOST = "http://langfuse.test"


class _Recorder:
    """MockTransport handler that records requests and answers ``status``."""

    def __init__(self, status: int = 200, raise_exc: Exception = None):
        self.status = status
        self.raise_exc = raise_exc
        self.requests: List[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.raise_exc is not None:
            raise self.raise_exc
        return httpx.Response(self.status, json={"id": "ignored"})

    @property
    def bodies(self) -> List[dict]:
        return [json.loads(r.content) for r in self.requests]


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(config, "FEATURE_LANGFUSE_SCORES", True)
    monkeypatch.setattr(config, "LANGFUSE_HOST", HOST)
    monkeypatch.setattr(config, "LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setattr(config, "LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "deployment.environment.name=local")


@pytest.fixture
def recorder(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(langfuse_scores, "_transport", httpx.MockTransport(rec))
    return rec


async def _drain() -> None:
    pending = list(langfuse_scores._pending)
    if pending:
        await asyncio.gather(*pending)


def _message(**fields):
    base = {
        "id": "msg-1",
        "trace_id": TRACE_ID,
        "feedback": None,
        "feedback_reason": None,
        "hallucination": None,
    }
    base.update(fields)
    return SimpleNamespace(**base)


# ─── client ───────────────────────────────────────────────────────────────────


@pytest.mark.no_db
def test_thumbs_down_score_shape(enabled):
    scores = langfuse_scores.build_scores(
        _message(feedback="negative", feedback_reason="wrong mission date")
    )
    assert scores == [
        {
            "id": "thumbs-msg-1",
            "traceId": TRACE_ID,
            "name": "thumbs",
            "dataType": "BOOLEAN",
            "value": 0,
            "comment": "wrong mission date",
            "environment": "local",
        }
    ]


@pytest.mark.no_db
def test_thumbs_up_without_reason_has_no_comment(enabled):
    (score,) = langfuse_scores.build_scores(_message(feedback="positive"))
    assert score["value"] == 1
    assert "comment" not in score


@pytest.mark.no_db
def test_hallucination_score_added_when_set(enabled):
    scores = langfuse_scores.build_scores(
        _message(
            feedback="positive",
            hallucination={"feedback": "negative", "feedback_reason": "made up"},
        )
    )
    assert [s["id"] for s in scores] == ["thumbs-msg-1", "hallucination-msg-1"]
    hallucination = scores[1]
    assert hallucination["name"] == "hallucination"
    assert hallucination["value"] == 0
    assert hallucination["comment"] == "made up"
    assert hallucination["traceId"] == TRACE_ID


@pytest.mark.no_db
def test_hallucination_without_vote_is_skipped(enabled):
    scores = langfuse_scores.build_scores(
        _message(feedback="negative", hallucination={"was_copied": True})
    )
    assert [s["id"] for s in scores] == ["thumbs-msg-1"]


@pytest.mark.no_db
def test_no_trace_id_no_scores(enabled):
    assert langfuse_scores.build_scores(_message(trace_id=None, feedback="negative")) == []


@pytest.mark.no_db
def test_environment_falls_back_to_app_environment(enabled, monkeypatch):
    monkeypatch.delenv("OTEL_RESOURCE_ATTRIBUTES")
    monkeypatch.setattr(config, "APP_ENVIRONMENT", "staging")
    (score,) = langfuse_scores.build_scores(_message(feedback="positive"))
    assert score["environment"] == "staging"


@pytest.mark.no_db
def test_environment_langfuse_rejects_is_omitted(enabled, monkeypatch):
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "deployment.environment.name=Local Dev")
    (score,) = langfuse_scores.build_scores(_message(feedback="positive"))
    assert "environment" not in score


@pytest.mark.no_db
async def test_disabled_sends_nothing(enabled, recorder, monkeypatch):
    monkeypatch.setattr(config, "FEATURE_LANGFUSE_SCORES", False)
    assert langfuse_scores.schedule_feedback_scores(_message(feedback="negative")) is None
    await _drain()
    assert recorder.requests == []


@pytest.mark.no_db
async def test_enabled_without_keys_sends_nothing(enabled, recorder, monkeypatch):
    monkeypatch.setattr(config, "LANGFUSE_SECRET_KEY", "")
    assert langfuse_scores.schedule_feedback_scores(_message(feedback="negative")) is None
    assert recorder.requests == []


@pytest.mark.no_db
async def test_no_trace_id_is_silent(enabled, recorder, caplog):
    caplog.set_level(logging.DEBUG, logger=langfuse_scores.__name__)
    task = langfuse_scores.schedule_feedback_scores(
        _message(trace_id=None, feedback="negative")
    )
    assert task is None
    assert recorder.requests == []
    assert caplog.records == []


@pytest.mark.no_db
async def test_posts_to_public_scores_with_basic_auth(enabled, recorder):
    task = langfuse_scores.schedule_feedback_scores(
        _message(feedback="negative", feedback_reason="off topic")
    )
    assert task is not None
    await task
    (request,) = recorder.requests
    assert request.method == "POST"
    assert str(request.url) == f"{HOST}/api/public/scores"
    expected = base64.b64encode(b"pk-test:sk-test").decode()
    assert request.headers["authorization"] == f"Basic {expected}"
    assert request.extensions["timeout"]["read"] == 3.0
    assert recorder.bodies[0]["id"] == "thumbs-msg-1"
    assert recorder.bodies[0]["comment"] == "off topic"


@pytest.mark.no_db
async def test_http_error_is_logged_not_raised(enabled, recorder, caplog):
    recorder.status = 500
    caplog.set_level(logging.WARNING, logger=langfuse_scores.__name__)
    task = langfuse_scores.schedule_feedback_scores(
        _message(feedback="negative", hallucination={"feedback": "positive"})
    )
    await task
    assert len(recorder.requests) == 2, "one failure must not stop the next score"
    messages = [r.getMessage() for r in caplog.records]
    assert any("thumbs-msg-1 not sent" in m and "500" in m for m in messages)
    assert all("sk-test" not in m for m in messages)


@pytest.mark.no_db
async def test_network_error_is_logged_not_raised(enabled, recorder, caplog):
    recorder.raise_exc = httpx.ConnectError("connection refused")
    caplog.set_level(logging.WARNING, logger=langfuse_scores.__name__)
    await langfuse_scores.schedule_feedback_scores(_message(feedback="positive"))
    assert any("ConnectError" in r.getMessage() for r in caplog.records)


@pytest.mark.no_db
async def test_schedule_never_raises(enabled, monkeypatch):
    def boom(_message):
        raise RuntimeError("bad message")

    monkeypatch.setattr(langfuse_scores, "build_scores", boom)
    assert langfuse_scores.schedule_feedback_scores(_message(feedback="positive")) is None


# ─── update_message wiring ────────────────────────────────────────────────────


async def _owned_message(trace_id=TRACE_ID):
    user, token = await create_test_user_and_token()
    conversation = await Conversation.create(user_id=user.id, name="scores")
    message = await Message.create(
        conversation_id=conversation.id,
        input="q",
        output="a",
        trace_id=trace_id,
    )
    return user, token, conversation, message


async def _patch(async_client, token, conversation, message, payload):
    return await async_client.patch(
        f"/conversations/{conversation.id}/messages/{message.id}",
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
    )


async def test_update_message_posts_thumbs_score(async_client, enabled, recorder):
    user, token, conversation, message = await _owned_message()
    try:
        resp = await _patch(
            async_client,
            token,
            conversation,
            message,
            {
                "feedback": "negative",
                "feedback_reason": "cites the wrong paper",
                "hallucination_feedback": "positive",
            },
        )
        assert resp.status_code == 200
        await _drain()
        assert recorder.bodies == [
            {
                "id": f"thumbs-{message.id}",
                "traceId": TRACE_ID,
                "name": "thumbs",
                "dataType": "BOOLEAN",
                "value": 0,
                "comment": "cites the wrong paper",
                "environment": "local",
            },
            {
                "id": f"hallucination-{message.id}",
                "traceId": TRACE_ID,
                "name": "hallucination",
                "dataType": "BOOLEAN",
                "value": 1,
                "environment": "local",
            },
        ]
    finally:
        await cleanup_models([message, conversation, user])


async def test_update_message_langfuse_down_still_saves(async_client, enabled, recorder):
    recorder.raise_exc = httpx.ConnectError("connection refused")
    user, token, conversation, message = await _owned_message()
    try:
        resp = await _patch(
            async_client, token, conversation, message, {"feedback": "negative"}
        )
        assert resp.status_code == 200
        await _drain()
        assert len(recorder.requests) == 1
        saved = await Message.find_by_id(message.id)
        assert saved.feedback == "negative"
    finally:
        await cleanup_models([message, conversation, user])


async def test_update_message_without_trace_id_sends_nothing(
    async_client, enabled, recorder
):
    user, token, conversation, message = await _owned_message(trace_id=None)
    try:
        resp = await _patch(
            async_client, token, conversation, message, {"feedback": "negative"}
        )
        assert resp.status_code == 200
        await _drain()
        assert recorder.requests == []
    finally:
        await cleanup_models([message, conversation, user])


async def test_update_message_flag_off_sends_nothing(
    async_client, enabled, recorder, monkeypatch
):
    monkeypatch.setattr(config, "FEATURE_LANGFUSE_SCORES", False)
    user, token, conversation, message = await _owned_message()
    try:
        resp = await _patch(
            async_client, token, conversation, message, {"feedback": "positive"}
        )
        assert resp.status_code == 200
        await _drain()
        assert recorder.requests == []
    finally:
        await cleanup_models([message, conversation, user])
