"""Findings of the B3 review on the AG-UI stream, one case per finding.

Pure: the adapter, the validator and the cancel manager, no route, no Mongo.
The route cases (disconnect reason, one cancel) are in ``test_agui_disconnect``.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List

import pytest

from src.services.agui.adapter import AguiAdapter, close_json_prefix, encode_sse
from src.services.cancel_manager import CancelManager
from tests.agui.test_adapter import (
    ENDPOINT,
    M,
    R,
    T,
    _assert_valid,
    _ctx,
    _of,
    _run,
    f_final,
    f_status,
    f_token,
    f_tool_call,
    f_tool_result,
)
from tests.agui.validator import validate_transcript, violation_codes

pytestmark = pytest.mark.no_db

_SURROGATES = range(0xD800, 0xE000)


def _wire(frames: List[Dict[str, Any]]) -> str:
    """The SSE body the route sends for ``frames``: through ``encode_sse``."""
    adapter = AguiAdapter(_ctx())
    events: List[Any] = list(adapter.start())
    for frame in frames:
        events.extend(adapter.adapt(frame))
    events.extend(adapter.finish())
    return encode_sse(events)


def _parsed(body: str) -> List[Dict[str, Any]]:
    return [json.loads(line[len("data: "):]) for line in body.splitlines() if line.startswith("data: ")]


def _result_turn(content: str) -> List[Dict[str, Any]]:
    result = f_tool_result("call_1", "search", "", server="s", duration_ms=5)
    result["content"] = content  # as sent, not cut again by the builder
    return [
        f_status(),
        f_tool_call("call_1", "search", {"query": "q"}, server="s", label="Search"),
        result,
        f_token("Done."),
        f_final("Done.", endpoint=ENDPOINT),
    ]


# ─── 1. a cut never fails serialization ──────────────────────────────────────

# What stringify_tool_content gives for a result with emoji, a math letter and
# escapes: json.dumps defaults, so non-ASCII travels as \u escapes and every
# astral char as a surrogate pair of escapes.
_ESCAPED_DOC = json.dumps(
    {"title": "Flood 😀 map 𝔸", "note": "a \"quoted\" line\nnext", "tags": ["🌊", "é"]}
)


@pytest.mark.parametrize("end", range(1, len(_ESCAPED_DOC) + 1))
def test_every_cut_of_an_escaped_result_serializes_and_stays_json(end):
    body = _wire(_result_turn(_ESCAPED_DOC[:end]))

    (result,) = [e for e in _parsed(body) if e["type"] == "TOOL_CALL_RESULT"]
    parsed = json.loads(result["content"])
    assert isinstance(parsed, (dict, list))
    assert not any(ord(ch) in _SURROGATES for ch in result["content"])
    assert validate_transcript(body, thread_id=T, run_id=R) == []


def test_cut_between_the_two_halves_of_a_pair_drops_the_lone_half():
    doc = json.dumps({"t": "ab😀"})
    cut = doc[: doc.index("\\ude00")]  # ends right after the high half
    closed = close_json_prefix(cut)

    assert closed is not None
    value = json.loads(closed)["t"]
    assert not any(ord(ch) in _SURROGATES for ch in value)
    assert value.startswith("ab")


@pytest.mark.parametrize(
    "frame",
    [
        {"type": "tool_result", "content": "plain text with a lone \ud83d half", "tool": "x", "id": "call_1"},
        {"type": "token", "content": "lone \udc00 low half"},
    ],
    ids=["tool_result_text", "token"],
)
def test_encode_sse_never_raises_on_a_lone_surrogate(frame):
    frames = [f_status(), f_tool_call("call_1", "x", {}, server="s", label="X")]
    frames.append(frame)
    frames.append(f_final("Done.", endpoint=ENDPOINT))

    body = _wire(frames)  # must not raise

    assert not any(ord(ch) in _SURROGATES for ch in body)
    assert _parsed(body)[-1]["type"] == "RUN_FINISHED"


# ─── 4. text after a tool call opens a new message, never reopens one ────────


def test_text_after_a_tool_call_in_one_model_call_opens_a_new_message_id():
    # The text-format flush: prose, the call frame, then leftover prose, with
    # no tool result between (runner.py text-format path).
    events = _run(
        [
            f_status(),
            f_token("Let me look. "),
            f_tool_call("call_1", "search", {"query": "a"}, server="s", label="Search"),
            f_token("More prose."),
            f_tool_result("call_1", "search", "1", server="s", duration_ms=5),
            f_token("Answer."),
            f_final("Let me look. More prose.\n\nAnswer.", endpoint=ENDPOINT),
        ]
    )

    ids = [e["messageId"] for e in _of(events, "TEXT_MESSAGE_START")]
    assert len(ids) == len(set(ids)) == 3
    assert ids[0] == M
    _assert_valid(events)


def _body(events: List[Dict[str, Any]]) -> str:
    return "".join(f"data: {json.dumps(e, separators=(',', ':'))}\n\n" for e in events)


_START = {"type": "RUN_STARTED", "threadId": T, "runId": R, "protocolVersion": "1.0",
          "metadata": {"eve.traceId": "a" * 32, "eve.conversationId": T, "eve.messageId": M}}
_FINISHED = {"type": "RUN_FINISHED", "threadId": T, "runId": R, "outcome": {"type": "success"}}


def _triad(mid: str, text: str) -> List[Dict[str, Any]]:
    return [
        {"type": "TEXT_MESSAGE_START", "messageId": mid, "role": "assistant"},
        {"type": "TEXT_MESSAGE_CONTENT", "messageId": mid, "delta": text},
        {"type": "TEXT_MESSAGE_END", "messageId": mid},
    ]


def test_validator_rejects_a_same_role_reopen_of_a_closed_message():
    body = _body([_START, *_triad(M, "one"), *_triad(M, "two"), _FINISHED])

    assert "text_reopen" in violation_codes(validate_transcript(body, thread_id=T, run_id=R))


def test_validator_rejects_a_skipped_segment_number():
    body = _body([_START, *_triad(M, "one"), *_triad(f"{M}:assistant:3", "two"), _FINISHED])

    assert "segment_order" in violation_codes(validate_transcript(body, thread_id=T, run_id=R))


def test_validator_accepts_segments_in_order():
    body = _body([_START, *_triad(M, "one"), *_triad(f"{M}:assistant:2", "two"), _FINISHED])

    assert validate_transcript(body, thread_id=T, run_id=R) == []


# ─── 5. closing a cut JSON document is linear ────────────────────────────────


def test_closing_a_cut_100kb_array_is_fast():
    doc = json.dumps([{"id": i, "name": f"scene-{i}"} for i in range(5000)])
    assert len(doc) > 100_000
    cut = doc[: len(doc) - 7]

    started = time.perf_counter()
    closed = close_json_prefix(cut)
    elapsed = time.perf_counter() - started

    assert closed is not None
    assert json.loads(closed)[0] == {"id": 0, "name": "scene-0"}
    assert elapsed < 0.05, f"{elapsed * 1000:.0f} ms"


# ─── 6. a runner-cut result is always flagged ────────────────────────────────


def test_plain_text_result_the_runner_cut_is_flagged():
    # The runner sends the first 200 chars of every result.
    text = ("Scene list: " + "S1A_IW_GRDH " * 40)[:200]
    body = _wire(_result_turn(text))

    (result,) = [e for e in _parsed(body) if e["type"] == "TOOL_CALL_RESULT"]
    assert result["content"] == text
    assert result["metadata"]["eve.contentTruncated"] is True


def test_short_plain_text_result_is_not_flagged():
    text = "x" * 199
    body = _wire(_result_turn(text))

    (result,) = [e for e in _parsed(body) if e["type"] == "TOOL_CALL_RESULT"]
    assert "eve.contentTruncated" not in result["metadata"]


def test_runner_content_truncated_field_is_honoured():
    frames = _result_turn("short")
    frames[2]["content_truncated"] = True
    body = _wire(frames)

    (result,) = [e for e in _parsed(body) if e["type"] == "TOOL_CALL_RESULT"]
    assert result["metadata"]["eve.contentTruncated"] is True


# ─── 2 and 3. cancel reason and one cancel ───────────────────────────────────


class _Task:
    def __init__(self) -> None:
        self.cancels = 0

    def done(self) -> bool:
        return False

    def cancel(self) -> bool:
        self.cancels += 1
        return True


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setattr("src.services.cancel_manager.REDIS_URL", "")
    return CancelManager()


async def test_disconnect_then_stop_cancels_the_task_once(manager):
    task = _Task()
    manager.create("m", task)  # type: ignore[arg-type]

    manager.cancel("m", reason="client_disconnected")
    manager.cancel("m")

    assert task.cancels == 1


async def test_stop_then_disconnect_cancels_the_task_once(manager):
    task = _Task()
    manager.create("m", task)  # type: ignore[arg-type]

    manager.cancel("m")
    manager.cancel("m", reason="client_disconnected")

    assert task.cancels == 1
    assert manager.cancel_reason("m") is None


async def test_disconnect_reason_is_kept_until_an_explicit_stop(manager):
    manager.create("m", _Task())  # type: ignore[arg-type]

    manager.cancel("m", reason="client_disconnected")
    assert manager.cancel_reason("m") == "client_disconnected"

    manager.cancel("m")
    assert manager.cancel_reason("m") is None


async def test_clear_forgets_the_reason(manager):
    manager.create("m", _Task())  # type: ignore[arg-type]
    manager.cancel("m", reason="client_disconnected")

    manager.clear("m")

    assert manager.cancel_reason("m") is None


async def test_a_disconnect_publishes_no_cross_worker_cancel(monkeypatch):
    monkeypatch.setattr("src.services.cancel_manager.REDIS_URL", "redis://unused:6379/0")
    manager = CancelManager()
    published: List[str] = []

    async def _publish(message_id: str) -> None:
        published.append(message_id)

    monkeypatch.setattr(manager, "_publish_cancel", _publish)
    monkeypatch.setattr(manager, "_subscribe_cancel_channel", lambda *a: asyncio.sleep(0))
    manager.create("m", _Task())  # type: ignore[arg-type]

    manager.cancel("m", reason="client_disconnected")
    await asyncio.sleep(0)
    assert published == []

    manager.cancel("m")
    await asyncio.sleep(0)
    assert published == ["m"]
