"""POST /conversations/{id}/agui (src/routers/agui.py).

The runner and the stream bus are fakes: the runner publishes a scripted list
of legacy frames once the route subscribed, then closes the bus, as the real
one does after the token charge. Every streamed body goes through the shared
transcript validator. The Stop case goes through the real Stop route.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytest

from src import config
from src.database.models.collection import Collection as CollectionModel
from src.database.models.conversation import Conversation
from src.database.models.message import Message
from src.routers import agui as agui_route
from tests.agui.validator import validate_transcript
from tests.utils.cleaner import cleanup_models
from tests.utils.utils import create_test_user_and_token

TRACE = "4481c48c462beed158f2fe9a26f5b456"


class FakeBus:
    """In-process bus: one queue per message id, ``None`` closes it."""

    def __init__(self) -> None:
        self.queues: Dict[str, asyncio.Queue] = {}

    def _q(self, key: str) -> asyncio.Queue:
        return self.queues.setdefault(key, asyncio.Queue())

    async def publish(self, key: str, data: str) -> None:
        await self._q(key).put(data)

    async def stop(self, key: str, event: str) -> None:
        await self._q(key).put(event)

    async def close(self, key: str) -> None:
        await self._q(key).put(None)

    async def subscribe(self, key: str, ready: Optional[asyncio.Event] = None):
        queue = self._q(key)
        if ready is not None:
            ready.set()
        while True:
            item = await queue.get()
            if item is None:
                return
            yield item


def sse(frame: Dict[str, Any]) -> str:
    return f"data: {json.dumps(frame)}\n\n"


def f_status() -> Dict[str, Any]:
    return {"type": "status", "content": "Thinking…"}


def f_token(text: str) -> Dict[str, Any]:
    return {"type": "token", "content": text}


def f_final(answer: str) -> Dict[str, Any]:
    return {
        "type": "final",
        "answer": answer,
        "latencies": {"total_seconds": 0.1},
        "artifact_ids": None,
        "trace_id": TRACE,
    }


class FakeRunner:
    """Stands in for run_agentic_generation_to_bus."""

    def __init__(self, bus: FakeBus, frames: List[Dict[str, Any]], *, wait_cancel=False, delay=0.0):
        self.bus = bus
        self.frames = frames
        self.wait_cancel = wait_cancel
        self.delay = delay
        self.calls: List[Dict[str, Any]] = []
        self.started = asyncio.Event()

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        message_id = kwargs["message_id"]
        await kwargs["subscriber_ready"].wait()
        self.started.set()
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            for frame in self.frames:
                await self.bus.publish(message_id, sse(frame))
            if self.wait_cancel:
                await kwargs["cancel_event"].wait()
        finally:
            await self.bus.close(message_id)


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setattr(config, "FEATURE_AGUI_STREAM", True)


@pytest.fixture
def bus(monkeypatch):
    fake = FakeBus()
    monkeypatch.setattr("src.routers.agui.get_stream_bus", lambda: fake)
    monkeypatch.setattr("src.routers.message.get_stream_bus", lambda: fake)
    return fake


@pytest.fixture
def runner(monkeypatch, bus):
    def _install(frames, **kwargs) -> FakeRunner:
        fake = FakeRunner(bus, frames, **kwargs)
        monkeypatch.setattr("src.routers.agui.run_agentic_generation_to_bus", fake)
        return fake

    return _install


@pytest.fixture
async def owner():
    user, token = await create_test_user_and_token()
    conversation = Conversation(user_id=user.id, name="agui")
    await conversation.save()
    yield user, token, conversation
    for message in await Message.find_all(filter_dict={"conversation_id": conversation.id}):
        await message.delete()
    await cleanup_models([user, conversation])


def run_input(thread_id: str, run_id: Optional[str] = None, text: str = "hello", **extra) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "threadId": thread_id,
        "runId": run_id or str(uuid.uuid4()),
        "protocolVersion": "1.0",
        "messages": [{"id": "u1", "role": "user", "content": text}],
        "tools": [],
        "context": [],
        "state": None,
        "forwardedProps": {},
    }
    body.update(extra)
    return body


async def post(client, conversation_id: str, token: str, body: Dict[str, Any]):
    return await client.post(
        f"/conversations/{conversation_id}/agui",
        json=body,
        headers={"Authorization": f"Bearer {token}", "Accept": "text/event-stream"},
    )


def events_of(body: str) -> List[Dict[str, Any]]:
    return [json.loads(line[len("data: "):]) for line in body.splitlines() if line.startswith("data: ")]


def assert_valid(response, conversation_id: str, run_id: str) -> List[Dict[str, Any]]:
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    violations = validate_transcript(response.text, thread_id=conversation_id, run_id=run_id)
    assert violations == [], violations
    return events_of(response.text)


# ─── Errors before the stream ────────────────────────────────────────────────


async def test_flag_off_answers_404_even_without_a_token(async_client, monkeypatch, runner):
    monkeypatch.setattr(config, "FEATURE_AGUI_STREAM", False)
    fake = runner([f_status(), f_token("x"), f_final("x")])
    conversation_id = "6a3f0c2e9b1d4e7f8a2b3c4d"
    response = await async_client.post(
        f"/conversations/{conversation_id}/agui", json=run_input(conversation_id)
    )
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "not_found"
    assert fake.calls == []


async def test_foreign_conversation_answers_404(async_client, flag_on, runner, owner):
    _, _, conversation = owner
    other, other_token = await create_test_user_and_token()
    fake = runner([f_status(), f_token("x"), f_final("x")])
    try:
        response = await post(async_client, conversation.id, other_token, run_input(conversation.id))
        assert response.status_code == 404, response.text
        assert response.json() == {
            "detail": {"code": "not_found", "message": "Conversation not found."}
        }
        assert fake.calls == []
        assert await Message.find_all(filter_dict={"conversation_id": conversation.id}) == []
    finally:
        await cleanup_models([other])


async def test_spent_budget_answers_429_before_the_stream(async_client, flag_on, runner, owner):
    user, token, conversation = owner
    user.rate_limit_tokens_used = 1000
    user.rate_limit_period_start = datetime.now(timezone.utc) - timedelta(days=1)
    user.rate_limit_period_end = datetime.now(timezone.utc) + timedelta(days=1)
    await user.save()
    fake = runner([f_status(), f_token("x"), f_final("x")])

    response = await post(async_client, conversation.id, token, run_input(conversation.id))

    assert response.status_code == 429, response.text
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["detail"]["code"] == "budget_exhausted"
    assert "retry-after" in response.headers
    assert fake.calls == []
    assert await Message.find_all(filter_dict={"conversation_id": conversation.id}) == []


async def test_thread_id_must_equal_the_path(async_client, flag_on, runner, owner):
    _, token, conversation = owner
    runner([])
    response = await post(async_client, conversation.id, token, run_input("6a3f0c2e9b1d4e7f8a2b3c4d"))
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "thread_mismatch"


@pytest.mark.parametrize(
    "messages",
    [[], [{"id": "a1", "role": "assistant", "content": "hi"}]],
)
async def test_no_trailing_user_message_answers_400(async_client, flag_on, runner, owner, messages):
    _, token, conversation = owner
    runner([])
    response = await post(async_client, conversation.id, token, run_input(conversation.id, messages=messages))
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "no_user_message"


async def test_major_version_two_answers_400(async_client, flag_on, runner, owner):
    _, token, conversation = owner
    runner([])
    response = await post(async_client, conversation.id, token, run_input(conversation.id, protocolVersion="2.0"))
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "protocol_version_unsupported"


async def test_forwarded_agent_answers_400(async_client, flag_on, runner, owner):
    _, token, conversation = owner
    runner([])
    body = run_input(conversation.id, forwardedProps={"eve": {"agent": "deep"}})
    response = await post(async_client, conversation.id, token, body)
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "agent_not_allowed"


async def test_bad_forwarded_value_answers_422(async_client, flag_on, runner, owner):
    _, token, conversation = owner
    runner([])
    body = run_input(conversation.id, forwardedProps={"eve": {"k": 99}})
    response = await post(async_client, conversation.id, token, body)
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)
    assert response.json()["detail"][0]["loc"][:4] == ["body", "forwardedProps", "eve", "k"]


async def test_foreign_private_collection_answers_403(async_client, flag_on, runner, owner):
    _, token, conversation = owner
    other, _ = await create_test_user_and_token()
    collection = CollectionModel(user_id=other.id, name="theirs")
    await collection.save()
    fake = runner([f_status(), f_token("x"), f_final("x")])
    try:
        body = run_input(conversation.id, forwardedProps={"eve": {"privateCollections": [collection.id]}})
        response = await post(async_client, conversation.id, token, body)
        assert response.status_code == 403, response.text
        assert response.json()["detail"]["code"] == "collection_forbidden"
        assert fake.calls == []
    finally:
        await cleanup_models([other, collection])


async def test_run_id_reuse_answers_409(async_client, flag_on, runner, owner):
    _, token, conversation = owner
    fake = runner([f_status(), f_token("Hi"), f_final("Hi")])
    run_id = str(uuid.uuid4())

    first = await post(async_client, conversation.id, token, run_input(conversation.id, run_id))
    assert_valid(first, conversation.id, run_id)
    second = await post(async_client, conversation.id, token, run_input(conversation.id, run_id, text="again"))

    assert second.status_code == 409, second.text
    assert second.json()["detail"]["code"] == "run_id_conflict"
    assert len(fake.calls) == 1
    rows = await Message.find_all(filter_dict={"conversation_id": conversation.id})
    assert [m.metadata.get("run_id") for m in rows] == [run_id]


# ─── Streamed runs ────────────────────────────────────────────────────────────


async def test_happy_transcript(async_client, flag_on, runner, owner):
    user, token, conversation = owner
    fake = runner([f_status(), f_token("Hello"), f_token(" world"), f_final("Hello world")])
    run_id = str(uuid.uuid4())
    body = run_input(
        conversation.id,
        run_id,
        text="first",
        messages=[
            {"id": "u0", "role": "user", "content": "old turn, ignored"},
            {"id": "u1", "role": "user", "content": [{"type": "text", "text": "what is"}, {"type": "text", "text": "SAR?"}]},
        ],
        forwardedProps={"eve": {"model": "fallback", "k": 3, "unknownKey": 1}},
    )

    response = await post(async_client, conversation.id, token, body)

    events = assert_valid(response, conversation.id, run_id)
    types = [e["type"] for e in events]
    assert types[0] == "RUN_STARTED" and types[-1] == "RUN_FINISHED"
    assert events[-1]["outcome"]["type"] == "success"
    text = "".join(e["delta"] for e in events if e["type"] == "TEXT_MESSAGE_CONTENT")
    assert text == "Hello world"
    # Only the trailing user message reaches the runner, with the allowlisted props.
    request = fake.calls[0]["request"]
    assert request.query == "what is\nSAR?"
    assert request.llm_type == "fallback" and request.k == 3
    message = await Message.find_by_id(fake.calls[0]["message_id"])
    assert message.metadata["run_id"] == run_id
    assert events[0]["metadata"]["eve.messageId"] == message.id


async def test_tool_call_transcript(async_client, flag_on, runner, owner):
    _, token, conversation = owner
    runner(
        [
            f_status(),
            {
                "type": "tool_call",
                "content": "Searching: SAR",
                "tool": "eve_retrieval_retrieve",
                "label": "Searching",
                "query": "SAR",
                "id": "call_1",
                "args": {"query": "SAR"},
            },
            {
                "type": "tool_result",
                "content": "three documents",
                "tool": "eve_retrieval_retrieve",
                "status": "ok",
                "id": "call_1",
            },
            f_token("SAR is radar."),
            f_final("SAR is radar."),
        ]
    )
    run_id = str(uuid.uuid4())

    response = await post(async_client, conversation.id, token, run_input(conversation.id, run_id))

    events = assert_valid(response, conversation.id, run_id)
    types = [e["type"] for e in events]
    for expected in ("TOOL_CALL_START", "TOOL_CALL_ARGS", "TOOL_CALL_END", "TOOL_CALL_RESULT"):
        assert expected in types
    assert events[-1]["outcome"]["type"] == "success"


async def test_stop_through_the_stop_route(async_client, flag_on, runner, owner):
    _, token, conversation = owner
    fake = runner([f_status(), f_token("Partial")], wait_cancel=True)
    run_id = str(uuid.uuid4())

    stream = asyncio.create_task(post(async_client, conversation.id, token, run_input(conversation.id, run_id)))
    await asyncio.wait_for(fake.started.wait(), 5)
    stop = await async_client.post(
        f"/conversations/{conversation.id}/stop", headers={"Authorization": f"Bearer {token}"}
    )
    assert stop.status_code == 200, stop.text
    assert stop.json()["status"] == "stopping"
    response = await asyncio.wait_for(stream, 10)

    events = assert_valid(response, conversation.id, run_id)
    assert events[-1]["type"] == "RUN_FINISHED"
    assert events[-1]["outcome"]["type"] == "cancelled"


async def test_error_frame_ends_with_run_error(async_client, flag_on, runner, owner):
    _, token, conversation = owner
    runner([f_status(), {"type": "error", "code": "timeout", "message": "raw upstream text"}])
    run_id = str(uuid.uuid4())

    response = await post(async_client, conversation.id, token, run_input(conversation.id, run_id))

    events = assert_valid(response, conversation.id, run_id)
    assert events[-1]["type"] == "RUN_ERROR"
    assert events[-1]["code"] == "timeout"
    assert "raw upstream text" not in response.text


async def test_bus_closing_early_ends_with_stream_incomplete(async_client, flag_on, runner, owner):
    _, token, conversation = owner
    runner([f_status(), f_token("cut")])
    run_id = str(uuid.uuid4())

    response = await post(async_client, conversation.id, token, run_input(conversation.id, run_id))

    events = assert_valid(response, conversation.id, run_id)
    assert events[-1]["type"] == "RUN_ERROR"
    assert events[-1]["code"] == "stream_incomplete"


async def test_slow_stream_carries_the_keepalive_comment(async_client, flag_on, runner, owner, monkeypatch):
    _, token, conversation = owner
    monkeypatch.setattr(agui_route, "KEEPALIVE_INTERVAL_S", 0.05)
    runner([f_status(), f_token("late"), f_final("late")], delay=0.3)
    run_id = str(uuid.uuid4())

    response = await post(async_client, conversation.id, token, run_input(conversation.id, run_id))

    assert ": keepalive\n\n" in response.text
    assert_valid(response, conversation.id, run_id)


async def test_in_flight_cap_answers_429_overloaded(async_client, flag_on, runner, owner, monkeypatch):
    from src.services import load_shedding
    from src.services.load_shedding import GenerationLimiter

    _, token, conversation = owner
    limiter = GenerationLimiter(1)
    monkeypatch.setattr(load_shedding, "_limiter", limiter)
    held = await limiter.try_acquire()
    fake = runner([f_status(), f_token("x"), f_final("x")])
    try:
        response = await post(async_client, conversation.id, token, run_input(conversation.id))
        assert response.status_code == 429
        assert response.json()["detail"]["code"] == "overloaded"
        assert response.headers["retry-after"] == "10"
        assert fake.calls == []
        assert limiter.in_flight == 1
    finally:
        held.release()
