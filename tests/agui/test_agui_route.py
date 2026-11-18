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


# ─── Failure modes after review ───────────────────────────────────────────────


class RaisingBus(FakeBus):
    """Delivers the first published frame, then the subscription raises (a
    Valkey error mid-stream)."""

    async def subscribe(self, key: str, ready: Optional[asyncio.Event] = None):
        queue = self._q(key)
        if ready is not None:
            ready.set()
        item = await queue.get()
        yield item
        raise ConnectionError("valkey went away with secret-ish detail")


@pytest.fixture
def limiter_one(monkeypatch):
    from src.services import load_shedding
    from src.services.load_shedding import GenerationLimiter

    limiter = GenerationLimiter(1)
    monkeypatch.setattr(load_shedding, "_limiter", limiter)
    return limiter


async def test_bus_raising_mid_stream_ends_with_stream_incomplete(
    async_client, flag_on, monkeypatch, owner, caplog
):
    _, token, conversation = owner
    bus = RaisingBus()
    monkeypatch.setattr("src.routers.agui.get_stream_bus", lambda: bus)
    fake = FakeRunner(bus, [f_status(), f_token("cut"), f_final("cut")])
    monkeypatch.setattr("src.routers.agui.run_agentic_generation_to_bus", fake)
    run_id = str(uuid.uuid4())

    response = await post(async_client, conversation.id, token, run_input(conversation.id, run_id))

    events = assert_valid(response, conversation.id, run_id)
    assert events[-1]["type"] == "RUN_ERROR"
    assert events[-1]["code"] == "stream_incomplete"
    assert "agui.stream_failed error_type=ConnectionError" in caplog.text
    assert "secret-ish" not in caplog.text
    # Nobody reads the run any more: it is stopped, not left running unseen.
    assert fake.calls[0]["cancel_event"].is_set()
    assert "reason=stream_failed" in caplog.text


async def test_lost_bus_close_ends_at_the_stream_deadline(async_client, flag_on, runner, owner, monkeypatch):
    _, token, conversation = owner
    monkeypatch.setattr(agui_route, "AGENTIC_TURN_DEADLINE", 0)
    monkeypatch.setattr(agui_route, "STREAM_DEADLINE_MARGIN_S", 0.3)
    fake = runner([f_status(), f_token("still going")], wait_cancel=True)
    run_id = str(uuid.uuid4())
    try:
        response = await asyncio.wait_for(
            post(async_client, conversation.id, token, run_input(conversation.id, run_id)), 10
        )
        events = assert_valid(response, conversation.id, run_id)
        assert events[-1]["code"] == "stream_incomplete"
    finally:
        fake.calls[0]["cancel_event"].set()


async def test_no_token_answers_401_with_the_flag_on(async_client, flag_on, runner, owner):
    _, _, conversation = owner
    response = await async_client.post(
        f"/conversations/{conversation.id}/agui", json=run_input(conversation.id)
    )
    assert response.status_code == 401


async def test_pending_user_answers_403(async_client, flag_on, runner):
    from src.services.approval import APPROVAL_PENDING

    user, token = await create_test_user_and_token(approval_status=APPROVAL_PENDING)
    conversation = Conversation(user_id=user.id, name="pending")
    await conversation.save()
    fake = runner([f_status(), f_final("x")])
    try:
        response = await post(async_client, conversation.id, token, run_input(conversation.id))
        assert response.status_code == 403, response.text
        assert fake.calls == []
    finally:
        await cleanup_models([user, conversation])


async def test_request_rate_limit_answers_429_rate_limited(async_client, flag_on, runner, owner, monkeypatch):
    from src.services import request_rate_limiter as rrl
    from tests.test_request_rate_limiter import FakeStore, limiter

    _, token, conversation = owner
    monkeypatch.setattr(rrl, "FEATURE_REQUEST_RATE_LIMIT", True)
    monkeypatch.setattr(
        rrl, "_limiter", limiter(FakeStore(), mode="enforce", limits={"chat": {"rate": 1, "burst": 1}})
    )
    fake = runner([f_status(), f_token("x"), f_final("x")])

    first = await post(async_client, conversation.id, token, run_input(conversation.id))
    second = await post(async_client, conversation.id, token, run_input(conversation.id))

    assert first.status_code == 200
    assert second.status_code == 429, second.text
    assert second.json()["detail"]["code"] == "rate_limited"
    assert int(second.headers["retry-after"]) >= 1
    assert len(fake.calls) == 1


async def test_slot_is_released_after_every_pre_stream_refusal(
    async_client, flag_on, runner, owner, limiter_one
):
    user, token, conversation = owner
    fake = runner([f_status(), f_token("x"), f_final("x")])
    used = str(uuid.uuid4())
    first = await post(async_client, conversation.id, token, run_input(conversation.id, used))
    assert first.status_code == 200
    for _ in range(100):
        if limiter_one.in_flight == 0:
            break
        await asyncio.sleep(0.01)

    other = Conversation(user_id="someone-else", name="x")
    await other.save()
    foreign = CollectionModel(user_id="someone-else", name="private")
    await foreign.save()
    try:
        refusals = [
            (await post(async_client, other.id, token, run_input(other.id)), 404),
            (
                await post(
                    async_client,
                    conversation.id,
                    token,
                    run_input(conversation.id, forwardedProps={"eve": {"privateCollections": [foreign.id]}}),
                ),
                403,
            ),
            (await post(async_client, conversation.id, token, run_input(conversation.id, used)), 409),
        ]
        for response, status in refusals:
            assert response.status_code == status, response.text
            assert limiter_one.in_flight == 0
    finally:
        await cleanup_models([other, foreign])
    assert len(fake.calls) == 1


async def test_two_posts_with_one_run_id_start_one_turn(async_client, flag_on, runner, owner, monkeypatch):
    # Both reads of the conflict check run before either insert (the race two
    # workers hit): the claim decides.
    _, token, conversation = owner
    monkeypatch.setattr(agui_route, "_run_id_used", lambda *a: asyncio.sleep(0, result=False))
    fake = runner([f_status(), f_token("x"), f_final("x")])
    run_id = str(uuid.uuid4())

    responses = await asyncio.gather(
        post(async_client, conversation.id, token, run_input(conversation.id, run_id)),
        post(async_client, conversation.id, token, run_input(conversation.id, run_id)),
    )

    assert sorted(r.status_code for r in responses) == [200, 409]
    assert len(fake.calls) == 1


class FakeClaimStore:
    """``SET NX EX`` and ``DELETE`` of the Valkey run id claim, shared by workers."""

    def __init__(self, fail: bool = False) -> None:
        self.keys: Dict[str, int] = {}
        self.fail = fail

    async def set(self, key, value, nx=False, ex=None):
        if self.fail:
            raise ConnectionError("down")
        if nx and key in self.keys:
            return None
        self.keys[key] = ex
        return True

    async def delete(self, key):
        self.keys.pop(key, None)


async def test_two_workers_cannot_claim_one_run_id(monkeypatch):
    store = FakeClaimStore()
    monkeypatch.setattr(agui_route, "_store_client", lambda: store)
    monkeypatch.setattr(agui_route, "_runs_in_admission", set())

    assert await agui_route._claim_run_id("c1", "r1") == "store"
    # Worker B has its own memory; only the shared store knows the claim.
    monkeypatch.setattr(agui_route, "_runs_in_admission", set())
    assert await agui_route._claim_run_id("c1", "r1") is None
    assert store.keys == {"agui-run:c1:r1": agui_route.RUN_ID_CLAIM_TTL_S}
    # A pre-stream failure gives the run id back; a created turn keeps it.
    await agui_route._release_run_id("c1", "r1", "store", created=False)
    assert store.keys == {}
    assert await agui_route._claim_run_id("c1", "r1") == "store"
    await agui_route._release_run_id("c1", "r1", "store", created=True)
    assert "agui-run:c1:r1" in store.keys


async def test_without_a_store_the_claim_is_per_worker(monkeypatch):
    monkeypatch.setattr(config, "REDIS_URL", "")
    monkeypatch.setattr(agui_route, "_runs_in_admission", set())

    assert await agui_route._claim_run_id("c1", "r1") == "local"
    assert await agui_route._claim_run_id("c1", "r1") is None
    await agui_route._release_run_id("c1", "r1", "local", created=True)
    assert await agui_route._claim_run_id("c1", "r1") == "local"


async def test_a_store_that_fails_falls_back_to_the_worker(monkeypatch, caplog):
    monkeypatch.setattr(agui_route, "_store_client", lambda: FakeClaimStore(fail=True))
    monkeypatch.setattr(agui_route, "_runs_in_admission", set())

    assert await agui_route._claim_run_id("c1", "r1") == "local"
    assert await agui_route._claim_run_id("c1", "r1") is None
    assert "agui.run_id_store_failed error_type=ConnectionError" in caplog.text


@pytest.mark.parametrize("run_id", ["x" * 65, "has space", "line\nbreak", ""])
async def test_bad_run_id_answers_422(async_client, flag_on, runner, owner, run_id):
    _, token, conversation = owner
    fake = runner([f_status(), f_final("x")])
    body = run_input(conversation.id)
    body["runId"] = run_id

    response = await post(async_client, conversation.id, token, body)

    assert response.status_code == 422, response.text
    assert fake.calls == []


async def test_missing_artifact_answers_the_contract_404_and_frees_the_run_id(
    async_client, flag_on, runner, owner
):
    _, token, conversation = owner
    fake = runner([f_status(), f_token("x"), f_final("x")])
    run_id = str(uuid.uuid4())
    missing = "6a3f0c4b9b1d4e7f8a2b3c99"
    parts = [
        {"type": "text", "text": "describe"},
        {"type": "image", "source": {"type": "url", "value": "https://x.invalid/a.png"}, "metadata": {"eve.artifactId": missing}},
    ]
    body = run_input(conversation.id, run_id, messages=[{"id": "u1", "role": "user", "content": parts}])

    response = await post(async_client, conversation.id, token, body)

    assert response.status_code == 404, response.text
    assert response.json()["detail"]["code"] == "artifact_not_found"
    assert fake.calls == []
    # No turn was created, so the client may resend the same run id.
    retry = await post(async_client, conversation.id, token, run_input(conversation.id, run_id))
    assert_valid(retry, conversation.id, run_id)


async def test_missing_custom_model_answers_the_contract_404(async_client, flag_on, runner, owner):
    _, token, conversation = owner
    fake = runner([f_status(), f_final("x")])
    body = run_input(conversation.id, forwardedProps={"eve": {"customModelId": "6a3f0c4b9b1d4e7f8a2b3c98"}})

    response = await post(async_client, conversation.id, token, body)

    assert response.status_code == 404, response.text
    assert response.json()["detail"]["code"] == "custom_model_not_found"
    assert fake.calls == []


async def test_skipped_input_part_is_reported_as_a_notice(async_client, flag_on, runner, owner):
    _, token, conversation = owner
    runner([f_status(), f_token("Hi."), f_final("Hi.")])
    run_id = str(uuid.uuid4())
    parts = [
        {"type": "text", "text": "look at this"},
        {"type": "image", "id": "part-2", "source": {"type": "data", "value": "aGVsbG8=", "mimeType": "image/png"}},
    ]
    body = run_input(conversation.id, run_id, messages=[{"id": "u1", "role": "user", "content": parts}])

    response = await post(async_client, conversation.id, token, body)

    events = assert_valid(response, conversation.id, run_id)
    assert events[1]["type"] == "CUSTOM"
    assert events[1]["name"] == "eve.notice"
    assert events[1]["value"]["code"] == "input_part_skipped"
    assert events[1]["value"]["partIds"] == ["part-2"]


async def test_client_protocol_version_text_is_not_logged(async_client, flag_on, runner, owner, caplog):
    _, token, conversation = owner
    runner([f_status(), f_token("Hi."), f_final("Hi.")])
    body = run_input(conversation.id, protocolVersion="1.9 private user words")

    response = await post(async_client, conversation.id, token, body)

    assert response.status_code == 200
    assert "private user words" not in caplog.text
    assert "agui.protocol_minor version=unrecognised" in caplog.text
