"""Per-worker cap on in-flight answer generations (src/services/load_shedding.py).

Past the cap a generation route answers 429 ``overloaded`` with Retry-After
before doing any work, and every admitted generation gives its slot back
whether it ends normally, fails or is cancelled. The streaming routes hand the
slot to the decoupled generation task, so a client that goes away does not
free it while the work is still running.
"""

import asyncio
import contextlib

import pytest

from src.database.models.conversation import Conversation
from src.database.models.message import Message
from src.services import load_shedding
from src.services.load_shedding import (
    SHED_RETRY_AFTER_SECONDS,
    GenerationLimiter,
    get_generation_limiter,
)
from tests.utils.cleaner import cleanup_models
from tests.utils.utils import create_test_user_and_token

CLASSIC_RESULT = ("answer", [], False, {"total_seconds": 0.1}, {}, {})
AGENTIC_RESULT = ("answer", [], False, {"total_seconds": 0.1}, {}, [], [])


@pytest.fixture
def cap(monkeypatch):
    """Install a fresh limiter with the given cap for the test."""

    def _install(limit: int) -> GenerationLimiter:
        limiter = GenerationLimiter(limit)
        monkeypatch.setattr(load_shedding, "_limiter", limiter)
        return limiter

    return _install


async def _wait_in_flight(limiter: GenerationLimiter, expected: int) -> None:
    for _ in range(500):
        if limiter.in_flight == expected:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"in flight {limiter.in_flight}, expected {expected}")


def _assert_shed(response) -> None:
    assert response.status_code == 429, response.text
    assert response.headers["retry-after"] == str(SHED_RETRY_AFTER_SECONDS)
    assert response.json()["detail"] == {
        "code": "overloaded",
        "message": "The service is busy, retry in a few seconds",
    }


class _GoneClientBus:
    """A stream bus whose subscriber leaves at once: the client went away."""

    async def subscribe(self, key, ready=None):
        if ready is not None:
            ready.set()
        return
        yield  # pragma: no cover

    async def publish(self, key, data):
        return None

    async def close(self, key):
        return None


# ─── Limiter ──────────────────────────────────────────────────────────────────


@pytest.mark.no_db
async def test_limiter_refuses_past_the_cap_and_admits_after_release():
    limiter = GenerationLimiter(2)
    first = await limiter.try_acquire()
    second = await limiter.try_acquire()
    assert first and second
    assert await limiter.try_acquire() is None
    assert limiter.in_flight == 2

    first.release()
    first.release()  # idempotent: a double release must not raise the cap
    assert limiter.in_flight == 1
    third = await limiter.try_acquire()
    assert third is not None
    assert await limiter.try_acquire() is None


@pytest.mark.no_db
@pytest.mark.parametrize("limit", [0, -1])
async def test_limiter_disabled_at_zero_or_below(limit):
    limiter = GenerationLimiter(limit)
    slots = [await limiter.try_acquire() for _ in range(100)]
    assert all(slots)
    assert limiter.in_flight == 100
    for slot in slots:
        slot.release()
    assert limiter.in_flight == 0


@pytest.mark.no_db
@pytest.mark.parametrize("outcome", ["ok", "error", "cancel"])
async def test_slot_handed_to_a_task_is_freed_however_it_ends(outcome):
    limiter = GenerationLimiter(1)
    gate = asyncio.Event()

    async def work():
        await gate.wait()
        if outcome == "error":
            raise RuntimeError("generation failed")

    slot = await limiter.try_acquire()
    task = asyncio.create_task(work())
    slot.release_when_done(task)
    await asyncio.sleep(0)
    assert limiter.in_flight == 1

    if outcome == "cancel":
        task.cancel()
    else:
        gate.set()
    with contextlib.suppress(BaseException):
        await task
    await asyncio.sleep(0)  # done callbacks run on the next loop iteration
    assert limiter.in_flight == 0
    assert await limiter.try_acquire() is not None


@pytest.mark.no_db
async def test_shed_logs_route_and_count_without_user_text(caplog):
    limiter = GenerationLimiter(1)
    load_shedding._limiter, previous = limiter, load_shedding._limiter
    try:
        await limiter.try_acquire()
        with pytest.raises(Exception) as exc_info:
            await load_shedding.acquire_generation_slot_or_raise("stream_messages")
        assert exc_info.value.status_code == 429
    finally:
        load_shedding._limiter = previous
    shed = [r for r in caplog.records if r.name == load_shedding.__name__]
    assert len(shed) == 1
    assert shed[0].levelname == "WARNING"
    assert "stream_messages" in shed[0].getMessage()
    assert "1 generations in flight" in shed[0].getMessage()


# ─── Routes ───────────────────────────────────────────────────────────────────


@pytest.fixture
async def conversation_owner():
    user, token = await create_test_user_and_token()
    conversation = Conversation(user_id=user.id, name="load-shedding")
    await conversation.save()
    yield user, token, conversation
    for message in await Message.find_all(filter_dict={"conversation_id": conversation.id}):
        await message.delete()
    await cleanup_models([user, conversation])


@pytest.mark.parametrize(
    "path",
    [
        "messages",
        "stream_messages",
        "generate-agentic",
        "stream-generate-agentic",
        "retry",
    ],
)
async def test_every_generation_route_sheds_at_the_cap_before_any_work(
    async_client, conversation_owner, cap, path
):
    user, token, conversation = conversation_owner
    limiter = cap(1)
    held = await limiter.try_acquire()
    url = (
        f"/conversations/{conversation.id}/messages/000000000000000000000000/retry"
        if path == "retry"
        else f"/conversations/{conversation.id}/{path}"
    )

    response = await async_client.post(
        url, json={"query": "hello"}, headers={"Authorization": f"Bearer {token}"}
    )

    _assert_shed(response)
    assert limiter.in_flight == 1
    assert await Message.find_all(filter_dict={"conversation_id": conversation.id}) == []
    held.release()


async def test_cap_plus_one_concurrent_messages_last_one_shed(
    async_client, conversation_owner, cap, monkeypatch
):
    user, token, conversation = conversation_owner
    limiter = cap(2)
    gate = asyncio.Event()

    async def slow_generate(*args, **kwargs):
        await gate.wait()
        return CLASSIC_RESULT

    monkeypatch.setattr("src.routers.message.generate_answer", slow_generate)
    url = f"/conversations/{conversation.id}/messages"
    headers = {"Authorization": f"Bearer {token}"}

    admitted = [
        asyncio.create_task(async_client.post(url, json={"query": f"q{i}"}, headers=headers))
        for i in range(2)
    ]
    await _wait_in_flight(limiter, 2)

    _assert_shed(await async_client.post(url, json={"query": "q2"}, headers=headers))

    gate.set()
    for response in await asyncio.gather(*admitted):
        assert response.status_code == 200, response.text
    assert limiter.in_flight == 0
    # The slot is back: the next request is admitted.
    response = await async_client.post(url, json={"query": "q3"}, headers=headers)
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("route", ["messages", "generate-agentic", "retry"])
async def test_sync_slot_released_after_an_exception(
    async_client, conversation_owner, cap, monkeypatch, route
):
    user, token, conversation = conversation_owner
    limiter = cap(1)

    async def boom(*args, **kwargs):
        raise RuntimeError("model exploded")

    monkeypatch.setattr("src.routers.message.generate_answer", boom)
    monkeypatch.setattr("src.routers.message.generate_answer_agentic", boom)
    headers = {"Authorization": f"Bearer {token}"}
    if route == "retry":
        from src.schemas.generation_request import GenerationRequest

        request_input = GenerationRequest(query="hello", llm_type="main")
        message = await Message.create(
            conversation_id=conversation.id,
            input="hello",
            output="",
            documents=[],
            use_rag=False,
            request_input=request_input,
            metadata={},
        )
        url = f"/conversations/{conversation.id}/messages/{message.id}/retry"
    else:
        url = f"/conversations/{conversation.id}/{route}"

    response = await async_client.post(url, json={"query": "hello"}, headers=headers)

    assert response.status_code == 500, response.text
    assert limiter.in_flight == 0


async def test_sync_slot_released_when_the_request_is_cancelled(
    async_client, conversation_owner, cap, monkeypatch
):
    user, token, conversation = conversation_owner
    limiter = cap(1)

    async def never_answers(*args, **kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr("src.routers.message.generate_answer", never_answers)
    request = asyncio.create_task(
        async_client.post(
            f"/conversations/{conversation.id}/messages",
            json={"query": "hello"},
            headers={"Authorization": f"Bearer {token}"},
        )
    )
    await _wait_in_flight(limiter, 1)

    request.cancel()
    with contextlib.suppress(BaseException):
        await request
    assert limiter.in_flight == 0


STREAM_ROUTES = [
    ("stream_messages", "src.routers.message.run_generation_to_bus"),
    ("stream-generate-agentic", "src.routers.message.run_agentic_generation_to_bus"),
]


@pytest.mark.parametrize("path,generator", STREAM_ROUTES)
async def test_stream_slot_follows_the_generation_after_the_client_left(
    async_client, conversation_owner, cap, monkeypatch, path, generator
):
    user, token, conversation = conversation_owner
    limiter = cap(1)
    gate = asyncio.Event()

    async def slow_generation(**kwargs):
        await gate.wait()

    monkeypatch.setattr(generator, slow_generation)
    monkeypatch.setattr("src.routers.message.get_stream_bus", lambda: _GoneClientBus())
    url = f"/conversations/{conversation.id}/{path}"
    headers = {"Authorization": f"Bearer {token}"}

    # The response ends at once (the subscriber left), the generation does not.
    first = await async_client.post(url, json={"query": "hello"}, headers=headers)
    assert first.status_code == 200, first.text
    assert limiter.in_flight == 1
    _assert_shed(await async_client.post(url, json={"query": "again"}, headers=headers))

    gate.set()
    await _wait_in_flight(limiter, 0)


@pytest.mark.parametrize("path,generator", STREAM_ROUTES)
async def test_stream_slot_released_when_the_generation_fails(
    async_client, conversation_owner, cap, monkeypatch, path, generator
):
    user, token, conversation = conversation_owner
    limiter = cap(1)

    async def failing_generation(**kwargs):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(generator, failing_generation)
    monkeypatch.setattr("src.routers.message.get_stream_bus", lambda: _GoneClientBus())

    response = await async_client.post(
        f"/conversations/{conversation.id}/{path}",
        json={"query": "hello"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200, response.text
    await _wait_in_flight(limiter, 0)


@pytest.mark.parametrize("path,generator", STREAM_ROUTES)
async def test_stream_slot_released_on_stop(
    async_client, conversation_owner, cap, monkeypatch, path, generator
):
    user, token, conversation = conversation_owner
    limiter = cap(1)

    async def never_ends(**kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(generator, never_ends)
    monkeypatch.setattr("src.routers.message.get_stream_bus", lambda: _GoneClientBus())
    headers = {"Authorization": f"Bearer {token}"}

    response = await async_client.post(
        f"/conversations/{conversation.id}/{path}", json={"query": "hello"}, headers=headers
    )
    assert response.status_code == 200, response.text
    assert limiter.in_flight == 1

    stop = await async_client.post(f"/conversations/{conversation.id}/stop", headers=headers)
    assert stop.json()["status"] == "stopping", stop.text
    await _wait_in_flight(limiter, 0)


@pytest.mark.parametrize("path,_generator", STREAM_ROUTES)
async def test_stream_slot_released_when_the_request_fails_before_generation(
    async_client, conversation_owner, cap, path, _generator
):
    user, token, conversation = conversation_owner
    limiter = cap(1)

    response = await async_client.post(
        f"/conversations/000000000000000000000000/{path}",
        json={"query": "hello"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 404, response.text
    assert limiter.in_flight == 0


@pytest.mark.no_db
def test_default_cap_reads_the_config():
    from src.config import MAX_INFLIGHT_GENERATIONS_PER_WORKER

    assert get_generation_limiter().limit == MAX_INFLIGHT_GENERATIONS_PER_WORKER
