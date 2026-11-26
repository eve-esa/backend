"""GET /conversations/{id} reports the turn still running as ``active_message_id``.

A reload during a turn closes the stream while the backend keeps generating.
The reloaded page reads the conversation and needs to know that the empty
assistant row is still being written, not failed. The only record of a running
turn is the cancel manager's conversation to message mapping (written at stream
start, cleared when the turn ends), so the route reads it, bounded, and never
fails because of it.
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from bson import ObjectId

from src.database.models.message import Message
from src.routers import conversation as conversation_router
from src.services.cancel_manager import CancelManager
from tests.utils.cleaner import cleanup_models
from tests.utils.utils import create_test_user_and_token


def _message_id(age_s: float = 0.0) -> str:
    """An ObjectId string generated ``age_s`` seconds ago."""
    born = datetime.now(timezone.utc) - timedelta(seconds=age_s)
    return str(ObjectId.from_datetime(born))


@pytest.fixture(autouse=True)
def _no_skip_window():
    """Each test starts and ends outside the lookup skip window."""
    conversation_router._lookup_skip_until = 0.0
    yield
    conversation_router._lookup_skip_until = 0.0


def _local_manager() -> CancelManager:
    """The in-process cancel manager, as with REDIS_URL unset."""
    mgr = CancelManager()
    mgr._is_redis_enabled = lambda: False  # type: ignore[method-assign]
    return mgr


class _RaisingRedis:
    async def hget(self, *_args, **_kwargs):
        raise ConnectionError("Error 111 connecting to valkey:6379. Connection refused.")


class _HangingRedis:
    def __init__(self):
        self.calls = 0

    async def hget(self, *_args, **_kwargs):
        self.calls += 1
        await asyncio.sleep(30)


def _redis_manager(fake) -> CancelManager:
    """A Valkey-backed manager whose client is ``fake``: the mapping lives on
    another worker, so this one has nothing locally."""
    mgr = CancelManager()
    mgr._is_redis_enabled = lambda: True  # type: ignore[method-assign]
    mgr._redis = fake
    return mgr


async def _create_conversation(async_client, token: str) -> str:
    resp = await async_client.post(
        "/conversations",
        json={"name": "Active turn"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    return resp.json()["id"]


async def _get(async_client, token: str, conversation_id: str):
    return await async_client.get(
        f"/conversations/{conversation_id}",
        headers={"Authorization": f"Bearer {token}"},
    )


@pytest.mark.asyncio
async def test_a_running_turn_is_reported(async_client):
    user, token = await create_test_user_and_token()
    try:
        conversation_id = await _create_conversation(async_client, token)
        mgr = _local_manager()
        message_id = _message_id()
        mgr.link_conversation(conversation_id, message_id)
        with patch.object(conversation_router, "get_cancel_manager", return_value=mgr):
            resp = await _get(async_client, token, conversation_id)
        assert resp.status_code == 200
        assert resp.json()["active_message_id"] == message_id
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_no_running_turn_reports_null(async_client):
    user, token = await create_test_user_and_token()
    try:
        conversation_id = await _create_conversation(async_client, token)
        mgr = _local_manager()
        # A turn that ran and ended clears its mapping.
        done = _message_id()
        mgr.link_conversation(conversation_id, done)
        mgr.clear_mapping_for(conversation_id, done)
        with patch.object(conversation_router, "get_cancel_manager", return_value=mgr):
            resp = await _get(async_client, token, conversation_id)
        assert resp.status_code == 200
        assert resp.json()["active_message_id"] is None
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_another_conversation_running_is_not_reported(async_client):
    user, token = await create_test_user_and_token()
    try:
        conversation_id = await _create_conversation(async_client, token)
        mgr = _local_manager()
        mgr.link_conversation("some-other-conversation", _message_id())
        with patch.object(conversation_router, "get_cancel_manager", return_value=mgr):
            resp = await _get(async_client, token, conversation_id)
        assert resp.status_code == 200
        assert resp.json()["active_message_id"] is None
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_a_store_error_with_no_messages_answers_null(async_client):
    """A Valkey error reaches the route as a failed lookup: with no stored
    row to fall back on, the answer is null."""
    user, token = await create_test_user_and_token()
    try:
        conversation_id = await _create_conversation(async_client, token)
        mgr = _redis_manager(_RaisingRedis())
        with patch.object(conversation_router, "get_cancel_manager", return_value=mgr):
            resp = await _get(async_client, token, conversation_id)
        assert resp.status_code == 200
        assert resp.json()["active_message_id"] is None
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_a_lookup_that_raises_answers_null_with_one_warning(async_client, caplog):
    user, token = await create_test_user_and_token()
    try:
        conversation_id = await _create_conversation(async_client, token)
        mgr = _local_manager()
        secret = "redis://:hunter2@valkey:6379"
        caplog.clear()
        with caplog.at_level(logging.WARNING), patch.object(
            conversation_router, "get_cancel_manager", return_value=mgr
        ), patch.object(
            mgr,
            "get_message_for_conversation_async",
            side_effect=RuntimeError(f"cannot reach {secret}"),
        ):
            resp = await _get(async_client, token, conversation_id)
        assert resp.status_code == 200
        assert resp.json()["active_message_id"] is None
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1, [r.getMessage() for r in warnings]
        assert warnings[0].name == conversation_router.logger.name
        line = warnings[0].getMessage()
        assert "conversation.active_message.lookup_failed" in line
        assert conversation_id in line
        assert "RuntimeError" in line
        assert secret not in line
        assert "Active turn" not in line
        assert user.id not in line
        assert user.email not in line
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_valkey_hang_is_bounded(async_client, caplog):
    user, token = await create_test_user_and_token()
    try:
        conversation_id = await _create_conversation(async_client, token)
        mgr = _redis_manager(_HangingRedis())
        caplog.clear()
        with caplog.at_level(logging.WARNING), patch.object(
            conversation_router, "get_cancel_manager", return_value=mgr
        ):
            started = time.monotonic()
            resp = await _get(async_client, token, conversation_id)
            elapsed = time.monotonic() - started
        assert resp.status_code == 200
        assert resp.json()["active_message_id"] is None
        assert elapsed < 3.0, elapsed
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1, [r.getMessage() for r in warnings]
        line = warnings[0].getMessage()
        assert conversation_id in line
        assert user.id not in line
        assert "Active turn" not in line
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_a_non_owner_still_gets_403_and_no_lookup(async_client):
    owner, owner_token = await create_test_user_and_token()
    other, other_token = await create_test_user_and_token()
    try:
        conversation_id = await _create_conversation(async_client, owner_token)
        mgr = _local_manager()
        message_id = _message_id()
        mgr.link_conversation(conversation_id, message_id)
        with patch.object(
            conversation_router, "get_cancel_manager", return_value=mgr
        ), patch.object(
            mgr, "get_message_for_conversation_async", wraps=mgr.get_message_for_conversation_async
        ) as lookup:
            resp = await _get(async_client, other_token, conversation_id)
        assert resp.status_code == 403
        assert message_id not in resp.text
        lookup.assert_not_called()
    finally:
        await cleanup_models([owner, other])


@pytest.mark.asyncio
async def test_a_missing_conversation_is_still_404(async_client):
    user, token = await create_test_user_and_token()
    try:
        resp = await _get(async_client, token, "64b7f0c2a1b2c3d4e5f60718")
        assert resp.status_code == 404
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_an_invalid_conversation_id_is_404_without_a_lookup(async_client):
    user, token = await create_test_user_and_token()
    try:
        mgr = _local_manager()
        with patch.object(
            conversation_router, "get_cancel_manager", return_value=mgr
        ), patch.object(
            mgr, "get_message_for_conversation_async", wraps=mgr.get_message_for_conversation_async
        ) as lookup:
            resp = await _get(async_client, token, "not-an-id")
        assert resp.status_code == 404
        lookup.assert_not_called()
    finally:
        await cleanup_models([user])


async def _add_message(conversation_id: str, output: str, **fields) -> None:
    await Message.create(
        conversation_id=conversation_id, input="question", output=output, **fields
    )


def _primary_flags(find_all) -> list:
    return [c.kwargs.get("primary", False) for c in find_all.call_args_list]


@pytest.mark.asyncio
async def test_a_normal_load_does_no_primary_read(async_client):
    user, token = await create_test_user_and_token()
    try:
        conversation_id = await _create_conversation(async_client, token)
        await _add_message(conversation_id, "a saved answer")
        with patch.object(
            conversation_router, "get_cancel_manager", return_value=_local_manager()
        ), patch.object(Message, "find_all", wraps=Message.find_all) as find_all:
            resp = await _get(async_client, token, conversation_id)
        assert resp.status_code == 200
        assert resp.json()["active_message_id"] is None
        assert _primary_flags(find_all) == [False]
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fields",
    [{"stopped": True}, {"metadata": {"error": {"code": "upstream_error"}}}],
    ids=["stopped", "error"],
)
async def test_an_ended_empty_row_does_no_primary_read(async_client, fields):
    user, token = await create_test_user_and_token()
    try:
        conversation_id = await _create_conversation(async_client, token)
        await _add_message(conversation_id, "", **fields)
        with patch.object(
            conversation_router, "get_cancel_manager", return_value=_local_manager()
        ), patch.object(Message, "find_all", wraps=Message.find_all) as find_all:
            resp = await _get(async_client, token, conversation_id)
        assert resp.status_code == 200
        assert _primary_flags(find_all) == [False]
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_an_empty_last_row_with_no_turn_is_read_again_on_the_primary(async_client):
    """A turn that just ended saved its answer on the primary: a secondary can
    still hold the empty row, and the client stops polling on this answer.
    The primary result is the one returned."""
    user, token = await create_test_user_and_token()
    try:
        conversation_id = await _create_conversation(async_client, token)
        await _add_message(conversation_id, "")
        stale = await Message.find_all(filter_dict={"conversation_id": conversation_id})
        saved = [m.model_copy(update={"output": "the saved answer"}) for m in stale]

        async def fake_find_all(*_args, primary=False, **_kwargs):
            return saved if primary else stale

        with patch.object(
            conversation_router, "get_cancel_manager", return_value=_local_manager()
        ), patch.object(Message, "find_all", side_effect=fake_find_all) as find_all:
            resp = await _get(async_client, token, conversation_id)
        assert resp.status_code == 200
        assert resp.json()["active_message_id"] is None
        assert _primary_flags(find_all) == [False, True]
        assert resp.json()["messages"][-1]["output"] == "the saved answer"
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_a_running_turn_does_no_primary_read(async_client):
    user, token = await create_test_user_and_token()
    try:
        conversation_id = await _create_conversation(async_client, token)
        await _add_message(conversation_id, "")
        mgr = _local_manager()
        mgr.link_conversation(conversation_id, _message_id())
        with patch.object(
            conversation_router, "get_cancel_manager", return_value=mgr
        ), patch.object(Message, "find_all", wraps=Message.find_all) as find_all:
            resp = await _get(async_client, token, conversation_id)
        assert resp.status_code == 200
        assert resp.json()["active_message_id"] is not None
        assert _primary_flags(find_all) == [False]
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_after_a_timeout_the_store_is_skipped_for_a_window(async_client, caplog):
    user, token = await create_test_user_and_token()
    try:
        conversation_id = await _create_conversation(async_client, token)
        hanging = _HangingRedis()
        mgr = _redis_manager(hanging)
        now = [1000.0]
        with patch.object(
            conversation_router, "get_cancel_manager", return_value=mgr
        ), patch.object(conversation_router, "_clock", lambda: now[0]):
            first = await _get(async_client, token, conversation_id)
            assert first.json()["active_message_id"] is None
            assert hanging.calls == 1

            # Inside the window: null without asking the store, no warning.
            now[0] += conversation_router.ACTIVE_MESSAGE_SKIP_S - 1
            caplog.clear()
            with caplog.at_level(logging.WARNING):
                second = await _get(async_client, token, conversation_id)
            assert second.status_code == 200
            assert second.json()["active_message_id"] is None
            assert hanging.calls == 1
            assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

            # Past the window the store is asked again.
            now[0] += 2
            third = await _get(async_client, token, conversation_id)
            assert third.status_code == 200
            assert hanging.calls == 2
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_after_a_failed_lookup_the_store_is_skipped_for_a_window(async_client):
    user, token = await create_test_user_and_token()
    try:
        conversation_id = await _create_conversation(async_client, token)
        mgr = _local_manager()
        now = [1000.0]
        with patch.object(
            conversation_router, "get_cancel_manager", return_value=mgr
        ), patch.object(conversation_router, "_clock", lambda: now[0]), patch.object(
            mgr, "get_message_for_conversation_async", side_effect=RuntimeError("down")
        ) as lookup:
            await _get(async_client, token, conversation_id)
            assert lookup.call_count == 1
            now[0] += conversation_router.ACTIVE_MESSAGE_SKIP_S - 1
            await _get(async_client, token, conversation_id)
            assert lookup.call_count == 1
            now[0] += 2
            await _get(async_client, token, conversation_id)
            assert lookup.call_count == 2
    finally:
        await cleanup_models([user])


@pytest.mark.parametrize(
    "agentic, classic, expected_longest",
    [
        (480, 300, 480),  # agentic longer
        (480, 900, 900),  # classic longer
        (480, 0, 480),  # classic unbounded: the agentic deadline bounds it
        (480, -1, 480),
    ],
)
def test_the_age_cap_follows_the_longer_turn_deadline(agentic, classic, expected_longest):
    assert conversation_router._max_turn_age_s(agentic, classic) == (
        expected_longest + conversation_router.ACTIVE_MESSAGE_AGE_MARGIN_S
    )


@pytest.mark.asyncio
async def test_a_classic_turn_past_the_agentic_deadline_is_still_reported(async_client):
    """With CLASSIC_GENERATION_TIMEOUT above AGENTIC_TURN_DEADLINE a classic
    turn can run longer than the agentic deadline and is still live."""
    user, token = await create_test_user_and_token()
    try:
        conversation_id = await _create_conversation(async_client, token)
        agentic, classic = 480, 900
        running = _message_id(age_s=agentic + 200)
        mgr = _redis_manager(_NoWriteRedis(running))
        with patch.object(conversation_router, "get_cancel_manager", return_value=mgr), patch.object(
            conversation_router,
            "ACTIVE_MESSAGE_MAX_AGE_S",
            conversation_router._max_turn_age_s(agentic, classic),
        ):
            resp = await _get(async_client, token, conversation_id)
        assert resp.status_code == 200
        assert resp.json()["active_message_id"] == running
    finally:
        await cleanup_models([user])


class _NoWriteRedis:
    """Answers the mapping read, fails the test on any write."""

    def __init__(self, value: str):
        self.value = value

    async def hget(self, *_args, **_kwargs):
        return self.value.encode()

    def __getattr__(self, name):
        raise AssertionError(f"unexpected Valkey call {name}")


@pytest.mark.asyncio
async def test_a_mapping_past_the_turn_deadline_is_null(async_client, caplog):
    """A worker that died mid-turn leaves its mapping behind: a message older
    than the turn deadline cannot still be generating."""
    user, token = await create_test_user_and_token()
    try:
        conversation_id = await _create_conversation(async_client, token)
        stale = _message_id(age_s=600)
        mgr = _redis_manager(_NoWriteRedis(stale))
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger=conversation_router.logger.name), patch.object(
            conversation_router, "get_cancel_manager", return_value=mgr
        ):
            resp = await _get(async_client, token, conversation_id)
        assert resp.status_code == 200
        assert resp.json()["active_message_id"] is None
        mine = [r for r in caplog.records if r.name == conversation_router.logger.name]
        assert [r.levelno for r in mine] == [logging.DEBUG], [r.getMessage() for r in mine]
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_a_mapping_inside_the_turn_deadline_is_reported(async_client):
    user, token = await create_test_user_and_token()
    try:
        conversation_id = await _create_conversation(async_client, token)
        young = _message_id(age_s=conversation_router.AGENTIC_TURN_DEADLINE - 5)
        mgr = _redis_manager(_NoWriteRedis(young))
        with patch.object(conversation_router, "get_cancel_manager", return_value=mgr):
            resp = await _get(async_client, token, conversation_id)
        assert resp.status_code == 200
        assert resp.json()["active_message_id"] == young
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_a_mapping_that_is_not_an_object_id_is_null(async_client, caplog):
    user, token = await create_test_user_and_token()
    try:
        conversation_id = await _create_conversation(async_client, token)
        mgr = _redis_manager(_NoWriteRedis("not-an-object-id"))
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger=conversation_router.logger.name), patch.object(
            conversation_router, "get_cancel_manager", return_value=mgr
        ):
            resp = await _get(async_client, token, conversation_id)
        assert resp.status_code == 200
        assert resp.json()["active_message_id"] is None
        mine = [r for r in caplog.records if r.name == conversation_router.logger.name]
        assert [r.levelno for r in mine] == [logging.DEBUG], [r.getMessage() for r in mine]
    finally:
        await cleanup_models([user])


def _real_redis_url() -> str:
    url = os.getenv("REDIS_URL", "")
    if not url:
        pytest.skip("REDIS_URL not set")
    import redis

    client = redis.Redis.from_url(url, socket_connect_timeout=0.5, socket_timeout=0.5)
    try:
        client.ping()
    except Exception:
        pytest.skip("no Redis answering at REDIS_URL")
    finally:
        client.close()
    return url


@pytest.mark.asyncio
async def test_a_turn_owned_by_another_worker_is_reported_through_valkey(async_client):
    _real_redis_url()
    user, token = await create_test_user_and_token()
    owner = CancelManager()
    reader = CancelManager()
    conversation_id = None
    try:
        if not owner._is_redis_enabled():
            pytest.skip("redis client not available")
        conversation_id = await _create_conversation(async_client, token)
        message_id = _message_id()
        # The owning worker's mapping write, awaited here instead of fired.
        await owner._hset_mapping(conversation_id, message_id)
        with patch.object(conversation_router, "get_cancel_manager", return_value=reader):
            resp = await _get(async_client, token, conversation_id)
        assert resp.status_code == 200
        assert resp.json()["active_message_id"] == message_id
    finally:
        if conversation_id:
            await owner._hdel_mapping(conversation_id)
        for mgr in (owner, reader):
            if mgr._redis is not None:
                await mgr._redis.aclose()
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_a_timed_out_read_leaves_the_shared_client_working(async_client):
    """A store that stalls past the lookup deadline: the route answers null,
    and the cancel manager's shared client still reads the mapping and
    publishes a Stop afterwards (no desynced reply, no leaked connection)."""
    url = _real_redis_url()
    import redis.asyncio as aioredis

    user, token = await create_test_user_and_token()
    reader = CancelManager()
    admin = aioredis.Redis.from_url(url)
    conversation_id = None
    try:
        if not reader._is_redis_enabled():
            pytest.skip("redis client not available")
        conversation_id = await _create_conversation(async_client, token)
        message_id = _message_id()
        await reader._hset_mapping(conversation_id, message_id)

        pause_ms = 1500
        await admin.execute_command("CLIENT", "PAUSE", pause_ms, "ALL")
        paused_at = time.monotonic()
        with patch.object(conversation_router, "get_cancel_manager", return_value=reader):
            resp = await _get(async_client, token, conversation_id)
        assert resp.status_code == 200
        assert resp.json()["active_message_id"] is None
        assert time.monotonic() - paused_at < pause_ms / 1000

        await asyncio.sleep(max(0.0, pause_ms / 1000 - (time.monotonic() - paused_at)) + 0.3)

        assert await reader.get_message_for_conversation_async(conversation_id) == message_id
        assert isinstance(await reader._redis.publish(f"cancel:{message_id}", "1"), int)
        assert await reader.get_message_for_conversation_async(conversation_id) == message_id
        assert not reader._redis.connection_pool._in_use_connections
    finally:
        await admin.execute_command("CLIENT", "UNPAUSE")
        if conversation_id:
            await reader._hdel_mapping(conversation_id)
        if reader._redis is not None:
            await reader._redis.aclose()
        await admin.aclose()
        await cleanup_models([user])


# When the lookup cannot answer (timeout, raised error, skip window), the
# route does not report "no turn": a slow store during a running turn would
# show the empty row as failed. It falls back to the stored rows instead.


async def _add_message_aged(conversation_id: str, output: str, age_s: float = 0.0, **fields) -> str:
    """A message row whose id was generated ``age_s`` seconds ago."""
    message = Message(
        conversation_id=conversation_id, input="question", output=output, **fields
    )
    born = datetime.now(timezone.utc) - timedelta(seconds=age_s)
    # from_datetime zeroes the random part: two rows born in the same second
    # would collide, so keep the timestamp and randomise the rest.
    message_id = str(ObjectId(int(born.timestamp()).to_bytes(4, "big") + os.urandom(8)))
    await Message.get_collection().insert_one(
        {**message.to_dict(), "_id": ObjectId(message_id)}
    )
    return message_id


def _timeout_manager():
    return _redis_manager(_HangingRedis())


def _raising_lookup_manager():
    mgr = _local_manager()
    mgr.get_message_for_conversation_async = _raise_down  # type: ignore[method-assign]
    return mgr


async def _raise_down(*_args, **_kwargs):
    raise RuntimeError("down")


def _swallowing_store_manager():
    return _redis_manager(_RaisingRedis())


_UNAVAILABLE_CASES = [
    pytest.param(_timeout_manager, False, id="timeout"),
    pytest.param(_raising_lookup_manager, False, id="raised"),
    pytest.param(_swallowing_store_manager, False, id="store_error"),
    pytest.param(_local_manager, True, id="skip_window"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("make_manager, skip_window", _UNAVAILABLE_CASES)
async def test_unavailable_lookup_reports_a_young_unsaved_last_row(
    async_client, make_manager, skip_window
):
    user, token = await create_test_user_and_token()
    try:
        conversation_id = await _create_conversation(async_client, token)
        await _add_message_aged(conversation_id, "an earlier answer", age_s=30)
        running = await _add_message_aged(conversation_id, "", age_s=5)
        if skip_window:
            conversation_router._lookup_skip_until = float("inf")
        with patch.object(
            conversation_router, "get_cancel_manager", return_value=make_manager()
        ), patch.object(Message, "find_all", wraps=Message.find_all) as find_all:
            resp = await _get(async_client, token, conversation_id)
        assert resp.status_code == 200
        assert resp.json()["active_message_id"] == running
        # A fallback id means the turn may still run: no primary re-read.
        assert _primary_flags(find_all) == [False]
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
@pytest.mark.parametrize("make_manager, skip_window", _UNAVAILABLE_CASES)
async def test_unavailable_lookup_with_a_saved_last_row_is_null(
    async_client, make_manager, skip_window
):
    user, token = await create_test_user_and_token()
    try:
        conversation_id = await _create_conversation(async_client, token)
        await _add_message_aged(conversation_id, "the saved answer", age_s=5)
        if skip_window:
            conversation_router._lookup_skip_until = float("inf")
        with patch.object(
            conversation_router, "get_cancel_manager", return_value=make_manager()
        ), patch.object(Message, "find_all", wraps=Message.find_all) as find_all:
            resp = await _get(async_client, token, conversation_id)
        assert resp.status_code == 200
        assert resp.json()["active_message_id"] is None
        assert _primary_flags(find_all) == [False]
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
@pytest.mark.parametrize("make_manager, skip_window", _UNAVAILABLE_CASES)
@pytest.mark.parametrize(
    "fields",
    [{"stopped": True}, {"metadata": {"error": {"code": "upstream_error"}}}],
    ids=["stopped", "error"],
)
async def test_unavailable_lookup_with_an_ended_empty_row_is_null(
    async_client, make_manager, skip_window, fields
):
    user, token = await create_test_user_and_token()
    try:
        conversation_id = await _create_conversation(async_client, token)
        await _add_message_aged(conversation_id, "", age_s=5, **fields)
        if skip_window:
            conversation_router._lookup_skip_until = float("inf")
        with patch.object(
            conversation_router, "get_cancel_manager", return_value=make_manager()
        ):
            resp = await _get(async_client, token, conversation_id)
        assert resp.status_code == 200
        assert resp.json()["active_message_id"] is None
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
@pytest.mark.parametrize("make_manager, skip_window", _UNAVAILABLE_CASES)
async def test_unavailable_lookup_with_an_unsaved_row_past_the_max_age_is_null(
    async_client, make_manager, skip_window
):
    user, token = await create_test_user_and_token()
    try:
        conversation_id = await _create_conversation(async_client, token)
        await _add_message_aged(
            conversation_id, "", age_s=conversation_router.ACTIVE_MESSAGE_MAX_AGE_S + 60
        )
        if skip_window:
            conversation_router._lookup_skip_until = float("inf")
        with patch.object(
            conversation_router, "get_cancel_manager", return_value=make_manager()
        ):
            resp = await _get(async_client, token, conversation_id)
        assert resp.status_code == 200
        assert resp.json()["active_message_id"] is None
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_unavailable_lookup_with_no_messages_is_null(async_client):
    user, token = await create_test_user_and_token()
    try:
        conversation_id = await _create_conversation(async_client, token)
        with patch.object(
            conversation_router, "get_cancel_manager", return_value=_timeout_manager()
        ):
            resp = await _get(async_client, token, conversation_id)
        assert resp.status_code == 200
        assert resp.json()["active_message_id"] is None
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_a_lookup_that_answers_no_turn_still_wins_over_an_unsaved_row(async_client):
    """When the store answers, its null is the truth: the empty row is read
    again on the primary and no fallback id is reported."""
    user, token = await create_test_user_and_token()
    try:
        conversation_id = await _create_conversation(async_client, token)
        await _add_message_aged(conversation_id, "", age_s=5)
        with patch.object(
            conversation_router, "get_cancel_manager", return_value=_local_manager()
        ), patch.object(Message, "find_all", wraps=Message.find_all) as find_all:
            resp = await _get(async_client, token, conversation_id)
        assert resp.status_code == 200
        assert resp.json()["active_message_id"] is None
        assert _primary_flags(find_all) == [False, True]
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_a_store_error_opens_the_skip_window(async_client):
    user, token = await create_test_user_and_token()
    try:
        conversation_id = await _create_conversation(async_client, token)
        now = [1000.0]
        with patch.object(
            conversation_router, "get_cancel_manager", return_value=_swallowing_store_manager()
        ), patch.object(conversation_router, "_clock", lambda: now[0]):
            await _get(async_client, token, conversation_id)
        assert conversation_router._lookup_skip_until == 1000.0 + conversation_router.ACTIVE_MESSAGE_SKIP_S
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_the_cancel_manager_still_swallows_store_errors_by_default():
    """Other callers (Stop, retry preempt) keep the old contract: None on error."""
    mgr = _swallowing_store_manager()
    assert await mgr.get_message_for_conversation_async("c1") is None
    with pytest.raises(ConnectionError):
        await mgr.get_message_for_conversation_async("c1", raise_errors=True)
