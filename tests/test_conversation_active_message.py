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
import uuid
from unittest.mock import patch

import pytest

from src.routers import conversation as conversation_router
from src.services.cancel_manager import CancelManager
from tests.utils.cleaner import cleanup_models
from tests.utils.utils import create_test_user_and_token


def _local_manager() -> CancelManager:
    """The in-process cancel manager, as with REDIS_URL unset."""
    mgr = CancelManager()
    mgr._is_redis_enabled = lambda: False  # type: ignore[method-assign]
    return mgr


class _RaisingRedis:
    async def hget(self, *_args, **_kwargs):
        raise ConnectionError("Error 111 connecting to valkey:6379. Connection refused.")


class _HangingRedis:
    async def hget(self, *_args, **_kwargs):
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
        mgr.link_conversation(conversation_id, "msg-running")
        with patch.object(conversation_router, "get_cancel_manager", return_value=mgr):
            resp = await _get(async_client, token, conversation_id)
        assert resp.status_code == 200
        assert resp.json()["active_message_id"] == "msg-running"
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_no_running_turn_reports_null(async_client):
    user, token = await create_test_user_and_token()
    try:
        conversation_id = await _create_conversation(async_client, token)
        mgr = _local_manager()
        # A turn that ran and ended clears its mapping.
        mgr.link_conversation(conversation_id, "msg-done")
        mgr.clear_mapping_for(conversation_id, "msg-done")
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
        mgr.link_conversation("some-other-conversation", "msg-elsewhere")
        with patch.object(conversation_router, "get_cancel_manager", return_value=mgr):
            resp = await _get(async_client, token, conversation_id)
        assert resp.status_code == 200
        assert resp.json()["active_message_id"] is None
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_valkey_error_answers_null_with_one_warning(async_client, caplog):
    user, token = await create_test_user_and_token()
    try:
        conversation_id = await _create_conversation(async_client, token)
        mgr = _redis_manager(_RaisingRedis())
        caplog.clear()
        with caplog.at_level(logging.WARNING), patch.object(
            conversation_router, "get_cancel_manager", return_value=mgr
        ):
            resp = await _get(async_client, token, conversation_id)
        assert resp.status_code == 200
        assert resp.json()["active_message_id"] is None
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1, [r.getMessage() for r in warnings]
        line = warnings[0].getMessage()
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
        assert (
            elapsed
            < conversation_router.ACTIVE_MESSAGE_LOOKUP_TIMEOUT_S + 1.0
        ), elapsed
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1, [r.getMessage() for r in warnings]
        line = warnings[0].getMessage()
        assert conversation_id in line
        assert user.id not in line
        assert "Active turn" not in line
    finally:
        await cleanup_models([user])


def test_the_lookup_deadline_is_short():
    assert 0 < conversation_router.ACTIVE_MESSAGE_LOOKUP_TIMEOUT_S <= 0.5


@pytest.mark.asyncio
async def test_a_non_owner_still_gets_403_and_no_lookup(async_client):
    owner, owner_token = await create_test_user_and_token()
    other, other_token = await create_test_user_and_token()
    try:
        conversation_id = await _create_conversation(async_client, owner_token)
        mgr = _local_manager()
        mgr.link_conversation(conversation_id, "msg-running")
        with patch.object(
            conversation_router, "get_cancel_manager", return_value=mgr
        ), patch.object(
            mgr, "get_message_for_conversation_async", wraps=mgr.get_message_for_conversation_async
        ) as lookup:
            resp = await _get(async_client, other_token, conversation_id)
        assert resp.status_code == 403
        assert "msg-running" not in resp.text
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
        message_id = f"msg-{uuid.uuid4().hex}"
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
