import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.routers import message as message_router
from src.services.cancel_manager import CancelManager


@pytest.fixture
def cancel_mgr():
    mgr = CancelManager()
    with patch.object(mgr, "_is_redis_enabled", return_value=False):
        yield mgr


def _register_turn(mgr, conversation_id, message_id, body):
    """Register a running turn the way the stream endpoints do."""
    event = mgr.create(message_id)
    mgr.link_conversation(conversation_id, message_id)

    async def _turn():
        try:
            await body(event)
        finally:
            mgr.clear_mapping_for(conversation_id, message_id)
            mgr.clear(message_id)

    task = asyncio.create_task(_turn())
    mgr.set_task(message_id, task)
    return task


@pytest.mark.asyncio
async def test_retry_cancels_the_running_turn_and_waits_for_it(cancel_mgr):
    unwound = asyncio.Event()

    async def _body(_event):
        try:
            await asyncio.sleep(60)
        finally:
            unwound.set()

    task = _register_turn(cancel_mgr, "c1", "m1", _body)
    await asyncio.sleep(0)
    bus = AsyncMock()

    with patch.object(message_router, "get_cancel_manager", return_value=cancel_mgr), patch.object(
        message_router, "get_stream_bus", return_value=bus
    ):
        await message_router._preempt_active_generation("c1")

    assert task.cancelled()
    assert unwound.is_set()
    assert await cancel_mgr.get_message_for_conversation_async("c1") is None
    bus.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_retry_without_a_running_turn_does_nothing(cancel_mgr):
    bus = AsyncMock()

    with patch.object(message_router, "get_cancel_manager", return_value=cancel_mgr), patch.object(
        message_router, "get_stream_bus", return_value=bus
    ):
        await message_router._preempt_active_generation("c1")

    bus.stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_turn_owned_by_another_worker_is_awaited_through_the_mapping(
    cancel_mgr,
):
    cancel_mgr.link_conversation("c1", "m1")

    async def _other_worker_finishes():
        await asyncio.sleep(0.3)
        cancel_mgr.clear_mapping_for("c1", "m1")

    finisher = asyncio.create_task(_other_worker_finishes())

    assert await cancel_mgr.wait_until_finished("c1", "m1", timeout=5)
    await finisher


@pytest.mark.asyncio
async def test_wait_reports_a_turn_that_outlives_the_timeout(cancel_mgr):
    cancel_mgr.link_conversation("c1", "m1")

    assert not await cancel_mgr.wait_until_finished("c1", "m1", timeout=0.3)
