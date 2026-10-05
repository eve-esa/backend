"""A Stop that lands before the first token still ends the turn as stopped.

The stream routes create the message, register it with the cancel manager and
start the generation task; the Stop route cancels that task. On dev the Stop
arrived 100 to 150 ms after the task started, in one of three windows the bus
runners did not cover: before the task's first step, while it waited for the
SSE subscriber, or while it published a chunk. In each the message stayed
``stopped == False`` and the spec polling for it timed out.
"""

import asyncio
import contextvars
import gc
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import REDIS_URL
from src.database.models.conversation import Conversation
from src.database.models.message import Message
from src.schemas.generation_request import GenerationRequest
from src.services.agents.core.runner import run_agentic_generation_to_bus
from src.services.cancel_manager import CancelManager, get_cancel_manager
from src.services.generate_answer import (
    finish_turn_if_cancelled_unstarted,
    run_generation_to_bus,
)
from src.services.mcp.artifact_context import (
    get_artifact_context,
    reset_artifact_context,
    set_artifact_context,
)
from src.services.mcp.retrieval_context import (
    get_retrieval_context,
    reset_retrieval_context,
    set_retrieval_context,
)
from src.services.stream_bus import StreamBus
from tests.utils.cleaner import cleanup_models
from tests.utils.utils import create_test_user_and_token

_RUNNER = "src.services.agents.core.runner"
_CLASSIC = "src.services.generate_answer"


class _PublishBlocksBus(StreamBus):
    """A bus whose first publish never returns: the Redis round trip the Stop lands on."""

    def __init__(self):
        super().__init__()
        self.entered = asyncio.Event()

    async def publish(self, key, data):
        self.entered.set()
        await asyncio.Event().wait()


@pytest.fixture
async def turn():
    user, _token = await create_test_user_and_token()
    conversation = Conversation(user_id=user.id, name="stop-early")
    await conversation.save()
    message = await Message.create(
        conversation_id=conversation.id, input="a long question", output=""
    )
    yield user, conversation, message
    for row in await Message.find_all(filter_dict={"conversation_id": conversation.id}):
        await row.delete()
    await cleanup_models([user, conversation])


def _start(kind, user, conversation, message, ready):
    """What the stream route does after Message.create, minus the response."""
    cm = get_cancel_manager()
    cancel_event = cm.create(message.id)
    cm.link_conversation(conversation.id, message.id)
    common = dict(
        request=GenerationRequest(query="a long question", agent="react"),
        conversation_id=conversation.id,
        message_id=message.id,
        user_id=user.id,
        cancel_event=cancel_event,
    )
    if kind == "agentic":
        coro = run_agentic_generation_to_bus(subscriber_ready=ready, **common)
    else:
        coro = run_generation_to_bus(stream_ready=ready, **common)
    task = asyncio.create_task(coro)
    cm.set_task(message.id, task)
    finish_turn_if_cancelled_unstarted(task, conversation.id, message.id)
    return cm, task


async def _wait_stopped(message_id) -> Message:
    for _ in range(200):
        row = await Message.find_by_id(message_id)
        if row.stopped:
            return row
        await asyncio.sleep(0.01)
    raise AssertionError("the message was never persisted as stopped")


def _patch_bus(bus):
    return (
        patch(f"{_RUNNER}.get_stream_bus", MagicMock(return_value=bus)),
        patch(f"{_CLASSIC}.get_stream_bus", MagicMock(return_value=bus)),
    )


@pytest.mark.parametrize("kind", ["agentic", "classic"])
async def test_stop_before_the_task_runs_its_first_step(turn, kind):
    user, conversation, message = turn
    p1, p2 = _patch_bus(StreamBus())
    with p1, p2:
        cm, task = _start(kind, user, conversation, message, asyncio.Event())
        # Same loop iteration as create_task: the coroutine never starts.
        cm.cancel(message.id)
        with pytest.raises(asyncio.CancelledError):
            await task
        row = await _wait_stopped(message.id)

    assert row.output == ""
    assert await cm.get_message_for_conversation_async(conversation.id) is None


@pytest.mark.parametrize("kind", ["agentic", "classic"])
async def test_stop_while_waiting_for_the_subscriber(turn, kind):
    user, conversation, message = turn
    p1, p2 = _patch_bus(StreamBus())
    with p1, p2:
        cm, task = _start(kind, user, conversation, message, asyncio.Event())
        await asyncio.sleep(0.05)  # started, parked on the ready event
        cm.cancel(message.id)
        await asyncio.wait_for(task, timeout=5)
        row = await Message.find_by_id(message.id)

    assert row.stopped is True
    assert await cm.get_message_for_conversation_async(conversation.id) is None


@pytest.mark.parametrize("kind", ["agentic", "classic"])
async def test_stop_while_publishing_the_first_event(turn, kind):
    """The dev attempt with the ContextVar error: the cancel hit ``bus.publish``,
    the generator was left open, and the loop closed it later from another task."""
    user, conversation, message = turn
    loop = asyncio.get_running_loop()
    loop_errors = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, ctx: loop_errors.append(ctx))
    bus = _PublishBlocksBus()
    rag_decision = (SimpleNamespace(requery="a long question", use_rag=True), {}, False)
    p1, p2 = _patch_bus(bus)
    try:
        with p1, p2, patch(
            f"{_CLASSIC}.should_use_rag", AsyncMock(return_value=rag_decision)
        ):
            cm, task = _start(kind, user, conversation, message, None)
            await asyncio.wait_for(bus.entered.wait(), timeout=5)
            cm.cancel(message.id)
            await asyncio.wait_for(task, timeout=5)
            row = await Message.find_by_id(message.id)
        gc.collect()
        await asyncio.sleep(0.05)
    finally:
        loop.set_exception_handler(previous_handler)

    assert row.stopped is True
    assert loop_errors == []


@pytest.mark.no_db
def test_context_resets_tolerate_a_token_from_another_context():
    _, retrieval_token = set_retrieval_context()
    _, artifact_token = set_artifact_context(user_id="u1")

    contextvars.copy_context().run(reset_retrieval_context, retrieval_token)
    contextvars.copy_context().run(reset_artifact_context, artifact_token)

    reset_retrieval_context(retrieval_token)
    reset_artifact_context(artifact_token)
    assert get_retrieval_context() is None
    assert get_artifact_context() is None


@pytest.mark.no_db
@pytest.mark.skipif(not REDIS_URL, reason="cross-worker Stop needs Redis")
async def test_stop_published_before_the_owner_subscribed_is_not_lost():
    """Stop on worker B before worker A's cancel channel is subscribed."""
    message_id = uuid.uuid4().hex
    owner, other = CancelManager(), CancelManager()
    other.cancel(message_id)
    await asyncio.sleep(0.2)  # the publish has gone out with nobody listening

    event = owner.create(message_id)
    try:
        await asyncio.wait_for(event.wait(), timeout=3)
    finally:
        owner.clear(message_id)
        if other._redis is not None:
            await other._redis.delete(f"cancelled:{message_id}")
            await other._redis.aclose()
