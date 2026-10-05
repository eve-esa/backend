"""A Stop that lands before the first token still ends the turn as stopped.

The stream routes create the message, register it with the cancel manager and
start the generation task; the Stop route cancels that task. On dev the Stop
arrived 100 to 150 ms after the task started, in one of three windows the bus
runners did not cover: before the task's first step, while it waited for the
SSE subscriber, or while it published a chunk. In each the message stayed
``stopped == False`` and the spec polling for it timed out.
"""

import asyncio
import contextlib
import contextvars
import gc
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from src.config import REDIS_URL
from src.database.models.conversation import Conversation
from src.database.models.message import Message
from src.schemas.generation_request import GenerationRequest
from src.services.agents.core.runner import (
    generate_answer_agentic_stream_helper,
    run_agentic_generation_to_bus,
)
from src.services.cancel_manager import CancelManager, get_cancel_manager
from src.services.generate_answer import (
    finish_turn_if_cancelled_unstarted,
    generate_answer_stream_generator_helper,
    persist_message_state,
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
from src.services.stream_bus import RedisStreamBus, StreamBus
from tests.test_agentic_fallback import _FakeStreamGraph, _patched_runner
from tests.test_endpoint_failover import _FakeGraph, _patch_pipeline
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


def _start(kind, user, conversation, message, ready, deadline=None):
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
        deadline_seconds=deadline,
    )
    if kind == "agentic":
        coro = run_agentic_generation_to_bus(subscriber_ready=ready, **common)
    else:
        coro = run_generation_to_bus(stream_ready=ready, **common)
    task = asyncio.create_task(coro)
    cm.set_task(message.id, task)
    finish_turn_if_cancelled_unstarted(task, conversation.id, message.id, cancel_event)
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


def _lagging_secondary(snapshot):
    """Reads off the primary see ``snapshot``: None (the row has not replicated
    yet) or the row as it was at Message.create."""
    return patch.object(Message, "find_by_id", AsyncMock(return_value=snapshot))


@pytest.mark.parametrize("kind", ["agentic", "classic"])
async def test_stop_is_kept_when_the_secondary_has_not_the_row_yet(turn, kind):
    user, conversation, message = turn
    p1, p2 = _patch_bus(StreamBus())
    with p1, p2, _lagging_secondary(None):
        cm, task = _start(kind, user, conversation, message, asyncio.Event())
        await asyncio.sleep(0.05)
        cm.cancel(message.id)
        await asyncio.wait_for(task, timeout=5)

    row = await Message.find_by_id_on_primary(message.id)
    assert row.stopped is True


@pytest.mark.parametrize(
    "kind,stream",
    [
        ("agentic", f"{_RUNNER}.generate_answer_agentic_json_stream"),
        ("classic", f"{_CLASSIC}.generate_answer_json_stream_generator"),
    ],
)
async def test_partial_output_saved_by_the_generator_survives_the_stop_write(
    turn, kind, stream
):
    """Stop on publish: the generator saves its partial answer while it is
    closed, then the runner marks the stop. The second write must not put back
    the stale copy a lagging secondary returns."""
    from src.services.generate_answer import persist_message_state

    async def _stream(*, message_id, **kwargs):
        try:
            yield 'data: {"type": "token", "content": "partial"}\n\n'
        except GeneratorExit:
            await persist_message_state(
                message_id, output="partial answer", stopped=True
            )
            raise

    user, conversation, message = turn
    bus = _PublishBlocksBus()
    p1, p2 = _patch_bus(bus)
    with p1, p2, patch(stream, _stream), _lagging_secondary(message):
        cm, task = _start(kind, user, conversation, message, None)
        await asyncio.wait_for(bus.entered.wait(), timeout=5)
        cm.cancel(message.id)
        await asyncio.wait_for(task, timeout=5)

    row = await Message.find_by_id_on_primary(message.id)
    assert row.stopped is True
    assert row.output == "partial answer"


@pytest.mark.parametrize("kind", ["agentic", "classic"])
async def test_a_cancel_that_is_not_a_stop_is_a_failed_turn(turn, kind):
    """Worker shutdown cancels the task without setting the Stop event."""
    user, conversation, message = turn
    p1, p2 = _patch_bus(StreamBus())
    with p1, p2:
        cm, task = _start(kind, user, conversation, message, asyncio.Event())
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.wait_for(task, timeout=5)

    row = await Message.find_by_id_on_primary(message.id)
    assert row.stopped is False
    assert row.metadata["error"]["type"] == "CancelledError"


@pytest.mark.parametrize("kind", ["agentic", "classic"])
async def test_a_shutdown_cancel_before_the_first_step_is_not_a_stop(turn, kind):
    user, conversation, message = turn
    p1, p2 = _patch_bus(StreamBus())
    with p1, p2:
        cm, task = _start(kind, user, conversation, message, asyncio.Event())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.1)
        cm.clear_mapping_for(conversation.id, message.id)
        cm.clear(message.id)

    row = await Message.find_by_id_on_primary(message.id)
    assert row.stopped is False


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


# ─── Stop against a streamed answer ───────────────────────────────────────────


class _BlocksOnBus(StreamBus):
    """Publishes until ``when`` matches an event, then blocks on it."""

    def __init__(self, when):
        super().__init__()
        self.when = when
        self.entered = asyncio.Event()

    async def publish(self, key, data):
        if self.when(data):
            self.entered.set()
            await asyncio.Event().wait()
        await super().publish(key, data)


def _is_type(kind):
    return lambda data: f'"type": "{kind}"' in data


_TOKENS = ("Rome ", "is here.")


class _HangingAgentGraph:
    """Agentic model that streams the first token, then never answers again."""

    async def astream(self, *args, **kwargs):
        yield "messages", (AIMessage(content=_TOKENS[0]), {"langgraph_node": "agent"})
        await asyncio.Event().wait()


class _HangingGraph:
    """Classic twin of ``_HangingAgentGraph``."""

    def astream(self, state, config=None, stream_mode=None):
        async def _stream():
            yield SimpleNamespace(content=_TOKENS[0]), {}
            await asyncio.Event().wait()

        return _stream()

    async def aclose(self):
        return None


@contextlib.contextmanager
def _answer_pipeline(kind, monkeypatch, hang=False):
    """Real stream helpers over a fake model that answers ``_TOKENS``, or with
    ``hang`` its first token only."""
    if kind == "agentic":
        graph = (
            _HangingAgentGraph()
            if hang
            else _FakeStreamGraph(
                messages=[
                    (AIMessage(content=token), {"langgraph_node": "agent"})
                    for token in _TOKENS
                ]
            )
        )
        with _patched_runner(
            _build_react_graph=MagicMock(return_value=graph),
            persist_message_state=persist_message_state,
        ):
            yield
    else:
        graph = _HangingGraph() if hang else _FakeGraph({"eve_jsc": list(_TOKENS)})
        _patch_pipeline(monkeypatch, graph)
        yield


async def _stop_on(kind, turn, monkeypatch, when):
    user, conversation, message = turn
    bus = _BlocksOnBus(when)
    p1, p2 = _patch_bus(bus)
    with p1, p2, _answer_pipeline(kind, monkeypatch):
        cm, task = _start(kind, user, conversation, message, None)
        await asyncio.wait_for(bus.entered.wait(), timeout=5)
        cm.cancel(message.id)
        await asyncio.wait_for(task, timeout=5)
    return await Message.find_by_id_on_primary(message.id)


@pytest.mark.parametrize("kind", ["agentic", "classic"])
async def test_stop_mid_answer_keeps_the_streamed_tokens(turn, kind, monkeypatch):
    seen = []

    def second_token(data):
        if '"type": "token"' in data:
            seen.append(data)
        return len(seen) == 2

    row = await _stop_on(kind, turn, monkeypatch, second_token)

    assert row.stopped is True
    assert row.output.startswith("Rome")


@pytest.mark.parametrize("kind", ["agentic", "classic"])
async def test_stop_on_the_final_event_keeps_the_answer(turn, kind, monkeypatch):
    row = await _stop_on(kind, turn, monkeypatch, _is_type("final"))

    assert row.stopped is False
    assert row.output == "".join(_TOKENS)


@pytest.mark.parametrize("kind", ["agentic", "classic"])
async def test_a_stop_after_the_final_event_still_charges_the_turn(
    turn, kind, monkeypatch
):
    """The Stop lands while the runner charges the tokens, after ``final``."""
    user, conversation, message = turn
    in_charge, charged = asyncio.Event(), asyncio.Event()

    async def _slow_charge(_user, _count):
        in_charge.set()
        await asyncio.sleep(0.1)
        charged.set()

    p1, p2 = _patch_bus(StreamBus())
    with p1, p2, _answer_pipeline(kind, monkeypatch), patch(
        f"{_RUNNER}.consume_tokens_for_user", _slow_charge
    ), patch(f"{_CLASSIC}.consume_tokens_for_user", _slow_charge):
        cm, task = _start(kind, user, conversation, message, None)
        await asyncio.wait_for(in_charge.wait(), timeout=5)
        # What the Stop route does.
        target = await cm.get_message_for_conversation_async(conversation.id)
        if target:
            cm.cancel(target)
        await asyncio.wait_for(task, timeout=5)

    row = await Message.find_by_id_on_primary(message.id)
    assert charged.is_set(), "the Stop cancelled the token charge"
    assert row.stopped is False
    assert row.output == "".join(_TOKENS)


class _TokenSeenBus(StreamBus):
    def __init__(self):
        super().__init__()
        self.token_seen = asyncio.Event()

    async def publish(self, key, data):
        if '"type": "token"' in data:
            self.token_seen.set()
        await super().publish(key, data)


@pytest.mark.parametrize("kind", ["agentic", "classic"])
async def test_a_shutdown_mid_answer_is_a_failed_turn_with_its_output(
    turn, kind, monkeypatch
):
    """Cancel without the Stop event while the model streams (ECS rollover)."""
    user, conversation, message = turn
    bus = _TokenSeenBus()
    p1, p2 = _patch_bus(bus)
    with p1, p2, _answer_pipeline(kind, monkeypatch, hang=True):
        cm, task = _start(kind, user, conversation, message, None)
        await asyncio.wait_for(bus.token_seen.wait(), timeout=5)
        await asyncio.sleep(0.05)  # parked in the model stream
        task.cancel()
        await asyncio.wait_for(task, timeout=5)

    row = await Message.find_by_id_on_primary(message.id)
    assert row.stopped is False
    assert row.output == _TOKENS[0]
    assert row.metadata["error"]["type"] == "CancelledError"


@pytest.mark.parametrize("kind", ["agentic", "classic"])
async def test_the_deadline_keeps_the_partial_output(turn, kind, monkeypatch):
    user, conversation, message = turn
    p1, p2 = _patch_bus(StreamBus())
    with p1, p2, _answer_pipeline(kind, monkeypatch, hang=True):
        cm, task = _start(kind, user, conversation, message, None, deadline=0.3)
        await asyncio.wait_for(task, timeout=5)

    row = await Message.find_by_id_on_primary(message.id)
    assert row.stopped is False
    assert row.output == _TOKENS[0]
    assert row.metadata["error"]["code"] == "timeout"


@pytest.mark.parametrize("kind", ["agentic", "classic"])
async def test_a_close_without_a_stop_leaves_the_message_alone(turn, kind, monkeypatch):
    """GeneratorExit with the Stop event unset: not a stop, nothing persisted."""
    user, conversation, message = turn
    request = GenerationRequest(query="a long question", agent="react")
    not_stopped = asyncio.Event()
    with _answer_pipeline(kind, monkeypatch):
        if kind == "agentic":
            events = generate_answer_agentic_stream_helper(
                request, conversation.id, message.id, user.id, "json", None, not_stopped
            )
        else:
            events = generate_answer_stream_generator_helper(
                request, conversation.id, message.id, "json", None, not_stopped, user.id
            )
        async for event in events:
            if '"type": "token"' in event:
                break
        await events.aclose()

    row = await Message.find_by_id_on_primary(message.id)
    assert row.stopped is False
    assert row.output == ""


@pytest.mark.no_db
@pytest.mark.skipif(not REDIS_URL, reason="the late subscriber case is the Redis bus")
async def test_a_subscriber_attached_after_the_stop_gets_the_stopped_event():
    bus = RedisStreamBus(REDIS_URL)
    key = uuid.uuid4().hex
    stopped = 'data: {"type": "stopped"}\n\n'
    try:
        await bus.stop(key, stopped)

        async def _drain():
            return [item async for item in bus.subscribe(key)]

        assert await asyncio.wait_for(_drain(), timeout=3) == [stopped]
    finally:
        await bus._redis.delete(f"sse-stopped:{key}")
        await bus._redis.aclose()


class _OrderRecordingRedis:
    """Records SUBSCRIBE, its reply and GET in the order the code issues them."""

    def __init__(self, value):
        self.calls = []
        self.value = value

    def pubsub(self):
        calls = self.calls

        class _PubSub:
            async def subscribe(self, channel):
                calls.append("subscribe")

            async def get_message(self, **kwargs):
                calls.append("subscribe-reply")
                return {"type": "subscribe"}

            async def unsubscribe(self, channel):
                return None

            async def close(self):
                return None

            async def aclose(self):
                return None

        return _PubSub()

    async def get(self, key):
        self.calls.append("get")
        return self.value

    async def aclose(self):
        return None


@pytest.mark.no_db
async def test_the_stream_reads_the_stop_key_after_the_subscribe_reply():
    fake = _OrderRecordingRedis(b'data: {"type": "stopped"}\n\n')
    bus = RedisStreamBus.__new__(RedisStreamBus)
    bus._redis = fake

    items = [item async for item in bus.subscribe("m1")]

    assert fake.calls == ["subscribe", "subscribe-reply", "get"]
    assert items == ['data: {"type": "stopped"}\n\n']


@pytest.mark.no_db
async def test_the_cancel_channel_reads_the_flag_after_the_subscribe_reply(monkeypatch):
    from src.services import cancel_manager

    fake = _OrderRecordingRedis(b"1")
    monkeypatch.setattr(
        cancel_manager.aioredis.Redis, "from_url", MagicMock(return_value=fake)
    )
    event = asyncio.Event()

    await CancelManager()._subscribe_cancel_channel("m1", event)

    assert fake.calls == ["subscribe", "subscribe-reply", "get"]
    assert event.is_set()
