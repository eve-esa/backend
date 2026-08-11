"""Unit tests for the stream bus subscriber handshake (src/services/stream_bus.py).

Neither bus implementation buffers, so a chunk published before a subscriber is
attached is gone. The producer therefore has to wait for the ``ready`` event the
consumer sets, or every event emitted at t=0 loses the race.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.schemas.generation_request import GenerationRequest
from src.services.agents.core.runner import run_agentic_generation_to_bus
from src.services.stream_bus import StreamBus

pytestmark = [pytest.mark.asyncio, pytest.mark.no_db]

_RUNNER = "src.services.agents.core.runner"


class TestStreamBusReadyHandshake:
    async def test_producer_waiting_on_ready_reaches_the_subscriber(self):
        bus = StreamBus()
        ready = asyncio.Event()

        async def produce():
            await ready.wait()
            await bus.publish("m1", "x")
            await bus.close("m1")

        task = asyncio.create_task(produce())
        received = [item async for item in bus.subscribe("m1", ready=ready)]
        await task

        assert received == ["x"]

    async def test_publish_before_subscribe_is_dropped(self):
        """Why the handshake exists: the bus has no buffer to catch up from."""
        bus = StreamBus()
        ready = asyncio.Event()
        await bus.publish("m1", "lost")

        async def produce():
            await ready.wait()
            await bus.publish("m1", "kept")
            await bus.close("m1")

        task = asyncio.create_task(produce())
        received = [item async for item in bus.subscribe("m1", ready=ready)]
        await task

        assert received == ["kept"]


def _absent_lookup() -> MagicMock:
    """Model stand-in whose ``find_by_id`` resolves to nothing, skipping Mongo."""
    model = MagicMock()
    model.find_by_id = AsyncMock(return_value=None)
    return model


class TestRunAgenticGenerationToBusHandshake:
    async def test_generation_waits_for_the_subscriber(self):
        started = asyncio.Event()

        async def fake_stream(**kwargs):
            started.set()
            yield "data: {}\n\n"

        subscriber_ready = asyncio.Event()
        bus = StreamBus()

        with patch(
            f"{_RUNNER}.generate_answer_agentic_json_stream", fake_stream
        ), patch(f"{_RUNNER}.get_stream_bus", MagicMock(return_value=bus)), patch(
            f"{_RUNNER}.User", _absent_lookup()
        ), patch(
            f"{_RUNNER}.Message", _absent_lookup()
        ):
            task = asyncio.create_task(
                run_agentic_generation_to_bus(
                    request=GenerationRequest(query="hi", agent="react"),
                    conversation_id="c1",
                    message_id="m1",
                    user_id="u1",
                    subscriber_ready=subscriber_ready,
                )
            )
            await asyncio.sleep(0.05)
            assert not started.is_set(), "generation started before the subscriber"

            subscriber_ready.set()
            await asyncio.wait_for(task, timeout=1)

        assert started.is_set()

    async def test_no_ready_event_starts_immediately(self):
        """The parameter is optional: callers that don't pass it are unaffected."""
        started = asyncio.Event()

        async def fake_stream(**kwargs):
            started.set()
            yield "data: {}\n\n"

        bus = StreamBus()

        with patch(
            f"{_RUNNER}.generate_answer_agentic_json_stream", fake_stream
        ), patch(f"{_RUNNER}.get_stream_bus", MagicMock(return_value=bus)), patch(
            f"{_RUNNER}.User", _absent_lookup()
        ), patch(
            f"{_RUNNER}.Message", _absent_lookup()
        ):
            await asyncio.wait_for(
                run_agentic_generation_to_bus(
                    request=GenerationRequest(query="hi", agent="react"),
                    conversation_id="c1",
                    message_id="m1",
                    user_id="u1",
                ),
                timeout=1,
            )

        assert started.is_set()
