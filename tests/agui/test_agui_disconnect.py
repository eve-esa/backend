"""A client disconnect on /agui stops the run like POST /stop (decision 0008).

AG-UI clients cancel a run by closing the request. The app is driven as an
ASGI callable with a hand-made ``receive`` that reports ``http.disconnect`` at
a chosen event, for both ASGI binding paths Starlette has: spec 2.3 (uvicorn
today: a task listens for the disconnect) and 2.4 (the next write fails).

The runner is a fake that, like the real one, persists a cancel through
``persist_runner_cancel``; so the stored message reads stopped only when the
cancel event was set, exactly as after POST /stop.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from typing import Any, Callable, Dict, List, Optional

import pytest

from server import app
from src.database.models.message import Message
from src.routers import agui as agui_route
from src.services.cancel_manager import get_cancel_manager
from src.services.generate_answer import persist_runner_cancel
from tests.agui.test_agui_route import (  # noqa: F401  (fixtures)
    FakeBus,
    bus,
    events_of,
    f_final,
    f_status,
    f_token,
    flag_on,
    owner,
    run_input,
    sse,
)
from tests.agui.validator import validate_transcript

STOPPED_FRAME = f"data: {json.dumps({'type': 'stopped'})}\n\n"


class CountingBus(FakeBus):
    def __init__(self) -> None:
        super().__init__()
        self.stops: List[tuple] = []

    async def stop(self, key: str, event: str) -> None:
        self.stops.append((key, event))
        await super().stop(key, event)


class CancellableRunner:
    """Publishes ``frames``, then holds the bus open until cancelled or released.

    On a cancel it does what ``run_agentic_generation_to_bus`` does: persist
    the cancel (stopped only when the cancel event is set) and close the bus.
    """

    def __init__(self, bus: FakeBus, frames: List[Dict[str, Any]]) -> None:
        self.bus = bus
        self.frames = frames
        self.calls: List[Dict[str, Any]] = []
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.done = asyncio.Event()

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        message_id = kwargs["message_id"]
        await kwargs["subscriber_ready"].wait()
        try:
            for frame in self.frames:
                await self.bus.publish(message_id, sse(frame))
            await self.release.wait()
        except asyncio.CancelledError as exc:
            self.cancelled.set()
            await persist_runner_cancel(message_id, kwargs["cancel_event"], exc)
        finally:
            await self.bus.close(message_id)
            self.done.set()


@pytest.fixture
def counting_bus(monkeypatch):
    fake = CountingBus()
    monkeypatch.setattr("src.routers.agui.get_stream_bus", lambda: fake)
    monkeypatch.setattr("src.routers.message.get_stream_bus", lambda: fake)
    return fake


@pytest.fixture
def cancellable(monkeypatch, counting_bus):
    def _install(frames) -> CancellableRunner:
        fake = CancellableRunner(counting_bus, frames)
        monkeypatch.setattr("src.routers.agui.run_agentic_generation_to_bus", fake)
        return fake

    return _install


@pytest.fixture
def cancel_calls(monkeypatch):
    manager = get_cancel_manager()
    calls: List[str] = []
    original = manager.cancel

    def counting(message_id: str) -> None:
        calls.append(message_id)
        original(message_id)

    monkeypatch.setattr(manager, "cancel", counting)
    return calls


@pytest.fixture(autouse=True)
def fast_keepalive(monkeypatch):
    # Under ASGI 2.4 the disconnect is seen on the next write: keep writes coming.
    monkeypatch.setattr(agui_route, "KEEPALIVE_INTERVAL_S", 0.05)


class Client:
    """One /agui request over raw ASGI, with a disconnect at a chosen event."""

    def __init__(self, spec_version: str) -> None:
        self.spec_version = spec_version
        self.events: List[Dict[str, Any]] = []
        self.raw: List[str] = []
        self.after_disconnect: List[str] = []
        self.gone = asyncio.Event()
        self.status: Optional[int] = None

    async def run(
        self,
        conversation_id: str,
        token: str,
        body: Dict[str, Any],
        disconnect_when: Callable[[List[Dict[str, Any]]], bool],
        on_event: Optional[Callable[[List[Dict[str, Any]]], Any]] = None,
    ) -> None:
        payload = json.dumps(body).encode()
        sent = False

        async def receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": payload, "more_body": False}
            await self.gone.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if self.gone.is_set():
                if message["type"] == "http.response.body" and message.get("body"):
                    self.after_disconnect.append(message["body"].decode())
                if self.spec_version == "2.4":
                    raise OSError("client went away")
                return
            if message["type"] == "http.response.start":
                self.status = message["status"]
                return
            chunk = message.get("body", b"").decode()
            self.raw.append(chunk)
            self.events.extend(events_of(chunk))
            if on_event is not None:
                await on_event(self.events)
            if disconnect_when(self.events):
                self.gone.set()

        path = f"/conversations/{conversation_id}/agui"
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": self.spec_version},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": [
                (b"host", b"test"),
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode()),
                (b"accept", b"text/event-stream"),
                (b"authorization", f"Bearer {token}".encode()),
            ],
            "client": ("127.0.0.1", 50000),
            "server": ("test", 80),
            "state": {},
        }
        # A 2.4 disconnect surfaces as ClientDisconnect out of the app.
        with contextlib.suppress(Exception):
            await asyncio.wait_for(app(scope, receive, send), 10)

    @property
    def body(self) -> str:
        return "".join(self.raw)


def _seen(kind: str) -> Callable[[List[Dict[str, Any]]], bool]:
    return lambda events: any(e["type"] == kind for e in events)


async def _stored(conversation_id: str) -> Message:
    (message,) = await Message.find_all(filter_dict={"conversation_id": conversation_id})
    return message


SPECS = pytest.mark.parametrize("spec_version", ["2.3", "2.4"])


def _assert_truncated_prefix(client: Client, conversation_id: str, run_id: str) -> None:
    assert client.status == 200
    assert not [e for e in client.events if e["type"] in ("RUN_FINISHED", "RUN_ERROR")]
    assert client.after_disconnect == [] or all(
        not line.startswith("data:") for chunk in client.after_disconnect for line in chunk.splitlines()
    )
    violations = validate_transcript(
        client.body, thread_id=conversation_id, run_id=run_id, require_terminal=False
    )
    assert violations == [], violations


@SPECS
async def test_disconnect_mid_text_stops_the_run(
    flag_on, cancellable, counting_bus, cancel_calls, owner, spec_version
):
    _, token, conversation = owner
    fake = cancellable([f_status(), f_token("Partial ")])
    run_id = str(uuid.uuid4())
    client = Client(spec_version)

    await client.run(conversation.id, token, run_input(conversation.id, run_id), _seen("TEXT_MESSAGE_CONTENT"))

    await asyncio.wait_for(fake.done.wait(), 5)
    message_id = fake.calls[0]["message_id"]
    assert fake.cancelled.is_set()
    assert fake.calls[0]["cancel_event"].is_set()
    assert cancel_calls == [message_id]
    assert counting_bus.stops == [(message_id, STOPPED_FRAME)]
    assert (await _stored(conversation.id)).stopped is True
    _assert_truncated_prefix(client, conversation.id, run_id)


@SPECS
async def test_disconnect_during_a_tool_call_stops_the_run(
    flag_on, cancellable, counting_bus, cancel_calls, owner, spec_version
):
    _, token, conversation = owner
    tool_call = {
        "type": "tool_call",
        "content": "Searching…",
        "tool": "eve_retrieval_retrieve",
        "label": "Searching EVE sources",
        "query": "floods",
        "id": "call_1",
        "args": {"query": "floods"},
    }
    fake = cancellable([f_status(), tool_call])
    run_id = str(uuid.uuid4())
    client = Client(spec_version)

    await client.run(conversation.id, token, run_input(conversation.id, run_id), _seen("TOOL_CALL_END"))

    await asyncio.wait_for(fake.done.wait(), 5)
    message_id = fake.calls[0]["message_id"]
    assert fake.cancelled.is_set()
    assert cancel_calls == [message_id]
    assert counting_bus.stops == [(message_id, STOPPED_FRAME)]
    assert (await _stored(conversation.id)).stopped is True
    _assert_truncated_prefix(client, conversation.id, run_id)


@SPECS
async def test_disconnect_after_run_finished_has_no_effect(
    flag_on, cancellable, counting_bus, cancel_calls, owner, spec_version
):
    _, token, conversation = owner
    # The runner keeps the bus open after `final` (the token charge), and the
    # client leaves in that window.
    fake = cancellable([f_status(), f_token("Done."), f_final("Done.")])
    run_id = str(uuid.uuid4())
    client = Client(spec_version)

    await client.run(conversation.id, token, run_input(conversation.id, run_id), _seen("RUN_FINISHED"))
    await asyncio.sleep(0.2)

    assert not fake.cancelled.is_set()
    assert not fake.calls[0]["cancel_event"].is_set()
    assert cancel_calls == []
    assert counting_bus.stops == []
    fake.release.set()
    await asyncio.wait_for(fake.done.wait(), 5)
    assert (await _stored(conversation.id)).stopped is not True
    violations = validate_transcript(client.body, thread_id=conversation.id, run_id=run_id)
    assert violations == [], violations


@SPECS
async def test_stop_then_disconnect_is_one_stop(
    async_client, flag_on, cancellable, counting_bus, cancel_calls, owner, spec_version
):
    _, token, conversation = owner
    fake = cancellable([f_status(), f_token("Partial ")])
    run_id = str(uuid.uuid4())
    client = Client(spec_version)
    stopped = []

    async def press_stop(events):
        if stopped or not any(e["type"] == "TEXT_MESSAGE_CONTENT" for e in events):
            return
        response = await async_client.post(
            f"/conversations/{conversation.id}/stop", headers={"Authorization": f"Bearer {token}"}
        )
        stopped.append(response.json()["status"])

    # Stop is pressed, then the client closes the request at once (what a
    # client that calls /stop and aborts its fetch does).
    await client.run(
        conversation.id,
        token,
        run_input(conversation.id, run_id),
        lambda events: bool(stopped),
        on_event=press_stop,
    )

    await asyncio.wait_for(fake.done.wait(), 5)
    message_id = fake.calls[0]["message_id"]
    assert stopped == ["stopping"]
    assert cancel_calls == [message_id]
    assert counting_bus.stops == [(message_id, STOPPED_FRAME)]
    assert (await _stored(conversation.id)).stopped is True
    violations = validate_transcript(
        client.body, thread_id=conversation.id, run_id=run_id, require_terminal=False
    )
    assert violations == [], violations
