"""Shared helpers for ``src.routers.openai_proxy`` tests.

Two mocking styles, matching the two things the proxy tests need:

* ``mock_client`` / ``mock_client_raising``: a ``MagicMock`` whose ``stream()``
  is an async context manager, used wherever a test needs to inject a fault
  (a connection error, a mid-stream exception) or does not care about the
  exact bytes on the wire.
* ``transport_client``: a real ``httpx.AsyncClient`` backed by
  ``httpx.MockTransport``, used for the budget/charging tests that want the
  actual request (method, url, headers, body) httpx would send, not just a
  mocked response.
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import httpx
from bson import ObjectId

from src.database.models.user import User

FAKE_UPSTREAM = "http://fake-upstream"
FAKE_JSC_UPSTREAM = "http://fake-jsc-upstream"


def enable_proxy(monkeypatch, proxy, *, runpod_url: str = FAKE_UPSTREAM, jsc_url: str = ""):
    """Turn the dispatcher on and point it at fake upstream(s) for a test."""
    monkeypatch.setattr(proxy, "_proxy_enabled", True)
    monkeypatch.setattr("src.routers.openai_proxy.OPENAI_PROXY_UPSTREAM_URL", runpod_url)
    monkeypatch.setattr("src.routers.openai_proxy.EVE_JSC_BASE_URL", jsc_url)
    monkeypatch.setattr("src.routers.openai_proxy.OPENAI_PROXY_API_KEY", "fake-runpod-key")
    monkeypatch.setattr("src.routers.openai_proxy.EVE_JSC_API_KEY", "fake-jsc-key")


def minimal_completion_body() -> bytes:
    return json.dumps(
        {
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    ).encode()


def mock_client(status: int = 200, body: bytes = b"{}", content_type: str = "application/json"):
    """Return a mock httpx.AsyncClient whose stream() acts as an async context manager."""
    resp = MagicMock()
    resp.status_code = status
    resp.headers = {"content-type": content_type}

    async def aiter_bytes():
        yield body

    resp.aiter_bytes = aiter_bytes

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=resp)
    cm.__aexit__ = AsyncMock(return_value=None)

    client = MagicMock()
    client.stream.return_value = cm
    return client, client.stream


def mock_client_raising(exc: Exception):
    """A mock client whose stream() raises ``exc`` on entry (e.g. a connection error)."""
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(side_effect=exc)
    cm.__aexit__ = AsyncMock(return_value=None)

    client = MagicMock()
    client.stream.return_value = cm
    return client, client.stream


def mock_client_failing_mid_stream(status: int = 200, exc: Optional[Exception] = None, content_type: str = "text/event-stream"):
    """A mock streaming client that yields one chunk, then raises while iterating.

    Models a client disconnect / upstream drop partway through an SSE body:
    the response has already started (status, headers sent), so this is what
    exercises the "charge whatever was produced so far" path in the finally.
    """
    resp = MagicMock()
    resp.status_code = status
    resp.headers = {"content-type": content_type}

    async def aiter_bytes():
        yield b'data: {"id":"c","choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}\n\n'
        raise exc or ConnectionResetError("connection reset mid-stream")

    resp.aiter_bytes = aiter_bytes

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=resp)
    cm.__aexit__ = AsyncMock(return_value=None)

    client = MagicMock()
    client.stream.return_value = cm
    return client, client.stream


def transport_client(handler) -> httpx.AsyncClient:
    """A real ``httpx.AsyncClient`` backed by ``httpx.MockTransport``.

    ``handler(request: httpx.Request) -> httpx.Response``. Leaves the
    dispatcher's ``_client_loop`` alone (callers assign only ``_client``), so
    ``OpenAIProxyDispatcher._get_client`` keeps reusing this instance instead
    of building a fresh real one.
    """
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def sse_handler(*, prompt_tokens: int, completion_tokens: int, content: str = "hi"):
    """An httpx.MockTransport handler returning a minimal SSE completion stream."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = (
            b'data: {"id":"c","object":"chat.completion.chunk",'
            b'"choices":[{"index":0,"delta":{"content":"' + content.encode() + b'"},"finish_reason":null}]}\n\n'
            b'data: {"id":"c","object":"chat.completion.chunk","choices":[],'
            b'"usage":{"prompt_tokens":' + str(prompt_tokens).encode() + b',"completion_tokens":'
            + str(completion_tokens).encode() + b',"total_tokens":'
            + str(prompt_tokens + completion_tokens).encode() + b"}}\n\n"
            b"data: [DONE]\n\n"
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    return handler


def json_handler(status: int, payload: dict):
    """An httpx.MockTransport handler that always answers with ``payload``."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return handler


async def seed_active_window(
    user: User,
    *,
    used_tokens: int,
    period_months: int = 1,
) -> None:
    """Give ``user`` an already-active, non-expired rate-limit window in Mongo.

    Writing straight to the collection (bypassing ``consume_tokens_for_user``)
    lets a test start a request already at, near, or over a cap without the
    proxy's own ``_ensure_active_window`` rolling a fresh, empty window over
    whatever the test just set.
    """
    now = datetime.now(timezone.utc)
    period_start = now - timedelta(minutes=1)
    period_end = now + timedelta(days=30 * period_months)
    await User.get_collection().update_one(
        {"_id": ObjectId(user.id)},
        {
            "$set": {
                "rate_limit_period_start": period_start,
                "rate_limit_period_end": period_end,
                "rate_limit_tokens_used": used_tokens,
            }
        },
    )
    user.rate_limit_period_start = period_start
    user.rate_limit_period_end = period_end
    user.rate_limit_tokens_used = used_tokens
