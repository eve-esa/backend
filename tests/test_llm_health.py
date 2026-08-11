"""Circuit breaker and failure classification for the endpoint chain.

The classification table is the load-bearing part: an endpoint that is merely
refusing our prompt (400) or our key (401) must stay in the chain, while one
that is timing out or 5xx-ing must drop out of it for the cooldown.
"""

import asyncio

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError

from src.core.llm_health import EndpointHealth, is_endpoint_failure


class _Clock:
    """Monotonic clock the tests move by hand."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _NodeTimeoutError(Exception):
    pass


_NodeTimeoutError.__name__ = "NodeTimeoutError"


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://endpoint.example/v1/chat/completions")


def _status_error(status: int) -> APIStatusError:
    return APIStatusError(
        "upstream said no",
        response=httpx.Response(status, request=_request()),
        body=None,
    )


# ─── classification ───────────────────────────────────────────────────────────


@pytest.mark.no_db
@pytest.mark.parametrize(
    "exc",
    [
        httpx.ConnectError("connection refused"),
        httpx.ConnectTimeout("connect timeout"),
        httpx.ReadTimeout("read timeout"),
        httpx.RemoteProtocolError("peer closed"),
        APIConnectionError(request=_request()),
        APITimeoutError(request=_request()),
        _status_error(500),
        _status_error(502),
        _status_error(504),
        _status_error(429),
        TimeoutError("first token budget"),
        asyncio.TimeoutError("first token budget"),
        _NodeTimeoutError("node 'agent' exceeded its idle timeout"),
    ],
)
def test_endpoint_failures_open_the_circuit(exc):
    assert is_endpoint_failure(exc) is True


@pytest.mark.no_db
@pytest.mark.parametrize(
    "exc",
    [
        _status_error(400),
        _status_error(401),
        _status_error(403),
        _status_error(422),
        asyncio.CancelledError(),
        ValueError("No generations found in stream."),
        RuntimeError("EVE_JSC_BASE_URL is not configured"),
    ],
)
def test_request_side_failures_leave_the_circuit_closed(exc):
    assert is_endpoint_failure(exc) is False


# ─── circuit transitions ──────────────────────────────────────────────────────


@pytest.mark.no_db
def test_unknown_endpoint_starts_closed():
    health = EndpointHealth(cooldown_s=120, clock=_Clock())

    assert health.is_open("eve_jsc") is False


@pytest.mark.no_db
def test_a_single_failure_opens_the_circuit():
    health = EndpointHealth(cooldown_s=120, clock=_Clock())

    health.record_failure("eve_jsc", TimeoutError("cold start"))

    assert health.is_open("eve_jsc") is True
    assert health.is_open("main") is False


@pytest.mark.no_db
def test_cooldown_expiry_closes_the_circuit():
    clock = _Clock()
    health = EndpointHealth(cooldown_s=120, clock=clock)
    health.record_failure("eve_jsc", TimeoutError("cold start"))

    clock.now = 119.0
    assert health.is_open("eve_jsc") is True

    clock.now = 120.0
    assert health.is_open("eve_jsc") is False
    # Expiry deletes the entry, so the next request is the half-open probe.
    assert health.snapshot() == {}


@pytest.mark.no_db
def test_record_success_closes_the_circuit():
    health = EndpointHealth(cooldown_s=120, clock=_Clock())
    health.record_failure("main", TimeoutError("cold start"))

    health.record_success("main")

    assert health.is_open("main") is False
    assert health.snapshot() == {}


@pytest.mark.no_db
def test_repeated_failures_count_and_restart_the_cooldown():
    clock = _Clock()
    health = EndpointHealth(cooldown_s=120, clock=clock)
    health.record_failure("main", TimeoutError("cold start"))

    clock.now = 119.0
    health.record_failure("main", httpx.ConnectError("connection refused"))

    snapshot = health.snapshot()
    assert snapshot["main"]["failures"] == 2
    assert snapshot["main"]["open"] is True
    assert "ConnectError" in snapshot["main"]["last_error"]

    clock.now = 238.0
    assert health.is_open("main") is True
