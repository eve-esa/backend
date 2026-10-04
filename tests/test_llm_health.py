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


# ─── shared state through Redis ───────────────────────────────────────────────


class _FakeRedis:
    """The four commands the breaker uses, with expiry on the test clock."""

    def __init__(self, clock: _Clock) -> None:
        self.clock = clock
        self.down = False
        self.calls: list[str] = []
        self._data: dict[str, tuple[str, float]] = {}

    def _check(self, command: str) -> None:
        self.calls.append(command)
        if self.down:
            raise ConnectionError("Error 111 connecting to valkey:6379")

    def _live(self, key: str):
        entry = self._data.get(key)
        if entry is not None and self.clock() >= entry[1]:
            del self._data[key]
            return None
        return entry

    def set(self, key: str, value: str, px: int) -> bool:
        self._check("set")
        self._data[key] = (value, self.clock() + px / 1000)
        return True

    def pttl(self, key: str) -> int:
        self._check("pttl")
        entry = self._live(key)
        if entry is None:
            return -2
        return int((entry[1] - self.clock()) * 1000)

    def delete(self, key: str) -> int:
        self._check("delete")
        return 1 if self._data.pop(key, None) is not None else 0

    def get(self, key: str):
        entry = self._live(key)
        return None if entry is None else entry[0]


def _pair(clock: _Clock, store: _FakeRedis):
    return (
        EndpointHealth(cooldown_s=120, clock=clock, redis_client=store),
        EndpointHealth(cooldown_s=120, clock=clock, redis_client=store),
    )


@pytest.mark.no_db
def test_one_worker_failure_opens_the_circuit_for_every_worker():
    clock = _Clock()
    store = _FakeRedis(clock)
    worker_a, worker_b = _pair(clock, store)

    worker_a.record_failure("eve_jsc", TimeoutError("first token budget"))

    # worker_b never failed on its own: the key alone keeps it off the provider.
    assert worker_b.is_open("eve_jsc") is True
    assert worker_b.is_open("main") is False
    assert "TimeoutError" in store.get("eve:breaker:eve_jsc")


@pytest.mark.no_db
def test_one_worker_success_closes_the_circuit_for_every_worker():
    clock = _Clock()
    store = _FakeRedis(clock)
    worker_a, worker_b = _pair(clock, store)
    worker_a.record_failure("eve_jsc", TimeoutError("first token budget"))
    assert worker_b.is_open("eve_jsc") is True

    worker_b.record_success("eve_jsc")
    clock.now = 1.0  # past worker_a's read cache

    # Redis is authoritative while it answers, over worker_a's own failure.
    assert store.get("eve:breaker:eve_jsc") is None
    assert worker_a.is_open("eve_jsc") is False


@pytest.mark.no_db
def test_key_expiry_closes_the_shared_circuit():
    clock = _Clock()
    store = _FakeRedis(clock)
    worker_a, worker_b = _pair(clock, store)
    worker_a.record_failure("eve_jsc", TimeoutError("first token budget"))

    clock.now = 119.0
    assert worker_b.is_open("eve_jsc") is True

    clock.now = 120.0
    assert worker_b.is_open("eve_jsc") is False
    assert worker_a.is_open("eve_jsc") is False


@pytest.mark.no_db
def test_reads_are_cached_between_refreshes():
    clock = _Clock()
    store = _FakeRedis(clock)
    worker_a, worker_b = _pair(clock, store)

    for _ in range(5):
        worker_b.is_open("eve_jsc")
    assert store.calls.count("pttl") == 1

    # A peer's failure is seen once the cache window has passed.
    worker_a.record_failure("eve_jsc", TimeoutError("first token budget"))
    assert worker_b.is_open("eve_jsc") is False
    clock.now = 1.0
    assert worker_b.is_open("eve_jsc") is True


@pytest.mark.no_db
def test_redis_down_falls_back_to_local_state_and_logs_once(caplog):
    clock = _Clock()
    store = _FakeRedis(clock)
    store.down = True
    worker_a, worker_b = _pair(clock, store)

    with caplog.at_level("WARNING", logger="src.core.llm_health"):
        worker_a.record_failure("eve_jsc", TimeoutError("first token budget"))
        assert worker_a.is_open("eve_jsc") is True
        assert worker_b.is_open("eve_jsc") is False
        clock.now = 31.0  # past the retry window, still down
        assert worker_a.is_open("eve_jsc") is True

    unreachable = [r for r in caplog.records if "store unreachable" in r.getMessage()]
    assert len(unreachable) == 2  # one per instance, not one per call
    # Inside the retry window the store is not even tried.
    calls_before = len(store.calls)
    worker_a.is_open("main")
    assert len(store.calls) == calls_before


@pytest.mark.no_db
def test_redis_recovery_shares_circuits_again():
    clock = _Clock()
    store = _FakeRedis(clock)
    store.down = True
    worker_a, worker_b = _pair(clock, store)
    worker_b.is_open("eve_jsc")  # worker_b learns the store is down

    store.down = False
    clock.now = 30.0
    worker_a.record_failure("eve_jsc", TimeoutError("first token budget"))

    assert worker_b.is_open("eve_jsc") is True


@pytest.mark.no_db
def test_unreachable_redis_url_never_raises():
    # Port 1 refuses at once: exercises the real redis-py client and its errors.
    health = EndpointHealth(cooldown_s=120, redis_url="redis://127.0.0.1:1/0")

    health.record_failure("eve_jsc", TimeoutError("first token budget"))

    assert health.is_open("eve_jsc") is True
    health.record_success("eve_jsc")
    assert health.is_open("eve_jsc") is False
