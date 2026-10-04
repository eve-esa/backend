"""Circuit breaker and failure classification for the endpoint chain.

The classification table is the load-bearing part: an endpoint that is merely
refusing our prompt (400) or our key (401) must stay in the chain, while one
that is timing out or 5xx-ing must drop out of it for the cooldown.
"""

import asyncio
import time

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError

from src.core.llm_health import EndpointHealth, is_endpoint_failure, reset_shared_stores


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

KEY = "eve:breaker:eve_jsc"


class _FakeRedis:
    """The commands the breaker uses, with expiry on the test clock."""

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

    def get(self, key: str):
        self._check("get")
        entry = self._live(key)
        return None if entry is None else entry[0]

    def set(self, key: str, value: str, px: int, nx: bool = False):
        self._check("set")
        if nx and self._live(key) is not None:
            return None
        self._data[key] = (value, self.clock() + px / 1000)
        return True

    def delete(self, *keys: str) -> int:
        self._check("delete")
        return sum(1 for key in keys if self._data.pop(key, None) is not None)

    def peek(self, key: str):
        entry = self._live(key)
        return None if entry is None else entry[0]


@pytest.fixture(autouse=True)
def _fresh_stores():
    reset_shared_stores()
    yield
    reset_shared_stores()


def _breaker(clock: _Clock, store: _FakeRedis) -> EndpointHealth:
    return EndpointHealth(
        cooldown_s=120, clock=clock, wall_clock=clock, redis_client=store, probe_s=30
    )


def _setup():
    clock = _Clock()
    clock.now = 1000.0
    store = _FakeRedis(clock)
    return clock, store, _breaker(clock, store), _breaker(clock, store)


@pytest.mark.no_db
def test_one_worker_failure_opens_the_circuit_for_every_worker():
    clock, store, worker_a, worker_b = _setup()

    worker_a.record_failure("eve_jsc", TimeoutError("first token budget"))

    # worker_b never failed on its own: the key alone keeps it off the provider.
    assert worker_b.is_open("eve_jsc") is True
    assert worker_b.is_open("main") is False
    assert store.peek(KEY).startswith("1000000:TimeoutError")


@pytest.mark.no_db
def test_one_worker_success_closes_the_circuit_for_every_worker():
    clock, store, worker_a, worker_b = _setup()
    worker_a.record_failure("eve_jsc", TimeoutError("first token budget"))
    assert worker_b.is_open("eve_jsc") is True

    worker_b.record_success("eve_jsc", started_at=1000.5)
    clock.now = 1001.0  # past worker_a's read cache

    # Redis is authoritative while it answers, over worker_a's own failure.
    assert store.peek(KEY) is None
    assert worker_a.is_open("eve_jsc") is False


@pytest.mark.no_db
def test_a_success_never_erases_a_newer_failure():
    clock, store, worker_a, worker_b = _setup()
    worker_a.record_failure("eve_jsc", TimeoutError("first token budget"))
    assert worker_b.is_open("eve_jsc") is True

    # worker_b's request started at 1001; worker_a re-opens at 1005.
    clock.now = 1005.0
    worker_a.record_failure("eve_jsc", TimeoutError("still down"))
    worker_b.record_success("eve_jsc", started_at=1001.0)

    assert store.peek(KEY).startswith("1005000:")
    clock.now = 1006.0
    assert worker_b.is_open("eve_jsc") is True


@pytest.mark.no_db
def test_a_success_on_a_closed_circuit_costs_no_round_trip():
    clock, store, worker_a, worker_b = _setup()

    worker_b.record_success("eve_jsc", started_at=1000.0)

    assert store.calls == []


@pytest.mark.no_db
def test_half_open_lets_one_worker_probe():
    clock, store, worker_a, worker_b = _setup()
    worker_a.record_failure("eve_jsc", TimeoutError("first token budget"))

    clock.now = 1120.0
    assert worker_b.is_open("eve_jsc") is False  # lease holder probes
    assert worker_a.is_open("eve_jsc") is True  # waits for the verdict

    worker_b.record_failure("eve_jsc", TimeoutError("probe failed"))
    clock.now = 1121.5
    assert worker_a.is_open("eve_jsc") is True
    assert store.peek(KEY + ":probe") is None

    clock.now = 1241.0
    assert worker_a.is_open("eve_jsc") is False
    assert worker_b.is_open("eve_jsc") is True
    worker_a.record_success("eve_jsc", started_at=1241.0)
    assert store.peek(KEY) is None and store.peek(KEY + ":probe") is None
    clock.now = 1242.5
    assert worker_b.is_open("eve_jsc") is False


@pytest.mark.no_db
def test_key_expiry_closes_the_shared_circuit():
    clock, store, worker_a, worker_b = _setup()
    worker_a.record_failure("eve_jsc", TimeoutError("first token budget"))

    clock.now = 1119.0
    assert worker_b.is_open("eve_jsc") is True

    # Cooldown plus probe budget gone and nobody reported: closed for all.
    clock.now = 1150.0
    assert worker_b.is_open("eve_jsc") is False
    assert worker_a.is_open("eve_jsc") is False
    assert store.peek(KEY + ":probe") is None


@pytest.mark.no_db
def test_reads_are_cached_between_refreshes():
    clock, store, worker_a, worker_b = _setup()

    for _ in range(5):
        worker_b.is_open("eve_jsc")
    assert store.calls.count("get") == 1

    # A peer's failure is seen once the cache window has passed.
    worker_a.record_failure("eve_jsc", TimeoutError("first token budget"))
    assert worker_b.is_open("eve_jsc") is False
    clock.now = 1001.0
    assert worker_b.is_open("eve_jsc") is True


@pytest.mark.no_db
def test_redis_down_falls_back_to_local_state_and_logs_once(caplog):
    clock, store, worker_a, worker_b = _setup()
    store.down = True

    with caplog.at_level("WARNING", logger="src.core.llm_health"):
        worker_a.record_failure("eve_jsc", TimeoutError("first token budget"))
        assert worker_a.is_open("eve_jsc") is True
        assert worker_b.is_open("eve_jsc") is False
        clock.now = 1031.0  # past the retry window, still down
        assert worker_a.is_open("eve_jsc") is True

    unreachable = [r for r in caplog.records if "store unreachable" in r.getMessage()]
    assert len(unreachable) == 1  # one per store and outage, not per call
    # Inside the retry window the store is not even tried, by any breaker on it.
    calls_before = len(store.calls)
    worker_b.is_open("main")
    assert len(store.calls) == calls_before


@pytest.mark.no_db
def test_a_failure_recorded_during_an_outage_survives_recovery():
    clock, store, worker_a, worker_b = _setup()
    store.down = True
    worker_a.record_failure("eve_jsc", TimeoutError("first token budget"))
    assert worker_b.is_open("eve_jsc") is False

    store.down = False
    clock.now = 1031.0
    # The empty store must not override what worker_a saw during the outage.
    assert worker_a.is_open("eve_jsc") is True
    assert store.peek(KEY).startswith("1000000:")
    assert worker_b.is_open("eve_jsc") is True


@pytest.mark.no_db
def test_breakers_on_one_url_share_client_and_outage_window():
    # Non-routable: the connect hangs until the 250 ms timeout.
    url = "redis://10.255.255.1:6379/0"
    worker_a = EndpointHealth(cooldown_s=120, redis_url=url)
    worker_b = EndpointHealth(cooldown_s=120, redis_url=url)
    assert worker_a._store is worker_b._store

    started = time.monotonic()
    worker_a.record_failure("eve_jsc", TimeoutError("first token budget"))
    assert time.monotonic() - started < 0.6
    assert worker_a._store.client is not None

    # worker_b is inside the shared retry window: no second stall.
    started = time.monotonic()
    assert worker_b.is_open("eve_jsc") is False
    assert time.monotonic() - started < 0.05
    assert worker_a.is_open("eve_jsc") is True


@pytest.mark.no_db
def test_unreachable_redis_url_never_raises():
    # Port 1 refuses at once: exercises the real redis-py client and its errors.
    health = EndpointHealth(cooldown_s=120, redis_url="redis://127.0.0.1:1/0")

    health.record_failure("eve_jsc", TimeoutError("first token budget"))

    assert health.is_open("eve_jsc") is True
    health.record_success("eve_jsc")
    assert health.is_open("eve_jsc") is False
