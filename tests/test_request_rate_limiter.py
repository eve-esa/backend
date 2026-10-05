"""Per-principal request rate limit (src/services/request_rate_limiter.py).

Unit tests run against an in-memory store that mirrors the Lua token bucket
with a clock the test moves, so the bucket maths are exact. The ``real store``
tests run the actual script against REDIS_URL and skip when nothing answers
there: they are the proof of the Lua, of database 1 and of exactness under
concurrency.
"""

import asyncio
import logging
import math
import os
import re
import uuid

import pytest
from fastapi import Depends, FastAPI
from httpx import ASGITransport, AsyncClient
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.resources import Resource

from src import config
from src.config import (
    DEFAULT_REQUEST_RATE_LIMITS,
    parse_request_rate_limit_mode,
    parse_request_rate_limits,
)
from src.database.models.user import User
from src.middlewares import auth as auth_module
from src.middlewares.auth import (
    AUTH_TYPE_API_KEY,
    AUTH_TYPE_OIDC,
    Principal,
    get_current_user,
)
from src.observability import metrics
from src.observability.metrics import RATE_LIMIT_METRIC, build_meter_provider
from src.services import request_rate_limiter as rrl
from src.services.request_rate_limiter import (
    ALLOWED,
    LIMITED,
    SKIPPED_STORE_DOWN,
    UNLIMITED,
    RequestRateLimiter,
    bucket_key,
    bucket_ttl_ms,
    check_or_raise,
    enforce_request_rate,
    retry_after_seconds,
    store_url,
)

pytestmark = pytest.mark.no_db

LIMITS = {"chat": {"rate": 60, "burst": 20}, "slow": {"rate": 1, "burst": 20}}
USER_ID = "6ac1aaaaaaaaaaaaaaaaaaaa"
API_KEY_ID = "6ac1bbbbbbbbbbbbbbbbbbbb"
OIDC = Principal(USER_ID, AUTH_TYPE_OIDC)
API_KEY = Principal(USER_ID, AUTH_TYPE_API_KEY, API_KEY_ID)


class FakeStore:
    """The Lua token bucket in Python, on a clock in ms the test moves."""

    def __init__(self) -> None:
        self.now_ms = 1_700_000_000_000
        self.buckets = {}
        self.ttls = {}
        self.calls = 0
        self.error = None
        self.hang = False
        self.scripts = []
        self.pings = 0
        self.ping_errors = []
        self.ping_hang = False
        self.connection_pool = CountingPool(self)

    def register_script(self, source):
        assert "TIME" in source and "PEXPIRE" in source
        self.scripts.append(source)
        return self._script

    async def _script(self, keys, args):
        self.calls += 1
        if self.hang:
            await asyncio.sleep(10)
        if self.error is not None:
            raise self.error
        rate, burst, ttl_ms = args
        rate_per_ms = rate / 60000
        tokens, ts = self.buckets.get(keys[0], (burst, self.now_ms))
        tokens = min(burst, tokens + max(0, self.now_ms - ts) * rate_per_ms)
        if tokens >= 1:
            tokens, allowed, retry_ms = tokens - 1, 1, 0
        else:
            allowed, retry_ms = 0, math.ceil((1 - tokens) / rate_per_ms)
        self.buckets[keys[0]] = (tokens, self.now_ms)
        self.ttls[keys[0]] = ttl_ms
        return [allowed, retry_ms]

    async def ping(self):
        self.pings += 1
        if self.ping_hang:
            await asyncio.sleep(10)
        if self.ping_errors:
            raise self.ping_errors.pop(0)
        return b"PONG"

    async def aclose(self):
        pass


class CountingConnection:
    """A pool connection whose PING answers like ``FakeStore.ping``."""

    def __init__(self, store) -> None:
        self.store = store
        self.sent = []

    async def send_command(self, *args):
        self.sent.append(args)

    async def read_response(self):
        return await self.store.ping()


class CountingPool:
    """Counts the connections the warm-up takes and gives back.

    Past ``open_limit`` connections a new one fails to open, like a store
    that accepts only so many TLS handshakes in time.
    """

    def __init__(self, store, open_limit=None) -> None:
        self.store = store
        self.open_limit = open_limit
        self.opened = 0
        self.released = 0
        self.in_use = 0
        self.max_in_use = 0

    async def get_connection(self):
        await asyncio.sleep(0)
        if self.open_limit is not None and self.opened >= self.open_limit:
            raise _redis_connection_error("Error connecting to 10.0.0.12:6379")
        self.opened += 1
        self.in_use += 1
        self.max_in_use = max(self.max_in_use, self.in_use)
        return CountingConnection(self.store)

    async def release(self, connection):
        self.released += 1
        self.in_use -= 1


class FakeConnection:
    def __init__(self) -> None:
        self.is_connected = False


class FakePool:
    def __init__(self) -> None:
        self._available_connections = [FakeConnection()]


class SlowConnectStore(FakeStore):
    """A pool with one connection: opening it takes 0.6 s, a command 0.01 s."""

    def __init__(self, connect_s=0.6, command_s=0.01) -> None:
        super().__init__()
        self.connection_pool = FakePool()
        self.connect_s = connect_s
        self.command_s = command_s

    async def _script(self, keys, args):
        conn = self.connection_pool._available_connections[0]
        if not conn.is_connected:
            await asyncio.sleep(self.connect_s)
            conn.is_connected = True
        await asyncio.sleep(self.command_s)
        return await super()._script(keys, args)


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def limiter(store=None, **kwargs) -> RequestRateLimiter:
    kwargs.setdefault("limits", LIMITS)
    return RequestRateLimiter(client=store or FakeStore(), **kwargs)


@pytest.fixture
def install(monkeypatch):
    """Turn the feature on with the given limiter as the process one."""

    def _install(instance: RequestRateLimiter) -> RequestRateLimiter:
        monkeypatch.setattr(rrl, "FEATURE_REQUEST_RATE_LIMIT", True)
        monkeypatch.setattr(rrl, "_limiter", instance)
        return instance

    return _install


# --- config ---------------------------------------------------------------


def test_config_defaults_match_the_plan(monkeypatch):
    assert parse_request_rate_limits("") == DEFAULT_REQUEST_RATE_LIMITS
    assert DEFAULT_REQUEST_RATE_LIMITS["chat"] == {"rate": 60, "burst": 20}
    assert DEFAULT_REQUEST_RATE_LIMITS["proxy"] == {"rate": 60, "burst": 60}
    assert set(DEFAULT_REQUEST_RATE_LIMITS) == set(rrl.KNOWN_CLASSES)
    # The parsers with the variables absent, whatever the process env says.
    for name in (
        "FEATURE_REQUEST_RATE_LIMIT",
        "REQUEST_RATE_LIMIT_FAIL_CLOSED",
        "REQUEST_RATE_LIMIT_SKIP_S",
        "REQUEST_RATE_LIMIT_CONNECT_S",
    ):
        monkeypatch.delenv(name, raising=False)
    assert config.getenv_or("FEATURE_REQUEST_RATE_LIMIT").lower() != "true"
    assert config.getenv_or("REQUEST_RATE_LIMIT_FAIL_CLOSED").lower() != "true"
    assert config._tolerant_int_env("REQUEST_RATE_LIMIT_SKIP_S", 30) == 30
    assert config.REQUEST_RATE_LIMIT_SKIP_S >= 1
    assert config._tolerant_positive_float_env("REQUEST_RATE_LIMIT_CONNECT_S", 1.0) == 1.0


@pytest.mark.parametrize("raw", ["abc", "0", "-1", "nan", "inf", " "])
def test_config_invalid_connect_s_keeps_the_default(raw, monkeypatch):
    monkeypatch.setenv("REQUEST_RATE_LIMIT_CONNECT_S", raw)
    assert config._tolerant_positive_float_env("REQUEST_RATE_LIMIT_CONNECT_S", 1.0) == 1.0


def test_config_connect_s_accepts_a_fraction(monkeypatch):
    monkeypatch.setenv("REQUEST_RATE_LIMIT_CONNECT_S", "1.5")
    assert config._tolerant_positive_float_env("REQUEST_RATE_LIMIT_CONNECT_S", 1.0) == 1.5


def test_config_connect_s_is_capped_at_5():
    assert config.REQUEST_RATE_LIMIT_CONNECT_S_MAX == 5.0
    assert 0 < config.REQUEST_RATE_LIMIT_CONNECT_S <= config.REQUEST_RATE_LIMIT_CONNECT_S_MAX


@pytest.mark.parametrize(
    "raw",
    [
        "{not json",
        "[1, 2]",
        '{"chat": {"rate": -1, "burst": 5}}',
        '{"chat": {"rate": 60, "burst": 0}}',
        '{"chat": {"rate": 60}}',
        '{"chat": {"rate": true, "burst": 5}}',
        '{"chat": {"rate": "60", "burst": 5}}',
        '{"chat": 60}',
    ],
)
def test_config_invalid_value_keeps_the_defaults(raw, caplog):
    with caplog.at_level(logging.WARNING, logger="src.config"):
        assert parse_request_rate_limits(raw) == DEFAULT_REQUEST_RATE_LIMITS
    assert "REQUEST_RATE_LIMITS" in caplog.text


def test_config_valid_value_merges_over_the_defaults_per_class():
    limits = parse_request_rate_limits(
        '{"chat": {"rate": 30, "burst": 5}, "upload": {"rate": 0}, "retrieve": {"rate": -1}}'
    )
    assert limits == {
        **DEFAULT_REQUEST_RATE_LIMITS,
        "chat": {"rate": 30, "burst": 5},
        "upload": {"rate": 0, "burst": 0},
    }


def test_config_unknown_class_is_ignored_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="src.config"):
        limits = parse_request_rate_limits('{"chats": {"rate": 1, "burst": 1}}')
    assert limits == DEFAULT_REQUEST_RATE_LIMITS
    assert "unknown class 'chats'" in caplog.text


def test_startup_log_names_the_unlimited_classes(monkeypatch, caplog):
    monkeypatch.setattr(rrl, "FEATURE_REQUEST_RATE_LIMIT", True)
    monkeypatch.setattr(rrl, "REDIS_URL", "")
    monkeypatch.setattr(rrl, "REQUEST_RATE_LIMIT_FAIL_CLOSED", True)
    monkeypatch.setattr(
        rrl,
        "REQUEST_RATE_LIMITS",
        {**DEFAULT_REQUEST_RATE_LIMITS, "upload": {"rate": 0, "burst": 0}},
    )
    with caplog.at_level(logging.INFO, logger=rrl.__name__):
        rrl.log_startup_config()
    assert "unlimited_classes=upload" in caplog.text
    assert "fail_closed=False" in caplog.text
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and "FAIL_CLOSED ignored" in errors[0].getMessage()


async def test_fail_closed_is_ignored_without_redis_url(monkeypatch):
    monkeypatch.setattr(rrl, "FEATURE_REQUEST_RATE_LIMIT", True)
    monkeypatch.setattr(rrl, "REDIS_URL", "")
    monkeypatch.setattr(rrl, "REQUEST_RATE_LIMIT_FAIL_CLOSED", True)
    monkeypatch.setattr(rrl, "_limiter", None)
    lim = rrl.get_request_rate_limiter()
    assert lim.fail_closed is False
    assert (await lim.check("user:a", "chat")).allowed is True


@pytest.mark.parametrize(
    "raw, mode",
    [("enforce", "enforce"), (" ENFORCE ", "enforce"), ("shadow", "shadow"), ("off", "shadow"), ("", "shadow")],
)
def test_config_mode_is_shadow_unless_enforce(raw, mode):
    assert parse_request_rate_limit_mode(raw) == mode


async def test_config_missing_class_and_rate_zero_are_unlimited():
    store = FakeStore()
    lim = limiter(store, limits={"chat": {"rate": 0, "burst": 0}})
    for cls in ("chat", "retrieve"):
        decision = await lim.check("user:x", cls)
        assert decision.allowed and decision.reason == UNLIMITED
    assert store.calls == 0


# --- bucket ---------------------------------------------------------------


async def test_bucket_burst_then_refill_is_exact_over_30_calls():
    store = FakeStore()
    lim = limiter(store)
    decisions = [await lim.check("user:a", "chat") for _ in range(30)]
    assert sum(d.allowed for d in decisions) == 20
    assert [d.reason for d in decisions[20:]] == [LIMITED] * 10
    # 60 per minute: the next token is one second away.
    assert {d.retry_after_s for d in decisions[20:]} == {1}
    store.now_ms += 999
    assert (await lim.check("user:a", "chat")).allowed is False
    store.now_ms += 1
    assert (await lim.check("user:a", "chat")).allowed is True
    assert (await lim.check("user:a", "chat")).allowed is False
    # Another subject and another class are separate buckets.
    assert (await lim.check("user:b", "chat")).allowed is True
    assert (await lim.check("user:a", "slow")).allowed is True


async def test_bucket_refill_never_exceeds_the_burst():
    store = FakeStore()
    lim = limiter(store)
    store.now_ms += 3_600_000
    decisions = [await lim.check("user:a", "chat") for _ in range(21)]
    assert sum(d.allowed for d in decisions) == 20


async def test_bucket_retry_after_is_the_time_to_the_next_token():
    lim = limiter()
    for _ in range(20):
        await lim.check("user:a", "slow")
    decision = await lim.check("user:a", "slow")
    assert decision == rrl.Decision(False, 60, LIMITED)


def test_bucket_key_ttl_and_retry_after_bounds():
    assert bucket_key("user:abc", "chat") == "eve:rl:{user:abc}:chat"
    assert bucket_ttl_ms(60, 20) == 80_000
    assert bucket_ttl_ms(1, 20) == (1200 + 60) * 1000
    assert retry_after_seconds(0) == 1
    assert retry_after_seconds(1001) == 2
    assert retry_after_seconds(600_000) == 60


async def test_bucket_one_script_call_per_check_with_the_ttl():
    store = FakeStore()
    lim = limiter(store)
    await lim.check("user:a", "chat")
    assert store.calls == 1
    assert store.ttls == {"eve:rl:{user:a}:chat": 80_000}
    assert len(store.scripts) == 1


@pytest.mark.parametrize(
    "url, expected",
    [
        ("redis://127.0.0.1:6379/0", "redis://127.0.0.1:6379/1"),
        ("rediss://cache.internal:6379", "rediss://cache.internal:6379/1"),
        ("redis://h:6379/0?db=0&ssl_cert_reqs=none", "redis://h:6379/1?ssl_cert_reqs=none"),
        ("unix:///var/run/redis.sock?db=0", "unix:///var/run/redis.sock?db=1"),
    ],
)
def test_store_url_points_at_database_1(url, expected):
    assert store_url(url) == expected


# --- store down -------------------------------------------------------------


def _redis_connection_error(message):
    from redis.exceptions import ConnectionError as RedisConnectionError

    return RedisConnectionError(message)


@pytest.mark.parametrize(
    "error",
    [
        ConnectionError("refused by 10.0.0.12:6379"),
        OSError("socket closed by 10.0.0.12"),
        _redis_connection_error("Error connecting to 10.0.0.12:6379"),
    ],
)
async def test_store_connection_error_opens_the_skip_window(error, caplog):
    store, clock = FakeStore(), Clock()
    store.error = error
    lim = limiter(store, clock=clock, skip_s=30)
    with caplog.at_level(logging.WARNING, logger=rrl.__name__):
        decisions = [await lim.check("user:a", "chat") for _ in range(5)]
    assert all(d.allowed and d.reason == SKIPPED_STORE_DOWN for d in decisions)
    assert store.calls == 1
    lines = [
        (r.levelno, r.getMessage())
        for r in caplog.records
        if r.getMessage().startswith("rate_limit.skipped_store_down")
    ]
    assert lines == [
        (
            logging.WARNING,
            f"rate_limit.skipped_store_down class=chat skip_s=30 type={type(error).__name__}",
        )
    ]
    assert "10.0.0.12" not in caplog.text
    # After the window the store is tried again, and recovers.
    store.error = None
    clock.t += 30
    assert (await lim.check("user:a", "chat")).reason == ALLOWED
    assert store.calls == 2


@pytest.mark.parametrize(
    "error",
    [TimeoutError(), _redis_connection_error("No connection available.")],
)
async def test_store_deadline_or_busy_pool_fails_open_for_that_request_only(error, caplog):
    store, clock = FakeStore(), Clock()
    store.error = error
    lim = limiter(store, clock=clock)
    with caplog.at_level(logging.WARNING, logger=rrl.__name__):
        for _ in range(2):
            assert (await lim.check("user:a", "chat")).reason == SKIPPED_STORE_DOWN
        store.error = None
        assert (await lim.check("user:a", "chat")).reason == ALLOWED
        store.error = error
        # Two more do not open the window: the success reset the count.
        for _ in range(2):
            await lim.check("user:a", "chat")
    assert store.calls == 5
    assert lim.skip_until == 0.0
    # Visible to the store-down filter, once per class and window.
    assert [r.getMessage() for r in caplog.records] == [
        f"rate_limit.skipped_store_down class=chat skip_s=0 type={type(error).__name__}"
    ]
    assert caplog.records[0].levelno == logging.WARNING


async def test_store_fail_open_line_is_sampled_per_class_and_window(caplog):
    store, clock = FakeStore(), Clock()
    store.error = TimeoutError()
    lim = limiter(
        store,
        clock=clock,
        limits={"chat": {"rate": 60, "burst": 20}, "errlog": {"rate": 30, "burst": 30}},
    )

    async def two_failures_then_success(route_class):
        store.error = TimeoutError()
        for _ in range(2):
            await lim.check("user:a", route_class)
        store.error = None
        await lim.check("user:a", route_class)

    with caplog.at_level(logging.WARNING, logger=rrl.__name__):
        for _ in range(3):
            await two_failures_then_success("chat")
        await two_failures_then_success("errlog")
        clock.t += rrl.LIMITED_LOG_WINDOW_S
        await two_failures_then_success("chat")
    assert [r.getMessage() for r in caplog.records] == [
        "rate_limit.skipped_store_down class=chat skip_s=0 type=TimeoutError",
        "rate_limit.skipped_store_down class=errlog skip_s=0 type=TimeoutError",
        "rate_limit.skipped_store_down class=chat skip_s=0 type=TimeoutError",
    ]
    assert lim.skip_until == 0.0


async def test_store_three_failures_in_a_row_open_the_window(caplog):
    store, clock = FakeStore(), Clock()
    store.error = TimeoutError()
    lim = limiter(store, clock=clock, skip_s=30)
    with caplog.at_level(logging.WARNING, logger=rrl.__name__):
        for _ in range(5):
            await lim.check("user:a", "chat")
    assert store.calls == 3
    assert lim.skip_until == clock.t + 30
    assert (
        caplog.text.count(
            "rate_limit.skipped_store_down class=chat skip_s=30 type=TimeoutError"
        )
        == 1
    )


async def test_store_hang_is_bounded_by_the_timeout():
    store = FakeStore()
    store.hang = True
    lim = limiter(store, timeout_s=0.05)
    loop = asyncio.get_running_loop()
    started = loop.time()
    decision = await lim.check("user:a", "chat")
    assert loop.time() - started < 1
    assert decision.reason == SKIPPED_STORE_DOWN and decision.allowed


async def test_store_first_connection_gets_the_connect_deadline():
    store = SlowConnectStore(connect_s=0.6, command_s=0.01)
    lim = limiter(store, timeout_s=0.25, connect_timeout_s=1.0)
    assert lim._deadline_s() == 1.25
    assert (await lim.check("user:a", "chat")).reason == ALLOWED
    assert (await lim.check("user:a", "chat")).reason == ALLOWED
    # Past connect plus command the check fails open.
    store.command_s = 3.0
    loop = asyncio.get_running_loop()
    started = loop.time()
    assert (await lim.check("user:a", "chat")).reason == SKIPPED_STORE_DOWN
    assert loop.time() - started < 2.5


async def test_store_first_connection_times_out_without_the_connect_deadline(caplog):
    # The dev finding: with one 0.25 s deadline the first connection never opens.
    store = SlowConnectStore(connect_s=0.6, command_s=0.01)
    lim = limiter(store, timeout_s=0.25, connect_timeout_s=0.25)
    with caplog.at_level(logging.WARNING, logger=rrl.__name__):
        assert (await lim.check("user:a", "chat")).reason == SKIPPED_STORE_DOWN
    assert "rate_limit.skipped_store_down class=chat skip_s=0 type=TimeoutError" in caplog.text


def test_store_client_opens_connections_with_the_connect_timeout():
    client = rrl._build_client("rediss://cache.example.invalid:6379/0", 0.25, 10, 1.0)
    pool = client.connection_pool
    assert pool.connection_kwargs["socket_connect_timeout"] == 1.0
    assert pool.connection_kwargs["socket_timeout"] == 0.25
    assert pool.connection_kwargs["db"] == 1
    assert pool.timeout == 0.25
    assert pool._available_connections == []


def test_store_connect_deadline_is_never_below_the_command_deadline():
    lim = limiter(FakeStore(), timeout_s=0.25, connect_timeout_s=0.1)
    assert lim.connect_timeout_s == 0.25


def test_process_limiter_takes_the_connect_knob(monkeypatch):
    monkeypatch.setattr(rrl, "FEATURE_REQUEST_RATE_LIMIT", True)
    monkeypatch.setattr(rrl, "REQUEST_RATE_LIMIT_CONNECT_S", 1.5)
    monkeypatch.setattr(rrl, "_limiter", None)
    assert rrl.get_request_rate_limiter().connect_timeout_s == 1.5


# --- warm-up ----------------------------------------------------------------


async def test_warm_up_opens_the_whole_pool_at_once_and_releases_it(caplog):
    store = FakeStore()
    lim = limiter(store, max_connections=10)
    with caplog.at_level(logging.INFO, logger=rrl.__name__):
        assert await lim.warm_up() == 10
    pool = store.connection_pool
    # Ten connections held together, one PING each, all back in the pool.
    assert pool.opened == 10 and pool.max_in_use == 10
    assert pool.released == 10 and pool.in_use == 0
    assert store.pings == 10
    lines = [(r.levelno, r.getMessage()) for r in caplog.records]
    assert len(lines) == 1 and lines[0][0] == logging.INFO
    assert re.fullmatch(
        r"rate_limit\.store_ready latency_ms=\d+ connections=10", lines[0][1]
    )


async def test_warm_up_partial_pool_logs_the_real_count(caplog):
    store = FakeStore()
    store.connection_pool = CountingPool(store, open_limit=4)
    lim = limiter(store, max_connections=10)
    with caplog.at_level(logging.INFO, logger=rrl.__name__):
        assert await lim.warm_up(deadline_s=0.5) == 4
    pool = store.connection_pool
    assert pool.opened == 4 and pool.released == 4 and pool.in_use == 0
    assert [r.getMessage().split(" connections=")[1] for r in caplog.records] == ["4"]
    assert "skipped_store_down" not in caplog.text


async def test_warm_up_retries_within_the_deadline(caplog):
    store = FakeStore()
    store.ping_errors = [_redis_connection_error("Error connecting to 10.0.0.12:6379")]
    lim = limiter(store)
    with caplog.at_level(logging.INFO, logger=rrl.__name__):
        assert await lim.warm_up(deadline_s=2.0) == lim.max_connections
    assert store.pings == lim.max_connections + 1
    assert store.connection_pool.in_use == 0
    assert "rate_limit.store_ready" in caplog.text
    assert "skipped_store_down" not in caplog.text


async def test_warm_up_failure_logs_the_startup_line_and_never_raises(caplog):
    store = FakeStore()
    store.ping_errors = [_redis_connection_error("Error connecting to 10.0.0.12:6379")] * 500
    lim = limiter(store)
    loop = asyncio.get_running_loop()
    started = loop.time()
    with caplog.at_level(logging.INFO, logger=rrl.__name__):
        assert await lim.warm_up(deadline_s=0.5) == 0
    assert loop.time() - started < 2.5
    assert store.connection_pool.in_use == 0
    assert [(r.levelno, r.getMessage()) for r in caplog.records] == [
        (
            logging.WARNING,
            "rate_limit.skipped_store_down class=startup skip_s=0 type=ConnectionError",
        )
    ]
    assert "10.0.0.12" not in caplog.text
    # A failed warm-up opens no skip window: requests still try the store.
    assert lim.skip_until == 0.0


async def test_warm_up_hang_is_bounded_by_its_deadline(caplog):
    store = FakeStore()
    store.ping_hang = True
    lim = limiter(store)
    with caplog.at_level(logging.WARNING, logger=rrl.__name__):
        assert await lim.warm_up(deadline_s=0.2) == 0
    # The hung PINGs were cancelled and their connections given back.
    assert store.connection_pool.in_use == 0
    assert "rate_limit.skipped_store_down class=startup skip_s=0 type=TimeoutError" in caplog.text


@pytest.mark.parametrize("feature, url", [(False, "redis://x:6379/0"), (True, "")])
async def test_warm_up_is_a_no_op_when_off_or_without_redis_url(feature, url, monkeypatch, caplog):
    store = FakeStore()
    calls = []
    monkeypatch.setattr(rrl, "FEATURE_REQUEST_RATE_LIMIT", feature)
    monkeypatch.setattr(rrl, "REDIS_URL", url)
    monkeypatch.setattr(
        rrl, "get_request_rate_limiter", lambda: calls.append(1) or limiter(store)
    )
    with caplog.at_level(logging.DEBUG, logger=rrl.__name__):
        await rrl.warm_up_request_rate_limiter()
    assert calls == [] and store.pings == 0
    assert [r for r in caplog.records if r.name == rrl.__name__] == []


def test_warm_up_deadline_is_at_most_3_s():
    assert 0 < rrl.WARM_UP_DEADLINE_S <= 3


async def test_warm_up_runs_for_the_process_limiter(monkeypatch, caplog):
    store = FakeStore()
    monkeypatch.setattr(rrl, "FEATURE_REQUEST_RATE_LIMIT", True)
    monkeypatch.setattr(rrl, "REDIS_URL", "redis://x:6379/0")
    monkeypatch.setattr(rrl, "_limiter", limiter(store))
    with caplog.at_level(logging.INFO, logger=rrl.__name__):
        await rrl.warm_up_request_rate_limiter()
    assert store.pings == rrl.STORE_MAX_CONNECTIONS
    assert f"connections={rrl.STORE_MAX_CONNECTIONS}" in caplog.text


async def test_store_down_fail_closed_refuses():
    store = FakeStore()
    store.error = ConnectionError("down")
    lim = limiter(store, fail_closed=True)
    decision = await lim.check("user:a", "chat")
    assert decision.allowed is False and decision.reason == SKIPPED_STORE_DOWN


async def test_store_missing_url_reads_as_store_down():
    lim = RequestRateLimiter(limits=LIMITS, url="")
    assert (await lim.check("user:a", "chat")).reason == SKIPPED_STORE_DOWN


async def test_store_no_client_when_the_feature_is_off(monkeypatch):
    built = []
    monkeypatch.setattr(rrl, "_build_client", lambda *a: built.append(a))
    monkeypatch.setattr(rrl, "FEATURE_REQUEST_RATE_LIMIT", False)
    monkeypatch.setattr(rrl, "_limiter", None)
    assert rrl.get_request_rate_limiter() is None
    assert await check_or_raise(OIDC, "chat") is None
    assert built == [] and rrl._limiter is None


# --- contract ---------------------------------------------------------------


def _app(route_class="chat") -> FastAPI:
    app = FastAPI()

    @app.post("/limited", dependencies=[Depends(enforce_request_rate(route_class))])
    async def limited():
        return {"ok": True}

    return app


def _principal_override(app: FastAPI, principal: Principal) -> None:
    user = User(id=principal.user_id, email="someone@example.com")

    async def _user():
        return user

    app.dependency_overrides[get_current_user] = _user
    app.state.bearer = (
        "eve_" + (principal.api_key_id or "key")
        if principal.auth_type == AUTH_TYPE_API_KEY
        else "header.payload.signature"
    )


async def _burst(app: FastAPI, n: int):
    headers = {"Authorization": f"Bearer {app.state.bearer}"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        return [await client.post("/limited", headers=headers) for _ in range(n)]


async def test_contract_shadow_logs_and_lets_every_request_through(install, caplog):
    install(limiter(mode="shadow"))
    app = _app()
    _principal_override(app, OIDC)
    with caplog.at_level(logging.INFO, logger=rrl.__name__):
        responses = await _burst(app, 25)
    assert [r.status_code for r in responses] == [200] * 25
    lines = [
        (r.levelno, r.getMessage())
        for r in caplog.records
        if r.getMessage().startswith("rate_limit.limited")
    ]
    # Five refusals, one line: the first refusal of the window.
    assert lines == [
        (logging.INFO, "rate_limit.limited subject_kind=user class=chat mode=shadow retry_after_s=1")
    ]


async def test_contract_enforce_answers_429_with_retry_after(install, caplog):
    install(limiter(mode="enforce"))
    app = _app("slow")
    _principal_override(app, API_KEY)
    with caplog.at_level(logging.INFO, logger=rrl.__name__):
        responses = await _burst(app, 22)
    assert [
        (r.levelno, r.getMessage())
        for r in caplog.records
        if r.getMessage().startswith("rate_limit.limited")
    ] == [
        (logging.WARNING, "rate_limit.limited subject_kind=api_key class=slow mode=enforce retry_after_s=60")
    ]
    assert [r.status_code for r in responses[:20]] == [200] * 20
    for refused in responses[20:]:
        assert refused.status_code == 429
        assert refused.json()["detail"] == {
            "code": "rate_limited",
            "message": "Too many requests, retry in 60 s",
        }
        assert refused.headers["Retry-After"] == "60"
        assert 1 <= int(refused.headers["Retry-After"]) <= 60


async def test_contract_one_bucket_per_user_across_credentials(install):
    install(limiter(mode="enforce"))
    for principal in (OIDC, API_KEY):
        app = _app("slow")
        _principal_override(app, principal)
        await _burst(app, 10)
    app = _app("slow")
    _principal_override(app, Principal(USER_ID, AUTH_TYPE_API_KEY, "another-key"))
    assert (await _burst(app, 1))[0].status_code == 429


@pytest.mark.parametrize("mode, status", [("enforce", 503), ("shadow", 200)])
async def test_contract_fail_closed_is_503_in_enforce_only(install, mode, status):
    store = FakeStore()
    store.error = ConnectionError("down")
    install(limiter(store, mode=mode, fail_closed=True))
    app = _app()
    _principal_override(app, OIDC)
    response = (await _burst(app, 1))[0]
    assert response.status_code == status
    if status == 503:
        assert response.json()["detail"]["code"] == "limiter_unavailable"


async def test_contract_feature_off_never_touches_the_store(monkeypatch):
    store = FakeStore()
    monkeypatch.setattr(rrl, "FEATURE_REQUEST_RATE_LIMIT", False)
    monkeypatch.setattr(rrl, "_limiter", limiter(store, mode="enforce"))
    app = _app()
    _principal_override(app, OIDC)
    assert {r.status_code for r in await _burst(app, 25)} == {200}
    assert store.calls == 0


async def test_contract_caller_resolved_once_with_get_current_user(install, monkeypatch):
    """The key's last_used_at stamp runs once, not once per dependency."""
    install(limiter(mode="enforce"))
    calls = []
    user = User(id=USER_ID, email="someone@example.com")

    async def fake_lookup(token):
        calls.append(token)
        return user, API_KEY_ID

    monkeypatch.setattr(auth_module, "_get_user_from_api_key", fake_lookup)
    app = FastAPI()

    @app.post("/limited", dependencies=[Depends(enforce_request_rate("chat"))])
    async def limited(current: User = Depends(get_current_user)):
        return {"id": current.id}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        response = await client.post("/limited", headers={"Authorization": "Bearer eve_test"})
    assert response.status_code == 200 and response.json() == {"id": USER_ID}
    assert calls == ["eve_test"]


async def test_contract_subject_kind_follows_the_bearer(install, caplog):
    for principal, kind in ((OIDC, "user"), (API_KEY, "api_key")):
        install(limiter(mode="shadow"))
        app = _app("slow")
        _principal_override(app, principal)
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=rrl.__name__):
            await _burst(app, 21)
        assert f"subject_kind={kind} " in caplog.text


async def test_contract_check_or_raise_for_the_dispatchers(install):
    install(limiter(mode="enforce"))
    for _ in range(20):
        assert (await check_or_raise(API_KEY, "slow")).allowed
    with pytest.raises(rrl.RequestRateLimited) as info:
        await check_or_raise(API_KEY, "slow")
    assert info.value.status_code == 429 and info.value.retry_after_s == 60


# --- counter and logs ---------------------------------------------------------


def _counter_points(reader):
    data = reader.get_metrics_data()
    return [
        point
        for rm in data.resource_metrics
        for sm in rm.scope_metrics
        for metric in sm.metrics
        if metric.name == RATE_LIMIT_METRIC
        for point in metric.data.data_points
    ]


async def test_counter_attributes(install, monkeypatch):
    monkeypatch.setattr(metrics, "_rate_limit_counter", None)
    reader = InMemoryMetricReader()
    provider = build_meter_provider(
        Resource.create({"service.name": "eve-backend-test"}), reader=reader
    )
    try:
        install(limiter(mode="shadow"))
        for _ in range(21):
            await check_or_raise(API_KEY, "chat")
        await check_or_raise(OIDC, "unlisted")
        points = {
            tuple(sorted(p.attributes.items())): p.value for p in _counter_points(reader)
        }
    finally:
        provider.shutdown()
    base = {"class": "chat", "mode": "shadow", "subject_kind": "api_key"}
    assert points == {
        tuple(sorted({**base, "decision": "allowed"}.items())): 20,
        tuple(sorted({**base, "decision": "limited"}.items())): 1,
    }


async def test_limited_line_once_per_subject_class_and_window(install, caplog):
    clock = Clock()
    install(limiter(mode="enforce", clock=clock))
    other = Principal("6ac1cccccccccccccccccccc", AUTH_TYPE_OIDC)

    async def refuse(principal, cls, n):
        for _ in range(n):
            with pytest.raises(rrl.RequestRateLimited):
                await check_or_raise(principal, cls)

    with caplog.at_level(logging.INFO, logger=rrl.__name__):
        for principal in (OIDC, other):
            for _ in range(20):
                await check_or_raise(principal, "slow")
        await refuse(OIDC, "slow", 50)
        await refuse(other, "slow", 5)
        clock.t += rrl.LIMITED_LOG_WINDOW_S - 1
        await refuse(OIDC, "slow", 5)
        clock.t += 1
        await refuse(OIDC, "slow", 5)
    lines = [r for r in caplog.records if r.getMessage().startswith("rate_limit.limited")]
    # OIDC twice (two windows), the other user once; 65 refusals in all.
    assert len(lines) == 3


def test_counter_is_a_no_op_without_a_meter_provider(monkeypatch):
    monkeypatch.setattr(metrics, "_rate_limit_counter", None)
    metrics.record_rate_limit_decision("chat", "allowed", "shadow", "user")


async def test_logs_carry_no_pii(install, caplog):
    store = FakeStore()
    install(limiter(store, mode="shadow"))
    with caplog.at_level(logging.DEBUG, logger=rrl.__name__):
        for principal in (OIDC, API_KEY):
            for _ in range(25):
                await check_or_raise(principal, "chat")
        store.error = ConnectionError("Error connecting to 10.1.2.3:6379")
        rrl._limiter.skip_until = 0
        await check_or_raise(OIDC, "chat")
    text = caplog.text
    assert "rate_limit.limited" in text and "rate_limit.skipped_store_down" in text
    assert USER_ID not in text and API_KEY_ID not in text
    assert not re.search(r"@|eve_[0-9a-f]{8}|eyJ|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", text)


# --- real store -------------------------------------------------------------


async def _real_limiter(**kwargs):
    url = os.getenv("REDIS_URL", "")
    if not url:
        pytest.skip("REDIS_URL not set")
    lim = RequestRateLimiter(url=url, timeout_s=1.0, **kwargs)
    probe = rrl._build_client(url, 1.0, 2)
    try:
        await asyncio.wait_for(probe.ping(), 1.0)
    except Exception:
        pytest.skip("no Redis answering at REDIS_URL")
    finally:
        await probe.aclose()
    return lim


async def test_real_store_exact_under_concurrency_in_database_1():
    lim = await _real_limiter(limits=LIMITS, max_connections=10)
    subject = f"user:test-{uuid.uuid4().hex}"
    try:
        decisions = await asyncio.gather(*(lim.check(subject, "slow") for _ in range(30)))
        assert sum(d.allowed for d in decisions) == 20
        assert {d.retry_after_s for d in decisions if not d.allowed} == {60}
        client = lim._client
        assert client.connection_pool.connection_kwargs["db"] == 1
        key = bucket_key(subject, "slow")
        ttl = await client.pttl(key)
        assert 0 < ttl <= bucket_ttl_ms(1, 20)
        # The script is reloaded after a flush (failover, restart).
        await client.script_flush()
        assert (await lim.check(subject, "slow")).reason == LIMITED
        assert (await lim.check(f"{subject}-2", "slow")).reason == ALLOWED
    finally:
        if lim._client is not None:
            await lim._client.delete(
                bucket_key(subject, "slow"), bucket_key(f"{subject}-2", "slow")
            )
        await lim.aclose()


async def test_real_store_refills_from_the_server_clock():
    lim = await _real_limiter(limits={"fast": {"rate": 60000, "burst": 2}})
    subject = f"user:test-{uuid.uuid4().hex}"
    try:
        assert [(await lim.check(subject, "fast")).allowed for _ in range(2)] == [True, True]
        await asyncio.sleep(0.01)
        # 1000 per second: 10 ms of server time refills the bucket.
        assert (await lim.check(subject, "fast")).allowed is True
    finally:
        if lim._client is not None:
            await lim._client.delete(bucket_key(subject, "fast"))
        await lim.aclose()


async def test_real_store_busy_pool_fails_open_without_the_window():
    lim = await _real_limiter(limits=LIMITS, max_connections=1)
    subject = f"user:test-{uuid.uuid4().hex}"
    try:
        assert (await lim.check(subject, "chat")).reason == ALLOWED
        pool = lim._client.connection_pool
        held = await pool.get_connection()
        try:
            loop = asyncio.get_running_loop()
            started = loop.time()
            decision = await lim.check(subject, "chat")
            assert loop.time() - started < 1.5
            assert decision.reason == SKIPPED_STORE_DOWN and decision.allowed
            assert lim.skip_until == 0.0
        finally:
            await pool.release(held)
        assert (await lim.check(subject, "chat")).reason == ALLOWED
    finally:
        if lim._client is not None:
            await lim._client.delete(bucket_key(subject, "chat"))
        await lim.aclose()


async def test_real_store_warm_up_leaves_the_whole_pool_open(caplog):
    lim = await _real_limiter(limits=LIMITS, connect_timeout_s=1.0, max_connections=10)
    subject = f"user:test-{uuid.uuid4().hex}"
    try:
        with caplog.at_level(logging.INFO, logger=rrl.__name__):
            assert await lim.warm_up() == 10
        assert "connections=10" in caplog.text
        pool = lim._client.connection_pool
        assert pool.connection_kwargs["socket_connect_timeout"] == 1.0
        idle = list(pool._available_connections)
        assert len(idle) == 10 and all(c.is_connected for c in idle)
        assert len(pool._in_use_connections) == 0
        assert (await lim.check(subject, "chat")).reason == ALLOWED
    finally:
        if lim._client is not None:
            await lim._client.delete(bucket_key(subject, "chat"))
        await lim.aclose()


async def test_real_store_dropped_connection_last_in_the_pool_reconnects_in_time():
    # redis-py hands out the tail of the idle list and puts a connection that
    # was dropped (command cancelled mid-read) back at the tail: with an open
    # connection first and the dropped one last, the check must reopen it.
    lim = await _real_limiter(limits=LIMITS, connect_timeout_s=1.0, max_connections=2)
    lim.timeout_s = 0.25
    subject = f"user:test-{uuid.uuid4().hex}"
    try:
        pool = lim._get_client().connection_pool
        opened = await pool.get_connection()
        dropped = await pool.get_connection()
        await pool.release(opened)
        await pool.release(dropped)
        await dropped.disconnect()
        assert pool._available_connections == [opened, dropped]
        assert opened.is_connected and not dropped.is_connected
        real_connect = dropped._connect

        async def slow_connect():
            # A TLS handshake slower than the 0.25 s command deadline.
            await asyncio.sleep(0.6)
            await real_connect()

        dropped._connect = slow_connect
        loop = asyncio.get_running_loop()
        started = loop.time()
        decision = await lim.check(subject, "chat")
        elapsed = loop.time() - started
        assert decision.reason == ALLOWED
        assert 0.6 <= elapsed < lim._deadline_s() + 1.0
        assert dropped.is_connected
        assert lim.consecutive_failures == 0
    finally:
        if lim._client is not None:
            await lim._client.delete(bucket_key(subject, "chat"))
        await lim.aclose()
