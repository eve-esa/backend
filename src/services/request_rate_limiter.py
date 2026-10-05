"""Per-principal request rate limit: one token bucket per user and route class.

The bucket lives in Valkey so every worker of every task spends from the same
one: a user with two browser tabs and three API keys has one bucket per class,
not one per credential or per worker. One Lua script call per request reads
the bucket, refills it from the server clock (``TIME``), takes a token or not
and writes it back, so concurrent requests can never both take the last token.

Off by default (``FEATURE_REQUEST_RATE_LIMIT``): no Redis client is built and
:func:`check_or_raise` returns at once. In ``shadow`` mode a refusal is logged
and counted but the request goes through; ``enforce`` answers 429
``rate_limited`` with ``Retry-After``.

Fail open: a store failure allows the request and marks it
``skipped_store_down``. A deadline overrun or a busy connection pool fails
open for that request only, since event loop lag under load looks the same;
it logs ``rate_limit.skipped_store_down class=<c> skip_s=0
type=<ExceptionClass>`` at WARNING, once per class and 60 s window of a
worker. A connection or socket error, or three failures in a row of any kind,
makes this worker skip the store for ``REQUEST_RATE_LIMIT_SKIP_S`` seconds,
with one WARNING per window (the same line with ``skip_s=<n>``). Every fail
open is therefore visible to the store-down metric filter in infra.
``REQUEST_RATE_LIMIT_FAIL_CLOSED`` turns a skipped check into 503
``limiter_unavailable`` in enforce mode (ignored without ``REDIS_URL``).

Deadlines: a new connection to ElastiCache costs a TCP and TLS handshake,
which alone can take longer than 0.25 s, and a check cannot know cheaply
whether the connection the pool hands out is open (redis-py drops one whose
command was cancelled and hands it out again). So every check runs under one
deadline of ``REQUEST_RATE_LIMIT_CONNECT_S`` (default 1 s, also the pool's
``socket_connect_timeout``) plus 0.25 s, while the pool wait and every socket
read (AUTH, SELECT, the script) stay bounded by 0.25 s each. On an open
connection a check therefore still fails within 0.25 s of a stalled read; only
a reconnect may use the extra second. At startup the lifespan starts a
background task that opens the pool ahead of traffic but for two slots
(:func:`start_request_rate_limiter_warm_up`: a PING on database 1 per
connection, at most two handshakes in flight per worker so the workers of a
small task do not pile TLS handshakes onto one vCPU, retried within 10 s),
since a cold burst would otherwise open them all under the check deadline.
The warm-up holds its connections until it is done, because the pool hands an
idle connection back before it opens a new one; the two slots it leaves free
are the ones a check arriving meanwhile takes. It
logs ``rate_limit.store_ready latency_ms=<n> connections=<k>`` at INFO, or
``rate_limit.skipped_store_down class=startup skip_s=0 type=<ExceptionClass>``
at WARNING. Readiness never waits for it, and a check that arrives meanwhile
runs under the usual connect plus command deadline.

A refusal logs ``rate_limit.limited subject_kind=<user|api_key> class=<c>
mode=<shadow|enforce> retry_after_s=<n>``, INFO in shadow and WARNING in
enforce, for the first refusal of a subject and class in each 60 s window of
a worker only: a flood must not turn into one log line per request. The
CloudWatch alarm built on that line therefore counts limited subject windows
per worker, not refused requests; the ``eve.rate_limit.decisions`` counter
counts every decision. Both log lines are matched by CloudWatch metric
filters in infra: change them only together. Log lines and metric attributes
carry the route class and whether the caller used a session or an API key,
never a user id, key, token or address.
"""

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from src.config import (
    FEATURE_REQUEST_RATE_LIMIT,
    REDIS_URL,
    REQUEST_RATE_LIMIT_CONNECT_S,
    REQUEST_RATE_LIMIT_FAIL_CLOSED,
    REQUEST_RATE_LIMIT_MODE,
    REQUEST_RATE_LIMIT_SKIP_S,
    REQUEST_RATE_LIMITS,
)
from src.database.models.user import User
from src.middlewares.auth import (
    AUTH_TYPE_API_KEY,
    AUTH_TYPE_OIDC,
    Principal,
    get_current_user,
    security,
)
from src.observability.metrics import record_rate_limit_decision

logger = logging.getLogger(__name__)

# Database 1: the stream bus, the Stop channel and the breaker use 0, so the
# limiter keys never mix with theirs.
STORE_DB = 1
STORE_TIMEOUT_S = 0.25
# Total time the startup warm-up may take, retries included.
WARM_UP_DEADLINE_S = 10.0
# Connections the warm-up opens at the same time, per worker.
WARM_UP_CONCURRENCY = 2
# Pool slots the warm-up leaves free for checks while it holds the rest.
WARM_UP_FREE_SLOTS = 2
WARM_UP_RETRY_PAUSE_S = 0.2
STORE_MAX_CONNECTIONS = 10
RETRY_AFTER_MIN_S = 1
RETRY_AFTER_MAX_S = 60
SPAN_ATTRIBUTE = "eve.rate_limit.decision"
# Failures in a row of any kind that open the skip window.
STORE_DOWN_AFTER_FAILURES = 3
# One rate_limit.limited line per subject and class per window, per worker;
# the same window samples the per-request skipped_store_down line per class.
LIMITED_LOG_WINDOW_S = 60
# Bound on the remembered subjects; expired entries go first.
LIMITED_LOG_MAX_SUBJECTS = 4096
KNOWN_CLASSES = ("chat", "retrieve", "proxy", "mcp", "upload", "errlog")

ALLOWED = "allowed"
LIMITED = "limited"
SKIPPED_STORE_DOWN = "skipped_store_down"
# Not a decision: the class has no limit, nothing is counted.
UNLIMITED = "unlimited"

RATE_LIMITED_CODE = "rate_limited"
LIMITER_UNAVAILABLE_DETAIL = {
    "code": "limiter_unavailable",
    "message": "The service cannot check request limits, retry in a few seconds",
}

# KEYS[1] bucket hash. ARGV: rate per minute, burst, ttl in ms.
# Returns {allowed (1 or 0), ms until the next token}. TIME is read inside the
# script so every worker refills against one clock, whatever its own says.
TOKEN_BUCKET_LUA = """
local rate_per_ms = tonumber(ARGV[1]) / 60000
local burst = tonumber(ARGV[2])
local ttl_ms = tonumber(ARGV[3])
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
local state = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(state[1])
local ts = tonumber(state[2])
if tokens == nil or ts == nil then
  tokens = burst
  ts = now
end
local elapsed = now - ts
if elapsed < 0 then
  elapsed = 0
end
tokens = math.min(burst, tokens + elapsed * rate_per_ms)
local allowed = 0
local retry_ms = 0
if tokens >= 1 then
  tokens = tokens - 1
  allowed = 1
else
  retry_ms = math.ceil((1 - tokens) / rate_per_ms)
end
redis.call('HSET', KEYS[1], 'tokens', tostring(tokens), 'ts', tostring(now))
redis.call('PEXPIRE', KEYS[1], ttl_ms)
return {allowed, retry_ms}
"""


@dataclass(frozen=True)
class Decision:
    allowed: bool
    retry_after_s: int
    reason: str  # allowed | limited | skipped_store_down | unlimited


def bucket_key(subject: str, route_class: str) -> str:
    """``eve:rl:{user:<id>}:<class>``; the braces are a cluster hash tag."""
    return f"eve:rl:{{{subject}}}:{route_class}"


def bucket_ttl_ms(rate_per_minute: int, burst: int) -> int:
    """Time to refill a full bucket plus 60 s: past it the key holds nothing."""
    return (math.ceil(burst * 60 / rate_per_minute) + 60) * 1000


def _format_seconds(value: float) -> str:
    """``30`` rather than ``30.0`` for a whole number of seconds."""
    return str(int(value)) if float(value).is_integer() else str(value)


def retry_after_seconds(retry_ms: int) -> int:
    return min(RETRY_AFTER_MAX_S, max(RETRY_AFTER_MIN_S, math.ceil(retry_ms / 1000)))


def store_url(url: str, db: int = STORE_DB) -> str:
    """``url`` pointed at database ``db``.

    redis-py lets the URL path and a ``db`` query option win over a ``db``
    keyword, so the database is rewritten in the URL itself.
    """
    parts = urlsplit(url)
    pairs = [(k, v) for k, v in parse_qsl(parts.query) if k != "db"]
    if parts.scheme in ("redis", "rediss"):
        return urlunsplit(
            (parts.scheme, parts.netloc, f"/{db}", urlencode(pairs), parts.fragment)
        )
    # unix://: the path is the socket, the database goes in the query.
    # Built by hand: urlunsplit drops the empty authority of unix:///path.
    pairs.append(("db", str(db)))
    return f"{parts.scheme}://{parts.netloc}{parts.path}?{urlencode(pairs)}"


def opens_skip_window(exc: BaseException) -> bool:
    """True for a connection or socket error: the store itself is unreachable.

    A deadline overrun (``asyncio.wait_for``, a socket read timeout) or a busy
    pool ("No connection available.") is False: under load the event loop
    alone can cause both, and skipping the store then would switch the limit
    off exactly when a flood arrives.
    """
    from redis.exceptions import ConnectionError as RedisConnectionError
    from redis.exceptions import MaxConnectionsError
    from redis.exceptions import TimeoutError as RedisTimeoutError

    if isinstance(exc, (TimeoutError, RedisTimeoutError, MaxConnectionsError)):
        return False
    if isinstance(exc, RedisConnectionError):
        return "No connection available" not in str(exc)
    return isinstance(exc, OSError)


def _build_client(
    url: str,
    timeout_s: float,
    max_connections: int,
    connect_timeout_s: Optional[float] = None,
) -> Any:
    """Async client with short timeouts, no retries and a bounded pool.

    The blocking pool waits up to ``timeout_s`` for a free connection instead
    of failing a burst past ``max_connections`` with "Too many connections",
    which would read as a store outage. Opening a connection (TCP, TLS
    handshake) may take ``connect_timeout_s``; every read takes ``timeout_s``.
    Building it opens no socket.
    """
    from redis.asyncio import BlockingConnectionPool, Redis
    from redis.asyncio.retry import Retry
    from redis.backoff import NoBackoff

    # redis-py retries a failed command by default, which would multiply the
    # stall the timeouts are meant to bound.
    pool = BlockingConnectionPool.from_url(
        store_url(url),
        max_connections=max_connections,
        timeout=timeout_s,
        socket_connect_timeout=connect_timeout_s or timeout_s,
        socket_timeout=timeout_s,
        retry=Retry(NoBackoff(), 0),
    )
    return Redis(connection_pool=pool, retry=Retry(NoBackoff(), 0))


class RequestRateLimiter:
    """Token buckets in one store, with a per-process skip window on errors."""

    def __init__(
        self,
        *,
        limits: Dict[str, Dict[str, int]],
        mode: str = "shadow",
        fail_closed: bool = False,
        skip_s: float = 30,
        url: str = "",
        client: Any = None,
        clock: Callable[[], float] = time.monotonic,
        timeout_s: float = STORE_TIMEOUT_S,
        connect_timeout_s: float = STORE_TIMEOUT_S,
        max_connections: int = STORE_MAX_CONNECTIONS,
    ) -> None:
        self.limits = limits
        self.mode = mode
        self.fail_closed = fail_closed
        self.skip_s = skip_s
        self.url = url
        self.clock = clock
        self.timeout_s = timeout_s
        self.connect_timeout_s = max(connect_timeout_s, timeout_s)
        self.max_connections = max_connections
        self.skip_until = 0.0
        self.consecutive_failures = 0
        self._client = client
        self._script = None
        self._limited_logged: Dict[tuple, float] = {}

    def _limit(self, route_class: str) -> Optional[Dict[str, int]]:
        spec = self.limits.get(route_class)
        if not spec or int(spec.get("rate") or 0) <= 0:
            return None
        return spec

    def _get_client(self) -> Any:
        if self._client is None:
            if not self.url:
                raise ConnectionError("REDIS_URL is not set")
            self._client = _build_client(
                self.url, self.timeout_s, self.max_connections, self.connect_timeout_s
            )
        return self._client

    def _deadline_s(self) -> float:
        """Connect plus command: the pool and socket timeouts bound the rest."""
        return self.connect_timeout_s + self.timeout_s

    def _get_script(self) -> Any:
        if self._script is None:
            self._get_client()
            # register_script reloads the script on NOSCRIPT (failover, flush).
            self._script = self._client.register_script(TOKEN_BUCKET_LUA)
        return self._script

    def _store_down(self) -> Decision:
        return Decision(
            allowed=not self.fail_closed, retry_after_s=0, reason=SKIPPED_STORE_DOWN
        )

    def _log_store_down(self, route_class: str, skip_s: float, exc: BaseException) -> None:
        # Contract with the CloudWatch metric filter (infra): keep the leading
        # token. The exception class name only, never its message, which can
        # carry the host.
        logger.warning(
            "rate_limit.skipped_store_down class=%s skip_s=%s type=%s",
            route_class,
            _format_seconds(skip_s),
            type(exc).__name__,
        )

    def _store_failed(self, exc: BaseException, route_class: str) -> None:
        self.consecutive_failures += 1
        # The exception class only: a redis error message can carry the host.
        logger.debug("rate_limit store error type=%s", type(exc).__name__)
        if not (
            opens_skip_window(exc)
            or self.consecutive_failures >= STORE_DOWN_AFTER_FAILURES
        ):
            # Fail open for this request only: one line per class and window.
            if self._sample(("skipped_store_down", route_class)):
                self._log_store_down(route_class, 0, exc)
            return
        now = self.clock()
        if now < self.skip_until:
            return
        self.skip_until = now + self.skip_s
        self.consecutive_failures = 0
        # One line per skip window.
        self._log_store_down(route_class, self.skip_s, exc)

    def should_log_limited(self, subject: str, route_class: str) -> bool:
        """True for the first refusal of ``subject`` in ``route_class`` per window."""
        return self._sample((subject, route_class))

    def _sample(self, key: tuple) -> bool:
        """True for the first call with ``key`` in each window of this worker."""
        now = self.clock()
        if self._limited_logged.get(key, 0.0) > now:
            return False
        if len(self._limited_logged) >= LIMITED_LOG_MAX_SUBJECTS:
            self._limited_logged = {
                k: until for k, until in self._limited_logged.items() if until > now
            }
            if len(self._limited_logged) >= LIMITED_LOG_MAX_SUBJECTS:
                self._limited_logged.clear()
        self._limited_logged[key] = now + LIMITED_LOG_WINDOW_S
        return True

    async def check(self, subject: str, route_class: str) -> Decision:
        """Take one token for ``subject`` in ``route_class``. Never raises."""
        spec = self._limit(route_class)
        if spec is None:
            return Decision(True, 0, UNLIMITED)
        if self.clock() < self.skip_until:
            return self._store_down()
        rate, burst = int(spec["rate"]), int(spec["burst"])
        try:
            script = self._get_script()
            allowed, retry_ms = await asyncio.wait_for(
                script(
                    keys=[bucket_key(subject, route_class)],
                    args=[rate, burst, bucket_ttl_ms(rate, burst)],
                ),
                timeout=self._deadline_s(),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - any store failure fails open
            self._store_failed(exc, route_class)
            return self._store_down()
        self.consecutive_failures = 0
        if int(allowed) == 1:
            return Decision(True, 0, ALLOWED)
        return Decision(False, retry_after_seconds(int(retry_ms)), LIMITED)

    async def warm_up(self, deadline_s: float = WARM_UP_DEADLINE_S) -> int:
        """Open the store pool but for two slots ahead of traffic. Never raises.

        Takes ``max_connections - WARM_UP_FREE_SLOTS`` connections from the
        request pool, opening at most ``WARM_UP_CONCURRENCY`` at a time, PINGs
        database 1 on each, holds them until all are open (or the deadline)
        and then releases them all; the free slots let a check arriving
        meanwhile take a connection without waiting, so the first burst of checks finds open connections instead
        of opening them (TLS) under the check deadline. Each connection is
        retried until ``deadline_s``; the ones still not open then are given
        up. Logs ``rate_limit.store_ready latency_ms=<n> connections=<k>``
        when at least one opened, else the startup ``skipped_store_down``
        line, and returns ``k``. Cancelled (shutdown), it gives back every
        connection it holds and re-raises.
        """
        loop = asyncio.get_running_loop()
        started = loop.time()
        last_exc: BaseException = TimeoutError()
        try:
            pool = self._get_client().connection_pool
        except Exception as exc:  # noqa: BLE001 - startup must not fail
            self._log_store_down("startup", 0, exc)
            return 0

        in_flight = asyncio.Semaphore(WARM_UP_CONCURRENCY)

        async def _open_one() -> Any:
            nonlocal last_exc
            while True:
                async with in_flight:
                    try:
                        conn = await pool.get_connection()
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:  # noqa: BLE001 - retried, then logged
                        last_exc = exc
                    else:
                        try:
                            await conn.send_command("PING")
                            await conn.read_response()
                            return conn
                        except asyncio.CancelledError:
                            await pool.release(conn)
                            raise
                        except Exception as exc:  # noqa: BLE001 - retried, then logged
                            last_exc = exc
                            await pool.release(conn)
                await asyncio.sleep(WARM_UP_RETRY_PAUSE_S)

        target = max(1, self.max_connections - WARM_UP_FREE_SLOTS)
        tasks = [asyncio.ensure_future(_open_one()) for _ in range(target)]
        try:
            await asyncio.wait(tasks, timeout=deadline_s)
        finally:
            # Also on cancellation: no connection stays held by the warm-up.
            for task in tasks:
                if not task.done():
                    task.cancel()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            opened = [r for r in results if not isinstance(r, BaseException)]
            for conn in opened:
                try:
                    await pool.release(conn)
                except Exception:  # noqa: BLE001 - startup must not fail
                    pass
        if not opened:
            self._log_store_down("startup", 0, last_exc)
            return 0
        logger.info(
            "rate_limit.store_ready latency_ms=%d connections=%d",
            round((loop.time() - started) * 1000),
            len(opened),
        )
        return len(opened)

    async def aclose(self) -> None:
        client, self._client, self._script = self._client, None, None
        if client is not None:
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001 - shutdown must not fail
                logger.debug("rate limit store close failed", exc_info=True)


class RequestRateLimited(HTTPException):
    """429 ``rate_limited``; an HTTPException so routes need no handler."""

    def __init__(self, retry_after_s: int) -> None:
        super().__init__(
            status_code=429,
            detail={
                "code": RATE_LIMITED_CODE,
                "message": f"Too many requests, retry in {retry_after_s} s",
            },
            headers={"Retry-After": str(retry_after_s)},
        )
        self.retry_after_s = retry_after_s


class RateLimiterUnavailable(HTTPException):
    """503 ``limiter_unavailable``: store down with fail closed in enforce mode."""

    def __init__(self) -> None:
        super().__init__(status_code=503, detail=LIMITER_UNAVAILABLE_DETAIL)


_limiter: Optional[RequestRateLimiter] = None


def get_request_rate_limiter() -> Optional[RequestRateLimiter]:
    """The process limiter, or None when the feature is off (no client built)."""
    global _limiter
    if not FEATURE_REQUEST_RATE_LIMIT:
        return None
    if _limiter is None:
        _limiter = RequestRateLimiter(
            limits=REQUEST_RATE_LIMITS,
            mode=REQUEST_RATE_LIMIT_MODE,
            # Without a store fail closed would refuse every covered request.
            fail_closed=REQUEST_RATE_LIMIT_FAIL_CLOSED and bool(REDIS_URL),
            skip_s=REQUEST_RATE_LIMIT_SKIP_S,
            url=REDIS_URL,
            connect_timeout_s=REQUEST_RATE_LIMIT_CONNECT_S,
        )
    return _limiter


def log_startup_config() -> None:
    """Log the limiter setup once at startup (app lifespan). Silent when off."""
    if not FEATURE_REQUEST_RATE_LIMIT:
        return
    unlimited = [
        cls
        for cls in KNOWN_CLASSES
        if int((REQUEST_RATE_LIMITS.get(cls) or {}).get("rate") or 0) <= 0
    ]
    logger.info(
        "rate_limit.config mode=%s fail_closed=%s unlimited_classes=%s",
        REQUEST_RATE_LIMIT_MODE,
        REQUEST_RATE_LIMIT_FAIL_CLOSED and bool(REDIS_URL),
        ",".join(unlimited) or "none",
    )
    if not REDIS_URL:
        logger.warning(
            "FEATURE_REQUEST_RATE_LIMIT is on without REDIS_URL: every check is "
            "skipped_store_down"
        )
        if REQUEST_RATE_LIMIT_FAIL_CLOSED:
            logger.error(
                "REQUEST_RATE_LIMIT_FAIL_CLOSED ignored: REDIS_URL is not set, so "
                "it would refuse every covered request"
            )


def start_request_rate_limiter_warm_up() -> Optional["asyncio.Task[int]"]:
    """Start the pool warm-up in the background (app lifespan). Never raises.

    Returns the task, for :func:`stop_request_rate_limiter_warm_up` at
    shutdown, or None when the feature is off or ``REDIS_URL`` is unset (the
    startup config line already warns about the missing store). The lifespan
    does not await it: readiness never waits for the store.
    """
    if not FEATURE_REQUEST_RATE_LIMIT or not REDIS_URL:
        return None
    limiter = get_request_rate_limiter()
    if limiter is None:
        return None
    return asyncio.create_task(limiter.warm_up(), name="rate_limit_warm_up")


async def stop_request_rate_limiter_warm_up(task: Optional["asyncio.Task[int]"]) -> None:
    """Cancel the warm-up if still running and wait for it (app lifespan)."""
    if task is None:
        return
    if not task.done():
        task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception as exc:  # noqa: BLE001 - shutdown must not fail
        # The class only: a redis error message can carry the host.
        logger.warning("rate_limit warm-up failed type=%s", type(exc).__name__)


async def aclose_request_rate_limiter() -> None:
    """Close the process limiter's store client (app lifespan shutdown)."""
    limiter = _limiter
    if limiter is not None:
        await limiter.aclose()


def _subject_kind(principal: Principal) -> str:
    return "api_key" if principal.auth_type == AUTH_TYPE_API_KEY else "user"


def _mark_span(reason: str) -> None:
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span.is_recording():
            span.set_attribute(SPAN_ATTRIBUTE, reason)
    except Exception:  # noqa: BLE001 - tracing must never fail a request
        pass


async def check_or_raise(principal: Principal, route_class: str) -> Optional[Decision]:
    """Spend one ``route_class`` token for the principal's user, or raise.

    Returns None when the feature is off. Raises :class:`RequestRateLimited`
    (429) or :class:`RateLimiterUnavailable` (503) in enforce mode only; shadow
    mode logs and counts the refusal and lets the request through. For the two
    ASGI dispatchers, which catch the exceptions and answer themselves.
    """
    limiter = get_request_rate_limiter()
    if limiter is None:
        return None
    subject = f"user:{principal.user_id}"
    decision = await limiter.check(subject, route_class)
    if decision.reason == UNLIMITED:
        return decision
    kind = _subject_kind(principal)
    mode = limiter.mode
    record_rate_limit_decision(route_class, decision.reason, mode, kind)
    _mark_span(decision.reason)
    if decision.reason == LIMITED and limiter.should_log_limited(subject, route_class):
        # Contract with the CloudWatch metric filter (infra): keep the literal
        # token and fields. INFO in shadow, WARNING when a request is refused.
        logger.log(
            logging.WARNING if mode == "enforce" else logging.INFO,
            "rate_limit.limited subject_kind=%s class=%s mode=%s retry_after_s=%d",
            kind,
            route_class,
            mode,
            decision.retry_after_s,
        )
    if mode != "enforce" or decision.allowed:
        return decision
    if decision.reason == LIMITED:
        raise RequestRateLimited(decision.retry_after_s)
    raise RateLimiterUnavailable()


def enforce_request_rate(route_class: str) -> Callable[..., Any]:
    """FastAPI dependency factory: ``Depends(enforce_request_rate("chat"))``.

    Depends on :func:`get_current_user`, which FastAPI resolves once per
    request, so on a route that also depends on it the caller is looked up and
    an API key's ``last_used_at`` stamped once. The bearer only tells a session
    from an ``eve_`` key, for the ``subject_kind`` attribute; the bucket is the
    user's either way. Not for a route that depends on ``get_auth_context``:
    that one would resolve the caller a second time.
    """

    async def _enforce(
        user: User = Depends(get_current_user),
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> None:
        is_key = credentials.credentials.startswith("eve_")
        principal = Principal(user.id, AUTH_TYPE_API_KEY if is_key else AUTH_TYPE_OIDC)
        await check_or_raise(principal, route_class)

    _enforce.__name__ = f"enforce_request_rate_{route_class}"
    return _enforce
