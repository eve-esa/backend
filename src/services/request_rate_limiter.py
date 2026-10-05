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

Fail open: a store error or timeout allows the request, marks it
``skipped_store_down`` and makes this worker skip the store for
``REQUEST_RATE_LIMIT_SKIP_S`` seconds, with one WARNING per window.
``REQUEST_RATE_LIMIT_FAIL_CLOSED`` turns that into 503 ``limiter_unavailable``
in enforce mode.

Log lines and metric attributes carry the route class and whether the caller
used a session or an API key, never a user id, key, token or address.
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
STORE_MAX_CONNECTIONS = 10
RETRY_AFTER_MIN_S = 1
RETRY_AFTER_MAX_S = 60
SPAN_ATTRIBUTE = "eve.rate_limit.decision"

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


def retry_after_seconds(retry_ms: int) -> int:
    return min(RETRY_AFTER_MAX_S, max(RETRY_AFTER_MIN_S, math.ceil(retry_ms / 1000)))


def store_url(url: str, db: int = STORE_DB) -> str:
    """``url`` pointed at database ``db``.

    redis-py lets the URL path and a ``db`` query option win over a ``db``
    keyword, so the database is rewritten in the URL itself.
    """
    parts = urlsplit(url)
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query) if k != "db"])
    return urlunsplit((parts.scheme, parts.netloc, f"/{db}", query, parts.fragment))


def _build_client(url: str, timeout_s: float, max_connections: int) -> Any:
    """Async client with short timeouts, no retries and a bounded pool.

    The blocking pool waits up to ``timeout_s`` for a free connection instead
    of failing a burst past ``max_connections`` with "Too many connections",
    which would read as a store outage. Building it opens no socket.
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
        socket_connect_timeout=timeout_s,
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
        max_connections: int = STORE_MAX_CONNECTIONS,
    ) -> None:
        self.limits = limits
        self.mode = mode
        self.fail_closed = fail_closed
        self.skip_s = skip_s
        self.url = url
        self.clock = clock
        self.timeout_s = timeout_s
        self.max_connections = max_connections
        self.skip_until = 0.0
        self._client = client
        self._script = None

    def _limit(self, route_class: str) -> Optional[Dict[str, int]]:
        spec = self.limits.get(route_class)
        if not spec or int(spec.get("rate") or 0) <= 0:
            return None
        return spec

    def _get_script(self) -> Any:
        if self._script is None:
            if self._client is None:
                if not self.url:
                    raise RuntimeError("REDIS_URL is not set")
                self._client = _build_client(self.url, self.timeout_s, self.max_connections)
            # register_script reloads the script on NOSCRIPT (failover, flush).
            self._script = self._client.register_script(TOKEN_BUCKET_LUA)
        return self._script

    def _store_down(self) -> Decision:
        return Decision(
            allowed=not self.fail_closed, retry_after_s=0, reason=SKIPPED_STORE_DOWN
        )

    def _store_failed(self, exc: BaseException) -> None:
        now = self.clock()
        if now < self.skip_until:
            return
        self.skip_until = now + self.skip_s
        # The class name only: a redis error message can carry the host.
        logger.warning(
            "rate_limit.store_down decision=%s skip_s=%s error=%s",
            SKIPPED_STORE_DOWN,
            self.skip_s,
            type(exc).__name__,
        )

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
                timeout=self.timeout_s,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - any store failure fails open
            self._store_failed(exc)
            return self._store_down()
        if int(allowed) == 1:
            return Decision(True, 0, ALLOWED)
        return Decision(False, retry_after_seconds(int(retry_ms)), LIMITED)

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
        if not REDIS_URL:
            logger.warning(
                "FEATURE_REQUEST_RATE_LIMIT is on without REDIS_URL: every check "
                "is skipped_store_down"
            )
        _limiter = RequestRateLimiter(
            limits=REQUEST_RATE_LIMITS,
            mode=REQUEST_RATE_LIMIT_MODE,
            fail_closed=REQUEST_RATE_LIMIT_FAIL_CLOSED,
            skip_s=REQUEST_RATE_LIMIT_SKIP_S,
            url=REDIS_URL,
        )
    return _limiter


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
    decision = await limiter.check(f"user:{principal.user_id}", route_class)
    if decision.reason == UNLIMITED:
        return decision
    kind = _subject_kind(principal)
    mode = limiter.mode
    record_rate_limit_decision(route_class, decision.reason, mode, kind)
    _mark_span(decision.reason)
    if decision.reason == LIMITED:
        logger.warning(
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
