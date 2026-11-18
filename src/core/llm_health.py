"""Circuit breaker and failure classification for the EVE endpoint chain.

Imports nothing from :mod:`src.core.llm_manager` on purpose: the manager owns
one :class:`EndpointHealth`, and the classification predicate must stay usable
from call sites that cannot import langgraph or the OpenAI SDK.

With a Redis URL the open circuits live in Valkey, so every worker of every
task sees the outage the first one found instead of paying its own probe.

A 429 is the endpoint asking to be left alone, and Blablador doubles a block
for every request it receives during it. So a 429 is not a circuit failure but
its own state, a rate limit window: nothing is sent to the endpoint until the
Retry-After has passed, then exactly one probe, and only a success of the
endpoint itself closes the window.
"""

import asyncio
import email.utils
import logging
import math
import os
import re
import threading
import time
import uuid
import weakref
from dataclasses import dataclass
from datetime import timezone
from typing import Any, Callable, Dict, NamedTuple, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

# Matched by class name so this module stays free of langgraph (NodeTimeoutError)
# and httpx imports. Status-carrying errors are classified by status instead.
_ENDPOINT_FAILURE_NAMES = {
    "APIConnectionError",
    "APITimeoutError",
    "ConnectError",
    "ConnectTimeout",
    "NodeTimeoutError",
    "ReadTimeout",
    "RemoteProtocolError",
    "TimeoutException",
}

# A 429 without a usable Retry-After keeps the endpoint out this long; Blablador
# starts its blocks at 60 s and doubles them up to an hour, the cap.
RATE_LIMIT_DEFAULT_S = 60.0
RATE_LIMIT_MAX_S = 3600.0
# How long a window that ended without a successful probe is remembered: until
# then only the holder of the probe lease may call the endpoint.
RATE_LIMIT_HOLD_S = 3600.0
# How long the one probe after a window holds the endpoint for itself. A 429
# comes back at once, so a probe still running after this was accepted; one
# that never reports (a Stop, a crash) frees the endpoint after it.
PROBE_LEASE_S = 30.0

# Retry-After as delay-seconds (RFC 9110 says digits; a decimal is read too).
# re.ASCII: str.isdigit() and \d without it accept digits such as "²".
_DELAY_SECONDS = re.compile(r"\d+(?:\.\d+)?", re.ASCII)


def parse_retry_after(raw: Any, now: float) -> Optional[float]:
    """Seconds a Retry-After value asks for, or None when it says nothing usable.

    *raw* is the header value: delay-seconds or an HTTP-date. *now* is the wall
    clock (epoch seconds) the HTTP-date is read against. A date that is not in
    the future (clock skew) is unusable, not zero: the block is likely still
    running. Callers apply their own default and cap.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if _DELAY_SECONDS.fullmatch(text):
        return float(text)
    try:
        when = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    try:
        seconds = when.timestamp() - now
    except (OverflowError, ValueError):
        return None
    return seconds if seconds > 0 else None


def rate_limit_window_s(raw: Any, now: float) -> float:
    """The window a 429 with Retry-After *raw* keeps the endpoint out.

    Missing or unusable: :data:`RATE_LIMIT_DEFAULT_S`. Always between 1 s and
    :data:`RATE_LIMIT_MAX_S`.
    """
    seconds = parse_retry_after(raw, now)
    if seconds is None:
        seconds = RATE_LIMIT_DEFAULT_S
    return min(max(seconds, 1.0), RATE_LIMIT_MAX_S)


def retry_after_header(exc: BaseException) -> Any:
    """The Retry-After header of the response an error carries, or None."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        return None
    try:
        return headers.get("retry-after")
    except Exception:  # noqa: BLE001 - a foreign headers object is not a window
        return None


class EndpointRateLimited(RuntimeError):
    """An endpoint answered 429, or its rate limit window keeps it out.

    ``status_code`` keeps it an endpoint failure that reports as
    ``upstream_error``. It carries no ``response`` and a class name no retry
    policy matches, so neither the SDK nor the agent graph's node retry sends
    another request into the block: the graph goes straight to its fallback
    node. ``breaker_recorded`` tells :meth:`EndpointHealth.record_failure` the
    window is already set. The message holds no upstream text.
    """

    status_code = 429
    breaker_recorded = True

    def __init__(self, llm_type: str, retry_after_s: float) -> None:
        self.llm_type = llm_type
        self.retry_after_s = max(float(retry_after_s), 1.0)
        super().__init__(
            f"Endpoint {llm_type} is rate limited (HTTP 429), "
            f"retry after {math.ceil(self.retry_after_s)}s"
        )


class RateLimitPermit(NamedTuple):
    """Leave to send one request, from :meth:`EndpointHealth.admit`.

    *probe* is true when the request is the one probe after a rate limit
    window: its success closes the window.
    """

    started_at: float
    probe: bool


def is_endpoint_failure(exc: BaseException) -> bool:
    """True when *exc* says the endpoint is unhealthy, rather than the request.

    Any 4xx stays out: a Blablador strict-template 400 is our prompt bug and a
    401/403/422 is a misconfiguration, and parking a reachable endpoint in a
    cooldown would hide both. Cancellation is the user's decision, never a
    failure.
    """
    if isinstance(exc, asyncio.CancelledError):
        return False
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    if isinstance(exc, TimeoutError):
        return True
    return bool({cls.__name__ for cls in type(exc).__mro__} & _ENDPOINT_FAILURE_NAMES)


@dataclass
class _EndpointState:
    opened_at: float
    opened_wall_ms: int
    failures: int
    last_error: str


class _Store:
    """One Redis client and one outage window, shared by every breaker on it.

    LLMManager is built per request in places (HallucinationDetector), so a
    client and a skip window per breaker would mean one connection pool and
    one stall per request during an outage instead of one per worker.
    """

    def __init__(
        self,
        *,
        url: str = "",
        client: Any = None,
        clock: Callable[[], float] = time.monotonic,
        timeout_s: float = 0.25,
        retry_s: float = 30.0,
    ) -> None:
        self.url = url
        self.client = client
        self.clock = clock
        self.timeout_s = timeout_s
        self.retry_s = retry_s
        self.skip_until = 0.0
        self.degraded = False
        self.lock = threading.Lock()

    def get_client(self) -> Any:
        """The client, or None inside the retry window after an error.

        The first command of a fresh client connects synchronously, name
        resolution included (getaddrinfo is not covered by the socket
        timeout). Acceptable here: the Valkey endpoint is a VPC-internal name
        resolved by the VPC resolver, and the skip window bounds the cost.
        """
        if self.clock() < self.skip_until:
            return None
        if self.client is None:
            try:
                import redis
                from redis.backoff import NoBackoff
                from redis.retry import Retry

                client = redis.Redis.from_url(
                    self.url,
                    socket_connect_timeout=self.timeout_s,
                    socket_timeout=self.timeout_s,
                    # redis-py retries timeouts three times by default, which
                    # would multiply the stall the timeout is meant to bound.
                    retry=Retry(NoBackoff(), 0),
                )
            except Exception as exc:  # noqa: BLE001 - a bad URL must not break the chain
                self.failed(exc)
                return None
            with self.lock:
                if self.client is None:
                    self.client = client
        return self.client

    def failed(self, exc: BaseException) -> None:
        with self.lock:
            self.skip_until = self.clock() + self.retry_s
            first = not self.degraded
            self.degraded = True
        if first:
            logger.warning(
                "Endpoint breaker store unreachable, using per-process state "
                "(retry every %ss): %s: %s",
                self.retry_s,
                type(exc).__name__,
                exc,
            )

    def answered(self) -> None:
        if not self.degraded:
            return
        with self.lock:
            recovered = self.degraded
            self.degraded = False
        if recovered:
            logger.info("Endpoint breaker store reachable again, circuits shared")


_STORES_LOCK = threading.Lock()
_URL_STORES: Dict[str, _Store] = {}
_CLIENT_STORES: "weakref.WeakKeyDictionary[Any, _Store]" = weakref.WeakKeyDictionary()


def _store_for(
    *,
    url: str,
    client: Any,
    clock: Callable[[], float],
    timeout_s: float,
    retry_s: float,
) -> Optional[_Store]:
    if client is None and not url:
        return None
    with _STORES_LOCK:
        if client is not None:
            store = _CLIENT_STORES.get(client)
            if store is None:
                store = _Store(client=client, clock=clock, timeout_s=timeout_s, retry_s=retry_s)
                _CLIENT_STORES[client] = store
            return store
        store = _URL_STORES.get(url)
        if store is None:
            store = _Store(url=url, timeout_s=timeout_s, retry_s=retry_s)
            _URL_STORES[url] = store
        return store


def reset_shared_stores() -> None:
    """Forget every shared client and outage window (tests only)."""
    with _STORES_LOCK:
        _URL_STORES.clear()
        _CLIENT_STORES.clear()


@dataclass
class _RateWindow:
    """A rate limit window as this process knows it, on the wall clock."""

    opened_wall_ms: int
    until: float
    count: int
    # This process holds the probe lease until then (0: it does not).
    lease_until: float = 0.0


class _CachedCircuit(NamedTuple):
    checked_at: float
    open_until: float


class _WindowView(NamedTuple):
    """A rate limit window as read: its end (wall clock) and whether any
    worker holds the probe lease."""

    until: float
    probe_out: bool


class _CachedWindow(NamedTuple):
    checked_at: float
    view: Optional[_WindowView]


# The store cannot answer: the per-process state decides.
_UNKNOWN: Any = object()

# Rate limit window values are "<opened ms>:<until ms>"; the probe lease is a
# separate key with its own expiry. Plain commands, no script. The race that
# matters, two requests taking the lease, is settled by SET NX. The window is
# read, then written: two 429s recorded within one round trip of each other can
# leave the shorter window, and the probe after it then meets the block once
# and reopens the window. A close reads, then deletes: while the probe is out
# nothing else is sent, so no new 429 can land in between.


def _write_window(
    client: Any, window_key: str, lease_key: str, opened_ms: int, until_ms: int, now_ms: int
) -> int:
    """Store a window that only grows; returns its end in ms.

    A shorter window than the one stored changes nothing. A longer one also
    drops the probe lease, so the probe that got the new 429 frees its place.
    """
    current = _decode_window(client.get(window_key))
    if current is not None and current[1] >= until_ms:
        return current[1]
    ttl_ms = max(until_ms - now_ms + int(RATE_LIMIT_HOLD_S * 1000), 1000)
    client.set(window_key, f"{opened_ms}:{until_ms}", px=ttl_ms)
    client.delete(lease_key)
    return until_ms


class EndpointHealth:
    """Circuit breaker over the endpoint chain, shared through Redis when it can be.

    One failure opens the circuit: a false positive costs a single answer routed
    to the next endpoint, while a second probe of a dead RunPod endpoint costs
    every request its full cold-start budget.

    Without Redis the state is per process and cooldown expiry deletes the
    entry, so the request after it is the half-open probe.

    With *redis_url* (or *redis_client*) set, an open circuit is the key
    ``<prefix><llm_type>`` holding ``<opened_at epoch ms>:<reason>``, alive for
    the cooldown plus the endpoint's probe budget:

    - before ``opened_at + cooldown`` every worker reads it as open;
    - after it the circuit is half-open: one worker wins
      ``SET <key>:probe NX PX <probe budget>`` and probes, every other worker
      keeps reading open until the probe reports (failure re-opens the key,
      success deletes both) or the lease and the key expire;
    - a success deletes the key only when this worker saw the circuit open
      and the key is not newer than the request that succeeded, so a slow
      success never erases a failure another worker recorded meanwhile.

    A 429 is a rate limit window instead, in its own key
    ``<prefix><llm_type>:rate-limit`` (``<opened ms>:<until ms>``) with its own
    probe lease ``<prefix><llm_type>:rate-limit:probe``. Nothing else writes
    it, so a timeout or a 5xx never touches it, and it only grows. Callers
    that send to the endpoint ask :meth:`admit` first, a fresh read: inside
    the window it refuses; after it, one request in the fleet wins the lease
    (``SET NX``, :data:`PROBE_LEASE_S`) and is the probe, every other one
    refuses. Only
    :meth:`close_rate_limit_window`, called when the endpoint itself answered
    the probe, closes it; a window nobody probes is forgotten
    :data:`RATE_LIMIT_HOLD_S` after it ended. :meth:`is_rate_limited` reads the
    same state (cached *refresh_s*) without ever taking the lease.

    While Redis answers it is authoritative. The per-process state is always
    written too and takes over whenever Redis is unset or unreachable; failures
    and windows recorded during an outage are written back (circuits ``NX``
    with the remaining cooldown, windows only when they are longer)
    before this breaker reads the store again. The breaker never raises, except
    :meth:`admit`, which raises :class:`EndpointRateLimited` to refuse.

    The public methods stay synchronous because ``resolve_chain`` and its
    callers are. A read is one command per endpoint at most every *refresh_s*,
    and :meth:`admit` one command per request (two or three right after a
    window), each bounded by
    *redis_timeout_s*; after a Redis error every breaker on that store skips it
    for *redis_retry_s*, so an outage costs one bounded stall per worker per
    retry window.
    """

    def __init__(
        self,
        cooldown_s: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        redis_url: Optional[str] = None,
        redis_client: Any = None,
        key_prefix: str = "eve:breaker:",
        probe_s: Union[float, Callable[[str], float]] = 30.0,
        refresh_s: float = 1.0,
        redis_timeout_s: float = 0.25,
        redis_retry_s: float = 30.0,
    ) -> None:
        self._cooldown_s = cooldown_s
        self._clock = clock
        self._wall = wall_clock
        self._lock = threading.Lock()
        self._states: Dict[str, _EndpointState] = {}
        self._windows: Dict[str, _RateWindow] = {}

        self._store = _store_for(
            url=(redis_url or "").strip(),
            client=redis_client,
            clock=clock,
            timeout_s=redis_timeout_s,
            retry_s=redis_retry_s,
        )
        self._key_prefix = key_prefix
        self._probe_s = probe_s
        self._refresh_s = refresh_s
        # Shared reads, on self._clock.
        self._remote: Dict[str, _CachedCircuit] = {}
        self._remote_windows: Dict[str, _CachedWindow] = {}
        # Half-open leases this breaker holds.
        self._probing: Set[str] = set()
        # Failures and windows recorded while the store was unreachable.
        self._unsynced: Set[str] = set()
        self._unsynced_windows: Set[str] = set()
        # Identifies this process's probe leases in the store.
        self._lease_token = f"{os.getpid()}:{uuid.uuid4().hex[:8]}"

    # ─── public API ──────────────────────────────────────────────────────────

    def is_open(self, llm_type: str) -> bool:
        """True while *llm_type* cools down after a failure, or a rate limit
        window keeps it out."""
        if self.is_rate_limited(llm_type):
            return True
        local_open = self._local_is_open(llm_type)
        shared_open = self._shared_is_open(llm_type)
        return local_open if shared_open is None else shared_open

    def is_rate_limited(self, llm_type: str) -> bool:
        """True while a 429 keeps *llm_type* out of new turns.

        Inside the Retry-After window, and after it while a probe is out. A
        window that ended with no probe out reads False, so the next turn may
        try: :meth:`admit` then picks the one request that probes.
        """
        view = self._window_view(llm_type)
        return view is not None and (self._wall() < view.until or view.probe_out)

    def rate_limit_remaining_s(self, llm_type: str) -> float:
        """Seconds left in the Retry-After window of *llm_type*, 0 outside one."""
        view = self._window_view(llm_type)
        return max(view.until - self._wall(), 0.0) if view is not None else 0.0

    def admit(self, llm_type: str) -> RateLimitPermit:
        """Leave to send one request to *llm_type*, or :class:`EndpointRateLimited`.

        Reads the store fresh, never the cache, so a 429 another worker just
        recorded holds here at once. Inside a window it refuses; after it the
        first caller in the fleet gets the probe lease and every other caller
        refuses until the probe reports or its lease expires. With the store
        unset or unreachable the per-process state decides.
        """
        started = self._wall()
        started_ms = int(started * 1000)
        lease_ms = int(PROBE_LEASE_S * 1000)
        window_key, lease_key = self._window_keys(llm_type)

        def admit(client: Any) -> Tuple[Optional[_WindowView], float]:
            """The window as read, and the seconds to refuse (0: send)."""
            window = _decode_window(client.get(window_key))
            if window is None:
                return None, 0.0
            until = window[1] / 1000.0
            if started_ms < window[1]:
                return _WindowView(until, False), (window[1] - started_ms) / 1000.0
            if client.set(lease_key, self._lease_token, nx=True, px=lease_ms):
                return _WindowView(until, True), 0.0
            return _WindowView(until, True), max(client.pttl(lease_key), 1) / 1000.0

        result = self._run(admit)
        if result is not None:
            view, wait_s = result
            self._cache_window(llm_type, view)
            if view is None:
                with self._lock:
                    self._windows.pop(llm_type, None)
                return RateLimitPermit(started, False)
            if wait_s > 0:
                raise EndpointRateLimited(llm_type, wait_s)
            return RateLimitPermit(started, True)

        with self._lock:
            window = self._windows.get(llm_type)
            if window is None or started >= window.until + RATE_LIMIT_HOLD_S:
                self._windows.pop(llm_type, None)
                return RateLimitPermit(started, False)
            if started < window.until:
                raise EndpointRateLimited(llm_type, window.until - started)
            if window.lease_until > started:
                raise EndpointRateLimited(llm_type, window.lease_until - started)
            window.lease_until = started + PROBE_LEASE_S
            return RateLimitPermit(started, True)

    def record_rate_limit(self, llm_type: str, retry_after: Any = None) -> float:
        """Open (or lengthen) the rate limit window of *llm_type* after a 429.

        *retry_after* is the response's Retry-After header. Returns the seconds
        left in the window, which is never shorter than one already running.
        """
        now = self._wall()
        window_s = rate_limit_window_s(retry_after, now)
        opened_ms = int(now * 1000)
        with self._lock:
            previous = self._windows.get(llm_type)
            until = max(now + window_s, previous.until if previous else 0.0)
            self._windows[llm_type] = _RateWindow(
                opened_wall_ms=opened_ms,
                until=until,
                count=previous.count + 1 if previous else 1,
            )
        window_key, lease_key = self._window_keys(llm_type)

        def record(client: Any) -> int:
            return _write_window(
                client, window_key, lease_key, opened_ms, int(until * 1000), opened_ms
            )

        shared_until = self._run(record)
        if shared_until is None:
            with self._lock:
                self._unsynced_windows.add(llm_type)
        else:
            until = max(until, shared_until / 1000.0)
            with self._lock:
                window = self._windows.get(llm_type)
                if window is not None:
                    window.until = max(window.until, until)
            self._cache_window(llm_type, _WindowView(until, False))
        remaining_s = max(until - now, 1.0)
        # No upstream text: a 429 body can echo the key or the prompt.
        logger.warning(
            "Endpoint %s circuit opened: rate limited (HTTP 429), retry after %ds",
            llm_type,
            math.ceil(remaining_s),
        )
        return remaining_s

    def close_rate_limit_window(self, llm_type: str, started_at: float) -> bool:
        """Close the rate limit window of *llm_type* after the endpoint answered.

        Call it only with a success produced by *llm_type* itself, never with
        an answer another endpoint gave in its place. *started_at* is the wall
        clock the request started: a window opened after it (a newer 429)
        stays. Returns True when a window was closed.
        """
        started_ms = int(started_at * 1000)
        local_closed = False
        with self._lock:
            window = self._windows.get(llm_type)
            if window is not None and window.opened_wall_ms <= started_ms:
                del self._windows[llm_type]
                self._unsynced_windows.discard(llm_type)
                local_closed = True
        window_key, lease_key = self._window_keys(llm_type)

        def close(client: Any) -> bool:
            window = _decode_window(client.get(window_key))
            if window is None or window[0] > started_ms:
                return False
            client.delete(window_key, lease_key)
            return True

        result = self._run(close)
        closed = local_closed if result is None else result
        if result is not None and closed:
            self._cache_window(llm_type, None)
        if closed:
            logger.info("Endpoint %s rate limit window closed: the probe answered", llm_type)
        return closed

    def record_failure(self, llm_type: str, exc: BaseException) -> Optional[float]:
        """Open (or re-open) the circuit for *llm_type*.

        A 429 opens the rate limit window instead (:meth:`record_rate_limit`)
        and returns its seconds left; anything else opens the circuit for the
        cooldown, leaves any window alone and returns None. An error that says
        it is already recorded changes nothing.
        """
        if getattr(exc, "breaker_recorded", False):
            return None
        if getattr(exc, "status_code", None) == 429:
            return self.record_rate_limit(llm_type, retry_after_header(exc))
        reason = f"{type(exc).__name__}: {exc}"[:200]
        wall_ms = int(self._wall() * 1000)
        with self._lock:
            previous = self._states.get(llm_type)
            self._states[llm_type] = _EndpointState(
                opened_at=self._clock(),
                opened_wall_ms=wall_ms,
                failures=previous.failures + 1 if previous else 1,
                last_error=reason,
            )
            self._probing.discard(llm_type)
        if self._cooldown_s > 0:
            key = self._key(llm_type)
            ttl_ms = int((self._cooldown_s + self._probe_budget(llm_type)) * 1000)

            def reopen(client: Any) -> bool:
                client.set(key, f"{wall_ms}:{reason}", px=ttl_ms)
                client.delete(f"{key}:probe")
                return True

            if self._run(reopen):
                self._cache(llm_type, self._cooldown_s)
            else:
                with self._lock:
                    self._unsynced.add(llm_type)
        logger.warning("Endpoint %s circuit opened: %s", llm_type, exc)
        return None

    def record_success(
        self, llm_type: str, *, started_at: Optional[float] = None
    ) -> None:
        """Close the circuit for *llm_type*.

        *started_at* is the wall-clock time (``time.time()``) the successful
        request started; a shared circuit opened after it stays open. A rate
        limit window is never closed here: callers report the turn, which a
        substituted client may have answered (see
        :meth:`close_rate_limit_window`).
        """
        now = self._clock()
        with self._lock:
            local_open = llm_type in self._states
            self._states.pop(llm_type, None)
            self._unsynced.discard(llm_type)
            probing = llm_type in self._probing
            self._probing.discard(llm_type)
            cached = self._remote.get(llm_type)
        cached_open = cached is not None and now < cached.open_until
        if self._store is None or not (local_open or probing or cached_open):
            return
        key = self._key(llm_type)

        def close(client: Any) -> bool:
            if started_at is not None:
                opened_ms = _opened_ms(client.get(key))
                if opened_ms is not None and opened_ms > started_at * 1000:
                    return False
            client.delete(key, f"{key}:probe")
            return True

        closed = self._run(close)
        if closed:
            self._cache(llm_type, 0.0)

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        """This process's circuit state, for operational readouts.

        Failure counts and errors are what this process saw; a circuit another
        worker opened shows up in :meth:`is_open`, not here.
        """
        now = self._clock()
        wall = self._wall()
        with self._lock:
            rows = {
                llm_type: {
                    "open": now - state.opened_at < self._cooldown_s,
                    "rate_limited": False,
                    "failures": state.failures,
                    "last_error": state.last_error,
                    "opened_since_s": round(now - state.opened_at, 3),
                }
                for llm_type, state in self._states.items()
            }
            for llm_type, window in self._windows.items():
                rows[llm_type] = {
                    "open": wall < window.until or window.lease_until > wall,
                    "rate_limited": True,
                    "failures": window.count,
                    "last_error": "RateLimited: HTTP 429",
                    "opened_since_s": round(wall - window.opened_wall_ms / 1000.0, 3),
                }
            return rows

    # ─── per-process state ───────────────────────────────────────────────────

    def _local_is_open(self, llm_type: str) -> bool:
        with self._lock:
            state = self._states.get(llm_type)
            if state is None:
                return False
            if self._clock() - state.opened_at >= self._cooldown_s:
                del self._states[llm_type]
                self._unsynced.discard(llm_type)
                return False
            return True

    def _local_window(self, llm_type: str) -> Optional[_WindowView]:
        wall = self._wall()
        with self._lock:
            window = self._windows.get(llm_type)
            if window is None:
                return None
            if wall >= window.until + RATE_LIMIT_HOLD_S:
                del self._windows[llm_type]
                self._unsynced_windows.discard(llm_type)
                return None
            return _WindowView(window.until, window.lease_until > wall)

    # ─── shared state ────────────────────────────────────────────────────────

    def _key(self, llm_type: str) -> str:
        return f"{self._key_prefix}{llm_type}"

    def _window_keys(self, llm_type: str) -> Tuple[str, str]:
        window_key = f"{self._key(llm_type)}:rate-limit"
        return window_key, f"{window_key}:probe"

    def _probe_budget(self, llm_type: str) -> float:
        budget = self._probe_s(llm_type) if callable(self._probe_s) else self._probe_s
        return max(float(budget), 1.0)

    def _cache(self, llm_type: str, open_for_s: float) -> None:
        now = self._clock()
        with self._lock:
            self._remote[llm_type] = _CachedCircuit(now, now + open_for_s)

    def _cache_window(self, llm_type: str, view: Optional[_WindowView]) -> None:
        if self._store is None:
            return
        now = self._clock()
        with self._lock:
            self._remote_windows[llm_type] = _CachedWindow(now, view)

    def _window_view(self, llm_type: str) -> Optional[_WindowView]:
        shared = self._shared_window(llm_type)
        return self._local_window(llm_type) if shared is _UNKNOWN else shared

    def _shared_window(self, llm_type: str) -> Any:
        """The shared window, None for none, or _UNKNOWN when only the local
        state can answer. Cached *refresh_s*; never takes the probe lease."""
        if self._store is None:
            return _UNKNOWN
        now = self._clock()
        with self._lock:
            cached = self._remote_windows.get(llm_type)
            if cached is not None and now - cached.checked_at < self._refresh_s:
                return cached.view
        window_key, lease_key = self._window_keys(llm_type)

        def read(client: Any) -> Optional[_WindowView]:
            window, lease = client.mget([window_key, lease_key])
            decoded = _decode_window(window)
            if decoded is None:
                return None
            return _WindowView(decoded[1] / 1000.0, lease is not None)

        result = self._run(read, default=_UNKNOWN)
        if result is _UNKNOWN:
            return _UNKNOWN
        self._cache_window(llm_type, result)
        return result

    def _shared_is_open(self, llm_type: str) -> Optional[bool]:
        """The shared verdict, or None when only the local state can answer."""
        if self._store is None:
            return None
        now = self._clock()
        with self._lock:
            cached = self._remote.get(llm_type)
            if cached is not None and now - cached.checked_at < self._refresh_s:
                return now < cached.open_until
        key = self._key(llm_type)
        probe_ms = int(self._probe_budget(llm_type) * 1000)

        def read(client: Any) -> float:
            """Seconds the circuit stays open for this worker (0 = closed)."""
            opened_ms = _opened_ms(client.get(key))
            if opened_ms is None:
                return 0.0
            remaining_ms = opened_ms + self._cooldown_s * 1000 - self._wall() * 1000
            if remaining_ms > 0:
                return remaining_ms / 1000.0
            # Half-open: one worker probes, the others wait for its verdict.
            if client.set(f"{key}:probe", str(os.getpid()), nx=True, px=probe_ms):
                with self._lock:
                    self._probing.add(llm_type)
                return 0.0
            return self._refresh_s

        open_for = self._run(read)
        if open_for is None:
            return None
        self._cache(llm_type, open_for)
        return open_for > 0

    def _run(self, op: Callable[[Any], Any], default: Any = None) -> Any:
        """Run *op* against the store; *default* when the store cannot answer.

        Failures and windows recorded during an outage go back first, so the
        store never contradicts what this worker saw.
        """
        if self._store is None:
            return default
        client = self._store.get_client()
        if client is None:
            return default
        try:
            self._resync(client)
            result = op(client)
        except Exception as exc:  # noqa: BLE001 - any store error means fall back
            self._store.failed(exc)
            with self._lock:
                self._remote.clear()
                self._remote_windows.clear()
            return default
        self._store.answered()
        return result

    def _resync(self, client: Any) -> None:
        with self._lock:
            if not self._unsynced and not self._unsynced_windows:
                return
            now = self._clock()
            wall = self._wall()
            pending = [
                (llm_type, self._states[llm_type])
                for llm_type in self._unsynced
                if llm_type in self._states
            ]
            pending_windows = [
                (llm_type, _RateWindow(**vars(self._windows[llm_type])))
                for llm_type in self._unsynced_windows
                if llm_type in self._windows
            ]
        for llm_type, state in pending:
            remaining_s = self._cooldown_s - (now - state.opened_at)
            if remaining_s <= 0:
                continue
            ttl_ms = int((remaining_s + self._probe_budget(llm_type)) * 1000)
            client.set(
                self._key(llm_type),
                f"{state.opened_wall_ms}:{state.last_error}",
                nx=True,
                px=ttl_ms,
            )
        for llm_type, window in pending_windows:
            if wall >= window.until + RATE_LIMIT_HOLD_S:
                continue
            # Only grows: a window this worker saw while the store was down
            # wins over a shorter one written meanwhile.
            window_key, lease_key = self._window_keys(llm_type)
            _write_window(
                client, window_key, lease_key,
                window.opened_wall_ms, int(window.until * 1000), int(wall * 1000),
            )
        with self._lock:
            self._unsynced.clear()
            self._unsynced_windows.clear()


def _text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _opened_ms(value: Any) -> Optional[int]:
    """``opened_at`` from a breaker key value, or None when absent or foreign."""
    text = _text(value)
    if text is None:
        return None
    head = text.split(":", 1)[0]
    try:
        return int(head)
    except ValueError:
        return None


def _decode_window(value: Any) -> Optional[Tuple[int, int]]:
    """``(opened ms, until ms)`` from a rate limit window value, or None."""
    text = _text(value)
    if text is None:
        return None
    opened, _, until = text.partition(":")
    try:
        return int(opened), int(until)
    except ValueError:
        return None
