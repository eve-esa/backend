"""Circuit breaker and failure classification for the EVE endpoint chain.

Imports nothing from :mod:`src.core.llm_manager` on purpose: the manager owns
one :class:`EndpointHealth`, and the classification predicate must stay usable
from call sites that cannot import langgraph or the OpenAI SDK.

With a Redis URL the open circuits live in Valkey, so every worker of every
task sees the outage the first one found instead of paying its own probe.

A 429 is the endpoint asking to be left alone: it opens the circuit for the
Retry-After it sent instead of the fixed cooldown, and while that window runs
the endpoint leaves the chain and its client refuses to send (Blablador doubles
a block on every request it receives during it).
"""

import asyncio
import email.utils
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from datetime import timezone
import weakref
from typing import Any, Callable, Dict, Optional, Set, Tuple, Union

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


class EndpointRateLimited(RuntimeError):
    """An endpoint answered 429, or is still inside the window a 429 set.

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
        self.retry_after_s = retry_after_s
        super().__init__(
            f"Endpoint {llm_type} is rate limited (HTTP 429) for {math.ceil(retry_after_s)}s"
        )


def _retry_after_s(raw: Any, now: float) -> Optional[float]:
    """Seconds from a Retry-After value (delta-seconds or HTTP-date), or None."""
    if raw is None:
        return None
    text = str(raw).strip()
    if text.isdigit():
        return float(text)
    try:
        when = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return float(math.ceil(when.timestamp() - now))


def rate_limit_window_s(exc: BaseException, now: float) -> Optional[float]:
    """How long a 429 asks the endpoint to be left alone; None for anything else.

    *now* is the wall clock (epoch seconds), for an HTTP-date Retry-After.
    Missing or unparsable: :data:`RATE_LIMIT_DEFAULT_S`. Always between 1 s and
    :data:`RATE_LIMIT_MAX_S`.
    """
    if getattr(exc, "status_code", None) != 429:
        return None
    headers = getattr(getattr(exc, "response", None), "headers", None)
    raw = headers.get("retry-after") if headers is not None else None
    seconds = _retry_after_s(raw, now)
    if seconds is None:
        seconds = RATE_LIMIT_DEFAULT_S
    return min(max(seconds, 1.0), RATE_LIMIT_MAX_S)


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
    cooldown_s: float
    # Opened by a 429: the window is the endpoint's own Retry-After.
    rate_limited: bool = False

    def remaining_s(self, now: float) -> float:
        return self.cooldown_s - (now - self.opened_at)

    def value(self) -> str:
        """The shared key value for this state."""
        return _encode(
            self.opened_wall_ms,
            self.cooldown_s if self.rate_limited else None,
            self.last_error,
        )


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

    A 429 writes ``<opened_at>:rl=<window ms>:<reason>`` instead: the window
    replaces the cooldown for every reader, and :meth:`is_rate_limited` is true
    while the circuit is open, the half-open wait of the other workers
    included, so only the worker holding the probe lease calls the endpoint.
    A later timeout or 5xx never shortens such a window.

    While Redis answers it is authoritative. The per-process state is always
    written too and takes over whenever Redis is unset or unreachable; failures
    recorded during an outage are written back (``NX``, remaining cooldown)
    before this breaker reads the store again. The breaker never raises.

    The public methods stay synchronous because ``resolve_chain`` and its
    callers are. A read is one ``GET`` per endpoint at most every *refresh_s*,
    bounded by *redis_timeout_s*; after a Redis error every breaker on that
    store skips it for *redis_retry_s*, so an outage costs one bounded stall
    per worker per retry window.
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
        # llm_type -> (checked_at, open_until, window_until, rate_limited), on
        # self._clock; window_until is the end of a 429 window (0 outside one).
        self._remote: Dict[str, Tuple[float, float, float, bool]] = {}
        # Half-open leases this breaker holds.
        self._probing: Set[str] = set()
        # Failures recorded while the store was unreachable.
        self._unsynced: Set[str] = set()

    # ─── public API ──────────────────────────────────────────────────────────

    def is_open(self, llm_type: str) -> bool:
        """True while *llm_type* is still cooling down after a failure."""
        return self._status(llm_type)[0]

    def is_rate_limited(self, llm_type: str) -> bool:
        """True while a 429 keeps *llm_type* out: it must not be called at all.

        Covers the Retry-After window and, after it, the half-open wait of the
        workers that did not win the probe lease.
        """
        is_open, rate_limited, _ = self._status(llm_type)
        return is_open and rate_limited

    def rate_limit_remaining_s(self, llm_type: str) -> float:
        """Seconds left in the Retry-After window of *llm_type*, 0 outside one."""
        _, rate_limited, window_s = self._status(llm_type)
        return window_s if rate_limited else 0.0

    def record_failure(self, llm_type: str, exc: BaseException) -> Optional[float]:
        """Open (or re-open) the circuit for *llm_type*.

        A 429 opens it for the Retry-After it carries and returns that window
        in seconds; anything else opens it for the cooldown and returns None.
        An error that says it is already recorded changes nothing.
        """
        if getattr(exc, "breaker_recorded", False):
            return None
        window_s = rate_limit_window_s(exc, self._wall())
        rate_limited = window_s is not None
        cooldown_s = window_s if window_s is not None else self._cooldown_s
        reason = (
            f"RateLimited: HTTP 429, retry after {window_s:.0f}s"
            if rate_limited
            else f"{type(exc).__name__}: {exc}"[:200]
        )
        wall_ms = int(self._wall() * 1000)
        now = self._clock()
        with self._lock:
            previous = self._states.get(llm_type)
            keeps_window = (
                not rate_limited
                and previous is not None
                and previous.rate_limited
                and previous.remaining_s(now) > cooldown_s
            )
            if keeps_window:
                previous.failures += 1
            else:
                self._states[llm_type] = _EndpointState(
                    opened_at=now,
                    opened_wall_ms=wall_ms,
                    failures=previous.failures + 1 if previous else 1,
                    last_error=reason,
                    cooldown_s=cooldown_s,
                    rate_limited=rate_limited,
                )
            self._probing.discard(llm_type)
        if cooldown_s > 0 and not keeps_window:
            key = self._key(llm_type)
            probe_s = self._probe_budget(llm_type)
            value = _encode(wall_ms, window_s, reason)

            def reopen(client: Any) -> Tuple[float, float, bool]:
                if not rate_limited:
                    # A timeout or a 5xx never shortens a 429 window another
                    # worker recorded.
                    current = _decode(client.get(key))
                    if current is not None and current[1] is not None:
                        left_s = (current[0] + current[1] - wall_ms) / 1000.0
                        if left_s > cooldown_s:
                            return left_s, left_s, True
                client.set(key, value, px=int((cooldown_s + probe_s) * 1000))
                client.delete(f"{key}:probe")
                return cooldown_s, (cooldown_s if rate_limited else 0.0), rate_limited

            opened = self._run(reopen)
            if opened is not None:
                self._cache(llm_type, *opened)
            else:
                with self._lock:
                    self._unsynced.add(llm_type)
        if rate_limited:
            # No upstream text: a 429 body can echo the key or the prompt.
            logger.warning(
                "Endpoint %s circuit opened: rate limited (HTTP 429), retry after %ds",
                llm_type,
                math.ceil(window_s),
            )
        else:
            logger.warning("Endpoint %s circuit opened: %s", llm_type, exc)
        return window_s

    def record_success(
        self,
        llm_type: str,
        *,
        started_at: Optional[float] = None,
        rate_limit_only: bool = False,
    ) -> None:
        """Close the circuit for *llm_type*.

        *started_at* is the wall-clock time (``time.time()``) the successful
        request started; a shared circuit opened after it stays open, and so
        does a local 429 window. With *rate_limit_only* only a circuit a 429
        opened is closed: the client that keeps an endpoint out of its 429
        window closes it after its own probe, and leaves every other circuit to
        the callers that manage it today.
        """
        now = self._clock()
        with self._lock:
            state = self._states.get(llm_type)
            cached = self._remote.get(llm_type)
            if rate_limit_only and not (
                (state is not None and state.rate_limited)
                or (cached is not None and cached[3])
            ):
                return
            local_open = state is not None
            keeps_window = (
                state is not None
                and state.rate_limited
                and started_at is not None
                and started_at * 1000 < state.opened_wall_ms
            )
            if not keeps_window:
                self._states.pop(llm_type, None)
                self._unsynced.discard(llm_type)
            probing = llm_type in self._probing
            self._probing.discard(llm_type)
        cached_open = cached is not None and now < cached[1]
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
        with self._lock:
            return {
                llm_type: {
                    "open": state.remaining_s(now) > 0,
                    "rate_limited": state.rate_limited,
                    "failures": state.failures,
                    "last_error": state.last_error,
                    "opened_since_s": round(now - state.opened_at, 3),
                }
                for llm_type, state in self._states.items()
            }

    # ─── per-process state ───────────────────────────────────────────────────

    def _status(self, llm_type: str) -> Tuple[bool, bool, float]:
        """(open, opened by a 429, seconds left in the 429 window)."""
        local = self._local_status(llm_type)
        shared = self._shared_status(llm_type)
        return local if shared is None else shared

    def _local_status(self, llm_type: str) -> Tuple[bool, bool, float]:
        with self._lock:
            state = self._states.get(llm_type)
            if state is None:
                return False, False, 0.0
            remaining_s = state.remaining_s(self._clock())
            if remaining_s <= 0:
                del self._states[llm_type]
                self._unsynced.discard(llm_type)
                return False, False, 0.0
            return True, state.rate_limited, remaining_s if state.rate_limited else 0.0

    # ─── shared state ────────────────────────────────────────────────────────

    def _key(self, llm_type: str) -> str:
        return f"{self._key_prefix}{llm_type}"

    def _probe_budget(self, llm_type: str) -> float:
        budget = self._probe_s(llm_type) if callable(self._probe_s) else self._probe_s
        return max(float(budget), 1.0)

    def _cache(
        self,
        llm_type: str,
        open_for_s: float,
        window_s: float = 0.0,
        rate_limited: bool = False,
    ) -> None:
        now = self._clock()
        with self._lock:
            self._remote[llm_type] = (now, now + open_for_s, now + window_s, rate_limited)

    def _shared_status(self, llm_type: str) -> Optional[Tuple[bool, bool, float]]:
        """The shared verdict, or None when only the local state can answer."""
        if self._store is None:
            return None
        now = self._clock()
        with self._lock:
            cached = self._remote.get(llm_type)
            if cached is not None and now - cached[0] < self._refresh_s:
                return now < cached[1], cached[3], max(cached[2] - now, 0.0)
        key = self._key(llm_type)
        probe_ms = int(self._probe_budget(llm_type) * 1000)

        def read(client: Any) -> Tuple[float, float, bool]:
            """Seconds open for this worker (0 = closed), seconds left in a 429
            window, and whether a 429 opened the circuit."""
            current = _decode(client.get(key))
            if current is None:
                return 0.0, 0.0, False
            opened_ms, window_ms = current
            rate_limited = window_ms is not None
            cooldown_ms = window_ms if window_ms is not None else self._cooldown_s * 1000
            remaining_ms = opened_ms + cooldown_ms - self._wall() * 1000
            if remaining_ms > 0:
                remaining_s = remaining_ms / 1000.0
                return remaining_s, (remaining_s if rate_limited else 0.0), rate_limited
            # Half-open: one worker probes, the others wait for its verdict.
            if client.set(f"{key}:probe", str(os.getpid()), nx=True, px=probe_ms):
                with self._lock:
                    self._probing.add(llm_type)
                return 0.0, 0.0, rate_limited
            return self._refresh_s, 0.0, rate_limited

        verdict = self._run(read)
        if verdict is None:
            return None
        open_for, window_s, rate_limited = verdict
        self._cache(llm_type, open_for, window_s, rate_limited)
        return open_for > 0, rate_limited, window_s

    def _run(self, op: Callable[[Any], Any]) -> Any:
        """Run *op* against the store; None when the store cannot answer.

        Failures recorded during an outage go back first (NX, remaining
        cooldown), so the store never contradicts what this worker saw.
        """
        if self._store is None:
            return None
        client = self._store.get_client()
        if client is None:
            return None
        try:
            self._resync(client)
            result = op(client)
        except Exception as exc:  # noqa: BLE001 - any store error means fall back
            self._store.failed(exc)
            with self._lock:
                self._remote.clear()
            return None
        self._store.answered()
        return result

    def _resync(self, client: Any) -> None:
        with self._lock:
            if not self._unsynced:
                return
            now = self._clock()
            pending = [
                (llm_type, self._states[llm_type])
                for llm_type in self._unsynced
                if llm_type in self._states
            ]
        for llm_type, state in pending:
            remaining_s = state.remaining_s(now)
            if remaining_s <= 0:
                continue
            ttl_ms = int((remaining_s + self._probe_budget(llm_type)) * 1000)
            client.set(self._key(llm_type), state.value(), nx=True, px=ttl_ms)
        with self._lock:
            self._unsynced.clear()


def _encode(opened_wall_ms: int, window_s: Optional[float], reason: str) -> str:
    """A breaker key value; *window_s* set marks a 429 window."""
    if window_s is None:
        return f"{opened_wall_ms}:{reason}"
    return f"{opened_wall_ms}:rl={int(window_s * 1000)}:{reason}"


def _decode(value: Any) -> Optional[Tuple[int, Optional[int]]]:
    """``(opened_at ms, 429 window ms or None)``, or None when absent or foreign.

    A reason never starts with ``rl=``: it starts with an exception class name.
    """
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    head, _, rest = str(value).partition(":")
    try:
        opened_ms = int(head)
    except ValueError:
        return None
    window_ms: Optional[int] = None
    if rest.startswith("rl="):
        try:
            window_ms = int(rest[3:].split(":", 1)[0])
        except ValueError:
            window_ms = None
    return opened_ms, window_ms


def _opened_ms(value: Any) -> Optional[int]:
    """``opened_at`` from a breaker key value, or None when absent or foreign."""
    decoded = _decode(value)
    return None if decoded is None else decoded[0]
