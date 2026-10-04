"""Circuit breaker and failure classification for the EVE endpoint chain.

Imports nothing from :mod:`src.core.llm_manager` on purpose: the manager owns
one :class:`EndpointHealth`, and the classification predicate must stay usable
from call sites that cannot import langgraph or the OpenAI SDK.

With a Redis URL the open circuits live in Valkey, so every worker of every
task sees the outage the first one found instead of paying its own probe.
"""

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass
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
        # llm_type -> (checked_at, open_until), both on self._clock.
        self._remote: Dict[str, Tuple[float, float]] = {}
        # Half-open leases this breaker holds.
        self._probing: Set[str] = set()
        # Failures recorded while the store was unreachable.
        self._unsynced: Set[str] = set()

    # ─── public API ──────────────────────────────────────────────────────────

    def is_open(self, llm_type: str) -> bool:
        """True while *llm_type* is still cooling down after a failure."""
        local_open = self._local_is_open(llm_type)
        shared_open = self._shared_is_open(llm_type)
        return local_open if shared_open is None else shared_open

    def record_failure(self, llm_type: str, exc: BaseException) -> None:
        """Open (or re-open) the circuit for *llm_type*."""
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

    def record_success(
        self, llm_type: str, *, started_at: Optional[float] = None
    ) -> None:
        """Close the circuit for *llm_type*.

        *started_at* is the wall-clock time (``time.time()``) the successful
        request started; a shared circuit opened after it stays open.
        """
        now = self._clock()
        with self._lock:
            local_open = llm_type in self._states
            self._states.pop(llm_type, None)
            self._unsynced.discard(llm_type)
            probing = llm_type in self._probing
            self._probing.discard(llm_type)
            cached = self._remote.get(llm_type)
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
                    "open": now - state.opened_at < self._cooldown_s,
                    "failures": state.failures,
                    "last_error": state.last_error,
                    "opened_since_s": round(now - state.opened_at, 3),
                }
                for llm_type, state in self._states.items()
            }

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

    # ─── shared state ────────────────────────────────────────────────────────

    def _key(self, llm_type: str) -> str:
        return f"{self._key_prefix}{llm_type}"

    def _probe_budget(self, llm_type: str) -> float:
        budget = self._probe_s(llm_type) if callable(self._probe_s) else self._probe_s
        return max(float(budget), 1.0)

    def _cache(self, llm_type: str, open_for_s: float) -> None:
        now = self._clock()
        with self._lock:
            self._remote[llm_type] = (now, now + open_for_s)

    def _shared_is_open(self, llm_type: str) -> Optional[bool]:
        """The shared verdict, or None when only the local state can answer."""
        if self._store is None:
            return None
        now = self._clock()
        with self._lock:
            cached = self._remote.get(llm_type)
            if cached is not None and now - cached[0] < self._refresh_s:
                return now < cached[1]
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
        with self._lock:
            self._unsynced.clear()


def _opened_ms(value: Any) -> Optional[int]:
    """``opened_at`` from a breaker key value, or None when absent or foreign."""
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    head = str(value).split(":", 1)[0]
    try:
        return int(head)
    except ValueError:
        return None
