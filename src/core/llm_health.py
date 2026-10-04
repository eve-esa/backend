"""Circuit breaker and failure classification for the EVE endpoint chain.

Imports nothing from :mod:`src.core.llm_manager` on purpose: the manager owns
one :class:`EndpointHealth`, and the classification predicate must stay usable
from call sites that cannot import langgraph or the OpenAI SDK.

With a Redis URL the open circuits live in Valkey, so every worker of every
task sees the outage the first one found instead of paying its own probe.
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

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
    failures: int
    last_error: str


class EndpointHealth:
    """Circuit breaker over the endpoint chain, shared through Redis when it can be.

    One failure opens the circuit: a false positive costs a single answer routed
    to the next endpoint, while a second probe of a dead RunPod endpoint costs
    every request its full cold-start budget. Cooldown expiry deletes the entry,
    so the request after it is the half-open probe.

    With *redis_url* (or *redis_client*) set, an open circuit is the key
    ``<prefix><llm_type>`` with the cooldown as its expiry: one worker's failure
    opens the circuit for every worker, one success closes it for all, and the
    key expiring is the fleet-wide half-open moment. While Redis answers it is
    authoritative. The per-process state is always written too and takes over
    whenever Redis is unset or unreachable, so the breaker never raises and
    never stops working because the store is down.

    The public methods stay synchronous because ``resolve_chain`` and its
    callers are. A read is one ``PTTL`` round trip per endpoint at most every
    *refresh_s*, bounded by *redis_timeout_s*; after a Redis error the store is
    skipped for *redis_retry_s*, so an outage costs one bounded stall per
    worker per retry window, not one per request.
    """

    def __init__(
        self,
        cooldown_s: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        redis_url: Optional[str] = None,
        redis_client: Any = None,
        key_prefix: str = "eve:breaker:",
        refresh_s: float = 1.0,
        redis_timeout_s: float = 0.25,
        redis_retry_s: float = 30.0,
    ) -> None:
        self._cooldown_s = cooldown_s
        self._clock = clock
        self._lock = threading.Lock()
        self._states: Dict[str, _EndpointState] = {}

        self._redis_url = (redis_url or "").strip()
        self._redis = redis_client
        self._shared = redis_client is not None or bool(self._redis_url)
        self._key_prefix = key_prefix
        self._refresh_s = refresh_s
        self._redis_timeout_s = redis_timeout_s
        self._redis_retry_s = redis_retry_s
        self._redis_skip_until = 0.0
        self._redis_degraded = False
        # llm_type -> (checked_at, open_until), both on self._clock.
        self._remote: Dict[str, Tuple[float, float]] = {}

    # ─── public API ──────────────────────────────────────────────────────────

    def is_open(self, llm_type: str) -> bool:
        """True while *llm_type* is still cooling down after a failure."""
        local_open = self._local_is_open(llm_type)
        shared_open = self._shared_is_open(llm_type)
        return local_open if shared_open is None else shared_open

    def record_failure(self, llm_type: str, exc: BaseException) -> None:
        """Open (or re-open) the circuit for *llm_type*."""
        reason = f"{type(exc).__name__}: {exc}"[:200]
        with self._lock:
            previous = self._states.get(llm_type)
            self._states[llm_type] = _EndpointState(
                opened_at=self._clock(),
                failures=previous.failures + 1 if previous else 1,
                last_error=reason,
            )
        if self._cooldown_s > 0:
            self._shared_call(
                lambda client: client.set(
                    self._key(llm_type), reason, px=int(self._cooldown_s * 1000)
                ),
                llm_type,
                open_for_s=self._cooldown_s,
            )
        logger.warning("Endpoint %s circuit opened: %s", llm_type, exc)

    def record_success(self, llm_type: str) -> None:
        """Close the circuit for *llm_type*."""
        with self._lock:
            self._states.pop(llm_type, None)
        self._shared_call(
            lambda client: client.delete(self._key(llm_type)),
            llm_type,
            open_for_s=0.0,
        )

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
                return False
            return True

    # ─── shared state ────────────────────────────────────────────────────────

    def _key(self, llm_type: str) -> str:
        return f"{self._key_prefix}{llm_type}"

    def _shared_is_open(self, llm_type: str) -> Optional[bool]:
        """The shared verdict, or None when only the local state can answer."""
        if not self._shared:
            return None
        now = self._clock()
        with self._lock:
            cached = self._remote.get(llm_type)
            if cached is not None and now - cached[0] < self._refresh_s:
                return now < cached[1]
        client = self._client()
        if client is None:
            return None
        try:
            ttl_ms = client.pttl(self._key(llm_type))
        except Exception as exc:  # noqa: BLE001 - any store error means fall back
            self._store_failed(exc)
            return None
        self._store_answered()
        now = self._clock()
        if ttl_ms is None or ttl_ms == -2:
            open_until = now
        elif ttl_ms == -1:
            # A key without expiry never comes from this class; honour it for
            # one cooldown rather than forever.
            open_until = now + self._cooldown_s
        else:
            open_until = now + ttl_ms / 1000.0
        with self._lock:
            self._remote[llm_type] = (now, open_until)
        return now < open_until

    def _shared_call(
        self, op: Callable[[Any], Any], llm_type: str, *, open_for_s: float
    ) -> None:
        client = self._client()
        if client is None:
            return
        try:
            op(client)
        except Exception as exc:  # noqa: BLE001 - any store error means fall back
            self._store_failed(exc)
            return
        self._store_answered()
        now = self._clock()
        with self._lock:
            self._remote[llm_type] = (now, now + open_for_s)

    def _client(self) -> Any:
        if not self._shared or self._clock() < self._redis_skip_until:
            return None
        if self._redis is None:
            try:
                import redis

                client = redis.Redis.from_url(
                    self._redis_url,
                    socket_connect_timeout=self._redis_timeout_s,
                    socket_timeout=self._redis_timeout_s,
                )
            except Exception as exc:  # noqa: BLE001 - a bad URL must not break the chain
                self._store_failed(exc)
                return None
            with self._lock:
                if self._redis is None:
                    self._redis = client
        return self._redis

    def _store_failed(self, exc: BaseException) -> None:
        with self._lock:
            self._redis_skip_until = self._clock() + self._redis_retry_s
            self._remote.clear()
            first = not self._redis_degraded
            self._redis_degraded = True
        if first:
            logger.warning(
                "Endpoint breaker store unreachable, using per-process state "
                "(retry every %ss): %s: %s",
                self._redis_retry_s,
                type(exc).__name__,
                exc,
            )

    def _store_answered(self) -> None:
        if not self._redis_degraded:
            return
        with self._lock:
            recovered = self._redis_degraded
            self._redis_degraded = False
        if recovered:
            logger.info("Endpoint breaker store reachable again, circuits shared")
