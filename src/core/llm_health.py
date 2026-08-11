"""Circuit breaker and failure classification for the EVE endpoint chain.

Imports nothing from :mod:`src.core.llm_manager` on purpose: the manager owns
one :class:`EndpointHealth`, and the classification predicate must stay usable
from call sites that cannot import langgraph or the OpenAI SDK.
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict

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
    """Per-process circuit breaker over the endpoint chain.

    One failure opens the circuit: a false positive costs a single answer routed
    to the next endpoint, while a second probe of a dead RunPod endpoint costs
    every request its full cold-start budget. Cooldown expiry deletes the entry,
    so the request after it is the half-open probe.
    """

    def __init__(
        self, cooldown_s: float, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._cooldown_s = cooldown_s
        self._clock = clock
        self._lock = threading.Lock()
        self._states: Dict[str, _EndpointState] = {}

    def is_open(self, llm_type: str) -> bool:
        """True while *llm_type* is still cooling down after a failure."""
        with self._lock:
            state = self._states.get(llm_type)
            if state is None:
                return False
            if self._clock() - state.opened_at >= self._cooldown_s:
                del self._states[llm_type]
                return False
            return True

    def record_failure(self, llm_type: str, exc: BaseException) -> None:
        """Open (or re-open) the circuit for *llm_type*."""
        with self._lock:
            previous = self._states.get(llm_type)
            self._states[llm_type] = _EndpointState(
                opened_at=self._clock(),
                failures=previous.failures + 1 if previous else 1,
                last_error=f"{type(exc).__name__}: {exc}"[:200],
            )
        logger.warning("Endpoint %s circuit opened: %s", llm_type, exc)

    def record_success(self, llm_type: str) -> None:
        """Close the circuit for *llm_type*."""
        with self._lock:
            self._states.pop(llm_type, None)

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Per-endpoint circuit state, for operational readouts."""
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
