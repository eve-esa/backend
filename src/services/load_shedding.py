"""Per-worker cap on in-flight answer generations.

Every gunicorn worker runs its own event loop, and each answer generation holds
model, retrieval and database calls open for tens of seconds. Past a point an
extra generation does not get served later, it makes every running stream
slower. So each worker admits at most ``MAX_INFLIGHT_GENERATIONS_PER_WORKER``
generations and refuses the next one at once with a 429 the client can retry.

A slot is held for the life of the generation, not of the HTTP request: the
streaming routes run generation in a task decoupled from the response, which
keeps working after the client goes away, so the slot follows that task.
"""

import asyncio
import logging
from typing import Optional

from fastapi import HTTPException

from src.config import MAX_INFLIGHT_GENERATIONS_PER_WORKER

logger = logging.getLogger(__name__)

SHED_RETRY_AFTER_SECONDS = 10
SHED_DETAIL = {
    "code": "overloaded",
    "message": "The service is busy, retry in a few seconds",
}


class GenerationSlot:
    """One admitted generation. ``release`` is idempotent."""

    __slots__ = ("_limiter", "_released")

    def __init__(self, limiter: "GenerationLimiter") -> None:
        self._limiter = limiter
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._limiter._release()

    def release_when_done(self, task: asyncio.Task) -> None:
        """Hand the slot to ``task``: freed when it ends, fails or is cancelled."""
        task.add_done_callback(lambda _task: self.release())


class GenerationLimiter:
    """Non-blocking admission: a request either gets a slot now or is shed."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        # Bounded, so a release without a matching acquire fails loudly
        # instead of silently raising the cap.
        self._semaphore: Optional[asyncio.BoundedSemaphore] = (
            asyncio.BoundedSemaphore(limit) if limit > 0 else None
        )
        self._in_flight = 0

    @property
    def in_flight(self) -> int:
        return self._in_flight

    async def try_acquire(self) -> Optional[GenerationSlot]:
        semaphore = self._semaphore
        if semaphore is not None:
            if semaphore.locked():
                return None
            # Never waits: locked() is False, so acquire() takes its fast path
            # and returns without yielding to the loop.
            await semaphore.acquire()
        self._in_flight += 1
        return GenerationSlot(self)

    def _release(self) -> None:
        self._in_flight -= 1
        if self._semaphore is not None:
            self._semaphore.release()


_limiter = GenerationLimiter(MAX_INFLIGHT_GENERATIONS_PER_WORKER)


def get_generation_limiter() -> GenerationLimiter:
    return _limiter


async def acquire_generation_slot_or_raise(route: str) -> GenerationSlot:
    """Admit one generation on ``route`` or raise 429 ``overloaded``."""
    limiter = _limiter
    slot = await limiter.try_acquire()
    if slot is None:
        logger.warning(
            "Load shed on %s: %d generations in flight, cap %d per worker",
            route,
            limiter.in_flight,
            limiter.limit,
        )
        raise HTTPException(
            status_code=429,
            detail=SHED_DETAIL,
            headers={"Retry-After": str(SHED_RETRY_AFTER_SECONDS)},
        )
    return slot
