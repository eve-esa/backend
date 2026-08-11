"""Keepalive wrapper for SSE response generators.

Every hop between the browser and this app enforces an inter-packet timeout:
CloudFront's origin_read_timeout is 60s (the maximum without a service quota
increase) and the frontend stream reader aborts after 60s without progress.
A model cold start keeps an SSE stream silent for minutes, so without traffic
the connection dies mid-generation even though the generation itself succeeds.

The wrapper interleaves an SSE comment line whenever the wrapped generator
stays silent for longer than ``interval`` seconds. Comment lines are invisible
to SSE parsers, and the frontend line parser skips anything that is not a JSON
object, but every hop's inter-packet timer resets on the bytes.
"""

import asyncio
import contextlib
from typing import AsyncIterator

SSE_KEEPALIVE_COMMENT = ": keepalive\n\n"

# Comfortably under the tightest inter-packet timeout in the chain (60s on
# both CloudFront and the frontend watchdog), with margin for scheduling lag.
DEFAULT_KEEPALIVE_INTERVAL_S = 15.0


async def with_sse_keepalive(
    source: AsyncIterator[str],
    interval: float = DEFAULT_KEEPALIVE_INTERVAL_S,
) -> AsyncIterator[str]:
    """Yield every item from ``source``, emitting keepalive comments during silence.

    Items pass through unchanged and in order. Exceptions raised by ``source``
    propagate to the caller. On early close (client disconnect) the pending
    read is cancelled and ``source`` is closed.
    """
    iterator = source.__aiter__()
    pending: asyncio.Task | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(iterator.__anext__())
            done, _ = await asyncio.wait({pending}, timeout=interval)
            if not done:
                yield SSE_KEEPALIVE_COMMENT
                continue
            task, pending = pending, None
            try:
                item = task.result()
            except StopAsyncIteration:
                return
            yield item
    finally:
        if pending is not None:
            pending.cancel()
            with contextlib.suppress(BaseException):
                await pending
        aclose = getattr(iterator, "aclose", None)
        if aclose is not None:
            with contextlib.suppress(BaseException):
                await aclose()
