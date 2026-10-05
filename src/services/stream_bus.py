import asyncio
import contextlib
from typing import Dict, Optional, Set, AsyncIterator
from src.config import REDIS_URL, redis_client_kwargs

try:
    # Requires redis>=4.2 for asyncio support
    from redis import asyncio as aioredis  # type: ignore
except Exception:
    aioredis = None  # type: ignore


class StreamBus:
    def __init__(self):
        self._subscribers: Dict[str, Set[asyncio.Queue[str]]] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

    def _get_lock(self, key: str) -> asyncio.Lock:
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    async def publish(self, key: str, data: str):
        async with self._get_lock(key):
            for q in list(self._subscribers.get(key, set())):
                try:
                    q.put_nowait(data)
                except asyncio.QueueFull:
                    # Drop if a slow consumer is lagging behind
                    pass

    async def stop(self, key: str, event: str):
        """Send the stopped ``event`` and end the stream."""
        await self.publish(key, event)
        await self.close(key)

    async def close(self, key: str):
        async with self._get_lock(key):
            for q in list(self._subscribers.get(key, set())):
                try:
                    q.put_nowait("[[__EOD__]]")
                except asyncio.QueueFull:
                    pass

    async def subscribe(
        self, key: str, ready: Optional[asyncio.Event] = None
    ) -> AsyncIterator[str]:
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=1000)
        async with self._get_lock(key):
            self._subscribers.setdefault(key, set()).add(q)
        if ready is not None:
            ready.set()

        try:
            while True:
                item = await q.get()
                if item == "[[__EOD__]]":
                    break
                yield item
        finally:
            async with self._get_lock(key):
                subs = self._subscribers.get(key)
                if subs:
                    subs.discard(q)
                if subs and len(subs) == 0:
                    self._subscribers.pop(key, None)
                    self._locks.pop(key, None)


# Long enough for a subscriber that attaches late, short enough to leave no trace.
_STOPPED_TTL_S = 300


def _stopped_key(key: str) -> str:
    return f"sse-stopped:{key}"


class RedisStreamBus:
    def __init__(self, url: str):
        if aioredis is None:
            raise RuntimeError("redis asyncio client not available")
        self._redis = aioredis.Redis.from_url(url, **redis_client_kwargs())

    async def publish(self, key: str, data: str):
        await self._redis.publish(f"sse:{key}", data)

    async def stop(self, key: str, event: str):
        """Send the stopped ``event`` and end the stream.

        Pub/sub keeps nothing, and the Stop can land before the route's
        subscriber attached: the event is also left under a key the
        subscriber reads once subscribed, set before the publish.
        """
        await self._redis.set(_stopped_key(key), event, ex=_STOPPED_TTL_S)
        await self.publish(key, event)
        await self.close(key)

    async def close(self, key: str):
        await self._redis.publish(f"sse:{key}", "[[__EOD__]]")

    async def subscribe(
        self, key: str, ready: Optional[asyncio.Event] = None
    ) -> AsyncIterator[str]:
        pubsub = self._redis.pubsub()
        channel = f"sse:{key}"
        try:
            await pubsub.subscribe(channel)
            # The SUBSCRIBE reply first: past it no later publish can be missed,
            # so a Stop is either delivered below or already under the key.
            await pubsub.get_message(timeout=1.0)
            if ready is not None:
                ready.set()
            stopped = await self._redis.get(_stopped_key(key))
            if stopped is not None:
                if isinstance(stopped, bytes):
                    stopped = stopped.decode("utf-8", errors="ignore")
                yield stopped
                return
            while True:
                msg = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=None
                )
                if not msg or msg.get("type") != "message":
                    continue
                data = msg.get("data")
                if isinstance(data, bytes):
                    data = data.decode("utf-8", errors="ignore")
                if data == "[[__EOD__]]":
                    break
                yield data
        finally:
            # A failed subscribe must not leave the producer waiting for it.
            if ready is not None:
                ready.set()
            with contextlib.suppress(Exception):
                await pubsub.unsubscribe(channel)
            with contextlib.suppress(Exception):
                await pubsub.close()


_bus = None
if REDIS_URL and aioredis is not None:
    try:
        _bus = RedisStreamBus(REDIS_URL)
    except Exception:
        _bus = StreamBus()
else:
    _bus = StreamBus()


def get_stream_bus():
    return _bus
