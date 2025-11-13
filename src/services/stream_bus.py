import asyncio
import contextlib
from typing import Dict, Set, AsyncIterator
from src.config import REDIS_URL

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

    async def close(self, key: str):
        async with self._get_lock(key):
            for q in list(self._subscribers.get(key, set())):
                try:
                    q.put_nowait("[[__EOD__]]")
                except asyncio.QueueFull:
                    pass

    async def subscribe(self, key: str) -> AsyncIterator[str]:
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=1000)
        async with self._get_lock(key):
            self._subscribers.setdefault(key, set()).add(q)

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


class RedisStreamBus:
    def __init__(self, url: str):
        if aioredis is None:
            raise RuntimeError("redis asyncio client not available")
        self._redis = aioredis.Redis.from_url(url)

    async def publish(self, key: str, data: str):
        await self._redis.publish(f"sse:{key}", data)

    async def close(self, key: str):
        await self._redis.publish(f"sse:{key}", "[[__EOD__]]")

    async def subscribe(self, key: str) -> AsyncIterator[str]:
        pubsub = self._redis.pubsub()
        channel = f"sse:{key}"
        await pubsub.subscribe(channel)
        try:
            async for msg in pubsub.listen():
                if not msg or msg.get("type") != "message":
                    continue
                data = msg.get("data")
                if isinstance(data, bytes):
                    data = data.decode("utf-8", errors="ignore")
                if data == "[[__EOD__]]":
                    break
                yield data
        finally:
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
