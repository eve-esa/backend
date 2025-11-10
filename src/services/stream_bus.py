import asyncio
from typing import Dict, Set, AsyncIterator


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


_bus = StreamBus()


def get_stream_bus() -> StreamBus:
    return _bus


