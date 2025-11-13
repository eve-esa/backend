import asyncio
import logging
from typing import Dict, Optional
from src.config import REDIS_URL

try:
    # Prefer redis>=4.2 which provides asyncio interface
    from redis import asyncio as aioredis  # type: ignore
except Exception:
    aioredis = None  # type: ignore


class CancelManager:
    def __init__(self):
        self._events: Dict[str, asyncio.Event] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._conv_to_msg: Dict[str, str] = {}
        self._logger = logging.getLogger(__name__)
        # Redis-related fields (lazy init)
        self._redis = None
        self._pubsub = None
        self._channel_tasks: Dict[str, asyncio.Task] = {}

    def _is_redis_enabled(self) -> bool:
        return bool(REDIS_URL) and aioredis is not None

    async def _ensure_redis(self):
        if not self._is_redis_enabled():
            return
        if self._redis is None:
            try:
                self._redis = aioredis.Redis.from_url(REDIS_URL)
            except Exception as e:
                self._logger.warning("Redis init failed: %s", str(e))
                self._redis = None
                return
        if self._pubsub is None and self._redis is not None:
            try:
                self._pubsub = self._redis.pubsub()
            except Exception as e:
                self._logger.warning("Redis pubsub init failed: %s", str(e))
                self._pubsub = None

    async def _subscribe_cancel_channel(self, message_id: str, ev: asyncio.Event):
        try:
            await self._ensure_redis()
            if self._pubsub is None:
                return
            channel = f"cancel:{message_id}"
            await self._pubsub.subscribe(channel)
            self._logger.info("cancel_manager.redis_subscribed channel=%s", channel)
            # Listen loop
            async for msg in self._pubsub.listen():
                if msg is None:
                    continue
                try:
                    if msg.get("type") != "message":
                        continue
                    data = msg.get("data")
                    if isinstance(data, bytes):
                        data = data.decode("utf-8", errors="ignore")
                    if data == "1" or data == "cancel":
                        ev.set()
                        self._logger.info(
                            "cancel_manager.redis_received_cancel message_id=%s",
                            message_id,
                        )
                        break
                except Exception:
                    continue
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._logger.warning("cancel_manager.redis_subscribe_error: %s", str(e))
        finally:
            try:
                if self._pubsub is not None:
                    await self._pubsub.unsubscribe(f"cancel:{message_id}")
            except Exception:
                pass

    def create(
        self, message_id: str, task: Optional[asyncio.Task] = None
    ) -> asyncio.Event:
        ev = self._events.get(message_id)
        if ev is None:
            ev = asyncio.Event()
            self._events[message_id] = ev
        if task is not None:
            self._tasks[message_id] = task
        # If Redis is enabled, start a background subscription for this message_id
        if self._is_redis_enabled() and message_id not in self._channel_tasks:
            try:
                loop = asyncio.get_running_loop()
                self._channel_tasks[message_id] = loop.create_task(
                    self._subscribe_cancel_channel(message_id, ev)
                )
            except Exception:
                pass
        try:
            self._logger.info("cancel_manager.create message_id=%s", message_id)
        except Exception:
            pass
        return ev

    def set_task(self, message_id: str, task: asyncio.Task) -> None:
        self._tasks[message_id] = task
        try:
            self._logger.info("cancel_manager.set_task message_id=%s", message_id)
        except Exception:
            pass

    def get_event(self, message_id: str) -> Optional[asyncio.Event]:
        return self._events.get(message_id)

    def cancel(self, message_id: str) -> None:
        ev = self._events.get(message_id)
        if ev is not None:
            ev.set()
        task = self._tasks.get(message_id)
        if task is not None and not task.done():
            task.cancel()
        # Publish cancel to Redis so other workers receive it
        if self._is_redis_enabled():
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._publish_cancel(message_id))
            except Exception:
                pass
        try:
            self._logger.info("cancel_manager.cancel message_id=%s", message_id)
        except Exception:
            pass

    async def _publish_cancel(self, message_id: str):
        try:
            await self._ensure_redis()
            if self._redis is None:
                return
            await self._redis.publish(f"cancel:{message_id}", "cancel")
        except Exception as e:
            self._logger.warning("cancel_manager.redis_publish_error: %s", str(e))

    def clear(self, message_id: str) -> None:
        self._events.pop(message_id, None)
        self._tasks.pop(message_id, None)
        # Stop channel subscription task if any
        task = self._channel_tasks.pop(message_id, None)
        if task is not None:
            try:
                task.cancel()
            except Exception:
                pass
        # Also clear any conversation mapping pointing to this message
        try:
            to_delete = []
            for conv_id, mid in self._conv_to_msg.items():
                if mid == message_id:
                    to_delete.append(conv_id)
            for conv_id in to_delete:
                self._conv_to_msg.pop(conv_id, None)
        except Exception:
            pass
        try:
            self._logger.info("cancel_manager.clear message_id=%s", message_id)
        except Exception:
            pass

    def link_conversation(self, conversation_id: str, message_id: str) -> None:
        self._conv_to_msg[conversation_id] = message_id
        # Also store mapping in Redis for cross-process lookup
        if self._is_redis_enabled():
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._hset_mapping(conversation_id, message_id))
            except Exception:
                pass
        try:
            self._logger.info(
                "cancel_manager.link conversation_id=%s message_id=%s",
                conversation_id,
                message_id,
            )
        except Exception:
            pass

    def get_message_for_conversation(self, conversation_id: str) -> Optional[str]:
        # Prefer local mapping first; if missing and Redis enabled, fetch from Redis synchronously via a helper
        local = self._conv_to_msg.get(conversation_id)
        if local:
            return local
        if self._is_redis_enabled():
            try:
                loop = asyncio.get_running_loop()
                return loop.run_until_complete(self._hget_mapping(conversation_id))  # type: ignore
            except Exception:
                return None
        return None

    async def get_message_for_conversation_async(
        self, conversation_id: str
    ) -> Optional[str]:
        local = self._conv_to_msg.get(conversation_id)
        if local:
            return local
        if self._is_redis_enabled():
            try:
                return await self._hget_mapping(conversation_id)
            except Exception:
                return None
        return None

    def cancel_by_conversation(self, conversation_id: str) -> Optional[str]:
        message_id = self._conv_to_msg.get(conversation_id)
        if message_id:
            self.cancel(message_id)
            try:
                self._logger.info(
                    "cancel_manager.cancel_by_conversation conversation_id=%s message_id=%s",
                    conversation_id,
                    message_id,
                )
            except Exception:
                pass
        return message_id

    def clear_mapping_for(self, conversation_id: str, message_id: str) -> None:
        current = self._conv_to_msg.get(conversation_id)
        if current == message_id:
            self._conv_to_msg.pop(conversation_id, None)
            # Remove from Redis mapping too
            if self._is_redis_enabled():
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._hdel_mapping(conversation_id))
                except Exception:
                    pass
            try:
                self._logger.info(
                    "cancel_manager.clear_mapping_for conversation_id=%s message_id=%s",
                    conversation_id,
                    message_id,
                )
            except Exception:
                pass

    async def _hset_mapping(self, conversation_id: str, message_id: str):
        try:
            await self._ensure_redis()
            if self._redis is None:
                return
            await self._redis.hset("conv_to_msg", conversation_id, message_id)
        except Exception as e:
            self._logger.warning("cancel_manager.redis_hset_error: %s", str(e))

    async def _hget_mapping(self, conversation_id: str) -> Optional[str]:
        try:
            await self._ensure_redis()
            if self._redis is None:
                return None
            val = await self._redis.hget("conv_to_msg", conversation_id)
            if isinstance(val, bytes):
                return val.decode("utf-8", errors="ignore")
            return str(val) if val is not None else None
        except Exception as e:
            self._logger.warning("cancel_manager.redis_hget_error: %s", str(e))
            return None

    async def _hdel_mapping(self, conversation_id: str):
        try:
            await self._ensure_redis()
            if self._redis is None:
                return
            await self._redis.hdel("conv_to_msg", conversation_id)
        except Exception as e:
            self._logger.warning("cancel_manager.redis_hdel_error: %s", str(e))


_cancel_mgr = CancelManager()


def get_cancel_manager() -> CancelManager:
    return _cancel_mgr
