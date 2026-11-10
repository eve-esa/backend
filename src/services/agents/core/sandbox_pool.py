"""One sandbox per conversation, shared by every worker, killed after it goes idle.

The map lives in Redis when ``REDIS_URL`` is set, because a later message can
land on another worker. Without it, the map is in this process. A turn with no
conversation does not come here: ``open_sandbox`` still creates and destroys
one for that turn.

The lock covers create, attach, and the reaper's kill. It is not held while
the turn runs. Two turns of one conversation therefore share the sandbox. The
deadline stays past the turn cap until the last of those turns releases, and
the release then restarts the idle timer.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from contextvars import ContextVar
from typing import Any, AsyncIterator, Optional, Protocol

from src.config import REDIS_URL, redis_client_kwargs
from src.services.agents.core.sandbox import (
    SandboxError,
    SandboxGone,
    SandboxSession,
    _open_sync,
    attach_session,
    kill_record,
    sandbox_busy_ttl_s,
    sandbox_idle_ttl_s,
    sandbox_provider,
    sandbox_reaper_interval_s,
    sandbox_reuse_limit_s,
    sync_skills,
)

logger = logging.getLogger(__name__)

_IDLE_KEY = "sandbox:idle"
_LOCK_TTL_MS = 180_000
_LOCK_POLL_S = 0.2
_UNLOCK_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
end
return 0
"""
# A create holds the lock for the whole provider start. The TTL is refreshed
# while that section runs, or a slow start would let a second worker in.
_REFRESH_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("pexpire", KEYS[1], ARGV[2])
end
return 0
"""
# The in-flight count and its deadline change together, and only while this
# worker owns the lock. A count that moved without its lease would never be
# released. KEYS: lock, record, idle zset. ARGV: token, delta, member,
# deadline, and "1" to set the deadline whatever the count is.
_TURN_SCRIPT = """
if redis.call("get", KEYS[1]) ~= ARGV[1] then
    return {0, 0}
end
local inflight = redis.call("hincrby", KEYS[2], "inflight", ARGV[2])
if ARGV[5] == "1" or inflight <= 0 then
    redis.call("zadd", KEYS[3], ARGV[4], ARGV[3])
end
return {1, inflight}
"""

try:
    from redis import asyncio as aioredis  # type: ignore
except Exception:
    aioredis = None  # type: ignore


def _conv_key(conversation_id: str) -> str:
    return f"sandbox:conv:{conversation_id}"


def _lock_key(conversation_id: str) -> str:
    return f"sandbox:lock:{conversation_id}"


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return "" if value is None else str(value)


class SandboxLockLost(SandboxError):
    """The Redis lock expired or was taken by another worker."""


class LockGuard:
    """A held conversation lock. ``check`` raises once the lock is gone.

    The holder awaits ``check`` before each write, or it can overwrite a
    record another worker now owns. An in-process lock is never lost.
    """

    def __init__(self) -> None:
        self.lost = False

    async def check(self) -> None:
        if self.lost:
            raise SandboxLockLost("sandbox lock lost")


class RedisLockGuard(LockGuard):
    """A Redis lock. ``check`` asks Redis rather than trusting the last refresh.

    A passing check also extends the key by ``_LOCK_TTL_MS``, so the write that
    follows it runs while this worker still owns the lock.
    """

    def __init__(self, client: Any, key: str, token: str) -> None:
        super().__init__()
        self.client = client
        self.key = key
        self.token = token

    async def extend(self) -> bool:
        """Reset the key's TTL. False, and ``lost``, when another token owns it."""
        extended = await self.client.eval(
            _REFRESH_SCRIPT, 1, self.key, self.token, str(_LOCK_TTL_MS)
        )
        if not extended:
            self.lost = True
        return bool(extended)

    async def check(self) -> None:
        await super().check()
        if not await self.extend():
            raise SandboxLockLost("sandbox lock lost")


async def _refresh_lock(guard: RedisLockGuard, conversation_id: str) -> None:
    """Keep a held Redis lock alive until this task is cancelled.

    Only a refresh that finds another token marks the lock lost. A Redis error
    is retried on the next tick: the key outlives two missed refreshes.
    """
    while True:
        await asyncio.sleep(_LOCK_TTL_MS / 3000)
        try:
            extended = await guard.extend()
        except Exception:
            logger.warning(
                "sandbox.lock_refresh_failed conversation_id=%s",
                conversation_id,
                exc_info=True,
            )
            continue
        if not extended:
            logger.warning("sandbox.lock_lost conversation_id=%s", conversation_id)
            return


def _within_lifetime(record: dict[str, str], now: float) -> bool:
    """False when a hard provider cap would run out during the next turn.

    OpenSandbox has no such cap: ``renew`` moves the expiry, so a record
    without ``created_at`` stays reusable. An unreadable cap is already
    expired, and that record is replaced.
    """
    created = record.get("created_at")
    lifetime = record.get("max_lifetime_s")
    if not created or not lifetime:
        return True
    try:
        remaining = float(created) + float(lifetime) - now
    except ValueError:
        return False
    return remaining > sandbox_reuse_limit_s()


def _stored_record(session: SandboxSession) -> dict[str, str]:
    record = {
        "provider": session.provider,
        "sandbox_id": session.sandbox_id,
        "skills_digest": session.skills_digest,
        "inflight": "0",
    }
    for key, value in session.metadata.items():
        record[key] = str(value)
    return record


class SandboxStore(Protocol):
    """The conversation map, the idle deadlines, and the per-conversation lock."""

    def hold(
        self, conversation_id: str, *, wait: bool
    ) -> AsyncIterator[Optional[LockGuard]]: ...

    async def get(self, conversation_id: str) -> Optional[dict[str, str]]: ...

    async def put(self, conversation_id: str, record: dict[str, str]) -> None: ...

    async def delete(self, conversation_id: str) -> None: ...

    async def add_inflight(self, conversation_id: str, delta: int) -> int: ...

    async def set_inflight(self, conversation_id: str, value: int) -> None: ...

    async def set_deadline(self, conversation_id: str, when: float) -> None: ...

    async def start_turn(
        self, conversation_id: str, guard: LockGuard, when: float
    ) -> int:
        """Count one more turn and set its deadline, or raise ``SandboxLockLost``."""
        ...

    async def finish_turn(
        self, conversation_id: str, guard: LockGuard, when: float
    ) -> int:
        """Count one turn less. The last one sets *when* as the idle deadline."""
        ...

    async def deadline(self, conversation_id: str) -> Optional[float]: ...

    async def due(self, now: float) -> list[str]: ...


class MemorySandboxStore:
    """In-process map. Used when Redis is unset, and by the unit tests."""

    def __init__(self) -> None:
        self._records: dict[str, dict[str, str]] = {}
        self._deadlines: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock(self, conversation_id: str) -> asyncio.Lock:
        lock = self._locks.get(conversation_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[conversation_id] = lock
        return lock

    @contextlib.asynccontextmanager
    async def hold(
        self, conversation_id: str, *, wait: bool
    ) -> AsyncIterator[Optional[LockGuard]]:
        lock = self._lock(conversation_id)
        if not wait and lock.locked():
            yield None
            return
        await lock.acquire()
        try:
            yield LockGuard()
        finally:
            lock.release()

    async def get(self, conversation_id: str) -> Optional[dict[str, str]]:
        record = self._records.get(conversation_id)
        return None if record is None else dict(record)

    async def put(self, conversation_id: str, record: dict[str, str]) -> None:
        self._records[conversation_id] = {key: str(value) for key, value in record.items()}

    async def delete(self, conversation_id: str) -> None:
        self._records.pop(conversation_id, None)
        self._deadlines.pop(conversation_id, None)

    async def add_inflight(self, conversation_id: str, delta: int) -> int:
        record = self._records.get(conversation_id)
        if record is None:
            record = {}
            self._records[conversation_id] = record
        value = int(record.get("inflight", "0")) + delta
        record["inflight"] = str(value)
        return value

    async def set_inflight(self, conversation_id: str, value: int) -> None:
        record = self._records.get(conversation_id)
        if record is None:
            return
        record["inflight"] = str(value)

    async def set_deadline(self, conversation_id: str, when: float) -> None:
        self._deadlines[conversation_id] = when

    async def start_turn(
        self, conversation_id: str, guard: LockGuard, when: float
    ) -> int:
        await guard.check()
        value = await self.add_inflight(conversation_id, 1)
        self._deadlines[conversation_id] = when
        return value

    async def finish_turn(
        self, conversation_id: str, guard: LockGuard, when: float
    ) -> int:
        await guard.check()
        value = await self.add_inflight(conversation_id, -1)
        if value <= 0:
            self._deadlines[conversation_id] = when
        return value

    async def deadline(self, conversation_id: str) -> Optional[float]:
        return self._deadlines.get(conversation_id)

    async def due(self, now: float) -> list[str]:
        return [key for key, when in self._deadlines.items() if when <= now]


class RedisSandboxStore:
    """The same map in Redis, so any worker can reattach."""

    def __init__(self) -> None:
        self._redis: Any = None

    async def _client(self) -> Any:
        if aioredis is None:
            raise SandboxError("redis is not available")
        if self._redis is None:
            self._redis = aioredis.Redis.from_url(REDIS_URL, **redis_client_kwargs())
        return self._redis

    @contextlib.asynccontextmanager
    async def hold(
        self, conversation_id: str, *, wait: bool
    ) -> AsyncIterator[Optional[LockGuard]]:
        client = await self._client()
        key = _lock_key(conversation_id)
        token: Optional[str] = None
        while token is None:
            candidate = uuid.uuid4().hex
            if await client.set(key, candidate, nx=True, px=_LOCK_TTL_MS):
                token = candidate
                break
            # A live holder refreshes the key and then releases it. A dead
            # holder stops refreshing, so the key expires within _LOCK_TTL_MS.
            if not wait:
                break
            await asyncio.sleep(_LOCK_POLL_S)
        if token is None:
            yield None
            return
        guard = RedisLockGuard(client, key, token)
        refresh = asyncio.create_task(_refresh_lock(guard, conversation_id))
        try:
            yield guard
        finally:
            refresh.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await refresh
            try:
                await client.eval(_UNLOCK_SCRIPT, 1, key, token)
            except Exception:
                logger.exception(
                    "sandbox.unlock_failed conversation_id=%s", conversation_id
                )

    async def get(self, conversation_id: str) -> Optional[dict[str, str]]:
        client = await self._client()
        raw = await client.hgetall(_conv_key(conversation_id))
        if not raw:
            return None
        return {_text(key): _text(value) for key, value in raw.items()}

    async def put(self, conversation_id: str, record: dict[str, str]) -> None:
        client = await self._client()
        mapping = {key: str(value) for key, value in record.items()}
        await client.hset(_conv_key(conversation_id), mapping=mapping)

    async def delete(self, conversation_id: str) -> None:
        client = await self._client()
        await client.delete(_conv_key(conversation_id))
        await client.zrem(_IDLE_KEY, conversation_id)

    async def add_inflight(self, conversation_id: str, delta: int) -> int:
        client = await self._client()
        value = await client.hincrby(_conv_key(conversation_id), "inflight", delta)
        return int(value)

    async def set_inflight(self, conversation_id: str, value: int) -> None:
        client = await self._client()
        await client.hset(_conv_key(conversation_id), "inflight", str(value))

    async def set_deadline(self, conversation_id: str, when: float) -> None:
        client = await self._client()
        await client.zadd(_IDLE_KEY, {conversation_id: when})

    async def _turn(
        self,
        conversation_id: str,
        guard: LockGuard,
        delta: int,
        when: float,
        always: bool,
    ) -> int:
        if not isinstance(guard, RedisLockGuard):
            raise SandboxError("a Redis turn needs the Redis lock guard")
        client = await self._client()
        owned, inflight = await client.eval(
            _TURN_SCRIPT,
            3,
            guard.key,
            _conv_key(conversation_id),
            _IDLE_KEY,
            guard.token,
            str(delta),
            conversation_id,
            repr(float(when)),
            "1" if always else "0",
        )
        if not int(owned):
            guard.lost = True
            raise SandboxLockLost("sandbox lock lost")
        return int(inflight)

    async def start_turn(
        self, conversation_id: str, guard: LockGuard, when: float
    ) -> int:
        return await self._turn(conversation_id, guard, 1, when, True)

    async def finish_turn(
        self, conversation_id: str, guard: LockGuard, when: float
    ) -> int:
        return await self._turn(conversation_id, guard, -1, when, False)

    async def deadline(self, conversation_id: str) -> Optional[float]:
        client = await self._client()
        score = await client.zscore(_IDLE_KEY, conversation_id)
        return None if score is None else float(score)

    async def due(self, now: float) -> list[str]:
        client = await self._client()
        found = await client.zrangebyscore(_IDLE_KEY, "-inf", now)
        return [_text(item) for item in found]


class SandboxLease:
    """A sandbox held for one turn. Release it; do not destroy it.

    ``fresh`` is true when this turn started the sandbox, including a
    replacement for one that was dead or past its hard cap. A reused sandbox
    already holds the tool results written into it, so it is not refilled.
    """

    def __init__(
        self,
        session: SandboxSession,
        conversation_id: str,
        *,
        fresh: bool = False,
    ) -> None:
        self.session = session
        self.conversation_id = conversation_id
        self.fresh = fresh

    @property
    def backend(self) -> Any:
        return self.session.backend


_default_store: Optional[SandboxStore] = None
_store_override: ContextVar[Optional[SandboxStore]] = ContextVar(
    "sandbox_store", default=None
)
_reaper_task: Optional[asyncio.Task[None]] = None


def get_store() -> SandboxStore:
    """The process store, or the store a test installed for this context."""
    override = _store_override.get()
    if override is not None:
        return override
    global _default_store
    if _default_store is None:
        if REDIS_URL and aioredis is not None:
            _default_store = RedisSandboxStore()
        else:
            _default_store = MemorySandboxStore()
    return _default_store


def override_sandbox_store(store: SandboxStore) -> Any:
    """Use *store* for this context. Return the token ``reset_sandbox_store`` takes."""
    return _store_override.set(store)


def reset_sandbox_store(token: Any) -> None:
    _store_override.reset(token)


async def _drop_stale_inflight(
    store: SandboxStore,
    conversation_id: str,
    record: dict[str, str],
    now: float,
    guard: LockGuard,
) -> dict[str, str]:
    """A deadline already past means the turn that counted inflight is gone.

    Leaving that count set skips the idle timer and also skips replacing an
    AgentCore session that is near its hard cap.
    """
    deadline = await store.deadline(conversation_id)
    if deadline is not None and deadline > now:
        return record
    if int(record.get("inflight") or "0") == 0:
        return record
    await guard.check()
    await store.set_inflight(conversation_id, 0)
    record["inflight"] = "0"
    return record


async def _retire(
    store: SandboxStore,
    conversation_id: str,
    record: dict[str, str],
    guard: LockGuard,
) -> bool:
    """Stop the sandbox and drop its record. False leaves the record for a retry."""
    await guard.check()
    try:
        await asyncio.to_thread(kill_record, record)
    except Exception:
        logger.exception(
            "sandbox.kill_failed conversation_id=%s sandbox_id=%s",
            conversation_id,
            record.get("sandbox_id"),
        )
        return False
    await guard.check()
    await store.delete(conversation_id)
    return True


async def _replace(
    store: SandboxStore,
    conversation_id: str,
    record: dict[str, str],
    guard: LockGuard,
) -> None:
    """Stop *record* before a new sandbox takes its place.

    A failed stop keeps the record. Creating a replacement would forget the
    id, and the reaper could not try the stop again.
    """
    if not await _retire(store, conversation_id, record, guard):
        raise SandboxError(f"could not stop sandbox {record.get('sandbox_id')}")


async def acquire_sandbox(
    conversation_id: str,
    skills_root: Any = None,
    on_fresh: Any = None,
) -> Optional[SandboxLease]:
    """Return the conversation's sandbox, starting one when it has none.

    ``None`` when ``SANDBOX_PROVIDER`` is unset or ``none``. A sandbox that is
    dead, from another provider, or inside its hard-TTL margin is destroyed
    and replaced. The idle deadline moves out by the turn cap plus the idle
    window, and the provider expiry moves with it when the provider allows.

    ``on_fresh`` runs only when this turn started the sandbox, and it runs
    before the lock is released. A reused sandbox does not call it. A failure
    is logged and swallowed so the lease is still returned.
    """
    provider = sandbox_provider()
    if provider in ("", "none"):
        return None
    if not conversation_id:
        raise SandboxError("a pooled sandbox needs a conversation id")
    store = get_store()
    async with store.hold(conversation_id, wait=True) as guard:
        if guard is None:
            raise SandboxError("sandbox lock was not acquired")
        lease = await _acquire_locked(
            store, conversation_id, provider, skills_root, guard
        )
        if lease.fresh and on_fresh is not None:
            try:
                await on_fresh(lease)
            except Exception:
                logger.warning(
                    "sandbox.on_fresh_failed conversation_id=%s",
                    conversation_id,
                    exc_info=True,
                )
        return lease


async def _acquire_locked(
    store: SandboxStore,
    conversation_id: str,
    provider: str,
    skills_root: Any,
    guard: LockGuard,
) -> SandboxLease:
    record = await store.get(conversation_id)
    now = time.time()
    if record is not None:
        record = await _drop_stale_inflight(
            store, conversation_id, record, now, guard
        )
    session: Optional[SandboxSession] = None
    fresh = False
    inflight = int(record.get("inflight", "0")) if record else 0
    same_provider = record is not None and record.get("provider") == provider
    if same_provider and record is not None and (
        _within_lifetime(record, now) or inflight > 0
    ):
        try:
            session = await asyncio.to_thread(attach_session, record)
        except SandboxGone:
            logger.warning(
                "sandbox.attach_failed conversation_id=%s sandbox_id=%s",
                conversation_id,
                record.get("sandbox_id"),
                exc_info=True,
            )
            await guard.check()
            await _replace(store, conversation_id, record, guard)
            record = None
    if session is None or record is None:
        if record is not None:
            await guard.check()
            await _replace(store, conversation_id, record, guard)
        session = await asyncio.to_thread(_open_sync, skills_root)
        fresh = True
        try:
            await guard.check()
            await store.put(conversation_id, _stored_record(session))
        except Exception:
            try:
                await asyncio.to_thread(session.destroy)
            except Exception:
                logger.exception(
                    "sandbox.destroy_failed conversation_id=%s sandbox_id=%s",
                    conversation_id,
                    session.sandbox_id,
                )
            raise
        logger.info(
            "sandbox.created conversation_id=%s provider=%s sandbox_id=%s",
            conversation_id,
            session.provider,
            session.sandbox_id,
        )
    else:
        try:
            digest = await asyncio.to_thread(sync_skills, session, skills_root)
            if digest != record.get("skills_digest"):
                record["skills_digest"] = digest
                await guard.check()
                await store.put(conversation_id, record)
            if session.can_renew:
                await asyncio.to_thread(session.renew, float(sandbox_busy_ttl_s()))
        except Exception:
            await asyncio.to_thread(session.detach)
            raise
        logger.info(
            "sandbox.reused conversation_id=%s provider=%s sandbox_id=%s",
            conversation_id,
            session.provider,
            session.sandbox_id,
        )
    try:
        await store.start_turn(
            conversation_id, guard, time.time() + sandbox_busy_ttl_s()
        )
    except Exception:
        try:
            await asyncio.to_thread(session.detach)
        except Exception:
            logger.exception(
                "sandbox.detach_failed conversation_id=%s", conversation_id
            )
        raise
    return SandboxLease(session, conversation_id, fresh=fresh)


async def release_sandbox(lease: Optional[SandboxLease]) -> None:
    """Restart the idle timer and drop local clients. Never raises.

    The remote sandbox stays up. The last in-flight turn of the conversation
    is the one that restarts the timer; an earlier release leaves the deadline
    that covers the turn still running.
    """
    if lease is None:
        return
    try:
        await _release(lease)
    except Exception:
        logger.exception(
            "sandbox.release_failed conversation_id=%s", lease.conversation_id
        )


async def _release(lease: SandboxLease) -> None:
    store = get_store()
    idle = float(sandbox_idle_ttl_s())
    try:
        async with store.hold(lease.conversation_id, wait=True) as guard:
            if guard is not None:
                record = await store.get(lease.conversation_id)
                if (
                    record is not None
                    and record.get("sandbox_id") == lease.session.sandbox_id
                ):
                    inflight = await store.finish_turn(
                        lease.conversation_id, guard, time.time() + idle
                    )
                    if inflight <= 0 and lease.session.can_renew:
                        await asyncio.to_thread(lease.session.renew, idle)
    except Exception:
        logger.exception(
            "sandbox.release_failed conversation_id=%s", lease.conversation_id
        )
    try:
        await asyncio.to_thread(lease.session.detach)
    except Exception:
        logger.exception(
            "sandbox.detach_failed conversation_id=%s", lease.conversation_id
        )


async def discard_conversation_sandbox(conversation_id: str) -> None:
    """Destroy the conversation's sandbox. Never raises.

    A conversation delete calls this. A failure leaves the provider's own
    expiry to stop the sandbox, and the reaper retries once the deadline passes
    when the record is still there.
    """
    if not conversation_id:
        return
    try:
        store = get_store()
        async with store.hold(conversation_id, wait=True) as guard:
            if guard is None:
                logger.warning(
                    "sandbox.discard_lock_timeout conversation_id=%s", conversation_id
                )
                return
            record = await store.get(conversation_id)
            if record is None:
                await guard.check()
                await store.delete(conversation_id)
                return
            if await _retire(store, conversation_id, record, guard):
                logger.info(
                    "sandbox.discarded conversation_id=%s provider=%s sandbox_id=%s",
                    conversation_id,
                    record.get("provider"),
                    record.get("sandbox_id"),
                )
    except Exception:
        logger.exception(
            "sandbox.discard_failed conversation_id=%s", conversation_id
        )


async def reap_expired(now: Optional[float] = None) -> None:
    """Destroy sandboxes whose idle deadline has passed.

    The deadline is read again under the lock, so a turn that reattached
    between the scan and the lock is left alone. A lock held by a turn is
    skipped; the next pass tries again.
    """
    store = get_store()
    moment = time.time() if now is None else now
    for conversation_id in await store.due(moment):
        try:
            async with store.hold(conversation_id, wait=False) as guard:
                if guard is None:
                    continue
                deadline = await store.deadline(conversation_id)
                record = await store.get(conversation_id)
                if deadline is None or deadline > time.time():
                    continue
                if record is None:
                    await guard.check()
                    await store.delete(conversation_id)
                    continue
                if not await _retire(store, conversation_id, record, guard):
                    continue
                logger.info(
                    "sandbox.reaped conversation_id=%s provider=%s sandbox_id=%s",
                    conversation_id,
                    record.get("provider"),
                    record.get("sandbox_id"),
                )
        except Exception:
            logger.exception(
                "sandbox.reap_failed conversation_id=%s", conversation_id
            )


async def _reaper_loop() -> None:
    while True:
        try:
            await asyncio.sleep(sandbox_reaper_interval_s())
            await reap_expired()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("sandbox.reaper_failed")


def start_sandbox_reaper() -> None:
    """Start this worker's idle reaper. A second call does nothing.

    Shutdown does not destroy sandboxes. Another worker may still reuse one,
    and the provider expiry stops it if every worker is gone.
    """
    global _reaper_task
    if _reaper_task is not None and not _reaper_task.done():
        return
    _reaper_task = asyncio.create_task(_reaper_loop(), name="sandbox-reaper")


async def stop_sandbox_reaper() -> None:
    """Stop this worker's reaper. Sandboxes it was tracking stay up."""
    global _reaper_task
    task = _reaper_task
    _reaper_task = None
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
