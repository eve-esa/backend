"""Conversation sandboxes: reuse, idle reap, and the skills digest."""

import asyncio
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.services.agents.core.sandbox import (
    SandboxError,
    SandboxGone,
    SandboxSession,
    sandbox_busy_ttl_s,
    sandbox_idle_ttl_s,
    sandbox_reuse_limit_s,
    sync_skills,
)
from src.services.agents.core.sandbox_pool import (
    acquire_sandbox,
    discard_conversation_sandbox,
    reap_expired,
    release_sandbox,
)

pytestmark = pytest.mark.no_db

_POOL = "src.services.agents.core.sandbox_pool"


def _session(state, sandbox_id, provider, skills_digest, metadata):
    def detach():
        if state["fail_detach"]:
            raise RuntimeError("detach boom")
        state["detached"].append(sandbox_id)

    def renew(seconds):
        if state["fail_renew"]:
            raise RuntimeError("renew boom")
        state["renewed"].append(seconds)

    return SandboxSession(
        sandbox_id,
        provider=provider,
        sandbox_id=sandbox_id,
        skills_digest=skills_digest,
        metadata=metadata,
        destroy=lambda: state["destroyed"].append(sandbox_id),
        detach=detach,
        renew=renew if provider == "opensandbox" else None,
    )


def _install(monkeypatch, clock):
    """Replace the provider calls. The store under test is the memory one."""
    state = {
        "n": 0,
        "dead": False,
        "lifetime": None,
        "killed": [],
        "detached": [],
        "renewed": [],
        "synced": [],
        "fail_detach": False,
        "fail_renew": False,
        "destroyed": [],
    }

    def open_sync(skills_root=None):
        state["n"] += 1
        sandbox_id = f"sb-{state['n']}"
        provider = os.environ.get("SANDBOX_PROVIDER", "")
        metadata = {}
        if state["lifetime"] is not None:
            metadata = {
                "created_at": str(int(clock["now"])),
                "max_lifetime_s": str(state["lifetime"]),
                "region": "eu-west-1",
                "identifier": "aws.codeinterpreter.v1",
            }
        return _session(state, sandbox_id, provider, "digest-1", metadata)

    def attach(record):
        if state["dead"]:
            raise SandboxGone("dead")
        if state.get("transient"):
            raise RuntimeError("blip")
        metadata = {
            key: record[key]
            for key in ("region", "identifier", "created_at", "max_lifetime_s")
            if key in record
        }
        return _session(
            state,
            record["sandbox_id"],
            record["provider"],
            record.get("skills_digest", ""),
            metadata,
        )

    def kill(record):
        state["killed"].append(record["sandbox_id"])

    def sync(session, skills_root=None):
        state["synced"].append(skills_root)
        if skills_root == "changed":
            session.skills_digest = "digest-2"
            return "digest-2"
        return session.skills_digest

    monkeypatch.setattr(f"{_POOL}._open_sync", open_sync)
    monkeypatch.setattr(f"{_POOL}.attach_session", attach)
    monkeypatch.setattr(f"{_POOL}.kill_record", kill)
    monkeypatch.setattr(f"{_POOL}.sync_skills", sync)
    monkeypatch.setattr(f"{_POOL}.time.time", lambda: clock["now"])
    return state


@pytest.fixture
def clock(monkeypatch):
    clock = {"now": 1_700_000_000.0}
    monkeypatch.setenv("SANDBOX_PROVIDER", "agentcore")
    return clock


async def test_unset_provider_acquires_nothing(monkeypatch):
    monkeypatch.delenv("SANDBOX_PROVIDER", raising=False)
    assert await acquire_sandbox("c1") is None


async def test_same_conversation_reuses_one_sandbox(monkeypatch, clock, sandbox_store):
    state = _install(monkeypatch, clock)

    first = await acquire_sandbox("c1", None)
    await release_sandbox(first)
    second = await acquire_sandbox("c1", None)
    await release_sandbox(second)

    assert first is not None and second is not None
    assert second.session.sandbox_id == first.session.sandbox_id
    assert state["n"] == 1
    assert state["killed"] == []
    assert state["renewed"] == []
    deadline = await sandbox_store.deadline("c1")
    assert deadline == clock["now"] + sandbox_idle_ttl_s()


async def test_conversations_get_different_sandboxes(monkeypatch, clock):
    state = _install(monkeypatch, clock)

    first = await acquire_sandbox("c1", None)
    second = await acquire_sandbox("c2", None)

    assert first is not None and second is not None
    assert first.session.sandbox_id != second.session.sandbox_id
    assert state["n"] == 2


async def test_parallel_turns_share_the_sandbox(monkeypatch, clock, sandbox_store):
    state = _install(monkeypatch, clock)

    first, second = await asyncio.gather(
        acquire_sandbox("c1", None),
        acquire_sandbox("c1", None),
    )

    assert first is not None and second is not None
    assert first.session.sandbox_id == second.session.sandbox_id
    assert state["n"] == 1
    record = await sandbox_store.get("c1")
    assert record is not None
    assert record["inflight"] == "2"
    await release_sandbox(first)
    # The other turn is still in flight, so the idle timer stays past the turn.
    assert await sandbox_store.deadline("c1") == clock["now"] + sandbox_busy_ttl_s()
    await release_sandbox(second)
    assert await sandbox_store.deadline("c1") == clock["now"] + sandbox_idle_ttl_s()


async def test_idle_expiry_reaps_and_the_next_turn_creates(monkeypatch, clock, sandbox_store):
    state = _install(monkeypatch, clock)
    lease = await acquire_sandbox("c1", None)
    assert lease is not None
    sandbox_id = lease.session.sandbox_id

    # Still inside the in-turn window: idle time alone does not kill it.
    clock["now"] += sandbox_busy_ttl_s() - 1
    await reap_expired()
    assert state["killed"] == []

    await release_sandbox(lease)
    clock["now"] += sandbox_idle_ttl_s()
    await reap_expired()

    assert state["killed"] == [sandbox_id]
    assert await sandbox_store.get("c1") is None
    again = await acquire_sandbox("c1", None)
    assert again is not None
    assert again.session.sandbox_id != sandbox_id
    assert state["n"] == 2


async def test_reaper_skips_a_conversation_whose_lock_is_held(
    monkeypatch, clock, sandbox_store
):
    state = _install(monkeypatch, clock)
    lease = await acquire_sandbox("c1", None)
    assert lease is not None
    await release_sandbox(lease)
    clock["now"] += sandbox_idle_ttl_s()

    async with sandbox_store.hold("c1", wait=True):
        await reap_expired()
    assert state["killed"] == []

    await reap_expired()
    assert state["killed"] == [lease.session.sandbox_id]


async def test_a_dead_sandbox_is_replaced(monkeypatch, clock):
    state = _install(monkeypatch, clock)
    lease = await acquire_sandbox("c1", None)
    assert lease is not None
    await release_sandbox(lease)
    state["dead"] = True

    again = await acquire_sandbox("c1", None)

    assert again is not None
    assert state["killed"] == [lease.session.sandbox_id]
    assert again.session.sandbox_id != lease.session.sandbox_id


async def test_agentcore_near_its_hard_ttl_is_replaced(monkeypatch, clock):
    state = _install(monkeypatch, clock)
    state["lifetime"] = sandbox_reuse_limit_s() - 1
    lease = await acquire_sandbox("c1", None)
    assert lease is not None
    await release_sandbox(lease)

    again = await acquire_sandbox("c1", None)

    assert again is not None
    assert state["killed"] == [lease.session.sandbox_id]
    assert again.session.sandbox_id != lease.session.sandbox_id
    assert state["n"] == 2


async def test_a_busy_sandbox_near_its_hard_ttl_is_kept(monkeypatch, clock):
    state = _install(monkeypatch, clock)
    state["lifetime"] = sandbox_reuse_limit_s() - 1
    lease = await acquire_sandbox("c1", None)
    assert lease is not None

    again = await acquire_sandbox("c1", None)

    assert again is not None
    assert again.session.sandbox_id == lease.session.sandbox_id
    assert state["killed"] == []
    assert state["n"] == 1


async def test_a_changed_skills_digest_is_stored(monkeypatch, clock, sandbox_store):
    state = _install(monkeypatch, clock)
    first = await acquire_sandbox("c1", None)
    await release_sandbox(first)
    second = await acquire_sandbox("c1", "changed")
    await release_sandbox(second)

    record = await sandbox_store.get("c1")
    assert record is not None
    assert record["skills_digest"] == "digest-2"
    assert state["synced"] == ["changed"]
    assert state["n"] == 1


async def test_a_provider_change_replaces_the_sandbox(monkeypatch, clock):
    state = _install(monkeypatch, clock)
    lease = await acquire_sandbox("c1", None)
    assert lease is not None
    await release_sandbox(lease)
    monkeypatch.setenv("SANDBOX_PROVIDER", "opensandbox")

    again = await acquire_sandbox("c1", None)

    assert again is not None
    assert again.session.provider == "opensandbox"
    assert state["killed"] == [lease.session.sandbox_id]
    assert state["n"] == 2


async def test_opensandbox_renews_on_reuse_and_on_release(monkeypatch, clock):
    state = _install(monkeypatch, clock)
    monkeypatch.setenv("SANDBOX_PROVIDER", "opensandbox")

    lease = await acquire_sandbox("c1", None)
    assert state["renewed"] == []
    await release_sandbox(lease)
    assert state["renewed"] == [float(sandbox_idle_ttl_s())]

    await acquire_sandbox("c1", None)
    assert state["renewed"][-1] == float(sandbox_busy_ttl_s())
    assert state["n"] == 1


async def test_delete_destroys_the_sandbox(monkeypatch, clock, sandbox_store):
    state = _install(monkeypatch, clock)
    lease = await acquire_sandbox("c1", None)
    assert lease is not None
    await release_sandbox(lease)

    await discard_conversation_sandbox("c1")

    assert state["killed"] == [lease.session.sandbox_id]
    assert await sandbox_store.get("c1") is None
    assert await sandbox_store.deadline("c1") is None


async def test_a_failed_stop_keeps_the_record_for_the_reaper(
    monkeypatch, clock, sandbox_store
):
    state = _install(monkeypatch, clock)
    lease = await acquire_sandbox("c1", None)
    assert lease is not None
    await release_sandbox(lease)
    clock["now"] += sandbox_idle_ttl_s()

    def boom(_record):
        raise RuntimeError("stop failed")

    monkeypatch.setattr(f"{_POOL}.kill_record", boom)
    await reap_expired()
    assert await sandbox_store.get("c1") is not None
    assert state["killed"] == []

    monkeypatch.setattr(
        f"{_POOL}.kill_record", lambda record: state["killed"].append(record["sandbox_id"])
    )
    await reap_expired()
    assert state["killed"] == [lease.session.sandbox_id]
    assert await sandbox_store.get("c1") is None


async def test_a_failed_stop_does_not_forget_the_sandbox(
    monkeypatch, clock, sandbox_store
):
    state = _install(monkeypatch, clock)
    state["lifetime"] = sandbox_reuse_limit_s() - 1
    lease = await acquire_sandbox("c1", None)
    assert lease is not None
    await release_sandbox(lease)

    def boom(_record):
        raise RuntimeError("stop failed")

    monkeypatch.setattr(f"{_POOL}.kill_record", boom)

    with pytest.raises(SandboxError, match="could not stop"):
        await acquire_sandbox("c1", None)

    record = await sandbox_store.get("c1")
    assert record is not None
    assert record["sandbox_id"] == lease.session.sandbox_id
    assert state["n"] == 1


async def test_a_past_deadline_clears_a_stuck_inflight(
    monkeypatch, clock, sandbox_store
):
    state = _install(monkeypatch, clock)
    state["lifetime"] = sandbox_reuse_limit_s() - 1
    lease = await acquire_sandbox("c1", None)
    assert lease is not None
    await release_sandbox(lease)
    record = await sandbox_store.get("c1")
    assert record is not None
    record["inflight"] = "3"
    await sandbox_store.put("c1", record)
    await sandbox_store.set_deadline("c1", clock["now"] - 1)

    again = await acquire_sandbox("c1", None)

    assert again is not None
    assert again.session.sandbox_id != lease.session.sandbox_id
    assert state["killed"] == [lease.session.sandbox_id]


async def test_a_failed_save_destroys_the_new_sandbox(monkeypatch, clock, sandbox_store):
    state = _install(monkeypatch, clock)

    async def fail_put(_conversation_id, _record):
        raise RuntimeError("redis down")

    monkeypatch.setattr(sandbox_store, "put", fail_put)
    with pytest.raises(RuntimeError, match="redis down"):
        await acquire_sandbox("c1", None)

    assert state["destroyed"] == ["sb-1"]
    assert await sandbox_store.get("c1") is None


async def test_the_lock_refresh_extends_a_held_key(monkeypatch):
    from src.services.agents.core.sandbox_pool import _refresh_lock

    monkeypatch.setattr(f"{_POOL}._LOCK_TTL_MS", 30)
    calls: list = []

    class _Redis:
        async def eval(self, _script, _nkeys, key, token, ttl):
            calls.append((key, token, ttl))
            return 1

    from src.services.agents.core.sandbox_pool import RedisLockGuard

    guard = RedisLockGuard(_Redis(), "sandbox:lock:c1", "tok")
    task = asyncio.create_task(_refresh_lock(guard, "c1"))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert calls
    assert calls[0] == ("sandbox:lock:c1", "tok", "30")
    assert guard.lost is False


async def test_a_transient_attach_error_keeps_the_sandbox(
    monkeypatch, clock, sandbox_store
):
    state = _install(monkeypatch, clock)
    lease = await acquire_sandbox("c1", None)
    assert lease is not None
    await release_sandbox(lease)
    state["transient"] = True

    with pytest.raises(RuntimeError, match="blip"):
        await acquire_sandbox("c1", None)

    record = await sandbox_store.get("c1")
    assert record is not None
    assert record["sandbox_id"] == lease.session.sandbox_id
    assert state["killed"] == []
    assert state["n"] == 1


def test_provider_attach_maps_gone_and_keeps_other_errors(monkeypatch):
    mode: dict = {"status": "TERMINATED", "error": None}

    class CodeInterpreter:
        def __init__(self, region, integration_source=None):
            self.identifier = None
            self.session_id = None

        def get_session(self):
            if mode["error"] is not None:
                raise mode["error"]
            return {"status": mode["status"]}

    client = types.ModuleType("bedrock_agentcore.tools.code_interpreter_client")
    client.CodeInterpreter = CodeInterpreter
    adapter = types.ModuleType("langchain_agentcore_codeinterpreter.sandbox")
    adapter.INTEGRATION_SOURCE = "test"
    monkeypatch.setitem(
        sys.modules, "bedrock_agentcore.tools.code_interpreter_client", client
    )
    monkeypatch.setitem(sys.modules, "langchain_agentcore_codeinterpreter.sandbox", adapter)

    from src.services.agents.core.sandbox import _attach_agentcore, _attach_opensandbox

    record = {
        "region": "eu-west-1",
        "identifier": "id",
        "sandbox_id": "s",
        "max_lifetime_s": "3600",
        "created_at": "1",
        "skills_digest": "",
    }
    with pytest.raises(SandboxGone):
        _attach_agentcore(record)

    ended = Exception("ended")
    ended.response = {"Error": {"Code": "ResourceNotFoundException", "Message": "nope"}}
    mode["error"] = ended
    with pytest.raises(SandboxGone):
        _attach_agentcore(record)

    throttled = Exception("throttled")
    throttled.response = {"Error": {"Code": "ThrottlingException", "Message": "slow"}}
    mode["error"] = throttled
    with pytest.raises(Exception) as caught:
        _attach_agentcore(record)
    assert not isinstance(caught.value, SandboxGone)

    monkeypatch.setenv("SANDBOX_OPENSANDBOX_DOMAIN", "sandbox.internal")
    monkeypatch.delenv("SANDBOX_OPENSANDBOX_PROTOCOL", raising=False)
    monkeypatch.delenv("SANDBOX_OPENSANDBOX_API_KEY", raising=False)

    class SandboxSync:
        @staticmethod
        def connect(sandbox_id, connection_config=None):
            missing = Exception("missing")
            missing.status_code = 404
            raise missing

    class ConnectionConfigSync:
        def __init__(self, domain=None, protocol="http", api_key=None):
            self.domain = domain

    remote = types.ModuleType("opensandbox")
    remote.SandboxSync = SandboxSync
    config = types.ModuleType("opensandbox.config")
    config.ConnectionConfigSync = ConnectionConfigSync
    monkeypatch.setitem(sys.modules, "opensandbox", remote)
    monkeypatch.setitem(sys.modules, "opensandbox.config", config)
    with pytest.raises(SandboxGone):
        _attach_opensandbox(record)


class _FakeRedis:
    """Enough of redis.asyncio for the lock, and nothing else."""

    def __init__(self, now):
        self._now = now
        self.values: dict[str, tuple[str, float]] = {}
        self.hashes: dict[str, dict[str, str]] = {}
        self.zsets: dict[str, dict[str, float]] = {}

    def _live(self, key):
        item = self.values.get(key)
        if item is None:
            return None
        token, expiry = item
        if self._now() >= expiry:
            self.values.pop(key, None)
            return None
        return token

    async def set(self, key, value, nx=False, px=None):
        if nx and self._live(key) is not None:
            return None
        self.values[key] = (value, self._now() + (px or 0) / 1000)
        return True

    async def eval(self, script, numkeys, *args):
        if "hincrby" in script:
            lock, record, idle = args[:numkeys]
            token, delta, member, deadline, always = args[numkeys:]
            if self._live(lock) != token:
                return [0, 0]
            inflight = await self.hincrby(record, "inflight", int(delta))
            if always == "1" or inflight <= 0:
                await self.zadd(idle, {member: float(deadline)})
            return [1, inflight]
        key, token = args[0], args[1]
        if self._live(key) != token:
            return 0
        if "pexpire" in script:
            self.values[key] = (token, self._now() + int(args[2]) / 1000)
            return 1
        if "del" in script:
            self.values.pop(key, None)
            return 1
        return 0

    async def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    async def hset(self, key, field=None, value=None, mapping=None):
        target = self.hashes.setdefault(key, {})
        if mapping:
            target.update(mapping)
        if field is not None:
            target[field] = value

    async def hincrby(self, key, field, delta):
        target = self.hashes.setdefault(key, {})
        target[field] = str(int(target.get(field, "0")) + int(delta))
        return int(target[field])

    async def zadd(self, key, mapping):
        self.zsets.setdefault(key, {}).update(mapping)

    async def delete(self, key):
        return 1 if self.hashes.pop(key, None) is not None else 0

    async def zrem(self, key, member):
        return 1 if self.zsets.get(key, {}).pop(member, None) is not None else 0

    async def zscore(self, key, member):
        return self.zsets.get(key, {}).get(member)

    async def zrangebyscore(self, key, low, high):
        return [m for m, score in self.zsets.get(key, {}).items() if score <= high]


async def test_a_waiter_outlasts_the_old_lock_window():
    from src.services.agents.core import sandbox_pool as pool

    clock = {"t": 0.0}
    redis = _FakeRedis(lambda: clock["t"])
    store = pool.RedisSandboxStore()
    store._redis = redis
    release = asyncio.Event()
    holding = asyncio.Event()

    async def holder():
        async with store.hold("c1", wait=True) as guard:
            assert guard is not None
            key = pool._lock_key("c1")
            token = redis._live(key)
            # Refresh at t=60, so the key now expires at t=240, past the
            # original t=180 deadline a waiter used to give up on.
            clock["t"] = 60
            await redis.eval(pool._REFRESH_SCRIPT, 1, key, token, str(pool._LOCK_TTL_MS))
            clock["t"] = 200
            holding.set()
            await release.wait()

    holder_task = asyncio.create_task(holder())
    await holding.wait()
    got = asyncio.Event()

    async def waiter():
        async with store.hold("c1", wait=True) as guard:
            assert guard is not None
            got.set()

    waiter_task = asyncio.create_task(waiter())
    await asyncio.sleep(0.05)
    assert clock["t"] > 180
    assert not waiter_task.done()
    release.set()
    await asyncio.wait_for(waiter_task, timeout=2)
    assert got.is_set()
    await holder_task


async def test_a_lost_lock_does_not_save_the_new_sandbox(monkeypatch, clock):
    import time

    from src.services.agents.core import sandbox_pool as pool

    state = _install(monkeypatch, clock)
    redis = _FakeRedis(time.monotonic)
    store = pool.RedisSandboxStore()
    store._redis = redis
    original = pool._open_sync

    # No refresh tick runs before the save. The check itself must see the key
    # another worker now owns.
    def open_sync(skills_root=None):
        key = pool._lock_key("c1")
        redis.values[key] = ("stolen", time.monotonic() + 60)
        return original(skills_root)

    monkeypatch.setattr(pool, "_open_sync", open_sync)
    token = pool.override_sandbox_store(store)
    try:
        with pytest.raises(pool.SandboxLockLost):
            await acquire_sandbox("c1", None)
    finally:
        pool.reset_sandbox_store(token)

    assert state["destroyed"] == ["sb-1"]
    assert redis.hashes == {}


async def test_a_redis_turn_moves_inflight_and_deadline_only_while_locked():
    import time

    from src.services.agents.core import sandbox_pool as pool

    redis = _FakeRedis(time.monotonic)
    store = pool.RedisSandboxStore()
    store._redis = redis
    await store.put("c1", {"sandbox_id": "sb-1", "inflight": "0"})

    async with store.hold("c1", wait=True) as guard:
        assert await store.start_turn("c1", guard, 100.0) == 1
        assert await store.start_turn("c1", guard, 200.0) == 2
        assert await store.deadline("c1") == 200.0
        assert await store.finish_turn("c1", guard, 300.0) == 1
        assert await store.deadline("c1") == 200.0
        assert await store.finish_turn("c1", guard, 400.0) == 0
        assert await store.deadline("c1") == 400.0

        redis.values[pool._lock_key("c1")] = ("stolen", time.monotonic() + 60)
        with pytest.raises(pool.SandboxLockLost):
            await store.start_turn("c1", guard, 500.0)

    assert (await store.get("c1"))["inflight"] == "0"
    assert await store.deadline("c1") == 400.0


async def test_a_redis_error_does_not_mark_the_lock_lost(monkeypatch):
    from src.services.agents.core.sandbox_pool import RedisLockGuard, _refresh_lock

    monkeypatch.setattr(f"{_POOL}._LOCK_TTL_MS", 30)
    calls: list = []

    class _Redis:
        async def eval(self, *_args):
            calls.append(1)
            if len(calls) == 1:
                raise ConnectionError("redis blip")
            return 1

    guard = RedisLockGuard(_Redis(), "sandbox:lock:c1", "tok")
    task = asyncio.create_task(_refresh_lock(guard, "c1"))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(calls) >= 2
    assert guard.lost is False
    await guard.check()


async def test_release_failure_does_not_raise(monkeypatch, clock):
    state = _install(monkeypatch, clock)
    state["fail_detach"] = True
    lease = await acquire_sandbox("c1", None)

    await release_sandbox(lease)

    monkeypatch.setenv("SANDBOX_PROVIDER", "opensandbox")
    state["fail_renew"] = True
    state["fail_detach"] = False
    other = await acquire_sandbox("c2", None)
    await release_sandbox(other)
    assert state["detached"][-1] == other.session.sandbox_id


def test_sync_skills_skips_an_unchanged_tree_and_clears_a_changed_one(tmp_path):
    tree = tmp_path / "skills"
    skill = tree / "roll" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_bytes(b"one")
    commands: list[str] = []
    uploads: list = []

    class _Backend:
        def execute(self, command, *args, **kwargs):
            commands.append(command)
            return SimpleNamespace(exit_code=0, output="")

        def upload_files(self, files):
            uploads.append(list(files))
            return [SimpleNamespace(error=None) for _ in files]

    session = SandboxSession(
        _Backend(),
        provider="agentcore",
        sandbox_id="s",
        destroy=lambda: None,
        detach=lambda: None,
    )
    first = sync_skills(session, tree)
    assert commands == ["rm -rf /skills"]
    assert uploads[0][0] == ("/skills/roll/SKILL.md", b"one")

    commands.clear()
    uploads.clear()
    assert sync_skills(session, tree) == first
    assert commands == []
    assert uploads == []

    skill.write_bytes(b"two")
    second = sync_skills(session, tree)
    assert second != first
    assert commands == ["rm -rf /skills"]
    assert uploads[0][0][1] == b"two"
    assert session.skills_digest == second


def test_sync_skills_reports_a_failed_clear(tmp_path: Path):
    class _Backend:
        def execute(self, command, *args, **kwargs):
            return SimpleNamespace(exit_code=1, output="")

    session = SandboxSession(
        _Backend(),
        provider="agentcore",
        sandbox_id="s",
        destroy=lambda: None,
        detach=lambda: None,
        skills_digest="stale",
    )
    with pytest.raises(SandboxError, match="clear"):
        sync_skills(session, tmp_path)
