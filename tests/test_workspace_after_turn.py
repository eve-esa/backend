"""The post-turn workspace save when the sandbox session is already gone.

A conversation delete discards the sandbox while the save of the turn that
just ended still runs. That save stops, writes nothing, and logs at INFO.
A session that ends while its record still names it (provider timeout or
crash) is a lost save and logs a WARNING. A session that expires during a
turn keeps its ERROR.
"""

import asyncio
import json
import logging

import pytest

from src.services.agents.core import workspace
from src.services.agents.core.sandbox import SandboxSession
from src.services.agents.core.sandbox_pool import RETIRING_FIELD
from src.services.agents.core.workspace import (
    save_workspace,
    sync_workspace,
    workspace_prefix,
    workspace_state_key,
)

pytestmark = pytest.mark.no_db

# The AgentCore integration logs this at ERROR, then returns exit code 1
# with the text below, when the session is gone.
_SDK_LOGGER = "langchain_agentcore_codeinterpreter.sandbox"
_SDK_EXPIRED = "Error: AgentCore session has expired. Start a new session to continue."


class _Store:
    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def list_key_records(self, prefix, max_keys):
        records = [
            {"key": key, "size": len(body)}
            for key, body in self.objects.items()
            if key.startswith(prefix)
        ]
        return records[:max_keys], False

    async def get_object(self, key):
        raise workspace.ObjectNotFoundError(key)

    async def put_object(self, key, body, content_type):
        self.objects[key] = body

    async def delete_object(self, key):
        self.objects.pop(key, None)


class _Result:
    def __init__(self, output, exit_code=0):
        self.output = output
        self.exit_code = exit_code


class _File:
    def __init__(self, content, error=None):
        self.content = content
        self.error = error


class _AgentCoreLike:
    """A sandbox that behaves like the AgentCore integration once closed."""

    def __init__(self, files):
        self.tree = dict(files)
        self.closed = False
        self.commands: list[str] = []
        self.on_download = None

    def execute(self, command):
        self.commands.append(command)
        if self.closed:
            logging.getLogger(_SDK_LOGGER).error(
                "AgentCore session expired while executing command: %s",
                command[:80],
            )
            return _Result(_SDK_EXPIRED, exit_code=1)
        if command.startswith("ps"):
            return _Result("1 init")
        lines = [
            f"{path}\t{len(data)}\t1.0" for path, data in sorted(self.tree.items())
        ]
        return _Result("\n".join(lines))

    def download_files(self, paths):
        if self.on_download is not None:
            self.on_download()
        if self.closed:
            logging.getLogger(_SDK_LOGGER).error(
                "AgentCore session expired while downloading files: %s", paths
            )
            return [_File(None, "permission_denied") for _ in paths]
        return [_File(self.tree[path[len("workspace/") :]]) for path in paths]


def _session(backend, detached):
    return SandboxSession(
        backend,
        provider="agentcore",
        sandbox_id="sb-1",
        destroy=lambda: None,
        detach=lambda: detached.append("sb-1"),
        metadata={"process_baseline": "1"},
    )


@pytest.fixture
def storage(monkeypatch):
    store = _Store()
    monkeypatch.setattr(workspace, "storage_service", store)
    monkeypatch.setattr(workspace, "_BACKOFF_S", (0, 0, 0, 0))
    return store


async def _record(sandbox_store, sandbox_id="sb-1", **extra):
    await sandbox_store.put(
        "c1",
        {"sandbox_id": sandbox_id, "provider": "agentcore", "user_id": "u1", **extra},
    )


def _warnings(caplog, text):
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and text in r.getMessage()
    ]


def _noisy(caplog):
    return [
        (record.name, record.getMessage())
        for record in caplog.records
        if record.levelno >= logging.WARNING
    ]


async def test_a_sync_whose_sandbox_was_discarded_after_the_turn_stops_quietly(
    monkeypatch, storage, sandbox_store, caplog
):
    await _record(sandbox_store)
    backend = _AgentCoreLike({"plot.png": b"plot"})
    detached: list[str] = []
    real_sleep = workspace.asyncio.sleep

    async def _delete_during_backoff(_seconds):
        # The conversation delete: the sandbox is stopped, then its record goes.
        backend.closed = True
        await sandbox_store.delete("c1")
        await real_sleep(0)

    monkeypatch.setattr(workspace.asyncio, "sleep", _delete_during_backoff)
    caplog.set_level(logging.DEBUG)

    await sync_workspace(_session(backend, detached), "u1", "c1")

    assert _noisy(caplog) == []
    assert storage.objects == {}
    assert detached == ["sb-1"]
    assert any(
        "workspace.sync_skipped" in r.getMessage() and "reason=" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.INFO
    )


async def test_a_discarded_session_that_expires_under_the_sync_is_info(
    storage, sandbox_store, caplog
):
    # The delete marks the record before the kill, so the expired session
    # reads as discarded even while the record is still there.
    await _record(sandbox_store, **{RETIRING_FIELD: "1"})
    backend = _AgentCoreLike({"plot.png": b"plot"})
    backend.closed = True
    detached: list[str] = []
    caplog.set_level(logging.DEBUG)

    await sync_workspace(_session(backend, detached), "u1", "c1")

    assert _noisy(caplog) == []
    assert storage.objects == {}
    assert detached == ["sb-1"]


async def test_an_expired_session_whose_record_still_names_it_is_a_lost_save(
    storage, sandbox_store, caplog
):
    # Provider timeout or crash: nobody discarded the sandbox, the save is lost.
    await _record(sandbox_store)
    backend = _AgentCoreLike({"plot.png": b"plot"})
    backend.closed = True
    detached: list[str] = []
    caplog.set_level(logging.DEBUG)

    await sync_workspace(_session(backend, detached), "u1", "c1")

    assert storage.objects == {}
    assert detached == ["sb-1"]
    lost = _warnings(caplog, "workspace.save_failed")
    assert lost and all("reason=session_expired" in line for line in lost)


async def test_a_snapshot_on_an_expired_session_is_a_lost_save(storage, caplog):
    # The reaper snapshot runs with no alive check: an ended session is a loss.
    backend = _AgentCoreLike({"plot.png": b"plot"})
    backend.closed = True
    caplog.set_level(logging.DEBUG)

    await save_workspace(backend, "u1", "c1")

    assert storage.objects == {}
    assert _warnings(caplog, "reason=session_expired")


def _erase(storage, sandbox_store):
    """The conversation delete: drop the record, then erase the prefix."""
    sandbox_store._records.pop("c1", None)
    prefix = workspace_prefix("u1", "c1")
    for key in [k for k in storage.objects if k.startswith(prefix)]:
        storage.objects.pop(key)


async def test_a_put_in_flight_during_the_delete_leaves_no_key(
    monkeypatch, storage, sandbox_store, caplog
):
    await _record(sandbox_store)
    backend = _AgentCoreLike({"a.txt": b"aaa", "b.txt": b"bbb"})
    real_put = storage.put_object

    async def _slow_put(key, body, content_type):
        # The delete runs while the write is on the wire; the write lands after.
        _erase(storage, sandbox_store)
        await asyncio.sleep(0)
        await real_put(key, body, content_type)

    monkeypatch.setattr(storage, "put_object", _slow_put)
    caplog.set_level(logging.DEBUG)

    await sync_workspace(_session(backend, []), "u1", "c1")

    assert storage.objects == {}
    assert _noisy(caplog) == []


async def test_a_delete_after_the_cleanup_writes_no_state(
    monkeypatch, storage, sandbox_store, caplog
):
    await _record(sandbox_store)
    backend = _AgentCoreLike({"a.txt": b"aaa"})
    real_delete_missing = workspace._delete_missing

    async def _then_delete(prefix, kept):
        await real_delete_missing(prefix, kept)
        _erase(storage, sandbox_store)

    monkeypatch.setattr(workspace, "_delete_missing", _then_delete)
    caplog.set_level(logging.DEBUG)

    await sync_workspace(_session(backend, []), "u1", "c1")

    assert workspace_state_key("u1", "c1") not in storage.objects
    assert storage.objects == {}
    assert _noisy(caplog) == []


async def test_a_state_write_in_flight_during_the_delete_is_removed(
    monkeypatch, storage, sandbox_store, caplog
):
    await _record(sandbox_store)
    backend = _AgentCoreLike({"a.txt": b"aaa"})
    real_put = storage.put_object
    state_key = workspace_state_key("u1", "c1")

    async def _slow_state_put(key, body, content_type):
        if key == state_key:
            _erase(storage, sandbox_store)
        await real_put(key, body, content_type)

    monkeypatch.setattr(storage, "put_object", _slow_state_put)
    caplog.set_level(logging.DEBUG)

    await sync_workspace(_session(backend, []), "u1", "c1")

    assert storage.objects == {}
    assert _noisy(caplog) == []


async def test_a_store_error_writes_nothing_and_warns(
    monkeypatch, storage, sandbox_store, caplog
):
    await _record(sandbox_store)
    backend = _AgentCoreLike({"plot.png": b"plot"})

    async def _down(_conversation_id):
        raise ConnectionError("valkey down")

    monkeypatch.setattr(sandbox_store, "get", _down)
    caplog.set_level(logging.DEBUG)

    await sync_workspace(_session(backend, []), "u1", "c1")

    assert storage.objects == {}
    assert _warnings(caplog, "reason=store_unavailable")


async def test_a_store_that_hangs_is_a_store_error(
    monkeypatch, storage, sandbox_store, caplog
):
    await _record(sandbox_store)
    backend = _AgentCoreLike({"plot.png": b"plot"})
    monkeypatch.setattr(workspace, "_ALIVE_TIMEOUT_S", 0.05)

    async def _hang(_conversation_id):
        await asyncio.Event().wait()

    monkeypatch.setattr(sandbox_store, "get", _hang)
    caplog.set_level(logging.DEBUG)

    await asyncio.wait_for(
        sync_workspace(_session(backend, []), "u1", "c1"), timeout=5
    )

    assert storage.objects == {}
    assert _warnings(caplog, "reason=store_unavailable")


async def test_a_failed_kill_on_delete_still_stops_the_save(
    monkeypatch, storage, sandbox_store, caplog
):
    from src.services.agents.core import sandbox_pool

    await _record(sandbox_store)
    backend = _AgentCoreLike({"plot.png": b"plot"})

    def _kill_fails(_record):
        raise RuntimeError("stop failed")

    monkeypatch.setattr(sandbox_pool, "kill_record", _kill_fails)
    await sandbox_pool.discard_conversation_sandbox("c1")
    assert await sandbox_store.get("c1") is not None
    caplog.set_level(logging.DEBUG)

    await sync_workspace(_session(backend, []), "u1", "c1")

    assert storage.objects == {}
    assert backend.commands == []
    assert _warnings(caplog, "workspace.") == []


async def test_a_conversation_deleted_during_the_save_writes_nothing_more(
    storage, sandbox_store, caplog
):
    await _record(sandbox_store)
    backend = _AgentCoreLike({"a.txt": b"aaa", "b.txt": b"bbb"})
    detached: list[str] = []

    def _delete_now():
        backend.closed = True
        sandbox_store._records.pop("c1", None)

    backend.on_download = _delete_now
    caplog.set_level(logging.DEBUG)

    await sync_workspace(_session(backend, detached), "u1", "c1")

    assert storage.objects == {}
    assert workspace_state_key("u1", "c1") not in storage.objects
    assert _noisy(caplog) == []
    assert detached == ["sb-1"]


async def test_a_sandbox_replaced_by_another_one_is_not_saved_over(
    storage, sandbox_store, caplog
):
    await _record(sandbox_store, sandbox_id="sb-2")
    backend = _AgentCoreLike({"plot.png": b"plot"})
    caplog.set_level(logging.DEBUG)

    await sync_workspace(_session(backend, []), "u1", "c1")

    assert storage.objects == {}
    assert backend.commands == []
    assert _noisy(caplog) == []


async def test_a_normal_post_turn_save_still_writes_the_workspace(
    storage, sandbox_store, caplog
):
    await _record(sandbox_store)
    backend = _AgentCoreLike({"plot.png": b"plot"})
    detached: list[str] = []
    caplog.set_level(logging.DEBUG)

    await sync_workspace(_session(backend, detached), "u1", "c1")

    prefix = workspace_prefix("u1", "c1")
    assert storage.objects[f"{prefix}plot.png"] == b"plot"
    state = json.loads(storage.objects[workspace_state_key("u1", "c1")])
    assert set(state["files"]) == {"plot.png"}
    assert detached == ["sb-1"]
    assert _noisy(caplog) == []


def test_an_expired_session_during_a_turn_stays_an_error(caplog):
    caplog.set_level(logging.DEBUG)
    backend = _AgentCoreLike({})
    backend.closed = True

    backend.execute("ls")

    sdk_lines = [r for r in caplog.records if r.name == _SDK_LOGGER]
    assert [r.levelno for r in sdk_lines] == [logging.ERROR]
