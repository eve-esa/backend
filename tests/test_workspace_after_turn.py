"""The post-turn workspace save when the sandbox session is already gone.

A conversation delete discards the sandbox while the save of the turn that
just ended still runs. That save stops, writes nothing, and logs at INFO.
A session that expires during a turn keeps its ERROR.
"""

import json
import logging

import pytest

from src.services.agents.core import workspace
from src.services.agents.core.sandbox import SandboxSession
from src.services.agents.core.workspace import (
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

    async def put_object(self, key, body, content_type, tagging=None):
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


async def _record(sandbox_store, sandbox_id="sb-1"):
    await sandbox_store.put(
        "c1", {"sandbox_id": sandbox_id, "provider": "agentcore", "user_id": "u1"}
    )


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


async def test_a_session_that_expires_under_the_sync_is_info_not_error(
    storage, sandbox_store, caplog
):
    await _record(sandbox_store)
    backend = _AgentCoreLike({"plot.png": b"plot"})
    backend.closed = True
    detached: list[str] = []
    caplog.set_level(logging.DEBUG)

    await sync_workspace(_session(backend, detached), "u1", "c1")

    assert _noisy(caplog) == []
    assert storage.objects == {}
    assert detached == ["sb-1"]
    sdk_lines = [r for r in caplog.records if r.name == _SDK_LOGGER]
    assert sdk_lines and all(r.levelno == logging.INFO for r in sdk_lines)
    assert all("reason=" in r.getMessage() for r in sdk_lines)


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
