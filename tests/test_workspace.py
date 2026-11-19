"""The sandbox working directory is saved under the conversation workspace prefix."""

import hashlib
import json

import pytest

from src.services.agents.core import workspace
from src.services.agents.core.sandbox import place_in_workspace
from src.services.agents.core.workspace import (
    publish_markdown_files,
    restore_workspace,
    save_workspace,
    session_path_for_markdown,
    settle_workspace,
    workspace_prefix,
    workspace_state_key,
)
from src.services.storage import ObjectNotFoundError


class _Body:
    def __init__(self, data: bytes):
        self.data = data

    def read(self):
        return self.data

    def close(self):
        return None


class _Store:
    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.types: dict[str, str] = {}
        self.deleted: list[str] = []

    def list_key_records(self, prefix, max_keys):
        records = [
            {"key": key, "size": len(body)}
            for key, body in self.objects.items()
            if key.startswith(prefix)
        ]
        truncated = len(records) > max_keys
        return records[:max_keys], truncated

    async def get_object(self, key):
        if key not in self.objects:
            raise ObjectNotFoundError(key)
        return {"Body": _Body(self.objects[key])}

    async def put_object(self, key, body, content_type, tagging=None):
        self.objects[key] = body
        self.types[key] = content_type

    async def delete_object(self, key):
        self.deleted.append(key)
        self.objects.pop(key, None)


class _File:
    def __init__(self, content):
        self.content = content
        self.error = None


class _Sandbox:
    def __init__(self, files=None, processes=None):
        self.tree = dict(files or {})
        self.processes = list(processes or [])
        self.uploaded: list[tuple[str, bytes]] = []
        self.commands: list[str] = []
        self.downloads: list[str] = []
        self.ps_calls = 0

    def execute(self, command):
        self.commands.append(command)
        if command.startswith("ps"):
            self.ps_calls += 1
            if self.ps_calls > 1:
                self.processes = [row for row in self.processes if row[0] == "1"]
            lines = [f"{pid} {args}" for pid, args in self.processes]
            return _Result("\n".join(lines))
        lines = [
            f"{path}\t{len(data)}\t{mtime}"
            for path, (data, mtime) in sorted(self.tree.items())
        ]
        return _Result("\n".join(lines))

    def download_files(self, paths):
        self.downloads.extend(paths)
        found = []
        for path in paths:
            key = path[len("workspace/") :] if path.startswith("workspace/") else path
            found.append(_File(self.tree[key][0]))
        return found

    def upload_files(self, files):
        self.uploaded.extend(files)
        return [{} for _ in files]


class _Result:
    def __init__(self, output, exit_code=0):
        self.output = output
        self.exit_code = exit_code


def _state(files: dict) -> bytes:
    return json.dumps({"files": files}).encode("utf-8")


def _install(monkeypatch, store):
    monkeypatch.setattr(workspace, "storage_service", store)
    monkeypatch.setattr(workspace, "_BACKOFF_S", (0, 0, 0, 0))


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_restore_uploads_workspace_files_and_skips_special_trees(monkeypatch):
    prefix = workspace_prefix("u1", "c1")
    store = _Store(
        {
            f"{prefix}notes.txt": b"hello",
            f"{prefix}skills/roll.md": b"skill",
            f"{prefix}large_tool_results/t1": b"big",
            f"{prefix}artifacts/abc/plot.png": b"png",
        }
    )
    _install(monkeypatch, store)
    sandbox = _Sandbox()

    await restore_workspace(sandbox, "u1", "c1")

    assert sandbox.uploaded == [("/workspace/notes.txt", b"hello")]


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_save_uploads_a_changed_file_and_deletes_a_removed_one(monkeypatch):
    prefix = workspace_prefix("u1", "c1")
    old = b"old"
    new = b"new!"
    store = _Store(
        {
            f"{prefix}plot.png": old,
            f"{prefix}gone.txt": b"bye",
            workspace_state_key("u1", "c1"): _state(
                {
                    "plot.png": {
                        "size": len(old),
                        "mtime": "1.0",
                        "sha256": hashlib.sha256(old).hexdigest(),
                    },
                    "gone.txt": {
                        "size": 3,
                        "mtime": "1.0",
                        "sha256": hashlib.sha256(b"bye").hexdigest(),
                    },
                }
            ),
        }
    )
    _install(monkeypatch, store)
    sandbox = _Sandbox({"plot.png": (new, "2.0"), "skills/roll.md": (b"x", "1.0")})

    await save_workspace(sandbox, "u1", "c1")

    assert sandbox.downloads == ["workspace/plot.png"]
    assert store.objects[f"{prefix}plot.png"] == new
    assert f"{prefix}gone.txt" not in store.objects
    assert store.deleted == [f"{prefix}gone.txt"]
    assert any("find " in command and "workspace" in command for command in sandbox.commands)
    state = json.loads(store.objects[workspace_state_key("u1", "c1")])
    assert "skills/roll.md" not in state["files"]
    assert state["files"]["plot.png"]["sha256"] == hashlib.sha256(new).hexdigest()


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_save_types_each_file_by_its_bytes(monkeypatch):
    prefix = workspace_prefix("u1", "c1")
    store = _Store()
    _install(monkeypatch, store)
    sandbox = _Sandbox(
        {
            "plot.png": (b"\x89PNG\r\n\x1a\n" + b"\x00" * 32, "1.0"),
            "fake.png": (b"<html><script>alert(1)</script></html>", "1.0"),
            "report.html": (b"<html><body>hi</body></html>", "1.0"),
            "notes.txt": (b"hello", "1.0"),
        }
    )

    await save_workspace(sandbox, "u1", "c1")

    assert store.types[f"{prefix}plot.png"] == "image/png"
    assert store.types[f"{prefix}fake.png"] == "application/octet-stream"
    assert store.types[f"{prefix}report.html"] == "application/octet-stream"
    assert store.types[f"{prefix}notes.txt"] == "text/plain"


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_save_skips_deletes_when_the_listing_is_truncated(monkeypatch):
    prefix = workspace_prefix("u1", "c1")
    store = _Store({f"{prefix}gone.txt": b"bye"})
    _install(monkeypatch, store)
    monkeypatch.setattr(workspace, "RESTORE_MAX_OBJECTS", 1)
    sandbox = _Sandbox(
        {"a.txt": (b"a", "1.0"), "b.txt": (b"bb", "1.0")}
    )

    await save_workspace(sandbox, "u1", "c1")

    assert store.deleted == []
    assert f"{prefix}gone.txt" in store.objects


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_settle_waits_while_a_job_is_still_running(monkeypatch):
    prefix = workspace_prefix("u1", "c1")
    store = _Store()
    _install(monkeypatch, store)
    slept: list[float] = []

    async def _sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(workspace.asyncio, "sleep", _sleep)
    sandbox = _Sandbox(
        {"plot.png": (b"plot", "1.0")},
        processes=[("1", "init"), ("99", "python job.py")],
    )

    await settle_workspace(
        sandbox, "u1", "c1", baseline="1", max_wait_s=30
    )

    assert slept
    assert sandbox.ps_calls >= 2
    assert store.objects[f"{prefix}plot.png"] == b"plot"


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_settle_saves_when_the_wait_runs_out(monkeypatch):
    store = _Store()
    _install(monkeypatch, store)
    sandbox = _Sandbox(
        {"plot.png": (b"plot", "1.0")},
        processes=[("1", "init"), ("99", "python job.py")],
    )

    await settle_workspace(sandbox, "u1", "c1", baseline="1", max_wait_s=0)

    assert store.objects[f"{workspace_prefix('u1', 'c1')}plot.png"] == b"plot"
    assert sandbox.ps_calls == 1


@pytest.mark.no_db
def test_a_model_file_goes_to_workspace_and_siblings_stay_put():
    assert place_in_workspace("ndvi.png") == "/workspace/ndvi.png"
    assert place_in_workspace("/ndvi.png") == "/workspace/ndvi.png"
    assert place_in_workspace("/workspace/ndvi.png") == "/workspace/ndvi.png"
    assert place_in_workspace("/skills/roll/SKILL.md") == "/skills/roll/SKILL.md"
    assert place_in_workspace("/artifacts/abc/a.png") == "/artifacts/abc/a.png"
    assert (
        place_in_workspace("/large_tool_results/call-1")
        == "/large_tool_results/call-1"
    )
    assert place_in_workspace("/opt/sandbox/ndvi.png", "/opt/sandbox") == (
        "/opt/sandbox/ndvi.png"
    )


@pytest.mark.no_db
def test_markdown_paths_map_onto_the_sibling_directories():
    assert session_path_for_markdown("./ndvi.png") == "workspace/ndvi.png"
    assert session_path_for_markdown("/workspace/ndvi.png") == "workspace/ndvi.png"
    assert (
        session_path_for_markdown("/opt/amazon/genesis1p-tools/var/workspace/x.png")
        == "workspace/x.png"
    )
    assert session_path_for_markdown("/skills/roll/SKILL.md") == "skills/roll/SKILL.md"
    assert session_path_for_markdown("/artifacts/abc/a.png") is None
    assert session_path_for_markdown("https://storage.googleapis.com/a.png") is None
    assert (
        session_path_for_markdown(
            "https://storage.googleapis.com/phi-lab-toolkits-public-010/workspace/ndvi_cropped.png"
        )
        == "workspace/ndvi_cropped.png"
    )


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_a_workspace_markdown_path_becomes_an_artifact(monkeypatch):
    from src.services.mcp.artifact_context import set_artifact_context, reset_artifact_context

    class _Store:
        async def put_object(self, key, body, content_type, tagging=None):
            self.key = key
            self.body = body

        def build_user_key(self, user_id, ext, prefix=None):
            return f"users/{user_id}/artifacts/file.{ext}"

    store = _Store()
    monkeypatch.setattr(workspace, "storage_service", store)
    sandbox = _Sandbox({"ndvi_cropped.png": (b"png", "1.0")})
    ctx, token = set_artifact_context("u1", "c1")
    ctx.sandbox_backend = sandbox
    try:
        answer = await publish_markdown_files(
            "![crop](https://storage.googleapis.com/phi-lab-toolkits-public-010/workspace/ndvi_cropped.png)"
        )
    finally:
        reset_artifact_context(token)

    assert answer == "![crop](/conversations/c1/workspace/ndvi_cropped.png)"
    assert store.body == b"png"
    assert (
        store.key
        == "users/u1/conversations/c1/workspace/ndvi_cropped.png"
    )
