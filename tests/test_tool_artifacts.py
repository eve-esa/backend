"""A new sandbox is filled from the conversation tool_artifacts prefix."""

import pytest

from src.services.mcp.tool_artifacts import restore_tool_artifacts


class _Body:
    def __init__(self, data: bytes):
        self.data = data

    def read(self):
        return self.data

    def close(self):
        return None


class _Store:
    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects

    def list_key_records(self, prefix, max_keys):
        records = [
            {"key": key, "size": len(body)}
            for key, body in self.objects.items()
            if key.startswith(prefix)
        ]
        return records[:max_keys], False

    async def get_object(self, key):
        return {"Body": _Body(self.objects[key])}


class _Sandbox:
    def __init__(self):
        self.files: list[tuple[str, bytes]] = []

    def upload_files(self, files):
        self.files.extend(files)
        return [{} for _ in files]


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_restore_uploads_the_conversation_prefix(monkeypatch):
    key = "users/u1/conversations/c1/tool_artifacts/abc/report.txt"
    store = _Store({key: b"report\n"})
    monkeypatch.setattr("src.services.mcp.tool_artifacts.storage_service", store)
    sandbox = _Sandbox()

    await restore_tool_artifacts(sandbox, "u1", "c1")

    assert sandbox.files == [("/tool_artifacts/abc/report.txt", b"report\n")]
