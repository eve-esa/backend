"""Conversation files and sandbox routes: flag, ownership, listings, live tree."""

from datetime import datetime, timezone

import pytest

from src.database.models.conversation import Conversation
from src.services.agents.core.sandbox import SandboxGone
from src.services.agents.core.sandbox_inspect import (
    _find_command,
    conversation_prefix,
)
from tests.utils.cleaner import cleanup_models
from tests.utils.fake_storage import FakeStorage
from tests.utils.utils import create_test_user_and_token

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


class ListingStorage(FakeStorage):
    """FakeStorage plus ``list_key_records``, newest first like StorageService."""

    def list_key_records(self, prefix, max_keys, *, scan_limit=1000):
        keys = sorted(
            (key for key in self.objects if key.startswith(prefix)), reverse=True
        )
        records = [
            {
                "key": key,
                "size": len(self.objects[key]["body"]),
                "last_modified": datetime(2026, 11, 18, tzinfo=timezone.utc),
            }
            for key in keys
        ]
        return records[:max_keys], len(records) > max_keys


class FakeSession:
    def __init__(self, output, exit_code=0):
        self.commands = []
        self.detached = False
        self.destroyed = False
        session = self

        class Backend:
            def execute(self, command):
                session.commands.append(command)
                return {"output": output, "exit_code": exit_code}

        self.backend = Backend()

    def detach(self):
        self.detached = True

    def destroy(self):
        self.destroyed = True


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def api_on(monkeypatch):
    monkeypatch.setattr("src.config.FEATURE_SANDBOX_API", True)


@pytest.fixture
def storage(monkeypatch):
    fake = ListingStorage()
    monkeypatch.setattr("src.services.agents.core.sandbox_inspect.storage_service", fake)
    monkeypatch.setattr("src.routers.conversation.storage_service", fake)
    return fake


@pytest.fixture
async def owned():
    user, token = await create_test_user_and_token()
    conversation = await Conversation.create(user_id=user.id, name="Sandbox API")
    try:
        yield user, token, conversation
    finally:
        await cleanup_models([user, conversation])


ROUTES = ("files", "files/workspace/a.txt", "sandbox", "sandbox/files")


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ROUTES)
async def test_flag_off_answers_404(async_client, monkeypatch, storage, owned, route):
    monkeypatch.setattr("src.config.FEATURE_SANDBOX_API", False)
    _user, token, conversation = owned
    resp = await async_client.get(
        f"/conversations/{conversation.id}/{route}", headers=_auth(token)
    )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Not Found"


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ROUTES)
async def test_another_users_conversation_answers_404(
    async_client, api_on, storage, owned, route
):
    _user, _token, conversation = owned
    other, other_token = await create_test_user_and_token()
    try:
        resp = await async_client.get(
            f"/conversations/{conversation.id}/{route}", headers=_auth(other_token)
        )
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Conversation not found"
    finally:
        await cleanup_models([other])


@pytest.mark.asyncio
async def test_files_are_relative_to_the_conversation(
    async_client, api_on, storage, owned
):
    user, token, conversation = owned
    base = conversation_prefix(user.id, conversation.id)
    await storage.put_object(f"{base}workspace/out/a.txt", b"hello", "text/plain")
    await storage.put_object(f"{base}artifacts/x/pic.png", PNG_BYTES, "image/png")
    await storage.put_object(f"{base}workspace_state", b"{}", "application/json")
    await storage.put_object(
        f"users/{user.id}/conversations/other/workspace/b.txt", b"no", "text/plain"
    )

    resp = await async_client.get(
        f"/conversations/{conversation.id}/files", headers=_auth(token)
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert [item["path"] for item in body["items"]] == [
        "artifacts/x/pic.png",
        "workspace/out/a.txt",
        "workspace_state",
    ]
    assert body["items"][1]["size"] == 5
    assert body["truncated"] is False


@pytest.mark.asyncio
async def test_files_prefix_and_limit(async_client, api_on, storage, owned):
    user, token, conversation = owned
    base = conversation_prefix(user.id, conversation.id)
    for name in ("a", "b", "c"):
        await storage.put_object(f"{base}workspace/{name}.txt", b"x", "text/plain")
    await storage.put_object(f"{base}artifacts/x/pic.png", PNG_BYTES, "image/png")

    resp = await async_client.get(
        f"/conversations/{conversation.id}/files",
        params={"prefix": "workspace/", "limit": 2},
        headers=_auth(token),
    )

    body = resp.json()
    assert len(body["items"]) == 2
    assert all(item["path"].startswith("workspace/") for item in body["items"])
    assert body["truncated"] is True


@pytest.mark.asyncio
async def test_file_download_uses_the_tool_allowlist(
    async_client, api_on, storage, owned
):
    user, token, conversation = owned
    base = conversation_prefix(user.id, conversation.id)
    await storage.put_object(f"{base}artifacts/x/pic.png", PNG_BYTES, "image/png")
    await storage.put_object(f"{base}workspace/page.html", b"<html/>", "text/html")

    png = await async_client.get(
        f"/conversations/{conversation.id}/files/artifacts/x/pic.png",
        headers=_auth(token),
    )
    html = await async_client.get(
        f"/conversations/{conversation.id}/files/workspace/page.html",
        headers=_auth(token),
    )
    missing = await async_client.get(
        f"/conversations/{conversation.id}/files/workspace/none.txt",
        headers=_auth(token),
    )

    assert png.status_code == 200
    assert png.content == PNG_BYTES
    assert png.headers["content-type"] == "image/png"
    assert html.headers["content-type"] == "application/octet-stream"
    assert html.headers["content-disposition"].startswith("attachment")
    assert missing.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route, params",
    [
        ("files/workspace/%2E%2E/%2E%2E/other/x", None),
        ("files/workspace/%2E/x", None),
        ("files", {"prefix": "workspace/../"}),
        ("sandbox/files", {"prefix": "../etc"}),
    ],
)
async def test_traversal_answers_404(
    async_client, api_on, storage, owned, sandbox_store, route, params
):
    _user, token, conversation = owned
    await sandbox_store.put(
        conversation.id, {"provider": "agentcore", "sandbox_id": "s-1"}
    )
    resp = await async_client.get(
        f"/conversations/{conversation.id}/{route}",
        params=params,
        headers=_auth(token),
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_sandbox_disabled_without_provider(
    async_client, api_on, monkeypatch, owned
):
    monkeypatch.delenv("SANDBOX_PROVIDER", raising=False)
    _user, token, conversation = owned
    resp = await async_client.get(
        f"/conversations/{conversation.id}/sandbox", headers=_auth(token)
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "disabled"


@pytest.mark.asyncio
async def test_sandbox_none_without_record(async_client, api_on, monkeypatch, owned):
    monkeypatch.setenv("SANDBOX_PROVIDER", "agentcore")
    _user, token, conversation = owned
    resp = await async_client.get(
        f"/conversations/{conversation.id}/sandbox", headers=_auth(token)
    )
    assert resp.json()["status"] == "none"


@pytest.mark.asyncio
@pytest.mark.parametrize("inflight, status", [("0", "idle"), ("2", "busy")])
async def test_sandbox_reads_the_pool_record(
    async_client, api_on, monkeypatch, owned, sandbox_store, inflight, status
):
    monkeypatch.setenv("SANDBOX_PROVIDER", "agentcore")
    _user, token, conversation = owned
    await sandbox_store.put(
        conversation.id,
        {
            "provider": "agentcore",
            "sandbox_id": "s-1",
            "skills_digest": "abc",
            "inflight": inflight,
            "created_at": "1800000000",
            "max_lifetime_s": "3600",
        },
    )
    await sandbox_store.set_deadline(conversation.id, 1800000600.0)

    resp = await async_client.get(
        f"/conversations/{conversation.id}/sandbox", headers=_auth(token)
    )

    body = resp.json()
    assert body["status"] == status
    assert body["sandbox_id"] == "s-1"
    assert body["inflight"] == int(inflight)
    assert datetime.fromisoformat(body["created_at"]).timestamp() == 1800000000
    assert datetime.fromisoformat(body["expires_at"]).timestamp() == 1800003600
    assert datetime.fromisoformat(body["idle_deadline"]).timestamp() == 1800000600


@pytest.mark.asyncio
async def test_sandbox_files_lists_and_detaches(
    async_client, api_on, monkeypatch, owned, sandbox_store
):
    _user, token, conversation = owned
    await sandbox_store.put(
        conversation.id, {"provider": "agentcore", "sandbox_id": "s-1"}
    )
    session = FakeSession(
        "workspace/b.txt\t3\t1800000000.5\n"
        "skills/demo/SKILL.md\t10\t1800000000\n"
        "garbage line\n"
    )
    monkeypatch.setattr(
        "src.services.agents.core.sandbox_inspect.attach_session",
        lambda record: session,
    )

    resp = await async_client.get(
        f"/conversations/{conversation.id}/sandbox/files", headers=_auth(token)
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert [item["path"] for item in body["items"]] == [
        "skills/demo/SKILL.md",
        "workspace/b.txt",
    ]
    assert body["truncated"] is False
    assert session.detached is True
    assert session.destroyed is False
    assert await sandbox_store.deadline(conversation.id) is None


@pytest.mark.asyncio
async def test_sandbox_files_marks_truncated(
    async_client, api_on, monkeypatch, owned, sandbox_store
):
    _user, token, conversation = owned
    await sandbox_store.put(
        conversation.id, {"provider": "agentcore", "sandbox_id": "s-1"}
    )
    lines = "".join(f"workspace/{i}.txt\t1\t1800000000\n" for i in range(3))
    monkeypatch.setattr(
        "src.services.agents.core.sandbox_inspect.attach_session",
        lambda record: FakeSession(lines),
    )

    resp = await async_client.get(
        f"/conversations/{conversation.id}/sandbox/files",
        params={"limit": 2},
        headers=_auth(token),
    )

    body = resp.json()
    assert len(body["items"]) == 2
    assert body["truncated"] is True


@pytest.mark.asyncio
async def test_sandbox_files_without_record_answers_404(async_client, api_on, owned):
    _user, token, conversation = owned
    resp = await async_client.get(
        f"/conversations/{conversation.id}/sandbox/files", headers=_auth(token)
    )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Sandbox not found"


@pytest.mark.asyncio
async def test_sandbox_files_gone_answers_410(
    async_client, api_on, monkeypatch, owned, sandbox_store
):
    _user, token, conversation = owned
    await sandbox_store.put(
        conversation.id, {"provider": "agentcore", "sandbox_id": "s-1"}
    )

    def gone(record):
        raise SandboxGone("ended")

    monkeypatch.setattr("src.services.agents.core.sandbox_inspect.attach_session", gone)

    resp = await async_client.get(
        f"/conversations/{conversation.id}/sandbox/files", headers=_auth(token)
    )
    assert resp.status_code == 410
    # A read never drops the record; the reaper or the next turn does.
    assert await sandbox_store.get(conversation.id) is not None


@pytest.mark.asyncio
async def test_sandbox_files_failed_listing_answers_502(
    async_client, api_on, monkeypatch, owned, sandbox_store
):
    _user, token, conversation = owned
    await sandbox_store.put(
        conversation.id, {"provider": "agentcore", "sandbox_id": "s-1"}
    )
    session = FakeSession("", exit_code=1)
    monkeypatch.setattr(
        "src.services.agents.core.sandbox_inspect.attach_session",
        lambda record: session,
    )

    resp = await async_client.get(
        f"/conversations/{conversation.id}/sandbox/files", headers=_auth(token)
    )
    assert resp.status_code == 502
    assert session.detached is True


def test_find_command_scopes_to_the_prefix_directory():
    everything = _find_command("", 10)
    assert "for d in workspace skills artifacts large_tool_results" in everything
    assert "awk" not in everything
    assert everything.endswith("head -n 11")

    scoped = _find_command("workspace/out/re", 5)
    assert "for d in workspace/out;" in scoped
    assert "awk -v p=workspace/out/re" in scoped


def test_find_command_quotes_the_prefix():
    command = _find_command("workspace/a b;rm -rf x", 5)
    assert "'workspace/a b;rm -rf x'" in command
