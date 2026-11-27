"""Per-user storage quota: the counter, every write path, and the fill command.

Counting is on even when ``FEATURE_STORAGE_QUOTA`` is off. The flag only
refuses a write and sends the notice.
"""

import asyncio
import base64
import io
import json
from types import SimpleNamespace

import pytest
from bson import ObjectId
from mcp.types import CallToolResult, ImageContent, TextContent

from src import config
from src.commands import reconcile_storage_usage as reconcile_cmd
from src.database.models.artifact import Artifact, ArtifactSource
from src.database.models.conversation import Conversation
from src.database.models.message import Message
from src.database.models.user import User
from src.schemas.rate_limit import RateLimitGroup
from src.services import artifact_cleanup
from src.services.agents.core import tool_results as tool_results_mod
from src.services.agents.core import workspace as workspace_mod
from src.services.agents.core.tool_results import S3ToolResultBackend
from src.services.mcp.artifact_context import (
    ArtifactRequestContext,
    reset_artifact_context,
    set_artifact_context,
)
from src.services.mcp.artifact_ingestion import (
    QUOTA_FAILURE_TEXT,
    QUOTA_REFUSED,
    ArtifactInterceptor,
)
from src.services.mcp.conversation_artifacts import file_path_for, object_key
from src.services.storage_quota import (
    GIB,
    STORAGE_QUOTA_CODE,
    count_storage,
    get_storage_usage_summary,
    opening_storage_notice,
    quota_bytes_for,
    release_storage,
    reserve_storage,
)
from tests.utils.cleaner import cleanup_models
from tests.utils.fake_storage import FakeStorage
from tests.utils.utils import create_test_user_and_token

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _enable(monkeypatch, gigabytes="1"):
    monkeypatch.setattr(config, "FEATURE_STORAGE_QUOTA", True)
    monkeypatch.setattr(config, "FEATURE_ARTIFACT_UPLOADS", True)
    monkeypatch.setenv("FREE_STORAGE_GB", gigabytes)


async def _used(user_id: str) -> int:
    fresh = await User.find_by_id(user_id)
    return int(fresh.storage_bytes_used or 0)


async def _set_used(user_id: str, nbytes: int) -> None:
    await User.get_collection().update_one(
        {"_id": ObjectId(user_id)}, {"$set": {"storage_bytes_used": nbytes}}
    )


class _SyncStore:
    """The sync methods the tool-result backend calls."""

    def __init__(self):
        self.objects = {}

    def exists_sync(self, key):
        return key in self.objects

    def put_object_sync(self, key, body, content_type):
        self.objects[key] = body


# ---------------- reserve ----------------


@pytest.mark.asyncio
async def test_reserve_under_at_and_over_the_cap(monkeypatch):
    _enable(monkeypatch)
    user, _ = await create_test_user_and_token()
    try:
        cap = quota_bytes_for(user)
        assert cap == GIB

        accepted, _ = await reserve_storage(user.id, 4, cap)
        assert accepted is True
        assert await _used(user.id) == 4

        await _set_used(user.id, cap)
        accepted, used = await reserve_storage(user.id, 1, cap)
        assert accepted is False
        assert used == cap
        assert await _used(user.id) == cap
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_two_writes_cannot_both_cross_the_cap(monkeypatch):
    _enable(monkeypatch)
    user, _ = await create_test_user_and_token()
    try:
        cap = 10
        first, second = await asyncio.gather(
            reserve_storage(user.id, 6, cap),
            reserve_storage(user.id, 6, cap),
        )
        assert sorted([first[0], second[0]]) == [False, True]
        assert await _used(user.id) == 6
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_no_cap_and_unlimited_are_never_refused(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.delenv("FREE_STORAGE_GB", raising=False)
    user, _ = await create_test_user_and_token()
    try:
        assert quota_bytes_for(user) is None
        accepted, _ = await reserve_storage(user.id, 50, None)
        assert accepted is True
        assert await _used(user.id) == 50

        await User.get_collection().update_one(
            {"_id": ObjectId(user.id)},
            {"$set": {"rate_limit_group": RateLimitGroup.UNLIMITED.value}},
        )
        fresh = await User.find_by_id(user.id)
        monkeypatch.setenv("FREE_STORAGE_GB", "1")
        assert quota_bytes_for(fresh) is None
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_flag_off_counts_and_does_not_refuse_or_notice(monkeypatch):
    monkeypatch.setattr(config, "FEATURE_STORAGE_QUOTA", False)
    monkeypatch.setenv("FREE_STORAGE_GB", "1")
    user, _ = await create_test_user_and_token()
    try:
        await _set_used(user.id, GIB)
        accepted, _ = await reserve_storage(user.id, 10, GIB)
        assert accepted is True
        assert await _used(user.id) == GIB + 10
        assert await opening_storage_notice(user.id) is None
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_release_may_go_negative_and_the_summary_clamps_it():
    user, _ = await create_test_user_and_token()
    try:
        await release_storage(user.id, 5)
        fresh = await User.find_by_id(user.id)
        assert fresh.storage_bytes_used == -5
        summary = await get_storage_usage_summary(fresh)
        assert summary["used_bytes"] == 0
        assert summary["used_gb"] == 0
    finally:
        await cleanup_models([user])


# ---------------- upload ----------------


async def _upload(async_client, token):
    return await async_client.post(
        "/artifacts",
        headers=_auth(token),
        files={"file": ("pic.png", io.BytesIO(PNG_BYTES), "image/png")},
    )


@pytest.mark.asyncio
async def test_upload_over_the_cap_is_403_and_stores_nothing(async_client, monkeypatch):
    _enable(monkeypatch)
    fake = FakeStorage()
    monkeypatch.setattr("src.routers.artifact.storage_service", fake)
    user, token = await create_test_user_and_token()
    try:
        await _set_used(user.id, GIB)
        resp = await _upload(async_client, token)
        assert resp.status_code == 403
        body = resp.json()["detail"]
        assert body["code"] == STORAGE_QUOTA_CODE
        assert body["max_bytes"] == GIB
        assert body["needed_bytes"] == len(PNG_BYTES)
        assert fake.objects == {}
        assert await _used(user.id) == GIB
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_upload_counts_when_the_flag_is_off_and_delete_releases(
    async_client, monkeypatch
):
    monkeypatch.setattr(config, "FEATURE_STORAGE_QUOTA", False)
    monkeypatch.setattr(config, "FEATURE_ARTIFACT_UPLOADS", True)
    monkeypatch.setenv("FREE_STORAGE_GB", "1")
    fake = FakeStorage()
    monkeypatch.setattr("src.routers.artifact.storage_service", fake)
    monkeypatch.setattr("src.services.artifact_cleanup.storage_service", fake)
    user, token = await create_test_user_and_token()
    try:
        await _set_used(user.id, GIB)
        resp = await _upload(async_client, token)
        assert resp.status_code == 200, resp.text
        assert await _used(user.id) == GIB + len(PNG_BYTES)

        gone = await async_client.delete(
            f"/artifacts/{resp.json()['id']}", headers=_auth(token)
        )
        assert gone.status_code == 200
        assert await _used(user.id) == GIB
    finally:
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user])


# ---------------- MCP ----------------


@pytest.mark.asyncio
async def test_mcp_persist_refuses_with_the_warning_and_the_flag(monkeypatch):
    _enable(monkeypatch)
    fake = FakeStorage()
    monkeypatch.setattr("src.services.mcp.artifact_ingestion.storage_service", fake)
    user, _ = await create_test_user_and_token()
    try:
        await _set_used(user.id, GIB)
        ctx = ArtifactRequestContext(user_id=user.id, conversation_id="c1")
        stored = await ArtifactInterceptor()._persist(
            b"png-bytes", "image/png", "plot.png", ctx, "draw", "server"
        )
        assert stored is QUOTA_REFUSED
        assert ctx.storage_quota_exceeded is True
        assert ctx.storage_max_bytes == GIB
        assert fake.objects == {}
        assert await _used(user.id) == GIB
    finally:
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_mcp_quota_refusal_fails_the_call_and_drops_the_output(monkeypatch):
    _enable(monkeypatch)
    fake = FakeStorage()
    monkeypatch.setattr("src.services.mcp.artifact_ingestion.storage_service", fake)
    user, _ = await create_test_user_and_token()
    payload = {
        "collection": "sentinel-2-l2a",
        "path": "/tmp/cdse_images/sentinel-2-l2a_x.jpg",
        "size_bytes": len(PNG_BYTES),
    }
    result = CallToolResult(
        content=[
            TextContent(type="text", text=json.dumps(payload)),
            ImageContent(
                type="image",
                data=base64.b64encode(PNG_BYTES).decode("ascii"),
                mimeType="image/png",
            ),
        ],
        structuredContent=payload,
    )

    async def handler(_request):
        return result

    ctx, token = set_artifact_context(user_id=user.id, conversation_id="c1")
    try:
        await _set_used(user.id, GIB)
        out = await ArtifactInterceptor()(
            SimpleNamespace(name="fetch_sentinel2", server_name="cdse"), handler
        )
        assert out.isError is True
        assert [block.text for block in out.content] == [QUOTA_FAILURE_TEXT]
        assert out.structuredContent is None
        assert "/tmp/" not in out.model_dump_json()
        assert ctx.storage_quota_exceeded is True
        assert ctx.storage_max_bytes == GIB
        assert fake.objects == {}
    finally:
        reset_artifact_context(token)
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user])


# ---------------- workspace and tool results ----------------


def _patch_workspace(monkeypatch, fake, listing):
    async def list_files(_backend):
        return listing, False

    async def download(_backend, _relative):
        return listing[0][1] and b"x" * listing[0][1]

    monkeypatch.setattr(workspace_mod, "storage_service", fake)
    monkeypatch.setattr(workspace_mod, "_list_files", list_files)
    monkeypatch.setattr(workspace_mod, "_download", download)


@pytest.mark.asyncio
async def test_workspace_skips_a_file_that_does_not_fit(monkeypatch):
    _enable(monkeypatch)
    user, _ = await create_test_user_and_token()
    fake = FakeStorage()
    _patch_workspace(monkeypatch, fake, [("note.txt", 100, "1")])
    try:
        await _set_used(user.id, GIB)
        await workspace_mod._save_locked(object(), user.id, "c1")
        assert not any(key.endswith("note.txt") for key in fake.objects)
        fresh = await User.find_by_id(user.id)
        assert fresh.storage_notice_pending is True
        assert await opening_storage_notice(user.id) is not None
        assert (await User.find_by_id(user.id)).storage_notice_pending is False
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_workspace_charges_only_the_growth_and_an_undo_releases(monkeypatch):
    _enable(monkeypatch, gigabytes="20")
    user, _ = await create_test_user_and_token()
    fake = FakeStorage()
    monkeypatch.setattr(workspace_mod, "storage_service", fake)
    key = f"users/{user.id}/conversations/c1/workspace/a.txt"
    try:
        assert await workspace_mod._store_bytes(key, b"a" * 10, "text/plain", None, user.id, GIB)
        assert await workspace_mod._store_bytes(key, b"b" * 15, "text/plain", None, user.id, GIB)
        assert await _used(user.id) == 15

        calls = {"n": 0}

        async def alive():
            calls["n"] += 1
            return calls["n"] == 1

        with pytest.raises(workspace_mod._SandboxClosed):
            await workspace_mod._store_bytes(
                key, b"c" * 15, "text/plain", alive, user.id, GIB
            )
        assert key not in fake.objects
        assert await _used(user.id) == 0
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_overlapping_saves_charge_the_file_once(monkeypatch):
    _enable(monkeypatch, gigabytes="20")
    user, _ = await create_test_user_and_token()
    fake = FakeStorage()
    _patch_workspace(monkeypatch, fake, [("a.txt", 20, "1")])
    try:
        await asyncio.gather(
            workspace_mod._save_locked(object(), user.id, "c1"),
            workspace_mod._save_locked(object(), user.id, "c1"),
        )
        once = await _used(user.id)
        await workspace_mod._save_locked(object(), user.id, "c1")
        assert await _used(user.id) == once
        assert once > 0
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_tool_results_are_counted_and_never_refused(monkeypatch):
    _enable(monkeypatch)
    user, _ = await create_test_user_and_token()
    store = _SyncStore()
    monkeypatch.setattr(tool_results_mod, "storage_service", store)
    await _set_used(user.id, GIB)
    try:
        backend = S3ToolResultBackend(user.id, "c1")
        result = backend.write("/large_tool_results/call_1", "hello")
        assert result.error is None
        assert await _used(user.id) == GIB + len(b"hello")
        assert store.objects
    finally:
        await cleanup_models([user])


# ---------------- cascade ----------------


@pytest.mark.asyncio
async def test_cascade_moves_a_shared_object_then_sweeps_and_releases(monkeypatch):
    user, _ = await create_test_user_and_token()
    fake = FakeStorage()
    monkeypatch.setattr("src.services.artifact_cleanup.storage_service", fake)
    monkeypatch.setattr("src.services.artifact_move.storage_service", fake)
    first = await Conversation.create(user_id=user.id, name="A")
    second = await Conversation.create(user_id=user.id, name="B")
    exclusive_key = object_key(user.id, first.id, file_path_for("only", "only.png"))
    orphan = f"users/{user.id}/conversations/{first.id}/stray.bin"
    await fake.put_object(exclusive_key, b"e" * 10, "image/png")
    await fake.put_object(orphan, b"o" * 30, "application/octet-stream")
    shared = await Artifact.create(
        user_id=user.id,
        key="pending",
        filename="pic.png",
        content_type="image/png",
        size_bytes=20,
        source=ArtifactSource(type="mcp_tool", mcp_server="s", tool_name="t"),
        conversation_id=first.id,
    )
    shared_key = object_key(user.id, first.id, file_path_for(shared.id, "pic.png"))
    await fake.put_object(shared_key, b"s" * 20, "image/png")
    await Artifact.get_collection().update_one(
        {"_id": ObjectId(shared.id)}, {"$set": {"key": shared_key}}
    )
    shared = await Artifact.find_by_id(shared.id)
    await _set_used(user.id, 20 + 10 + 30)
    exclusive = await Artifact.create(
        user_id=user.id,
        key=exclusive_key,
        filename="only.png",
        content_type="image/png",
        size_bytes=10,
        source=ArtifactSource(type="upload"),
        conversation_id=first.id,
    )
    reuse = await Message.create(
        conversation_id=second.id, input="q", output="a", artifact_ids=[shared.id]
    )
    try:
        await artifact_cleanup.delete_conversation_artifacts(user.id, first.id)
        kept = await Artifact.find_by_id(shared.id)
        dest = object_key(user.id, second.id, file_path_for(shared.id, "pic.png"))
        assert kept is not None and kept.key == dest
        assert dest in fake.objects
        assert exclusive_key not in fake.objects
        assert orphan not in fake.objects
        assert await Artifact.find_by_id(exclusive.id) is None
        # The shared object moved. The exclusive file and the stray were released.
        assert await _used(user.id) == 20
    finally:
        await Artifact.delete_many({"user_id": user.id})
        await Message.delete_many({"conversation_id": second.id})
        await cleanup_models([user, first, second, reuse])


# ---------------- usage endpoint ----------------


@pytest.mark.asyncio
async def test_storage_usage_endpoint(async_client, monkeypatch):
    _enable(monkeypatch)
    user, token = await create_test_user_and_token()
    try:
        await _set_used(user.id, 2 * GIB)
        resp = await async_client.get("/users/me/storage-usage", headers=_auth(token))
        assert resp.status_code == 200
        body = resp.json()
        assert body["unlimited"] is False
        assert body["rate_limit_group"] == "free"
        assert body["used_bytes"] == 2 * GIB
        assert body["max_bytes"] == GIB
        assert body["remaining_bytes"] == 0
        assert body["used_ratio"] == 1.0
        assert body["used_gb"] == 2
        assert body["max_gb"] == 1
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_recount_endpoint_applies_the_listing(async_client, monkeypatch):
    _enable(monkeypatch)
    user, token = await create_test_user_and_token()

    class _Listing:
        async def sum_prefix(self, prefix):
            assert prefix == f"users/{user.id}/"
            return 5 * GIB

    try:
        await _set_used(user.id, 0)
        monkeypatch.setattr("src.services.storage_quota.storage_service", _Listing())
        resp = await async_client.post(
            "/users/me/storage-usage/recount", headers=_auth(token)
        )
        assert resp.status_code == 200
        assert resp.json()["used_bytes"] == 5 * GIB
        assert await _used(user.id) == 5 * GIB
    finally:
        await cleanup_models([user])


# ---------------- reconcile ----------------


@pytest.mark.asyncio
async def test_reconcile_dry_run_apply_and_a_write_during_the_listing(monkeypatch):
    user, _ = await create_test_user_and_token()

    class _Listing:
        def __init__(self, listed, extra=0):
            self.listed = listed
            self.extra = extra

        async def sum_prefix(self, prefix):
            assert prefix == f"users/{user.id}/"
            if self.extra:
                await count_storage(user.id, self.extra)
            return self.listed

    try:
        await _set_used(user.id, 10)
        monkeypatch.setattr(
            "src.services.storage_quota.storage_service", _Listing(50)
        )
        await reconcile_cmd.reconcile_storage_usage(user_id=user.id, apply=False)
        assert await _used(user.id) == 10

        await reconcile_cmd.reconcile_storage_usage(user_id=user.id, apply=True)
        assert await _used(user.id) == 50

        await _set_used(user.id, 0)
        monkeypatch.setattr(
            "src.services.storage_quota.storage_service", _Listing(100, extra=7)
        )
        await reconcile_cmd.reconcile_storage_usage(user_id=user.id, apply=True)
        # The 7 landed after the counter was read and is still there.
        assert await _used(user.id) == 107
    finally:
        await cleanup_models([user])
