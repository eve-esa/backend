"""Tests for the src.commands.gc_mcp_artifacts retention/GC command."""

from datetime import datetime, timedelta, timezone

import pytest

from src.commands.gc_mcp_artifacts import gc_mcp_artifacts
from src.database.models.artifact import Artifact, ArtifactSource
from src.database.models.conversation import Conversation
from src.database.models.message import Message
from tests.utils.cleaner import cleanup_models
from tests.utils.fake_storage import FakeStorage
from tests.utils.utils import create_test_user_and_token

OLD = datetime.now(timezone.utc) - timedelta(days=45)
RECENT = datetime.now(timezone.utc) - timedelta(days=1)


def _use_fake_storage(monkeypatch) -> FakeStorage:
    fake = FakeStorage()
    monkeypatch.setattr("src.commands.gc_mcp_artifacts.storage_service", fake)
    return fake


async def _make_artifact(
    fake: FakeStorage, user_id: str, source_type: str, timestamp: datetime
) -> Artifact:
    key = fake.build_user_key(user_id, "bin", prefix="artifacts")
    await fake.put_object(key, b"payload", "application/octet-stream")
    kwargs = {"mcp_server": "wiley", "tool_name": "search"} if source_type == "mcp_tool" else {}
    return await Artifact.create(
        user_id=user_id,
        key=key,
        filename="artifact.bin",
        content_type="application/octet-stream",
        size_bytes=7,
        source=ArtifactSource(type=source_type, **kwargs),
        timestamp=timestamp,
    )


@pytest.mark.asyncio
async def test_dry_run_reports_without_deleting(monkeypatch):
    """The default dry-run mode leaves storage and Mongo untouched."""
    fake = _use_fake_storage(monkeypatch)
    user, _token = await create_test_user_and_token()
    try:
        old_artifact = await _make_artifact(fake, user.id, "mcp_tool", OLD)

        await gc_mcp_artifacts(days=30, apply=False)

        assert await Artifact.find_by_id(old_artifact.id) is not None
        assert old_artifact.key in fake.objects
    finally:
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_apply_deletes_old_mcp_artifacts_only(monkeypatch):
    """--apply deletes old mcp_tool artifacts but spares recent ones and uploads."""
    fake = _use_fake_storage(monkeypatch)
    user, _token = await create_test_user_and_token()
    try:
        old_mcp = await _make_artifact(fake, user.id, "mcp_tool", OLD)
        recent_mcp = await _make_artifact(fake, user.id, "mcp_tool", RECENT)
        old_upload = await _make_artifact(fake, user.id, "upload", OLD)

        await gc_mcp_artifacts(days=30, apply=True)

        assert await Artifact.find_by_id(old_mcp.id) is None
        assert old_mcp.key not in fake.objects

        assert await Artifact.find_by_id(recent_mcp.id) is not None
        assert recent_mcp.key in fake.objects

        # source.type == "upload" is never eligible, regardless of age.
        assert await Artifact.find_by_id(old_upload.id) is not None
        assert old_upload.key in fake.objects
    finally:
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_apply_scrubs_deleted_ids_from_messages(monkeypatch):
    """Deleted artifact ids are $pull-ed from every message's artifact_ids array."""
    fake = _use_fake_storage(monkeypatch)
    user, _token = await create_test_user_and_token()
    conversation = await Conversation.create(user_id=user.id, name="GC test")
    try:
        old_mcp = await _make_artifact(fake, user.id, "mcp_tool", OLD)
        recent_mcp = await _make_artifact(fake, user.id, "mcp_tool", RECENT)

        message = await Message.create(
            conversation_id=conversation.id,
            input="q",
            output="a",
            artifact_ids=[old_mcp.id, recent_mcp.id],
        )

        await gc_mcp_artifacts(days=30, apply=True)

        refreshed = await Message.find_by_id(message.id)
        assert refreshed.artifact_ids == [recent_mcp.id]
    finally:
        await Message.delete_many({"conversation_id": conversation.id})
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([conversation, user])


@pytest.mark.asyncio
async def test_storage_failure_skips_and_keeps_mongo_record(monkeypatch):
    """A storage delete failure fails closed: the Mongo record is kept, not orphaned."""
    fake = _use_fake_storage(monkeypatch)
    user, _token = await create_test_user_and_token()
    try:
        old_mcp = await _make_artifact(fake, user.id, "mcp_tool", OLD)

        async def _boom(key):
            raise RuntimeError("storage unavailable")

        monkeypatch.setattr(fake, "delete_object", _boom)

        await gc_mcp_artifacts(days=30, apply=True)

        # Fail closed: the DB record survives so a retry can reclaim the object.
        assert await Artifact.find_by_id(old_mcp.id) is not None
    finally:
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user])
