"""Move a legacy MCP artifact into the one conversation object."""

import pytest

from src.commands.migrate_conversation_artifacts import (
    migrate_conversation_artifacts,
)
from src.database.models.artifact import Artifact, ArtifactSource
from src.database.models.conversation import Conversation
from tests.utils.cleaner import cleanup_models
from tests.utils.fake_storage import FakeStorage
from tests.utils.utils import create_test_user_and_token


def _use_fake_storage(monkeypatch) -> FakeStorage:
    fake = FakeStorage()
    monkeypatch.setattr(
        "src.commands.migrate_conversation_artifacts.storage_service", fake
    )
    return fake


async def _legacy(fake, user_id, conversation_id, *, filename="report.txt", body=b"user"):
    key = fake.build_user_key(user_id, "bin", prefix="artifacts")
    await fake.put_object(key, body, "text/plain")
    artifact = await Artifact.create(
        user_id=user_id,
        key=key,
        filename=filename,
        content_type="text/plain",
        size_bytes=len(body),
        source=ArtifactSource(
            type="mcp_tool", mcp_server="dummy", tool_name="get_file"
        ),
        conversation_id=conversation_id,
    )
    return artifact


@pytest.mark.asyncio
async def test_dry_run_changes_nothing(monkeypatch):
    fake = _use_fake_storage(monkeypatch)
    user, _token = await create_test_user_and_token()
    conversation = await Conversation.create(user_id=user.id, name="move")
    try:
        artifact = await _legacy(fake, user.id, conversation.id)
        before = set(fake.objects)

        await migrate_conversation_artifacts(apply=False)

        stored = await Artifact.find_by_id(artifact.id)
        assert stored.key == artifact.key
        assert set(fake.objects) == before
    finally:
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([conversation, user])


@pytest.mark.asyncio
async def test_apply_moves_the_record_and_prefers_the_conversation_copy(monkeypatch):
    fake = _use_fake_storage(monkeypatch)
    user, _token = await create_test_user_and_token()
    conversation = await Conversation.create(user_id=user.id, name="move")
    try:
        artifact = await _legacy(
            fake, user.id, conversation.id, filename="report.txt", body=b"user-copy"
        )
        plain = await _legacy(
            fake, user.id, conversation.id, filename="plain.txt", body=b"only-user"
        )
        copy_key = (
            f"users/{user.id}/conversations/{conversation.id}"
            f"/tool_artifacts/{artifact.id}/report.txt"
        )
        await fake.put_object(copy_key, b"sandbox-copy", "text/plain")
        upload = await Artifact.create(
            user_id=user.id,
            key=fake.build_user_key(user.id, "png", prefix="artifacts"),
            filename="pic.png",
            content_type="image/png",
            size_bytes=3,
            source=ArtifactSource(type="upload"),
        )
        await fake.put_object(upload.key, b"png", "image/png")
        orphan = await _legacy(fake, user.id, None, filename="loose.txt", body=b"loose")

        await migrate_conversation_artifacts(apply=True)

        moved = await Artifact.find_by_id(artifact.id)
        new_key = (
            f"users/{user.id}/conversations/{conversation.id}"
            f"/artifacts/{artifact.id}/report.txt"
        )
        assert moved.key == new_key
        assert fake.objects[new_key]["body"] == b"sandbox-copy"
        assert artifact.key not in fake.objects
        assert copy_key not in fake.objects
        plain_key = (
            f"users/{user.id}/conversations/{conversation.id}"
            f"/artifacts/{plain.id}/plain.txt"
        )
        assert (await Artifact.find_by_id(plain.id)).key == plain_key
        assert fake.objects[plain_key]["body"] == b"only-user"
        assert plain.key not in fake.objects
        assert (await Artifact.find_by_id(upload.id)).key == upload.key
        assert upload.key in fake.objects
        assert (await Artifact.find_by_id(orphan.id)).key == orphan.key
        assert orphan.key in fake.objects
    finally:
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([conversation, user])


@pytest.mark.asyncio
async def test_a_failed_save_leaves_the_old_key_serving(monkeypatch):
    fake = _use_fake_storage(monkeypatch)
    user, _token = await create_test_user_and_token()
    conversation = await Conversation.create(user_id=user.id, name="move")
    try:
        artifact = await _legacy(fake, user.id, conversation.id, body=b"kept")

        async def _boom(self):
            raise RuntimeError("save failed")

        monkeypatch.setattr(Artifact, "save", _boom)

        await migrate_conversation_artifacts(apply=True)

        stored = await Artifact.find_by_id(artifact.id)
        assert stored.key == artifact.key
        assert fake.objects[artifact.key]["body"] == b"kept"
        new_key = (
            f"users/{user.id}/conversations/{conversation.id}"
            f"/artifacts/{artifact.id}/report.txt"
        )
        assert new_key not in fake.objects
    finally:
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([conversation, user])
