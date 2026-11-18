"""Files attached to a message reach the agent by sandbox path."""

from types import SimpleNamespace

import pytest
from bson import ObjectId

from src.database.models.artifact import Artifact
from src.services.agents.core.runner import _attached_files, _build_initial_messages
from src.services.mcp.conversation_artifacts import attached_file_paths
from tests.test_conversation_artifacts import _Sandbox, _Store


@pytest.mark.no_db
def test_human_message_lists_the_attached_paths():
    request = SimpleNamespace(query="What is in this image?")

    messages = _build_initial_messages(
        request,
        [],
        attached_files=["/artifacts/a1/scoring.png", "/artifacts/a2/notes.csv"],
    )

    assert messages[-1].content == (
        "What is in this image?\n\n"
        "## Attached files\n"
        "The user attached the following files:\n"
        "/artifacts/a1/scoring.png\n"
        "/artifacts/a2/notes.csv"
    )


@pytest.mark.no_db
def test_human_message_without_attachments_is_the_query():
    request = SimpleNamespace(query="Hello")

    assert _build_initial_messages(request, [])[-1].content == "Hello"


@pytest.mark.no_db
def test_a_key_outside_the_conversation_folder_has_no_path():
    inside = "users/u1/conversations/c1/artifacts/a1/pic.png"
    pending = "users/u1/artifacts/0f3c.png"

    assert attached_file_paths("u1", "c1", [inside, pending]) == [
        ("/artifacts/a1/pic.png", inside)
    ]


async def _seed(user_id: str, conversation_id: str, filename: str) -> str:
    oid = ObjectId()
    key = f"users/{user_id}/conversations/{conversation_id}/artifacts/{oid}/{filename}"
    await Artifact.get_collection().insert_one(
        {
            "_id": oid,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "key": key,
            "filename": filename,
            "content_type": "image/png",
            "size_bytes": 3,
            "source": {"type": "upload"},
        }
    )
    return str(oid)


@pytest.mark.asyncio
async def test_reused_sandbox_gets_the_attached_files(monkeypatch):
    user_id, conversation_id = str(ObjectId()), str(ObjectId())
    mine = await _seed(user_id, conversation_id, "pic.png")
    other = await _seed(str(ObjectId()), conversation_id, "theirs.png")
    try:
        key = f"users/{user_id}/conversations/{conversation_id}/artifacts/{mine}/pic.png"
        monkeypatch.setattr(
            "src.services.mcp.conversation_artifacts.storage_service",
            _Store({key: b"png"}),
        )
        sandbox = _Sandbox()
        lease = SimpleNamespace(backend=sandbox, fresh=False)
        request = SimpleNamespace(artifact_ids=[mine, other])

        paths = await _attached_files(request, user_id, conversation_id, lease)

        assert paths == [f"/artifacts/{mine}/pic.png"]
        assert sandbox.files == [(f"/artifacts/{mine}/pic.png", b"png")]
    finally:
        await Artifact.get_collection().delete_many(
            {"_id": {"$in": [ObjectId(mine), ObjectId(other)]}}
        )


@pytest.mark.asyncio
async def test_fresh_sandbox_is_not_copied_into_again(monkeypatch):
    user_id, conversation_id = str(ObjectId()), str(ObjectId())
    mine = await _seed(user_id, conversation_id, "pic.png")
    try:
        monkeypatch.setattr(
            "src.services.mcp.conversation_artifacts.storage_service", _Store({})
        )
        sandbox = _Sandbox()
        lease = SimpleNamespace(backend=sandbox, fresh=True)
        request = SimpleNamespace(artifact_ids=[mine])

        paths = await _attached_files(request, user_id, conversation_id, lease)

        assert paths == [f"/artifacts/{mine}/pic.png"]
        assert sandbox.files == []
        assert await _attached_files(request, user_id, conversation_id, None) == []
    finally:
        await Artifact.get_collection().delete_many({"_id": ObjectId(mine)})
