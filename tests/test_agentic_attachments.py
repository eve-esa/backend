"""User attachments must survive the agentic generation path.

Regression test: both agentic endpoints created the Message without
resolving ``request.artifact_ids``, so files a user attached in chat were
silently dropped whenever an MCP tool was selected (the classic /messages
endpoint always persisted them).
"""

import pytest

from src.database.models.artifact import Artifact
from tests.utils.cleaner import cleanup_models
from tests.utils.utils import create_test_user_and_token
from tests.test_artifacts import PNG_BYTES, _upload, _use_fake_storage


async def _mock_generate_answer_agentic(request, user_id=None, conversation_id=None):
    # (answer, tool_results, use_rag, latencies, prompts, trace, artifact_ids)
    return "Agentic answer", [], False, {}, {}, [], []


@pytest.mark.asyncio
async def test_agentic_message_persists_user_attachments(async_client, monkeypatch):
    _use_fake_storage(monkeypatch)
    monkeypatch.setattr(
        "src.routers.message.generate_answer_agentic", _mock_generate_answer_agentic
    )

    user, token = await create_test_user_and_token()
    try:
        image_id = (
            await _upload(async_client, token, "pic.png", PNG_BYTES, "image/png")
        ).json()["id"]

        conv_id = (
            await async_client.post(
                "/conversations",
                json={"name": "Agentic attach"},
                headers={"Authorization": f"Bearer {token}"},
            )
        ).json()["id"]

        msg_resp = await async_client.post(
            f"/conversations/{conv_id}/generate-agentic",
            json={"query": "look at this", "artifact_ids": [image_id]},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert msg_resp.status_code == 200

        detail = await async_client.get(
            f"/conversations/{conv_id}", headers={"Authorization": f"Bearer {token}"}
        )
        assert detail.status_code == 200
        messages = detail.json()["messages"]
        assert len(messages) == 1
        attachments = messages[0]["attachments"]
        assert attachments and attachments[0]["image_id"] == image_id
        assert attachments[0]["url"] == f"/artifacts/{image_id}"

        artifact = await Artifact.find_by_id(image_id)
        assert artifact.conversation_id == conv_id

        await async_client.delete(
            f"/conversations/{conv_id}", headers={"Authorization": f"Bearer {token}"}
        )
    finally:
        await cleanup_models([user])
