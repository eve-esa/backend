import pytest

from tests.utils.utils import create_test_user_and_token
from tests.utils.cleaner import cleanup_models
from src.database.models.message import Message


# Mock the generate_answer for test speed reasons, remove the mock when actually want to test the answer generation from the LLM
async def mock_generate_answer(request, conversation_id=None):
    return "Test answer", [], False, None, {}


@pytest.mark.asyncio
async def test_message_flow(async_client, monkeypatch):
    """Full flow: create conversation → add message → update feedback."""

    monkeypatch.setattr("src.routers.message.generate_answer", mock_generate_answer)

    user, token = await create_test_user_and_token()
    try:
        create_conv_payload = {"name": "Msg Test Conversation"}
        conv_resp = await async_client.post(
            "/conversations",
            json=create_conv_payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert conv_resp.status_code == 200
        conv_id = conv_resp.json()["id"]

        create_msg_payload = {"query": "Hello"}
        msg_resp = await async_client.post(
            f"/conversations/{conv_id}/messages",
            json=create_msg_payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert msg_resp.status_code == 200
        msg_body = msg_resp.json()
        assert isinstance(msg_body["answer"], str) and msg_body["answer"]
        assert msg_body["conversation_id"] == conv_id
        msg_id = msg_body["id"]

        detail_resp = await async_client.get(
            f"/conversations/{conv_id}", headers={"Authorization": f"Bearer {token}"}
        )
        assert detail_resp.status_code == 200
        assert len(detail_resp.json()["messages"]) == 1

        patch_payload = {"feedback": "positive", "was_copied": True}
        patch_resp = await async_client.patch(
            f"/conversations/{conv_id}/messages/{msg_id}",
            json=patch_payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert patch_resp.status_code == 200
        assert patch_resp.json() == {"message": "Feedback updated successfully"}

        updated = await Message.find_by_id(msg_id)
        assert updated is not None
        assert updated.feedback == patch_payload["feedback"]
        assert updated.was_copied is True

        del_resp = await async_client.delete(
            f"/conversations/{conv_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert del_resp.status_code == 200

        # Message should be gone after conversation deletion
        assert await Message.find_by_id(msg_id) is None
    finally:
        await cleanup_models([user])


# ---------------- Negative cases -----------------


@pytest.mark.asyncio
async def test_create_message_nonexistent_conversation(async_client, monkeypatch):
    """Attempting to create a message for a non-existing conversation should return 404."""

    monkeypatch.setattr("src.routers.message.generate_answer", mock_generate_answer)

    user, token = await create_test_user_and_token()
    try:
        fake_conv_id = "000000000000000000000000"
        payload = {"query": "Hello"}
        resp = await async_client.post(
            f"/conversations/{fake_conv_id}/messages",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Conversation not found"
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_create_message_not_owner(async_client, monkeypatch):
    """A user cannot add messages to someone else's conversation (403)."""

    monkeypatch.setattr("src.routers.message.generate_answer", mock_generate_answer)

    owner, owner_token = await create_test_user_and_token()
    intruder, intr_token = await create_test_user_and_token()
    try:
        conv_resp = await async_client.post(
            "/conversations",
            json={"name": "Owner Conversation"},
            headers={"Authorization": f"Bearer {owner_token}"},
        )
        conv_id = conv_resp.json()["id"]

        payload = {"query": "Hi"}
        resp = await async_client.post(
            f"/conversations/{conv_id}/messages",
            json=payload,
            headers={"Authorization": f"Bearer {intr_token}"},
        )
        assert resp.status_code == 403
        assert (
            resp.json()["detail"]
            == "You are not allowed to add a message to this conversation"
        )

        # cleanup conversation
        await async_client.delete(
            f"/conversations/{conv_id}",
            headers={"Authorization": f"Bearer {owner_token}"},
        )
    finally:
        await cleanup_models([owner, intruder])


@pytest.mark.asyncio
async def test_update_feedback_wrong_conversation(async_client, monkeypatch):
    """Updating feedback with mismatched conversation/message should yield 404."""

    monkeypatch.setattr("src.routers.message.generate_answer", mock_generate_answer)

    user, token = await create_test_user_and_token()
    try:
        # First conversation with a message
        conv1_resp = await async_client.post(
            "/conversations",
            json={"name": "Conv1"},
            headers={"Authorization": f"Bearer {token}"},
        )
        conv1_id = conv1_resp.json()["id"]

        msg_resp = await async_client.post(
            f"/conversations/{conv1_id}/messages",
            json={"query": "hello"},
            headers={"Authorization": f"Bearer {token}"},
        )
        msg_id = msg_resp.json()["id"]

        # Second conversation
        conv2_id = (
            await async_client.post(
                "/conversations",
                json={"name": "Conv2"},
                headers={"Authorization": f"Bearer {token}"},
            )
        ).json()["id"]

        patch = {"feedback": "positive"}
        resp = await async_client.patch(
            f"/conversations/{conv2_id}/messages/{msg_id}",
            json=patch,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Message not found in this conversation"

        # cleanup
        await async_client.delete(
            f"/conversations/{conv1_id}", headers={"Authorization": f"Bearer {token}"}
        )
        await async_client.delete(
            f"/conversations/{conv2_id}", headers={"Authorization": f"Bearer {token}"}
        )
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_update_feedback_not_owner(async_client, monkeypatch):
    """User cannot update feedback in conversation they don't own → 403."""

    monkeypatch.setattr("src.routers.message.generate_answer", mock_generate_answer)

    owner, owner_token = await create_test_user_and_token()
    intruder, intr_token = await create_test_user_and_token()
    try:
        conv_resp = await async_client.post(
            "/conversations",
            json={"name": "Owner Conv"},
            headers={"Authorization": f"Bearer {owner_token}"},
        )
        conv_id = conv_resp.json()["id"]

        msg_id = (
            await async_client.post(
                f"/conversations/{conv_id}/messages",
                json={"query": "hi"},
                headers={"Authorization": f"Bearer {owner_token}"},
            )
        ).json()["id"]

        patch_payload = {"feedback": "negative"}
        resp = await async_client.patch(
            f"/conversations/{conv_id}/messages/{msg_id}",
            json=patch_payload,
            headers={"Authorization": f"Bearer {intr_token}"},
        )
        assert resp.status_code == 403
        assert (
            resp.json()["detail"]
            == "You are not allowed to update feedback for this message"
        )

        # cleanup
        await async_client.delete(
            f"/conversations/{conv_id}",
            headers={"Authorization": f"Bearer {owner_token}"},
        )
    finally:
        await cleanup_models([owner, intruder])
