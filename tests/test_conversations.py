import pytest

from tests.utils.utils import create_test_user_and_token
from tests.utils.cleaner import cleanup_models


@pytest.mark.asyncio
async def test_list_conversations_empty(async_client):
    user, token = await create_test_user_and_token()
    try:
        response = await async_client.get(
            "/conversations", headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["meta"]["total_count"] == 0
        assert body["data"] == []
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_conversation_crud(async_client):
    user, token = await create_test_user_and_token()
    try:
        create_payload = {"name": "My Conversation"}
        create_response = await async_client.post(
            "/conversations",
            json=create_payload,
            headers={"Authorization": f"Bearer {token}"},
        )

        assert create_response.status_code == 200
        created = create_response.json()
        assert created["name"] == create_payload["name"]
        assert created["user_id"] == user.id
        conversation_id = created["id"]

        list_response = await async_client.get(
            "/conversations", headers={"Authorization": f"Bearer {token}"}
        )

        assert list_response.status_code == 200
        list_body = list_response.json()
        assert list_body["meta"]["total_count"] == 1
        assert list_body["data"][0]["id"] == conversation_id

        detail_response = await async_client.get(
            f"/conversations/{conversation_id}",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert detail_response.status_code == 200
        detail_body = detail_response.json()
        assert detail_body["id"] == conversation_id
        assert detail_body["messages"] == []

        patch_payload = {"name": "Updated Name"}
        patch_response = await async_client.patch(
            f"/conversations/{conversation_id}",
            json=patch_payload,
            headers={"Authorization": f"Bearer {token}"},
        )

        assert patch_response.status_code == 200
        patched = patch_response.json()
        assert patched["name"] == patch_payload["name"]

        delete_response = await async_client.delete(
            f"/conversations/{conversation_id}",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert delete_response.status_code == 200

        list_after_delete = await async_client.get(
            "/conversations", headers={"Authorization": f"Bearer {token}"}
        )

        assert list_after_delete.status_code == 200
        after_body = list_after_delete.json()
        assert after_body["meta"]["total_count"] == 0
    finally:
        await cleanup_models([user])
