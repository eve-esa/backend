import pytest
from tests.utils.cleaner import cleanup_models
from tests.utils.utils import create_test_user_and_token


@pytest.mark.asyncio
async def test_get_current_user(async_client):
    user, token = await create_test_user_and_token()
    try:
        response = await async_client.get(
            "/users/me", headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["id"] == user.id
        assert body["email"] == user.email
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_update_user_names(async_client):
    user, token = await create_test_user_and_token()
    payload = {"first_name": "Patched", "last_name": "User"}
    try:
        response = await async_client.patch(
            "/users/", json=payload, headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["first_name"] == payload["first_name"]
        assert body["last_name"] == payload["last_name"]
        assert body["id"] == user.id
    finally:
        await cleanup_models([user])
