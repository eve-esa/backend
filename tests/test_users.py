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
async def test_get_my_token_usage(async_client):
    user, token = await create_test_user_and_token()
    try:
        response = await async_client.get(
            "/users/me/token-usage",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["rate_limit_group"] == user.rate_limit_group.value
        assert body["used_tokens"] == 0
        assert body["unlimited"] is False
        assert body["max_tokens"] == 1000
        assert body["remaining_tokens"] == 1000
        assert body["used_ratio"] == 0.0
        assert body["remaining_ratio"] == 1.0
        assert body["period_start"] is not None
        assert body["period_end"] is not None
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_update_user_names(async_client):
    user, token = await create_test_user_and_token()
    payload = {"first_name": "Patched", "last_name": "User"}
    try:
        response = await async_client.patch(
            "/users", json=payload, headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["first_name"] == payload["first_name"]
        assert body["last_name"] == payload["last_name"]
        assert body["id"] == user.id
    finally:
        await cleanup_models([user])
