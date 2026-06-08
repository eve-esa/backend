from datetime import datetime, timedelta, timezone

import pytest

from tests.utils.cleaner import cleanup_models
from tests.utils.utils import create_test_user_and_token


@pytest.mark.asyncio
async def test_create_access_token(async_client):
    user, token = await create_test_user_and_token()
    expires_at = datetime.now(timezone.utc) + timedelta(days=30)
    try:
        response = await async_client.post(
            "/access-token",
            json={"expires_at": expires_at.isoformat()},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["access_token"]
        assert body["expires_at"]

        me_response = await async_client.get(
            "/users/me",
            headers={"Authorization": f"Bearer {body['access_token']}"},
        )
        assert me_response.status_code == 200
        assert me_response.json()["id"] == user.id
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_create_access_token_requires_auth(async_client):
    expires_at = datetime.now(timezone.utc) + timedelta(days=30)
    response = await async_client.post(
        "/access-token",
        json={"expires_at": expires_at.isoformat()},
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_create_access_token_rejects_past_expiry(async_client):
    user, token = await create_test_user_and_token()
    try:
        response = await async_client.post(
            "/access-token",
            json={
                "expires_at": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 422
    finally:
        await cleanup_models([user])
