from datetime import datetime, timedelta, timezone

import pytest

from src.database.models.api_key import ApiKey
from src.services.auth import generate_api_key
from tests.utils.cleaner import cleanup_models
from tests.utils.utils import create_test_user_and_token


@pytest.mark.asyncio
async def test_create_api_key(async_client):
    user, token = await create_test_user_and_token()
    try:
        response = await async_client.post(
            "/users/api-keys",
            json={"name": "EVA integration"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 201
        body = response.json()
        assert body["id"]
        assert body["name"] == "EVA integration"
        assert body["token"].startswith("eve_")
        assert body["expires_at"] is None
        assert body["created_at"]
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_create_api_key_with_expiry(async_client):
    user, token = await create_test_user_and_token()
    expires_at = datetime.now(timezone.utc) + timedelta(days=30)
    try:
        response = await async_client.post(
            "/users/api-keys",
            json={"name": "Short-lived key", "expires_at": expires_at.isoformat()},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 201
        body = response.json()
        assert body["expires_at"] is not None
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_api_key_authenticates_requests(async_client):
    user, token = await create_test_user_and_token()
    try:
        create_resp = await async_client.post(
            "/users/api-keys",
            json={"name": "test key"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert create_resp.status_code == 201
        raw_token = create_resp.json()["token"]

        me_resp = await async_client.get(
            "/users/me",
            headers={"Authorization": f"Bearer {raw_token}"},
        )
        assert me_resp.status_code == 200
        assert me_resp.json()["id"] == user.id
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_create_api_key_requires_auth(async_client):
    response = await async_client.post(
        "/users/api-keys",
        json={"name": "test"},
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_create_api_key_rejects_past_expiry(async_client):
    user, token = await create_test_user_and_token()
    try:
        response = await async_client.post(
            "/users/api-keys",
            json={
                "name": "bad key",
                "expires_at": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 422
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_list_api_keys(async_client):
    user, token = await create_test_user_and_token()
    try:
        for name in ("key-a", "key-b"):
            await async_client.post(
                "/users/api-keys",
                json={"name": name},
                headers={"Authorization": f"Bearer {token}"},
            )

        list_resp = await async_client.get(
            "/users/api-keys",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert list_resp.status_code == 200
        names = {k["name"] for k in list_resp.json()}
        assert {"key-a", "key-b"}.issubset(names)
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_revoke_api_key(async_client):
    user, token = await create_test_user_and_token()
    try:
        create_resp = await async_client.post(
            "/users/api-keys",
            json={"name": "to-revoke"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert create_resp.status_code == 201
        body = create_resp.json()
        key_id = body["id"]
        raw_token = body["token"]

        revoke_resp = await async_client.delete(
            f"/users/api-keys/{key_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert revoke_resp.status_code == 204

        me_resp = await async_client.get(
            "/users/me",
            headers={"Authorization": f"Bearer {raw_token}"},
        )
        assert me_resp.status_code == 401
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_expired_api_key_is_rejected(async_client):
    """An already-expired key stored directly in the DB must be rejected at auth time.

    The request validator prevents creating expired keys via the API, so this
    test bypasses it and writes the record directly to cover the ``is_valid``
    expiry branch and the naive-UTC comparison path.
    """
    user, token = await create_test_user_and_token()
    raw_token, key_hash = generate_api_key()
    try:
        await ApiKey.create(
            user_id=user.id,
            name="already-expired",
            key_hash=key_hash,
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )

        me_resp = await async_client.get(
            "/users/me",
            headers={"Authorization": f"Bearer {raw_token}"},
        )
        assert me_resp.status_code == 401
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_revoke_api_key_ownership(async_client):
    user_a, token_a = await create_test_user_and_token()
    user_b, token_b = await create_test_user_and_token()
    try:
        create_resp = await async_client.post(
            "/users/api-keys",
            json={"name": "user-a key"},
            headers={"Authorization": f"Bearer {token_a}"},
        )
        key_id = create_resp.json()["id"]

        revoke_resp = await async_client.delete(
            f"/users/api-keys/{key_id}",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert revoke_resp.status_code == 403
    finally:
        await ApiKey.delete_many({"user_id": user_a.id})
        await cleanup_models([user_a, user_b])
