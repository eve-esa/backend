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
        # UserPublic, not the full row: neither of these ever leaves the backend.
        assert "password_hash" not in body
        assert "activation_code" not in body
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
        if body["unlimited"]:
            assert body["max_tokens"] is None
            assert body["remaining_tokens"] is None
            assert body["used_ratio"] is None
            assert body["remaining_ratio"] is None
        else:
            assert isinstance(body["max_tokens"], int)
            assert body["max_tokens"] > 0
            assert body["remaining_tokens"] == body["max_tokens"]
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
        assert "password_hash" not in body
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_update_user_names_does_not_clobber_other_fields(async_client):
    """PATCH is a $set of the two name fields, not a full-document replace.

    A concurrent write to another field (here: the token counter, standing in
    for what B2's atomic $inc does on the same row) must survive a PATCH that
    ran around the same time.
    """
    from bson import ObjectId

    from src.database.models.user import User

    user, token = await create_test_user_and_token()
    try:
        await User.get_collection().update_one(
            {"_id": ObjectId(user.id)}, {"$inc": {"rate_limit_tokens_used": 42}}
        )

        response = await async_client.patch(
            "/users",
            json={"first_name": "Patched", "last_name": "User"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200

        stored = await User.get_collection().find_one({"_id": ObjectId(user.id)})
        assert stored["rate_limit_tokens_used"] == 42
        assert stored["first_name"] == "Patched"
    finally:
        await cleanup_models([user])
