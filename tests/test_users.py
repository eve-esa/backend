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


async def _stored(user_id: str) -> dict:
    from bson import ObjectId

    from src.database.models.user import User

    return await User.get_collection().find_one({"_id": ObjectId(user_id)})


@pytest.mark.asyncio
async def test_update_country_and_institution_round_trip(async_client):
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = await async_client.patch(
            "/users",
            json={
                "first_name": "Astro",
                "last_name": "User",
                "country": "  Italy  ",
                "institution": " European Space Agency ",
            },
            headers=headers,
        )
        assert response.status_code == 200
        body = response.json()
        assert body["country"] == "Italy"
        assert body["institution"] == "European Space Agency"

        me = (await async_client.get("/users/me", headers=headers)).json()
        assert me["country"] == "Italy"
        assert me["institution"] == "European Space Agency"

        stored = await _stored(user.id)
        assert stored["country"] == "Italy"
        assert stored["institution"] == "European Space Agency"
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_update_names_only_keeps_country_and_institution(async_client):
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        await async_client.patch(
            "/users",
            json={"first_name": "A", "last_name": "B", "country": "Italy", "institution": "ESA"},
            headers=headers,
        )
        response = await async_client.patch(
            "/users", json={"first_name": "C", "last_name": "D"}, headers=headers
        )
        assert response.status_code == 200
        body = response.json()
        assert body["first_name"] == "C"
        assert body["country"] == "Italy"
        assert body["institution"] == "ESA"

        stored = await _stored(user.id)
        assert stored["country"] == "Italy"
        assert stored["institution"] == "ESA"
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
@pytest.mark.parametrize("cleared", ["", "   ", None])
async def test_empty_or_null_clears_country_and_institution(async_client, cleared):
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        await async_client.patch(
            "/users",
            json={"first_name": "A", "last_name": "B", "country": "Italy", "institution": "ESA"},
            headers=headers,
        )
        response = await async_client.patch(
            "/users",
            json={"first_name": "A", "last_name": "B", "country": cleared, "institution": cleared},
            headers=headers,
        )
        assert response.status_code == 200
        body = response.json()
        assert body["country"] is None
        assert body["institution"] is None

        stored = await _stored(user.id)
        assert stored["country"] is None
        assert stored["institution"] is None
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
@pytest.mark.parametrize("field,limit", [("country", 100), ("institution", 200)])
async def test_country_and_institution_length_limits(async_client, field, limit):
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    names = {"first_name": "A", "last_name": "B"}
    try:
        too_long = await async_client.patch(
            "/users", json={**names, field: "x" * (limit + 1)}, headers=headers
        )
        assert too_long.status_code == 422
        assert (await _stored(user.id)).get(field) is None

        # The limit applies after stripping: padding does not count.
        at_limit = await async_client.patch(
            "/users", json={**names, field: "  " + "x" * limit + "  "}, headers=headers
        )
        assert at_limit.status_code == 200
        assert at_limit.json()[field] == "x" * limit
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_legacy_user_without_profile_fields_reads_null(async_client):
    from bson import ObjectId

    from src.database.models.user import User

    user, token = await create_test_user_and_token()
    try:
        await User.get_collection().update_one(
            {"_id": ObjectId(user.id)}, {"$unset": {"country": "", "institution": ""}}
        )
        stored = await _stored(user.id)
        assert "country" not in stored
        assert "institution" not in stored

        response = await async_client.get(
            "/users/me", headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        body = response.json()
        assert "country" in body and body["country"] is None
        assert "institution" in body and body["institution"] is None
    finally:
        await cleanup_models([user])
