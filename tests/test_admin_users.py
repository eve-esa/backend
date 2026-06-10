import pytest
from tests.utils.cleaner import cleanup_models

from src.database.models.user import User

ADMIN_HEADERS = {"X-Admin-Api-Key": "test-admin-key"}


@pytest.fixture
def admin_api_key(monkeypatch):
    monkeypatch.setattr("src.middlewares.admin.ADMIN_API_KEY", "test-admin-key")


@pytest.mark.asyncio
async def test_admin_create_user_success(async_client, admin_api_key):
    email = "admin-created@example.com"
    response = await async_client.post(
        "/admin/users",
        json={"email": email},
        headers=ADMIN_HEADERS,
    )

    assert response.status_code == 201
    body = response.json()
    assert body["email"] == email
    assert body["is_active"] is True
    assert body["rate_limit_group"] == "eve_free"
    assert len(body["password"]) == 20

    user = await User.find_one({"email": email})
    try:
        assert user is not None
        assert user.is_active is True
        assert user.activation_code is None
    finally:
        if user:
            await cleanup_models([user])


@pytest.mark.asyncio
async def test_admin_create_user_with_custom_password(async_client, admin_api_key):
    email = "admin-custom-pass@example.com"
    response = await async_client.post(
        "/admin/users",
        json={"email": email, "password": "custom-secret"},
        headers=ADMIN_HEADERS,
    )

    assert response.status_code == 201
    assert response.json()["password"] == "custom-secret"

    user = await User.find_one({"email": email})
    try:
        assert user is not None
    finally:
        if user:
            await cleanup_models([user])


@pytest.mark.asyncio
async def test_admin_create_user_duplicate_email(async_client, admin_api_key):
    email = "admin-duplicate@example.com"
    first = await async_client.post(
        "/admin/users",
        json={"email": email},
        headers=ADMIN_HEADERS,
    )
    assert first.status_code == 201

    second = await async_client.post(
        "/admin/users",
        json={"email": email},
        headers=ADMIN_HEADERS,
    )
    assert second.status_code == 400
    assert "already exists" in second.json()["detail"]

    user = await User.find_one({"email": email})
    try:
        assert user is not None
    finally:
        if user:
            await cleanup_models([user])


@pytest.mark.asyncio
async def test_admin_create_user_forbidden_without_key(async_client, admin_api_key):
    response = await async_client.post(
        "/admin/users",
        json={"email": "no-key@example.com"},
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_admin_create_user_forbidden_wrong_key(async_client, admin_api_key):
    response = await async_client.post(
        "/admin/users",
        json={"email": "wrong-key@example.com"},
        headers={"X-Admin-Api-Key": "wrong-key"},
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_admin_create_user_not_configured(async_client, monkeypatch):
    monkeypatch.setattr("src.middlewares.admin.ADMIN_API_KEY", "")
    response = await async_client.post(
        "/admin/users",
        json={"email": "not-configured@example.com"},
        headers=ADMIN_HEADERS,
    )
    assert response.status_code == 503
