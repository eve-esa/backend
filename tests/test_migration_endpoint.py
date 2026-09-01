"""The temporary migration endpoint, and the three walls in front of it.

It reads legacy password hashes, so its guards matter more than its feature:
unset configuration must be closed rather than open, a wrong secret must be
rejected by a constant-time comparison, and a correct secret must still not buy
unlimited guesses.

Delete this file with src/routers/migration.py in the cleanup PR.
"""

import uuid

import pytest

from src.database.models.user import User
from src.routers import migration
from src.routers.migration import hash_password
from tests.utils.cleaner import cleanup_models

SECRET = "migration-secret-for-tests"


@pytest.fixture(autouse=True)
def _configured_secret(monkeypatch):
    monkeypatch.setattr(migration, "MIGRATION_SHARED_SECRET", SECRET)
    migration._reset_attempts_for_tests()
    yield
    migration._reset_attempts_for_tests()


def unique_email() -> str:
    return f"{uuid.uuid4().hex[:10]}@example.com"


async def legacy_user(password: str = "legacy-password") -> User:
    return await User.create(
        email=unique_email(),
        password_hash=hash_password(password),
        first_name="Legacy",
        last_name="User",
    )


@pytest.mark.asyncio
async def test_verifies_a_legacy_password(async_client):
    user = await legacy_user()
    try:
        response = await async_client.post(
            "/internal/migration/verify-credentials",
            json={"email": user.email, "password": "legacy-password"},
            headers={"X-Migration-Secret": SECRET},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["email"] == user.email
        assert body["first_name"] == "Legacy"
        # A string, because that is what Cognito's attribute is. A boolean here
        # migrates the account unverified and the link rule then refuses it.
        assert body["email_verified"] == "true"
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_email_lookup_without_a_password_is_allowed(async_client):
    """The ForgotPassword trigger has no password to send at that point."""
    user = await legacy_user()
    try:
        response = await async_client.post(
            "/internal/migration/verify-credentials",
            json={"email": user.email.upper()},
            headers={"X-Migration-Secret": SECRET},
        )
        assert response.status_code == 200
        assert response.json()["email"] == user.email
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_wrong_password_is_rejected(async_client):
    user = await legacy_user()
    try:
        response = await async_client.post(
            "/internal/migration/verify-credentials",
            json={"email": user.email, "password": "not-the-password"},
            headers={"X-Migration-Secret": SECRET},
        )
        assert response.status_code == 401
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_unknown_email_is_not_found(async_client):
    response = await async_client.post(
        "/internal/migration/verify-credentials",
        json={"email": unique_email(), "password": "anything"},
        headers={"X-Migration-Secret": SECRET},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_user_without_a_legacy_hash_cannot_be_migrated(async_client):
    """A provider-provisioned account has no password to verify."""
    user = await User.create(email=unique_email())
    try:
        response = await async_client.post(
            "/internal/migration/verify-credentials",
            json={"email": user.email, "password": "anything"},
            headers={"X-Migration-Secret": SECRET},
        )
        assert response.status_code == 404
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_missing_or_wrong_secret_is_forbidden(async_client):
    user = await legacy_user()
    try:
        for headers in ({}, {"X-Migration-Secret": "wrong"}):
            response = await async_client.post(
                "/internal/migration/verify-credentials",
                json={"email": user.email, "password": "legacy-password"},
                headers=headers,
            )
            assert response.status_code == 403
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_unconfigured_secret_closes_the_endpoint(async_client, monkeypatch):
    """Absent configuration must not mean an open door onto password hashes."""
    monkeypatch.setattr(migration, "MIGRATION_SHARED_SECRET", "")
    user = await legacy_user()
    try:
        response = await async_client.post(
            "/internal/migration/verify-credentials",
            json={"email": user.email, "password": "legacy-password"},
            headers={"X-Migration-Secret": SECRET},
        )
        assert response.status_code == 503
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_attempts_are_capped_per_email(async_client):
    """A leaked secret must not turn the hashes into an unlimited oracle."""
    user = await legacy_user()
    try:
        statuses = []
        for _ in range(migration._MAX_ATTEMPTS_PER_WINDOW + 2):
            response = await async_client.post(
                "/internal/migration/verify-credentials",
                json={"email": user.email, "password": "wrong"},
                headers={"X-Migration-Secret": SECRET},
            )
            statuses.append(response.status_code)

        assert statuses[: migration._MAX_ATTEMPTS_PER_WINDOW] == [401] * (
            migration._MAX_ATTEMPTS_PER_WINDOW
        )
        assert statuses[migration._MAX_ATTEMPTS_PER_WINDOW :] == [429, 429]

        # The cap is per email: another address is unaffected.
        other = await legacy_user()
        try:
            response = await async_client.post(
                "/internal/migration/verify-credentials",
                json={"email": other.email, "password": "legacy-password"},
                headers={"X-Migration-Secret": SECRET},
            )
            assert response.status_code == 200
        finally:
            await cleanup_models([other])
    finally:
        await cleanup_models([user])
