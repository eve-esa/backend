"""``GET /users/me`` reads the caller from the primary.

On a replica set with ``readPreference=secondaryPreferred`` the auth dependency
loads the user from a secondary, which can lag behind a ``PATCH /users`` sent a
moment earlier. These tests stand in for that lag: the dependency's row is the
one in the test database, the "primary" is a stub that returns a newer copy.
"""

import logging

import pytest

from src.database.models.user import User
from tests.utils.cleaner import cleanup_models
from tests.utils.utils import create_test_user_and_token


def _stub_primary(monkeypatch, result=None, exc=None):
    calls = []

    async def _primary(document_id):
        calls.append(document_id)
        if exc is not None:
            raise exc
        return result

    monkeypatch.setattr(User, "find_by_id_on_primary", classmethod(lambda cls, i: _primary(i)))
    return calls


@pytest.mark.asyncio
async def test_me_answers_the_primary_row_when_the_auth_row_is_stale(async_client, monkeypatch):
    user, token = await create_test_user_and_token()
    try:
        newer = user.model_copy(update={"first_name": "Saved", "institution": "Primary Institute"})
        calls = _stub_primary(monkeypatch, result=newer)

        response = await async_client.get("/users/me", headers={"Authorization": f"Bearer {token}"})

        assert response.status_code == 200
        body = response.json()
        assert body["first_name"] == "Saved"
        assert body["institution"] == "Primary Institute"
        assert calls == [user.id]
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_me_falls_back_to_the_auth_row_when_the_primary_has_none(async_client, monkeypatch):
    user, token = await create_test_user_and_token()
    try:
        _stub_primary(monkeypatch, result=None)

        response = await async_client.get("/users/me", headers={"Authorization": f"Bearer {token}"})

        assert response.status_code == 200
        body = response.json()
        assert body["id"] == user.id
        assert body["first_name"] == user.first_name
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_me_falls_back_to_the_auth_row_when_the_primary_read_fails(
    async_client, monkeypatch, caplog
):
    """A failed primary read must not turn a profile read into a 5xx.

    The dependency already authenticated the caller and holds a valid row, so
    the worst case is the value from before this fix, with a WARNING.
    """
    user, token = await create_test_user_and_token()
    try:
        _stub_primary(monkeypatch, exc=RuntimeError(f"boom {user.email} {user.first_name}"))

        with caplog.at_level(logging.WARNING, logger="src.routers.user"):
            response = await async_client.get(
                "/users/me", headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        assert response.json()["id"] == user.id
        records = [r for r in caplog.records if r.name == "src.routers.user"]
        assert any(r.levelno == logging.WARNING for r in records)
        for record in records:
            text = record.getMessage()
            assert user.email not in text
            if user.first_name:
                assert user.first_name not in text
            if user.last_name:
                assert user.last_name not in text
    finally:
        await cleanup_models([user])
