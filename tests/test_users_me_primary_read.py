"""``GET /users/me`` reads the caller from the primary.

On a replica set with ``readPreference=secondaryPreferred`` the auth dependency
loads the user from a secondary, which can lag behind a ``PATCH /users`` sent a
moment earlier. Most tests stand in for that lag with a stubbed "primary" that
returns a newer copy. The last one keeps the real helper on the test database
and makes the dependency's read the stale one instead.
"""

import asyncio
import logging

import pytest
from bson import ObjectId
from pymongo import ReadPreference

import src.routers.user as user_router

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


@pytest.mark.asyncio
async def test_me_falls_back_to_the_auth_row_when_the_primary_read_is_slow(
    async_client, monkeypatch, caplog
):
    """A primary that does not answer (failover, election) is bounded, not awaited for 30 s."""
    user, token = await create_test_user_and_token()
    try:

        async def _hang(document_id):
            await asyncio.sleep(30)

        monkeypatch.setattr(User, "find_by_id_on_primary", classmethod(lambda cls, i: _hang(i)))
        monkeypatch.setattr(user_router, "PRIMARY_READ_TIMEOUT_S", 0.05)

        with caplog.at_level(logging.WARNING, logger="src.routers.user"):
            response = await asyncio.wait_for(
                async_client.get("/users/me", headers={"Authorization": f"Bearer {token}"}),
                timeout=5,
            )

        assert response.status_code == 200
        assert response.json()["id"] == user.id
        assert any(
            r.levelno == logging.WARNING and "TimeoutError" in r.getMessage()
            for r in caplog.records
            if r.name == "src.routers.user"
        )
    finally:
        await cleanup_models([user])


def test_primary_collection_reads_the_primary():
    assert User.get_primary_collection().read_preference == ReadPreference.PRIMARY


@pytest.mark.asyncio
async def test_patch_returns_the_stored_row_and_me_reads_it_through_the_real_helper(
    async_client, monkeypatch
):
    """No stub on the primary read: the real helper on the test Mongo.

    The auth dependency is made stale (it answers the row as it was before the
    write, like a lagging secondary). A field the PATCH does not touch is
    changed in the database first, so the PATCH answer can only carry it if it
    is the stored row and not the auth row with the written fields set.
    """
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        stale = user.model_copy()
        await User.get_collection().update_one(
            {"_id": ObjectId(user.id)}, {"$set": {"institution": "Stored Institute"}}
        )

        real_find_by_id = User.find_by_id.__func__

        async def _stale_find_by_id(cls, document_id):
            if document_id == user.id:
                return stale.model_copy()
            return await real_find_by_id(cls, document_id)

        monkeypatch.setattr(User, "find_by_id", classmethod(_stale_find_by_id))

        patched = await async_client.patch(
            "/users", json={"first_name": "Written", "last_name": "Row"}, headers=headers
        )
        assert patched.status_code == 200
        body = patched.json()
        assert body["first_name"] == "Written"
        assert body["last_name"] == "Row"
        assert body["institution"] == "Stored Institute"

        me = await async_client.get("/users/me", headers=headers)
        assert me.status_code == 200
        body = me.json()
        assert body["first_name"] == "Written"
        assert body["last_name"] == "Row"
        assert body["institution"] == "Stored Institute"
    finally:
        await cleanup_models([user])
