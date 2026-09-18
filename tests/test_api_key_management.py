"""Self-service API key management: create, list, revoke.

Covers the algorithms in ``src/services/api_keys.py`` (clamp, throttle, cap,
cascade revoke) and the request/response contract in ``src/schemas/auth.py``.
The plan file's API contract wins on any conflict with this file.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import pytest
from bson import ObjectId
from fastapi import HTTPException

from src.database.models.api_key import ApiKey
from src.middlewares.auth import AUTH_TYPE_API_KEY, AuthContext, Principal
from src.services import api_keys
from src.services.auth import generate_api_key
from tests.utils.cleaner import cleanup_models
from tests.utils.utils import create_test_user_and_token


async def create_key_row(user_id: str, **kwargs) -> tuple[str, ApiKey]:
    """Insert a key directly, bypassing the service (for legacy-row shapes)."""
    raw_token, key_hash = generate_api_key()
    kwargs.setdefault("name", "direct-row")
    kwargs.setdefault("token_suffix", raw_token[-6:])
    key = await ApiKey.create(user_id=user_id, key_hash=key_hash, **kwargs)
    return raw_token, key


async def create_via_api(client, headers: dict, **body) -> dict:
    resp = await client.post("/users/api-keys", json=body, headers=headers)
    assert resp.status_code == 201, resp.text
    return resp.json()


# ── Request body matrix ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_body_and_empty_body_take_every_default(async_client):
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        no_body_resp = await async_client.post("/users/api-keys", headers=headers)
        assert no_body_resp.status_code == 201
        body = no_body_resp.json()
        assert body["name"].startswith("Key created ")
        assert body["name"].endswith(" UTC")

        empty_body_resp = await async_client.post("/users/api-keys", json={}, headers=headers)
        assert empty_body_resp.status_code == 201
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_name_null_and_blank_both_generate_a_default_name(async_client):
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        for name in (None, "   "):
            body = await create_via_api(async_client, headers, name=name)
            assert body["name"].startswith("Key created ")
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_name_with_control_char_is_422(async_client):
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = await async_client.post(
            "/users/api-keys", json={"name": "foo\nbar"}, headers=headers
        )
        assert resp.status_code == 422
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_name_with_bidi_override_is_422(async_client):
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = await async_client.post(
            "/users/api-keys", json={"name": "exe‮gpj.txt"}, headers=headers
        )
        assert resp.status_code == 422
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_name_over_100_chars_is_422(async_client):
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = await async_client.post(
            "/users/api-keys", json={"name": "a" * 101}, headers=headers
        )
        assert resp.status_code == 422
    finally:
        await cleanup_models([user])


# ── Expiry matrix ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_expires_in_days_thirty(async_client):
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        body = await create_via_api(async_client, headers, expires_in_days=30)
        expires_at = datetime.fromisoformat(body["expires_at"])
        expected = datetime.now(timezone.utc) + timedelta(days=30)
        assert abs((expires_at - expected).total_seconds()) < 60
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_expires_in_days_null_means_never(async_client):
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        body = await create_via_api(async_client, headers, expires_in_days=None)
        assert body["expires_at"] is None
        assert body["status"] == "active"
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [0, -1, 3651, "30", True])
async def test_expires_in_days_invalid_values_are_422(async_client, value):
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = await async_client.post(
            "/users/api-keys", json={"expires_in_days": value}, headers=headers
        )
        assert resp.status_code == 422
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_expires_at_null_means_never(async_client):
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        body = await create_via_api(async_client, headers, expires_at=None)
        assert body["expires_at"] is None
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_expires_at_future_is_accepted(async_client):
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        future = (datetime.now(timezone.utc) + timedelta(days=5)).isoformat()
        body = await create_via_api(async_client, headers, expires_at=future)
        assert body["expires_at"] is not None
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_expires_at_past_is_422(async_client):
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        resp = await async_client.post(
            "/users/api-keys", json={"expires_at": past}, headers=headers
        )
        assert resp.status_code == 422
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_expires_at_over_ten_years_is_422(async_client):
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        far = (datetime.now(timezone.utc) + timedelta(days=3651)).isoformat()
        resp = await async_client.post(
            "/users/api-keys", json={"expires_at": far}, headers=headers
        )
        assert resp.status_code == 422
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_naive_expires_at_is_treated_as_utc(async_client):
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        naive = (datetime.now(timezone.utc) + timedelta(days=5)).replace(tzinfo=None)
        resp = await async_client.post(
            "/users/api-keys",
            json={"expires_at": naive.isoformat()},
            headers=headers,
        )
        assert resp.status_code == 201
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_both_expiry_fields_is_422(async_client):
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = await async_client.post(
            "/users/api-keys",
            json={"expires_in_days": 30, "expires_at": None},
            headers=headers,
        )
        assert resp.status_code == 422
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_omitted_expiry_uses_the_configured_default(async_client, monkeypatch):
    monkeypatch.setattr(api_keys, "API_KEY_DEFAULT_EXPIRES_IN_DAYS", 7)
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        body = await create_via_api(async_client, headers)
        expires_at = datetime.fromisoformat(body["expires_at"])
        expected = datetime.now(timezone.utc) + timedelta(days=7)
        assert abs((expires_at - expected).total_seconds()) < 60
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_omitted_expiry_with_zero_default_means_never(async_client, monkeypatch):
    monkeypatch.setattr(api_keys, "API_KEY_DEFAULT_EXPIRES_IN_DAYS", 0)
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        body = await create_via_api(async_client, headers)
        assert body["expires_at"] is None
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


# ── Provenance and clamp ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_session_created_key_has_no_parent_and_via_oidc(async_client):
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        body = await create_via_api(async_client, headers)
        assert body["created_via"] == "oidc"
        assert body["created_by_key_id"] is None
        assert body["created_by"] is None
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_key_created_key_carries_parent_provenance(async_client):
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        parent = await create_via_api(async_client, headers, name="Parent")
        parent_headers = {"Authorization": f"Bearer {parent['token']}"}

        child = await create_via_api(async_client, parent_headers, name="Child")
        assert child["created_via"] == "api_key"
        assert child["created_by_key_id"] == parent["id"]
        assert child["created_by"]["id"] == parent["id"]
        assert child["created_by"]["name"] == "Parent"
        assert child["created_by"]["token_suffix"] == parent["token_suffix"]
        assert child["created_by"]["status"] == "active"
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_expired_parent_is_still_resolved_on_the_child(async_client):
    """A key does not have to be usable to be shown as a child's provenance."""
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        raw_parent, parent = await create_key_row(
            user.id,
            name="Expiring soon",
            expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
        )
        # A grandchild created via a session but pointing at this parent id,
        # written directly since the parent itself can no longer authenticate.
        raw_child, child = await create_key_row(
            user.id,
            name="Child of expired",
            created_by_key_id=parent.id,
            created_via="api_key",
        )

        list_resp = await async_client.get("/users/api-keys", headers=headers)
        items = {i["id"]: i for i in list_resp.json()}
        assert items[child.id]["created_by"]["status"] == "expired"
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_child_is_clamped_to_a_thirty_day_parent(async_client):
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        parent = await create_via_api(async_client, headers, expires_in_days=10)
        parent_expires = datetime.fromisoformat(parent["expires_at"])
        parent_headers = {"Authorization": f"Bearer {parent['token']}"}

        # Default expiry (90d) must be clamped down to the parent's 10d.
        default_child = await create_via_api(async_client, parent_headers)
        default_expires = datetime.fromisoformat(default_child["expires_at"])
        assert default_expires <= parent_expires + timedelta(seconds=1)

        # Explicit "never" must also be clamped to the parent's expiry.
        never_child = await create_via_api(async_client, parent_headers, expires_in_days=None)
        never_expires = datetime.fromisoformat(never_child["expires_at"])
        assert never_expires <= parent_expires + timedelta(seconds=1)
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


# ── Cap ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cap_refuses_the_fourth_key(async_client, monkeypatch):
    monkeypatch.setattr(api_keys, "API_KEY_MAX_ACTIVE_PER_USER", 3)
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        for _ in range(3):
            await create_via_api(async_client, headers)
        resp = await async_client.post("/users/api-keys", json={}, headers=headers)
        assert resp.status_code == 409
        assert resp.json()["detail"]["code"] == "api_key_limit_reached"
        assert resp.json()["detail"]["limit"] == 3
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_revoked_and_expired_keys_do_not_count_against_the_cap(async_client, monkeypatch):
    monkeypatch.setattr(api_keys, "API_KEY_MAX_ACTIVE_PER_USER", 2)
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        await create_key_row(user.id, revoked_at=datetime.now(timezone.utc))
        await create_key_row(
            user.id, expires_at=datetime.now(timezone.utc) - timedelta(seconds=1)
        )
        # Neither of the two rows above is active, so both slots are free.
        first = await create_via_api(async_client, headers)
        second = await create_via_api(async_client, headers)
        assert first["status"] == "active"
        assert second["status"] == "active"
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_cap_of_zero_refuses_every_create(async_client, monkeypatch):
    monkeypatch.setattr(api_keys, "API_KEY_MAX_ACTIVE_PER_USER", 0)
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = await async_client.post("/users/api-keys", json={}, headers=headers)
        assert resp.status_code == 409
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_concurrent_creates_never_exceed_the_cap(async_client, monkeypatch):
    monkeypatch.setattr(api_keys, "API_KEY_MAX_ACTIVE_PER_USER", 3)
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        responses = await asyncio.gather(
            *(async_client.post("/users/api-keys", json={}, headers=headers) for _ in range(6))
        )
        statuses = [r.status_code for r in responses]
        assert set(statuses) <= {201, 409}
        # Lower bound: starting from 0 active keys, 6 concurrent creates
        # against a cap of 3 must land exactly 3 survivors, not fewer. A
        # check-then-act recount that lets every over-cap inserter delete
        # its own row can spuriously drop all of them to zero.
        assert statuses.count(201) == 3

        active = await ApiKey.count_documents({"user_id": user.id, "revoked_at": None})
        assert active == 3
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_parent_revoked_between_auth_and_insert_leaves_no_row(monkeypatch):
    """The post-insert parent check catches a revoke that races the create.

    Simulated by making the *second* ``ApiKey.find_by_id`` call inside the
    service (the post-insert re-check) report the parent as revoked, without
    touching the real row.
    """
    user, _ = await create_test_user_and_token()
    try:
        _, parent = await create_key_row(user.id, name="Parent")
        auth = AuthContext(
            user=user, principal=Principal(user.id, AUTH_TYPE_API_KEY, parent.id)
        )

        real_find_by_id = ApiKey.find_by_id
        calls = {"n": 0}

        async def _flaky_find_by_id(key_id):
            calls["n"] += 1
            found = await real_find_by_id(key_id)
            if calls["n"] >= 2 and found is not None and found.id == parent.id:
                found.revoked_at = datetime.now(timezone.utc)
            return found

        monkeypatch.setattr(api_keys.ApiKey, "find_by_id", _flaky_find_by_id)

        with pytest.raises(HTTPException) as exc_info:
            await api_keys.create_api_key(None, auth)
        assert exc_info.value.status_code == 401

        remaining = await ApiKey.find_all(
            filter_dict={"user_id": user.id, "created_by_key_id": parent.id}
        )
        assert remaining == []
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


# ── Throttle ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_throttle_429_with_retry_after(async_client, monkeypatch):
    monkeypatch.setattr(api_keys, "API_KEY_CREATE_MAX_PER_HOUR", 2)
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        await create_via_api(async_client, headers)
        await create_via_api(async_client, headers)
        resp = await async_client.post("/users/api-keys", json={}, headers=headers)
        assert resp.status_code == 429
        assert resp.json()["detail"]["code"] == "api_key_create_rate_limited"
        assert "Retry-After" in resp.headers
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


# ── List ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_hides_revoked_by_default_and_shows_expired(async_client):
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        active = await create_via_api(async_client, headers, name="active-one")
        revoke_resp = await async_client.delete(
            f"/users/api-keys/{active['id']}", headers=headers
        )
        assert revoke_resp.status_code == 204
        _, expired = await create_key_row(
            user.id,
            name="expired-one",
            expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
        )

        hidden_resp = await async_client.get("/users/api-keys", headers=headers)
        ids = {i["id"] for i in hidden_resp.json()}
        assert active["id"] not in ids
        assert expired.id in ids
        assert next(i for i in hidden_resp.json() if i["id"] == expired.id)["status"] == "expired"

        shown_resp = await async_client.get(
            "/users/api-keys", params={"include_revoked": "true"}, headers=headers
        )
        shown_ids = {i["id"] for i in shown_resp.json()}
        assert active["id"] in shown_ids
        assert next(i for i in shown_resp.json() if i["id"] == active["id"])["status"] == "revoked"
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_legacy_row_without_new_fields_lists_cleanly(async_client):
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        raw, key_hash = generate_api_key()
        legacy = await ApiKey.create(user_id=user.id, name="pre-B1 key", key_hash=key_hash)
        assert legacy.token_suffix is None
        assert legacy.created_via is None

        resp = await async_client.get("/users/api-keys", headers=headers)
        item = next(i for i in resp.json() if i["id"] == legacy.id)
        assert item["token_suffix"] is None
        assert item["created_via"] is None
        assert item["created_by_key_id"] is None
        assert item["status"] == "active"
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_list_never_leaks_the_token_or_the_hash(async_client):
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        await create_via_api(async_client, headers)
        resp = await async_client.get("/users/api-keys", headers=headers)
        raw_text = resp.text
        assert "key_hash" not in raw_text
        assert '"token"' not in raw_text
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_list_is_isolated_per_user(async_client):
    user_a, token_a = await create_test_user_and_token()
    user_b, token_b = await create_test_user_and_token()
    try:
        await create_via_api(async_client, {"Authorization": f"Bearer {token_a}"}, name="a-key")
        resp = await async_client.get(
            "/users/api-keys", headers={"Authorization": f"Bearer {token_b}"}
        )
        assert resp.json() == []
    finally:
        await ApiKey.delete_many({"user_id": user_a.id})
        await cleanup_models([user_a, user_b])


@pytest.mark.asyncio
async def test_is_current_flags_the_authenticating_key(async_client):
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        key = await create_via_api(async_client, headers)
        key_headers = {"Authorization": f"Bearer {key['token']}"}

        via_session = await async_client.get("/users/api-keys", headers=headers)
        assert all(not i["is_current"] for i in via_session.json())

        via_key = await async_client.get("/users/api-keys", headers=key_headers)
        flagged = {i["id"]: i["is_current"] for i in via_key.json()}
        assert flagged[key["id"]] is True
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_list_timestamps_are_tz_aware(async_client):
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        await create_via_api(async_client, headers)
        resp = await async_client.get("/users/api-keys", headers=headers)
        item = resp.json()[0]
        created_at = datetime.fromisoformat(item["created_at"])
        assert created_at.tzinfo is not None
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


# ── Revoke ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_revoke_own_key_then_it_stops_working(async_client):
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        key = await create_via_api(async_client, headers)
        resp = await async_client.delete(f"/users/api-keys/{key['id']}", headers=headers)
        assert resp.status_code == 204

        me_resp = await async_client.get(
            "/users/me", headers={"Authorization": f"Bearer {key['token']}"}
        )
        assert me_resp.status_code == 401
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_revoke_missing_and_malformed_ids_share_one_404_body(async_client):
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        missing = await async_client.delete(
            f"/users/api-keys/{ObjectId()}", headers=headers
        )
        malformed = await async_client.delete(
            "/users/api-keys/not-an-object-id", headers=headers
        )
        assert missing.status_code == malformed.status_code == 404
        assert missing.json() == malformed.json()
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_second_revoke_is_204_and_keeps_the_original_revoked_at(async_client):
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        key = await create_via_api(async_client, headers)
        first = await async_client.delete(f"/users/api-keys/{key['id']}", headers=headers)
        assert first.status_code == 204
        stored_first = await ApiKey.get_collection().find_one({"_id": ObjectId(key["id"])})

        await asyncio.sleep(0.01)
        second = await async_client.delete(f"/users/api-keys/{key['id']}", headers=headers)
        assert second.status_code == 204
        stored_second = await ApiKey.get_collection().find_one({"_id": ObjectId(key["id"])})
        assert stored_first["revoked_at"] == stored_second["revoked_at"]
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_cascade_revoke_a_to_b_to_c_leaves_d_from_session(async_client):
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        a = await create_via_api(async_client, headers, name="A")
        a_headers = {"Authorization": f"Bearer {a['token']}"}
        b = await create_via_api(async_client, a_headers, name="B")
        b_headers = {"Authorization": f"Bearer {b['token']}"}
        c = await create_via_api(async_client, b_headers, name="C")
        d = await create_via_api(async_client, headers, name="D")  # from the session, sibling of A

        resp = await async_client.delete(f"/users/api-keys/{a['id']}", headers=headers)
        assert resp.status_code == 204

        listing = await async_client.get(
            "/users/api-keys", params={"include_revoked": "true"}, headers=headers
        )
        status_by_id = {i["id"]: i["status"] for i in listing.json()}
        assert status_by_id[a["id"]] == "revoked"
        assert status_by_id[b["id"]] == "revoked"
        assert status_by_id[c["id"]] == "revoked"
        assert status_by_id[d["id"]] == "active"
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_revoking_the_middle_key_does_not_touch_its_parent(async_client):
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        a = await create_via_api(async_client, headers, name="A")
        a_headers = {"Authorization": f"Bearer {a['token']}"}
        b = await create_via_api(async_client, a_headers, name="B")
        b_headers = {"Authorization": f"Bearer {b['token']}"}
        c = await create_via_api(async_client, b_headers, name="C")

        resp = await async_client.delete(f"/users/api-keys/{b['id']}", headers=headers)
        assert resp.status_code == 204

        listing = await async_client.get(
            "/users/api-keys", params={"include_revoked": "true"}, headers=headers
        )
        status_by_id = {i["id"]: i["status"] for i in listing.json()}
        assert status_by_id[a["id"]] == "active"
        assert status_by_id[b["id"]] == "revoked"
        assert status_by_id[c["id"]] == "revoked"
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_another_users_row_with_the_same_parent_id_is_untouched(async_client):
    """created_by_key_id is not namespaced by user; the cascade must filter by user_id too."""
    user_a, token_a = await create_test_user_and_token()
    user_b, token_b = await create_test_user_and_token()
    try:
        a = await create_via_api(
            async_client, {"Authorization": f"Bearer {token_a}"}, name="A"
        )
        # A row belonging to B that happens to carry A's key id as its parent.
        _, foreign_child = await create_key_row(
            user_b.id, name="B's own key", created_by_key_id=a["id"], created_via="api_key"
        )

        resp = await async_client.delete(
            f"/users/api-keys/{a['id']}", headers={"Authorization": f"Bearer {token_a}"}
        )
        assert resp.status_code == 204

        stored = await ApiKey.find_by_id(foreign_child.id)
        assert stored.revoked_at is None
    finally:
        await ApiKey.delete_many({"user_id": user_a.id})
        await ApiKey.delete_many({"user_id": user_b.id})
        await cleanup_models([user_a, user_b])


@pytest.mark.asyncio
async def test_a_key_can_revoke_itself(async_client):
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        key = await create_via_api(async_client, headers)
        key_headers = {"Authorization": f"Bearer {key['token']}"}
        resp = await async_client.delete(f"/users/api-keys/{key['id']}", headers=key_headers)
        assert resp.status_code == 204
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_a_child_key_can_revoke_its_own_parent(async_client):
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        parent = await create_via_api(async_client, headers, name="Parent")
        parent_headers = {"Authorization": f"Bearer {parent['token']}"}
        child = await create_via_api(async_client, parent_headers, name="Child")
        child_headers = {"Authorization": f"Bearer {child['token']}"}

        resp = await async_client.delete(
            f"/users/api-keys/{parent['id']}", headers=child_headers
        )
        assert resp.status_code == 204

        listing = await async_client.get(
            "/users/api-keys", params={"include_revoked": "true"}, headers=headers
        )
        status_by_id = {i["id"]: i["status"] for i in listing.json()}
        assert status_by_id[parent["id"]] == "revoked"
        assert status_by_id[child["id"]] == "revoked"
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_repeated_revoke_sweeps_a_straggler_created_mid_cascade(async_client, monkeypatch):
    """A child whose create landed after the first cascade pass is still caught.

    Simulated the way the design's unit test would: revoke, then insert a
    straggler row pointing at the (already revoked) parent, then revoke again
    and check the straggler is now revoked too, without a third call.
    """
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        parent = await create_via_api(async_client, headers, name="Parent")

        resp = await async_client.delete(f"/users/api-keys/{parent['id']}", headers=headers)
        assert resp.status_code == 204

        _, straggler = await create_key_row(
            user.id,
            name="Straggler",
            created_by_key_id=parent["id"],
            created_via="api_key",
        )
        assert straggler.revoked_at is None

        resp2 = await async_client.delete(f"/users/api-keys/{parent['id']}", headers=headers)
        assert resp2.status_code == 204

        stored = await ApiKey.find_by_id(straggler.id)
        assert stored.revoked_at is not None
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


# ── Terminal (key-only) flow ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_terminal_flow_with_a_key_only(async_client):
    """No browser session anywhere: create with a session once, then do
    everything else (create, list, revoke) with only the ``eve_`` key.
    """
    user, token = await create_test_user_and_token()
    try:
        parent = await create_via_api(
            async_client, {"Authorization": f"Bearer {token}"}, name="terminal parent"
        )
        key_headers = {"Authorization": f"Bearer {parent['token']}"}

        create_resp = await async_client.post("/users/api-keys", json={}, headers=key_headers)
        assert create_resp.status_code == 201
        child = create_resp.json()
        assert child["created_by_key_id"] == parent["id"]

        list_resp = await async_client.get("/users/api-keys", headers=key_headers)
        assert list_resp.status_code == 200

        revoke_resp = await async_client.delete(
            f"/users/api-keys/{child['id']}", headers=key_headers
        )
        assert revoke_resp.status_code == 204
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


# ── Mass assignment ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mass_assignment_fields_in_the_body_are_ignored(async_client):
    user, token = await create_test_user_and_token()
    other, other_token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = await async_client.post(
            "/users/api-keys",
            json={
                "user_id": other.id,
                "created_by_key_id": "not-a-real-id",
                "created_via": "api_key",
                "revoked_at": "2020-01-01T00:00:00Z",
                "id": "ffffffffffffffffffffffff",
                "token": "eve_" + "0" * 64,
                "key_hash": "a" * 64,
                "status": "revoked",
                "is_current": True,
            },
            headers=headers,
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["created_by_key_id"] is None
        assert body["created_via"] == "oidc"
        assert body["status"] == "active"
        assert body["is_current"] is False

        stored = await ApiKey.find_by_id(body["id"])
        assert stored.user_id == user.id
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user, other])


# ── Audit ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_audit_logs_have_no_secret_material(async_client, caplog):
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        with caplog.at_level(logging.INFO, logger="eve.audit"):
            key = await create_via_api(async_client, headers, name="secret-name-not-logged")
            await async_client.delete(f"/users/api-keys/{key['id']}", headers=headers)

        audit_records = [r for r in caplog.records if r.name == "eve.audit"]
        events = [r.getMessage() for r in audit_records]
        assert any("api_key.created" in e for e in events)
        assert any("api_key.revoked" in e for e in events)

        for record in audit_records:
            text = record.getMessage()
            assert "secret-name-not-logged" not in text
            assert key["token"] not in text
            payload = getattr(record, "audit", {})
            assert "name" not in payload
            assert "token" not in payload
            assert "key_hash" not in payload
            assert "token_suffix" not in payload
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])
