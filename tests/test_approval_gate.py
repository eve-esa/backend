"""The sign-up gate: N self-registered accounts get in, the rest wait.

Two seams are under test and nothing else. Provisioning decides once whether a
brand-new account holds a seat, and every authenticated request re-reads that
decision. The second half is where the value is: an account that is pending must
be stopped at every door, REST and both proxies, whether it arrives with a
provider token or with an ``eve_`` API key.

``SIGNUP_AUTO_APPROVE_LIMIT`` is monkeypatched on ``src.services.approval``
rather than in the environment, because config is read once at import.
"""

import asyncio
import uuid

import pytest
from bson import ObjectId

from server import app as _root_app
from src.database.models.api_key import ApiKey
from src.database.models.external_identity import ExternalIdentity
from src.database.models.user import User
from src.services import approval, identity
from src.services.approval import (
    APPROVAL_APPROVED,
    APPROVAL_PENDING,
    next_approval_status,
)
from src.services.auth import generate_api_key
from src.services.identity import resolve_user_id
from tests.utils.cleaner import cleanup_models
from tests.utils.utils import TEST_SUBJECT_PREFIX, create_test_user_and_token

_proxy = _root_app.main_app

ISSUER = "https://idp.test.invalid/realms/eve"


def claims_for(subject: str) -> dict:
    return {"iss": ISSUER, "sub": subject}


def new_subject() -> str:
    return f"{TEST_SUBJECT_PREFIX}gate-{uuid.uuid4().hex}"


def stub_userinfo_by_token(monkeypatch, profiles: dict[str, dict]) -> None:
    """Answer userinfo per token, so several sign-ins can run at once."""

    async def _fetch(token: str) -> dict:
        return profiles[token]

    monkeypatch.setattr(identity, "fetch_userinfo", _fetch)


def set_limit(monkeypatch, limit: int) -> None:
    monkeypatch.setattr(approval, "SIGNUP_AUTO_APPROVE_LIMIT", limit)


async def stored_status(user_id: str):
    doc = await User.get_collection().find_one({"_id": ObjectId(user_id)})
    return doc.get("approval_status") if doc else None


@pytest.fixture(autouse=True)
async def _no_seats_left_behind():
    """Seats are counted across the whole collection, so leftovers would lie.

    Only these tests ever write the field, and the suite runs against a
    dedicated database, so dropping every gated row before and after each test
    is enough to make the count deterministic.
    """
    await User.delete_many({"approval_status": {"$ne": None}})
    try:
        yield
    finally:
        await User.delete_many({"approval_status": {"$ne": None}})


def _enable_openai_proxy(monkeypatch) -> None:
    monkeypatch.setattr(_proxy, "_proxy_enabled", True)
    monkeypatch.setattr(
        "src.routers.openai_proxy.OPENAI_PROXY_UPSTREAM_URL", "http://fake-upstream"
    )
    monkeypatch.setattr("src.routers.openai_proxy.EVE_JSC_BASE_URL", "")
    monkeypatch.setattr("src.routers.openai_proxy.OPENAI_PROXY_API_KEY", "fake-key")


async def make_api_key(user_id: str) -> str:
    raw_token, key_hash = generate_api_key()
    await ApiKey.create(user_id=user_id, name="gate-test", key_hash=key_hash)
    return raw_token


def assert_pending_body(response) -> None:
    assert response.status_code == 403
    assert response.json()["detail"] == {
        "code": "pending_approval",
        "message": "Your account is awaiting approval",
    }


# ── Provisioning: who gets a seat ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_third_signup_is_pending_when_the_limit_is_two(monkeypatch):
    set_limit(monkeypatch, 2)
    subjects = [new_subject() for _ in range(3)]
    profiles = {
        subject: {"email": f"{uuid.uuid4().hex[:10]}@example.com", "email_verified": True}
        for subject in subjects
    }
    stub_userinfo_by_token(monkeypatch, profiles)

    created: list[User] = []
    try:
        statuses = []
        for subject in subjects:
            user_id = await resolve_user_id(claims_for(subject), subject)
            statuses.append(await stored_status(user_id))
            user = await User.find_by_id(user_id)
            if user:
                created.append(user)

        assert statuses == [APPROVAL_APPROVED, APPROVAL_APPROVED, APPROVAL_PENDING]
    finally:
        await ExternalIdentity.delete_many({"subject": {"$in": subjects}})
        await cleanup_models(created)


@pytest.mark.asyncio
async def test_unlimited_never_counts_the_collection(monkeypatch):
    """The zero case must cost nothing: no query, not even a cheap one."""
    set_limit(monkeypatch, 0)

    async def _boom(*args, **kwargs):
        raise AssertionError("count_documents must not run when seats are unlimited")

    monkeypatch.setattr(User, "count_documents", _boom)

    assert await next_approval_status() == APPROVAL_APPROVED

    subject = new_subject()
    email = f"{uuid.uuid4().hex[:10]}@example.com"
    stub_userinfo_by_token(
        monkeypatch, {subject: {"email": email, "email_verified": True}}
    )
    created = None
    try:
        user_id = await resolve_user_id(claims_for(subject), subject)
        assert await stored_status(user_id) == APPROVAL_APPROVED
        created = await User.find_by_id(user_id)
    finally:
        await ExternalIdentity.delete_many({"subject": subject})
        await cleanup_models([created] if created else [])


@pytest.mark.asyncio
async def test_concurrent_first_signins_overshoot_is_bounded(monkeypatch):
    """Six first sign-ins at once, two seats.

    The count is not atomic, so more than two may be approved. What must hold is
    that the overshoot stays bounded by the number of racers and that each
    subject still ends up with exactly one account.
    """
    limit = 2
    racers = 6
    set_limit(monkeypatch, limit)

    subjects = [new_subject() for _ in range(racers)]
    profiles = {
        subject: {"email": f"{uuid.uuid4().hex[:10]}@example.com", "email_verified": True}
        for subject in subjects
    }
    stub_userinfo_by_token(monkeypatch, profiles)

    created: list[User] = []
    try:
        user_ids = await asyncio.gather(
            *(resolve_user_id(claims_for(subject), subject) for subject in subjects)
        )

        assert len(set(user_ids)) == racers
        for subject in subjects:
            assert await ExternalIdentity.count_documents({"subject": subject}) == 1

        statuses = [await stored_status(user_id) for user_id in user_ids]
        approved = statuses.count(APPROVAL_APPROVED)
        assert approved >= limit
        assert approved <= limit + racers
        assert set(statuses) <= {APPROVAL_APPROVED, APPROVAL_PENDING}

        for user_id in user_ids:
            user = await User.find_by_id(user_id)
            if user:
                created.append(user)
    finally:
        await ExternalIdentity.delete_many({"subject": {"$in": subjects}})
        await cleanup_models(created)


# ── Enforcement: every door ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pending_user_is_refused_on_rest_routes(async_client):
    user, token = await create_test_user_and_token(approval_status=APPROVAL_PENDING)
    try:
        headers = {"Authorization": f"Bearer {token}"}
        assert_pending_body(await async_client.get("/users/me", headers=headers))
        assert_pending_body(await async_client.get("/conversations", headers=headers))
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_pending_user_is_refused_on_the_openai_proxy(async_client, monkeypatch):
    _enable_openai_proxy(monkeypatch)
    user, token = await create_test_user_and_token(approval_status=APPROVAL_PENDING)
    try:
        resp = await async_client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert_pending_body(resp)
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_pending_user_is_refused_on_the_mcp_proxy(async_client):
    user, token = await create_test_user_and_token(approval_status=APPROVAL_PENDING)
    try:
        resp = await async_client.post(
            "/mcp/any-server",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert_pending_body(resp)
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_approving_in_mongo_takes_effect_on_the_next_request(async_client):
    """No cache to clear: the check is a projected read on every request."""
    user, token = await create_test_user_and_token(approval_status=APPROVAL_PENDING)
    headers = {"Authorization": f"Bearer {token}"}
    try:
        assert_pending_body(await async_client.get("/users/me", headers=headers))

        await User.get_collection().update_one(
            {"_id": ObjectId(user.id)},
            {"$set": {"approval_status": APPROVAL_APPROVED}},
        )

        resp = await async_client.get("/users/me", headers=headers)
        assert resp.status_code == 200
        assert resp.json()["id"] == user.id
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_document_without_the_field_is_let_through(async_client):
    """Every legacy row looks like this, and none of them may be locked out."""
    user, token = await create_test_user_and_token()
    try:
        await User.get_collection().update_one(
            {"_id": ObjectId(user.id)}, {"$unset": {"approval_status": ""}}
        )
        stored = await User.get_collection().find_one({"_id": ObjectId(user.id)})
        assert "approval_status" not in stored

        resp = await async_client.get(
            "/users/me", headers={"Authorization": f"Bearer {token}"}
        )
        assert resp.status_code == 200
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_unknown_status_string_is_allowed(async_client):
    """Only "pending" blocks. Anything else is somebody else's vocabulary."""
    user, token = await create_test_user_and_token(approval_status="grandfathered")
    try:
        resp = await async_client.get(
            "/users/me", headers={"Authorization": f"Bearer {token}"}
        )
        assert resp.status_code == 200
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_api_key_of_a_pending_user_is_refused(async_client, monkeypatch):
    """The machine door too: an issued key must not outlive its owner's block."""
    _enable_openai_proxy(monkeypatch)
    user, _ = await create_test_user_and_token(approval_status=APPROVAL_PENDING)
    try:
        raw_key = await make_api_key(user.id)
        headers = {"Authorization": f"Bearer {raw_key}"}

        assert_pending_body(await async_client.get("/users/me", headers=headers))
        assert_pending_body(
            await async_client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
                headers=headers,
            )
        )
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


# ── The mail that goes with the wait ──────────────────────────────────────────
# Sign-in must not depend on mail, so the notifier is scheduled rather than
# awaited. That makes "was it scheduled" and "does a failure leak" the two
# things worth pinning.


async def drain_pending_mail() -> None:
    """Let the fire-and-forget sends finish before asserting on them."""
    while identity._pending_mail_tasks:
        await asyncio.gather(
            *list(identity._pending_mail_tasks), return_exceptions=True
        )


@pytest.mark.asyncio
async def test_a_pending_signup_is_told_it_is_waiting(monkeypatch):
    set_limit(monkeypatch, 1)
    notified: list[tuple[str, str]] = []

    async def _notify(user_id: str, email: str) -> None:
        notified.append((user_id, email))

    monkeypatch.setattr(identity, "notify_account_pending", _notify)

    subjects = [new_subject() for _ in range(2)]
    profiles = {
        subject: {"email": f"{uuid.uuid4().hex[:10]}@example.com", "email_verified": True}
        for subject in subjects
    }
    stub_userinfo_by_token(monkeypatch, profiles)

    created: list[User] = []
    try:
        user_ids = [
            await resolve_user_id(claims_for(subject), subject) for subject in subjects
        ]
        await drain_pending_mail()

        # The first sign-up took the only seat, so only the second is told to wait.
        assert notified == [(user_ids[1], profiles[subjects[1]]["email"])]

        for user_id in user_ids:
            user = await User.find_by_id(user_id)
            if user:
                created.append(user)
    finally:
        await ExternalIdentity.delete_many({"subject": {"$in": subjects}})
        await cleanup_models(created)


@pytest.mark.asyncio
async def test_a_broken_mailer_never_breaks_a_sign_in(monkeypatch, caplog):
    """The account is the outcome; the mail is a courtesy that may fail."""
    set_limit(monkeypatch, 1)
    seat_taker = await User.create(
        email=f"{uuid.uuid4().hex[:10]}@example.com", approval_status=APPROVAL_APPROVED
    )

    async def _explode(user_id: str, email: str) -> None:
        raise RuntimeError("relay refused the message")

    monkeypatch.setattr(identity, "notify_account_pending", _explode)

    subject = new_subject()
    email = f"{uuid.uuid4().hex[:10]}@example.com"
    stub_userinfo_by_token(
        monkeypatch, {subject: {"email": email, "email_verified": True}}
    )

    created = None
    try:
        with caplog.at_level("ERROR", logger=identity.logger.name):
            user_id = await resolve_user_id(claims_for(subject), subject)
            await drain_pending_mail()

        assert await stored_status(user_id) == APPROVAL_PENDING
        created = await User.find_by_id(user_id)
        assert created is not None

        logged = "\n".join(record.getMessage() for record in caplog.records)
        assert "signup_pending_mail_failed" in logged
    finally:
        await ExternalIdentity.delete_many({"subject": subject})
        await cleanup_models(([created] if created else []) + [seat_taker])
