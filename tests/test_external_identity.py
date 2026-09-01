"""Linking a provider account to the EVE user it belongs to.

These tests care about one question: which EVE row does this token resolve to,
and when does it refuse to resolve at all. The rules being pinned down are the
ones where a wrong answer is not an error message but somebody reading somebody
else's conversations.

``userinfo`` is stubbed rather than served over a transport: the network shape
is already covered in test_oidc_validation.py, and what matters here is the
value the provider returns and the decision it produces.
"""

import uuid

import pytest
from bson import ObjectId

from src.database.models.external_identity import ExternalIdentity
from src.database.models.user import User
from src.services import identity
from src.services.identity import (
    normalize_email,
    normalize_email_verified,
    resolve_user_id,
)
from tests.utils.cleaner import cleanup_models

ISSUER = "https://idp.test.invalid/realms/eve"


def claims_for(subject: str, issuer: str = ISSUER) -> dict:
    return {"iss": issuer, "sub": subject}


def stub_userinfo(monkeypatch, profile: dict, calls: list | None = None):
    """Make the provider answer with ``profile`` and record that it was asked."""

    async def _fetch(token: str) -> dict:
        if calls is not None:
            calls.append(token)
        return profile

    monkeypatch.setattr(identity, "fetch_userinfo", _fetch)


def unique_email() -> str:
    return f"{uuid.uuid4().hex[:10]}@example.com"


async def drop_identities(*subjects: str) -> None:
    await ExternalIdentity.delete_many({"subject": {"$in": list(subjects)}})


# ── normalisation ─────────────────────────────────────────────────────────────


@pytest.mark.no_db
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("  Person@Example.COM ", "person@example.com"),
        ("person@example.com", "person@example.com"),
        ("   ", None),
        (None, None),
        (12345, None),
    ],
)
def test_email_normalisation(raw, expected):
    assert normalize_email(raw) == expected


@pytest.mark.no_db
@pytest.mark.parametrize(
    "raw,expected",
    [
        (True, True),
        (False, False),
        ("true", True),
        ("false", False),
        ("TRUE", True),
        # Everything else fails closed. This value decides whether a stranger
        # may adopt an existing account, so an unfamiliar shape is a "no".
        ("yes", False),
        ("1", False),
        (1, False),
        (None, False),
        ({}, False),
    ],
)
def test_email_verified_normalisation(raw, expected):
    assert normalize_email_verified(raw) is expected


# ── linking ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_first_sign_in_links_to_the_existing_account(monkeypatch):
    """The migration case: a pre-OIDC user keeps their id and their history."""
    email = unique_email()
    legacy = await User.create(email=email, first_name="Legacy")
    subject = f"link-{uuid.uuid4().hex}"
    stub_userinfo(monkeypatch, {"email": email, "email_verified": True})
    try:
        resolved = await resolve_user_id(claims_for(subject), "token")
        assert resolved == legacy.id

        row = await ExternalIdentity.find_one({"issuer": ISSUER, "subject": subject})
        assert row is not None
        assert row.user_id == legacy.id
        assert row.email == email
    finally:
        await drop_identities(subject)
        await cleanup_models([legacy])


@pytest.mark.asyncio
async def test_linking_matches_the_account_case_insensitively(monkeypatch):
    """Providers are inconsistent about case; a missed match provisions a twin."""
    email = unique_email()
    legacy = await User.create(email=email)
    subject = f"case-{uuid.uuid4().hex}"
    stub_userinfo(
        monkeypatch, {"email": f"  {email.upper()}  ", "email_verified": "true"}
    )
    try:
        assert await resolve_user_id(claims_for(subject), "token") == legacy.id
    finally:
        await drop_identities(subject)
        await cleanup_models([legacy])


@pytest.mark.asyncio
async def test_unverified_email_provisions_instead_of_linking(monkeypatch):
    """An unverified address is a claim, not a proof, so it earns a new account."""
    email = unique_email()
    legacy = await User.create(email=email, first_name="Legacy")
    subject = f"unverified-{uuid.uuid4().hex}"
    stub_userinfo(monkeypatch, {"email": email, "email_verified": "false"})
    provisioned = None
    try:
        resolved = await resolve_user_id(claims_for(subject), "token")
        assert resolved != legacy.id
        provisioned = await User.find_by_id(resolved)
        assert provisioned is not None
        assert provisioned.email == email
    finally:
        await drop_identities(subject)
        await cleanup_models([u for u in (legacy, provisioned) if u is not None])


@pytest.mark.asyncio
async def test_unrecognised_email_verified_shape_fails_closed(monkeypatch):
    """"yes" is not "true". Guessing here is how a recycled address takes over."""
    email = unique_email()
    legacy = await User.create(email=email)
    subject = f"weird-{uuid.uuid4().hex}"
    stub_userinfo(monkeypatch, {"email": email, "email_verified": "yes"})
    provisioned = None
    try:
        resolved = await resolve_user_id(claims_for(subject), "token")
        assert resolved != legacy.id
        provisioned = await User.find_by_id(resolved)
    finally:
        await drop_identities(subject)
        await cleanup_models([u for u in (legacy, provisioned) if u is not None])


@pytest.mark.asyncio
async def test_account_already_linked_is_never_relinked(monkeypatch):
    """The recycled-address takeover, refused.

    The address is verified and it matches a real account, but that account
    already belongs to another provider subject. Linking would hand the new
    subject somebody else's conversations.
    """
    email = unique_email()
    owner = await User.create(email=email)
    first_subject = f"owner-{uuid.uuid4().hex}"
    await ExternalIdentity.create(
        user_id=owner.id, issuer=ISSUER, subject=first_subject, email=email
    )

    intruder_subject = f"intruder-{uuid.uuid4().hex}"
    stub_userinfo(monkeypatch, {"email": email, "email_verified": True})
    try:
        with pytest.raises(PermissionError):
            await resolve_user_id(claims_for(intruder_subject), "token")
        assert (
            await ExternalIdentity.find_one({"subject": intruder_subject})
        ) is None
    finally:
        await drop_identities(first_subject, intruder_subject)
        await cleanup_models([owner])


@pytest.mark.asyncio
async def test_provisions_a_new_user_when_nothing_matches(monkeypatch):
    email = unique_email()
    subject = f"new-{uuid.uuid4().hex}"
    stub_userinfo(
        monkeypatch,
        {
            "email": email.upper(),
            "email_verified": True,
            "given_name": "Fresh",
            "family_name": "Account",
        },
    )
    created = None
    try:
        resolved = await resolve_user_id(claims_for(subject), "token")
        created = await User.find_by_id(resolved)
        assert created is not None
        assert created.email == email
        assert created.first_name == "Fresh"
        assert created.last_name == "Account"
        # Nothing writes a credential any more.
        assert created.password_hash is None

        row = await ExternalIdentity.find_one({"subject": subject})
        assert row is not None and row.user_id == resolved
    finally:
        await drop_identities(subject)
        await cleanup_models([created] if created else [])


@pytest.mark.asyncio
async def test_provider_without_an_email_is_refused(monkeypatch):
    """No address means no way to tell who this is. Refuse rather than guess."""
    subject = f"anon-{uuid.uuid4().hex}"
    stub_userinfo(monkeypatch, {"email_verified": True})
    try:
        with pytest.raises(PermissionError):
            await resolve_user_id(claims_for(subject), "token")
    finally:
        await drop_identities(subject)


# ── races and caching ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_lost_race_reuses_the_winning_user(monkeypatch):
    """Two workers, one first sign-in.

    The identity row is written before the user row, so the duplicate key is
    what decides the winner. The loser must adopt the winner's user id, not
    create a second account for the same person.
    """
    email = unique_email()
    subject = f"race-{uuid.uuid4().hex}"
    winner_user_id = str(ObjectId())
    await ExternalIdentity.create(
        user_id=winner_user_id, issuer=ISSUER, subject=subject, email=email
    )

    # The loser starts from a cold cache and misses the lookup, exactly as it
    # would if the winner's insert landed between the two.
    original_find_one = ExternalIdentity.find_one
    calls = {"n": 0}

    async def find_one_missing_first(filter_dict):
        calls["n"] += 1
        if calls["n"] == 1:
            return None
        return await original_find_one(filter_dict)

    monkeypatch.setattr(ExternalIdentity, "find_one", find_one_missing_first)
    stub_userinfo(monkeypatch, {"email": email, "email_verified": True})
    try:
        assert await resolve_user_id(claims_for(subject), "token") == winner_user_id
        assert await ExternalIdentity.count_documents({"subject": subject}) == 1
        # No orphan user was left behind by the losing branch.
        assert await User.count_documents({"email": email}) == 0
    finally:
        monkeypatch.setattr(ExternalIdentity, "find_one", original_find_one)
        await drop_identities(subject)


@pytest.mark.asyncio
async def test_existing_identity_resolves_without_asking_the_provider(monkeypatch):
    """The steady state: a known subject costs a lookup, never a userinfo call."""
    email = unique_email()
    user = await User.create(email=email)
    subject = f"known-{uuid.uuid4().hex}"
    await ExternalIdentity.create(
        user_id=user.id, issuer=ISSUER, subject=subject, email=email
    )
    calls: list[str] = []
    stub_userinfo(monkeypatch, {"email": email, "email_verified": True}, calls)
    try:
        assert await resolve_user_id(claims_for(subject), "token") == user.id
        assert calls == []

        row = await ExternalIdentity.find_one({"subject": subject})
        assert row.last_seen_at is not None
    finally:
        await drop_identities(subject)
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_second_resolution_is_served_from_the_cache(monkeypatch):
    email = unique_email()
    user = await User.create(email=email)
    subject = f"cached-{uuid.uuid4().hex}"
    await ExternalIdentity.create(
        user_id=user.id, issuer=ISSUER, subject=subject, email=email
    )
    try:
        assert await resolve_user_id(claims_for(subject), "token") == user.id

        # The row is gone; the answer must still come back, from the cache.
        await ExternalIdentity.delete_many({"subject": subject})
        assert await resolve_user_id(claims_for(subject), "token") == user.id

        # ... and stop coming back once the cache is cleared, which is the only
        # correction mechanism there is: a TTL lapse or a restart.
        identity.clear_identity_cache()
        stub_userinfo(monkeypatch, {"email": email, "email_verified": False})
        resolved = await resolve_user_id(claims_for(subject), "token")
        assert resolved != user.id
        await User.delete_many({"_id": ObjectId(resolved)})
    finally:
        await drop_identities(subject)
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_the_same_subject_from_another_issuer_is_a_different_identity(monkeypatch):
    """The key is the pair. A subject alone says nothing outside its issuer."""
    email = unique_email()
    user = await User.create(email=email)
    subject = f"shared-{uuid.uuid4().hex}"
    await ExternalIdentity.create(
        user_id=user.id, issuer=ISSUER, subject=subject, email=email
    )
    other_issuer = "https://other-idp.test.invalid/realms/eve"
    stub_userinfo(monkeypatch, {"email": unique_email(), "email_verified": True})
    provisioned = None
    try:
        resolved = await resolve_user_id(claims_for(subject, other_issuer), "token")
        assert resolved != user.id
        provisioned = await User.find_by_id(resolved)
    finally:
        await ExternalIdentity.delete_many({"subject": subject})
        await cleanup_models([u for u in (user, provisioned) if u is not None])


@pytest.mark.asyncio
async def test_claims_without_a_subject_are_refused():
    with pytest.raises(PermissionError):
        await resolve_user_id({"iss": ISSUER}, "token")
