"""The door the back office knocks on after it approves somebody.

Three things matter and the rest is plumbing: an unconfigured secret closes the
endpoint rather than opening it, the endpoint re-reads the approval status
instead of trusting the caller, and a transport failure is reported as such so
the back office knows it may retry.

``INTERNAL_API_SECRET`` is monkeypatched on the router module, because config is
read once at import.
"""

import uuid

import pytest
from bson import ObjectId

from src.database.models.user import User
from src.routers import internal_notifications
from src.services import account_notifications
from src.services.approval import APPROVAL_APPROVED, APPROVAL_PENDING
from tests.utils.cleaner import cleanup_models

SECRET = "internal-secret-for-tests"
ENDPOINT = "/internal/notifications/account-approved"


@pytest.fixture(autouse=True)
def _configured_secret(monkeypatch):
    monkeypatch.setattr(internal_notifications, "INTERNAL_API_SECRET", SECRET)


@pytest.fixture
def sent(monkeypatch):
    """Record every send instead of letting one leave the process."""
    calls = []

    async def _capture(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(account_notifications, "send_mail", _capture)
    return calls


def unique_email() -> str:
    return f"{uuid.uuid4().hex[:10]}@example.com"


async def user_with_status(status) -> User:
    return await User.create(email=unique_email(), approval_status=status)


@pytest.mark.asyncio
async def test_an_approved_user_is_mailed_once(async_client, sent):
    user = await user_with_status(APPROVAL_APPROVED)
    try:
        response = await async_client.post(
            ENDPOINT,
            json={"user_id": user.id},
            headers={"X-Internal-Secret": SECRET},
        )
        assert response.status_code == 200
        assert response.json() == {"sent": True}

        assert len(sent) == 1
        assert sent[0]["to"] == user.email
        assert sent[0]["subject"] == "Your EVE account is ready"
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_a_pending_user_is_a_conflict(async_client, sent):
    """The status is the source of truth, not the caller's say-so.

    A "your account is ready" mail to somebody still in the queue is worse than
    no mail: it invites a sign-in that then hits the wall.
    """
    user = await user_with_status(APPROVAL_PENDING)
    try:
        response = await async_client.post(
            ENDPOINT,
            json={"user_id": user.id},
            headers={"X-Internal-Secret": SECRET},
        )
        assert response.status_code == 409
        assert sent == []
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_unknown_and_malformed_ids_are_not_found(async_client, sent):
    for user_id in (str(ObjectId()), "not-an-object-id"):
        response = await async_client.post(
            ENDPOINT,
            json={"user_id": user_id},
            headers={"X-Internal-Secret": SECRET},
        )
        assert response.status_code == 404
    assert sent == []


@pytest.mark.asyncio
async def test_missing_or_wrong_secret_is_forbidden(async_client, sent):
    user = await user_with_status(APPROVAL_APPROVED)
    try:
        for headers in ({}, {"X-Internal-Secret": "wrong"}):
            response = await async_client.post(
                ENDPOINT, json={"user_id": user.id}, headers=headers
            )
            assert response.status_code == 403
        assert sent == []
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_unconfigured_secret_closes_the_endpoint(
    async_client, sent, monkeypatch
):
    """Nobody may trigger mail because a variable was forgotten."""
    monkeypatch.setattr(internal_notifications, "INTERNAL_API_SECRET", "")
    user = await user_with_status(APPROVAL_APPROVED)
    try:
        response = await async_client.post(
            ENDPOINT,
            json={"user_id": user.id},
            headers={"X-Internal-Secret": SECRET},
        )
        assert response.status_code == 503
        assert sent == []
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_a_transport_failure_is_reported_as_a_bad_gateway(
    async_client, monkeypatch
):
    """502, not 500: the request was fine, the thing downstream was not."""

    async def _fail(**kwargs):
        raise RuntimeError("relay refused the message")

    monkeypatch.setattr(account_notifications, "send_mail", _fail)

    user = await user_with_status(APPROVAL_APPROVED)
    try:
        response = await async_client.post(
            ENDPOINT,
            json={"user_id": user.id},
            headers={"X-Internal-Secret": SECRET},
        )
        assert response.status_code == 502
        assert response.json()["detail"] == "Mail delivery failed"
    finally:
        await cleanup_models([user])
