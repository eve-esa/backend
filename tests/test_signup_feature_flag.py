"""Self-signup is gated on the server, not only in the UI.

The frontend declines to register the /signup route when the feature is off, but
that decides what is rendered, not who may register: the endpoint answers
whatever a client sends it. Before FEATURE_SELF_SIGNUP existed, hiding the link
was the only "protection" there was, and anyone who knew the URL could create an
account on any environment.
"""

import pytest

from tests.utils.cleaner import cleanup_models


_SIGNUP_PAYLOAD = {
    "email": "feature-flag-probe@picampus-school.com",
    "password": "a-sufficiently-long-password",
    "first_name": "Feature",
    "last_name": "Probe",
}


@pytest.mark.asyncio
async def test_signup_is_not_found_when_feature_is_off(async_client, monkeypatch):
    """404, not 403: a probe must not be able to tell a disabled feature from an absent one."""
    monkeypatch.setattr("src.routers.auth.FEATURE_SELF_SIGNUP", False)

    resp = await async_client.post("/signup", json=_SIGNUP_PAYLOAD)

    assert resp.status_code == 404
    assert resp.json()["detail"] == "Not Found"


@pytest.mark.asyncio
async def test_signup_rejects_before_creating_anything(async_client, monkeypatch):
    """The guard runs before create_user, so a refused signup leaves no trace."""
    monkeypatch.setattr("src.routers.auth.FEATURE_SELF_SIGNUP", False)
    called = False

    async def _fail(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("create_user must not run when self-signup is off")

    monkeypatch.setattr("src.routers.auth.create_user", _fail)

    resp = await async_client.post("/signup", json=_SIGNUP_PAYLOAD)

    assert resp.status_code == 404
    assert called is False


@pytest.mark.asyncio
async def test_signup_works_when_feature_is_on(async_client, monkeypatch):
    monkeypatch.setattr("src.routers.auth.FEATURE_SELF_SIGNUP", True)
    monkeypatch.setattr(
        "src.routers.auth.email_service.send_email", lambda **kwargs: None
    )

    resp = await async_client.post("/signup", json=_SIGNUP_PAYLOAD)

    try:
        assert resp.status_code == 200, resp.text
        assert resp.json()["email"] == _SIGNUP_PAYLOAD["email"]
    finally:
        from src.database.models.user import User

        user = await User.find_one({"email": _SIGNUP_PAYLOAD["email"]})
        if user:
            await cleanup_models([user])
