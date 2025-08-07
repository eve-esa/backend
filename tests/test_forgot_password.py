import pytest

from tests.utils.utils import create_test_user_and_token
from tests.utils.cleaner import cleanup_models
from src.database.models.forgot_password import ForgotPassword
from src.services.utils import hash_password


# mocks for testing, if we want to test the email service we can remove this
def _mock_send_email(*_args, **_kwargs):
    """No-op email sender used to bypass SMTP calls during tests."""
    return None


@pytest.mark.asyncio
async def test_send_forgot_password_code(async_client, monkeypatch):
    """A POST to /forgot-password/code should create a code when the user exists."""

    monkeypatch.setattr(
        "src.routers.forgot_password.email_service.send_email", _mock_send_email
    )

    user, _token = await create_test_user_and_token()  # token not needed here
    try:
        payload = {"email": user.email}
        resp = await async_client.post("/forgot-password/code", json=payload)

        assert resp.status_code == 200
        assert resp.json() == {"message": "Code sent"}

        # A ForgotPassword doc should now exist
        fp_doc = await ForgotPassword.find_one({"email": user.email})
        assert fp_doc is not None
        assert fp_doc.email == user.email
        assert len(fp_doc.code) == 6
    finally:
        # Clean up created documents
        fp_doc = await ForgotPassword.find_one({"email": user.email})
        models = [user]
        if fp_doc:
            models.append(fp_doc)
        await cleanup_models(models)


@pytest.mark.asyncio
async def test_confirm_new_password_success(async_client, monkeypatch):
    """User can reset their password given a valid code and matching passwords."""

    monkeypatch.setattr(
        "src.routers.forgot_password.email_service.send_email", _mock_send_email
    )

    user, _token = await create_test_user_and_token()
    try:
        # Step 1: request code
        await async_client.post("/forgot-password/code", json={"email": user.email})
        fp_doc = await ForgotPassword.find_one({"email": user.email})
        assert fp_doc is not None

        # Step 2: confirm new password
        new_pwd = "NewPassword123!"
        confirm_payload = {
            "new_password": new_pwd,
            "confirm_password": new_pwd,
            "code": fp_doc.code,
        }
        resp = await async_client.post("/forgot-password/confirm", json=confirm_payload)

        assert resp.status_code == 200
        assert resp.json() == {"message": "Password changed"}

        # ForgotPassword doc should be deleted
        assert await ForgotPassword.find_by_id(fp_doc.id) is None

        # User password hash should have changed
        await user.refresh()  # ensure we have latest data
        assert user.password_hash == hash_password(new_pwd)
    finally:
        await cleanup_models([user])


# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_code_nonexistent_user(async_client, monkeypatch):
    """Sending a code for an unknown email should return 404."""

    monkeypatch.setattr(
        "src.routers.forgot_password.email_service.send_email", _mock_send_email
    )

    resp = await async_client.post(
        "/forgot-password/code", json={"email": "noone@example.com"}
    )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "User not found"


@pytest.mark.asyncio
async def test_confirm_password_invalid_code(async_client, monkeypatch):
    """Confirming with an invalid code returns 404."""

    monkeypatch.setattr(
        "src.routers.forgot_password.email_service.send_email", _mock_send_email
    )

    # Create a user (needed for password update path, though it won't be used)
    user, _token = await create_test_user_and_token()
    try:
        payload = {
            "new_password": "Pwd1!",
            "confirm_password": "Pwd1!",
            "code": "INVALID",
        }
        resp = await async_client.post("/forgot-password/confirm", json=payload)
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Invalid code"
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_confirm_password_mismatch(async_client, monkeypatch):
    """Mismatched new/confirm passwords yields 400."""

    monkeypatch.setattr(
        "src.routers.forgot_password.email_service.send_email", _mock_send_email
    )

    user, _token = await create_test_user_and_token()
    try:
        # Request code first
        await async_client.post("/forgot-password/code", json={"email": user.email})
        fp_doc = await ForgotPassword.find_one({"email": user.email})
        assert fp_doc is not None

        payload = {
            "new_password": "PwdA!",
            "confirm_password": "PwdB!",  # mismatch
            "code": fp_doc.code,
        }
        resp = await async_client.post("/forgot-password/confirm", json=payload)
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Passwords do not match"

    finally:
        fp_doc = await ForgotPassword.find_one({"email": user.email})
        models = [user]
        if fp_doc:
            models.append(fp_doc)
        await cleanup_models(models)
