"""``AuthContext`` / ``get_auth_context``: the one place that resolves who is calling.

``get_current_user`` is a thin wrapper over the same function (see
``src/middlewares/auth.py``), so every failure mode is proven once here rather
than once per dependency.
"""

import pytest
from fastapi import HTTPException

from src.middlewares import auth as auth_module
from src.middlewares.auth import (
    AUTH_TYPE_API_KEY,
    AUTH_TYPE_OIDC,
    get_auth_context,
    get_current_user,
)
from src.services import approval
from src.services.auth import generate_api_key
from src.services.oidc import IdentityProviderUnavailable
from tests.utils.cleaner import cleanup_models
from tests.utils.utils import create_test_user_and_token


class _Credentials:
    def __init__(self, token: str):
        self.credentials = token


async def make_api_key(user_id: str) -> str:
    from src.database.models.api_key import ApiKey

    raw_token, key_hash = generate_api_key()
    await ApiKey.create(user_id=user_id, name="auth-context-test", key_hash=key_hash)
    return raw_token


@pytest.mark.asyncio
async def test_api_key_id_is_set_for_a_key_and_none_for_oidc(async_client):
    user, token = await create_test_user_and_token()
    from src.database.models.api_key import ApiKey

    try:
        raw_key = await make_api_key(user.id)

        oidc_ctx = await get_auth_context(_Credentials(token))
        assert oidc_ctx.principal.api_key_id is None
        assert oidc_ctx.principal.auth_type == AUTH_TYPE_OIDC
        assert oidc_ctx.user.id == user.id

        key_ctx = await get_auth_context(_Credentials(raw_key))
        assert key_ctx.principal.api_key_id is not None
        assert key_ctx.principal.auth_type == AUTH_TYPE_API_KEY
        assert key_ctx.user.id == user.id
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_malformed_bearer_token_is_401_on_both_dependencies():
    for dependency in (get_auth_context, get_current_user):
        with pytest.raises(HTTPException) as exc_info:
            await dependency(_Credentials("not-a-real-token"))
        assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_revoked_api_key_is_401_on_both_dependencies():
    user, _ = await create_test_user_and_token()
    from src.database.models.api_key import ApiKey

    try:
        raw_key = await make_api_key(user.id)
        keys = await ApiKey.find_all(filter_dict={"user_id": user.id})
        await keys[0].delete()  # gone entirely; same PermissionError path as revoked

        for dependency in (get_auth_context, get_current_user):
            with pytest.raises(HTTPException) as exc_info:
                await dependency(_Credentials(raw_key))
            assert exc_info.value.status_code == 401
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_pending_user_gets_the_403_body_on_both_dependencies(monkeypatch):
    monkeypatch.setattr(approval, "SIGNUP_AUTO_APPROVE_LIMIT", 0)
    user, token = await create_test_user_and_token(approval_status="pending")
    try:
        for dependency in (get_auth_context, get_current_user):
            with pytest.raises(HTTPException) as exc_info:
                await dependency(_Credentials(token))
            assert exc_info.value.status_code == 403
            assert exc_info.value.detail == approval.PENDING_APPROVAL_DETAIL
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_identity_provider_unavailable_is_503_on_both_dependencies(monkeypatch):
    user, token = await create_test_user_and_token()

    async def _boom(_token: str) -> dict:
        raise IdentityProviderUnavailable("provider is down")

    monkeypatch.setattr(auth_module, "verify_access_token", _boom)
    try:
        for dependency in (get_auth_context, get_current_user):
            with pytest.raises(HTTPException) as exc_info:
                await dependency(_Credentials(token))
            assert exc_info.value.status_code == 503
    finally:
        await cleanup_models([user])
