import asyncio
import logging

from bson import ObjectId
from fastapi import APIRouter, Body, Depends, HTTPException, Query, Response
from pymongo import ReturnDocument

from src.config import API_KEY_MAX_ACTIVE_PER_USER
from src.database.models.user import User
from src.middlewares.auth import AuthContext, get_auth_context, get_current_user
from src.schemas.auth import ApiKeyItem, CreateApiKeyRequest, CreateApiKeyResponse
from src.schemas.user import TokenUsageResponse, UpdateUserRequest, UserPublic, to_user_public
from src.services import api_keys
from src.services.token_rate_limiter import get_token_usage_summary

router = APIRouter(prefix="/users")
logger = logging.getLogger(__name__)

# Upper bound on the primary read in ``me``. ``MONGO_PARAMS`` sets no
# ``serverSelectionTimeoutMS``, so during a failover the driver would wait its
# 30 s default for a primary. Past this the auth row is served instead.
PRIMARY_READ_TIMEOUT_S = 2.0


@router.get("/me", response_model=UserPublic)
async def me(user: User = Depends(get_current_user)) -> UserPublic:
    """
    Return the authenticated user's profile.

    The dependency may have read the row from a secondary, which can lag behind a
    ``PATCH /users`` sent just before. The profile is re-read on the primary so
    the caller sees what they saved. Only this route pays the primary read, the
    auth dependency stays on the secondary for every other request.

    If the primary has no row, the read fails or it takes longer than
    ``PRIMARY_READ_TIMEOUT_S``, the dependency's row is returned: the caller is
    already authenticated and that row is valid, at worst a moment old, which
    beats failing a profile read.

    Args:
        user (User): Authenticated user injected by dependency.

    Returns:
        The public subset of the current user's profile.
    """
    try:
        fresh = await asyncio.wait_for(
            User.find_by_id_on_primary(user.id), timeout=PRIMARY_READ_TIMEOUT_S
        )
    except Exception as exc:  # TimeoutError included
        # Class name only: the exception text may carry the row or the address.
        logger.warning("users.me primary read failed, serving the auth row: %s", type(exc).__name__)
        fresh = None
    return to_user_public(fresh or user)


@router.get("/me/token-usage", response_model=TokenUsageResponse)
async def get_my_token_usage(user: User = Depends(get_current_user)) -> TokenUsageResponse:
    """Current user's token budget for the active rate-limit window (see ``TokenUsageResponse``)."""
    return TokenUsageResponse.model_validate(await get_token_usage_summary(user))


@router.patch("", response_model=UserPublic)
async def update_user(
    request: UpdateUserRequest, user: User = Depends(get_current_user)
) -> UserPublic:
    """
    Update the authenticated user's profile.

    The two names are always written. ``country`` and ``institution`` are
    written only when the body carries them, so a client that sends just the
    names leaves them untouched; an empty string or null clears one. Only these
    fields are ever written: a ``$set`` of the request body, never a
    full-document replace, so a concurrent token-budget update on the same row
    is never clobbered.

    The answer is the row as stored after the write, returned by the same
    atomic update on the primary, so every field reflects the database and not
    the auth row, which may have been read from a lagging secondary.

    Args:
        request (UpdateUserRequest): New names and, optionally, country and institution.
        user (User): Authenticated user injected by dependency.

    Returns:
        The public subset of the updated user's profile.

    Raises:
        HTTPException: 404 if the row is gone.
    """
    fields = request.model_dump(
        include={"first_name", "last_name"} | (request.model_fields_set & {"country", "institution"})
    )
    stored = await User.get_collection().find_one_and_update(
        {"_id": ObjectId(user.id)},
        {"$set": fields},
        return_document=ReturnDocument.AFTER,
    )
    if stored is None:
        raise HTTPException(status_code=404, detail="User not found")
    return to_user_public(User.from_dict(stored))


@router.post("/api-keys", response_model=CreateApiKeyResponse, status_code=201)
async def create_api_key(
    response: Response,
    request: CreateApiKeyRequest | None = Body(default=None),
    auth: AuthContext = Depends(get_auth_context),
) -> CreateApiKeyResponse:
    """
    Create a new opaque API key for programmatic access.

    The raw token is returned exactly once and never stored. Every field is
    optional; an omitted expiry falls back to ``API_KEY_DEFAULT_EXPIRES_IN_DAYS``.

    Args:
        request (CreateApiKeyRequest | None): Optional name and expiry.
        auth (AuthContext): Authenticated caller, key or session.

    Returns:
        Key metadata and the raw token (shown once).
    """
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    return await api_keys.create_api_key(request, auth)


@router.get("/api-keys", response_model=list[ApiKeyItem])
async def list_api_keys(
    response: Response,
    include_revoked: bool = Query(default=False),
    auth: AuthContext = Depends(get_auth_context),
) -> list[ApiKeyItem]:
    """
    List the authenticated user's API keys, newest first.

    Revoked keys are hidden by default.

    Args:
        include_revoked (bool): Include revoked keys in the response.
        auth (AuthContext): Authenticated caller, key or session.

    Returns:
        Up to 200 key metadata rows (no raw tokens).
    """
    response.headers["X-API-Key-Limit"] = str(API_KEY_MAX_ACTIVE_PER_USER)
    return await api_keys.list_api_keys(auth, include_revoked=include_revoked)


@router.delete("/api-keys/{key_id}", status_code=204)
async def revoke_api_key(
    key_id: str,
    auth: AuthContext = Depends(get_auth_context),
) -> None:
    """
    Revoke an API key immediately, cascading to every key it created.

    Idempotent: revoking an already-revoked key still answers 204. A missing
    id and a foreign id both answer the same 404, so the response is never an
    existence oracle for another user's key.

    Args:
        key_id (str): ID of the key to revoke.
        auth (AuthContext): Authenticated caller, key or session.

    Raises:
        HTTPException: 404 if the key does not exist or belongs to another user.
    """
    await api_keys.revoke_api_key(key_id, auth)
