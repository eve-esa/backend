import logging
from datetime import datetime, timezone

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException

from src.database.models.api_key import ApiKey
from src.database.models.user import User
from src.middlewares.auth import get_current_user
from src.schemas.auth import ApiKeyItem, CreateApiKeyRequest, CreateApiKeyResponse
from src.schemas.user import TokenUsageResponse, UpdateUserRequest
from src.services.auth import generate_api_key
from src.services.token_rate_limiter import get_token_usage_summary

router = APIRouter(prefix="/users")
logger = logging.getLogger(__name__)


@router.get("/me", response_model=User)
async def me(user: User = Depends(get_current_user)) -> User:
    """
    Return the authenticated user's profile.

    Args:
        user (User): Authenticated user injected by dependency.

    Returns:
        Current user.
    """
    return user


@router.get("/me/token-usage", response_model=TokenUsageResponse)
async def get_my_token_usage(user: User = Depends(get_current_user)) -> TokenUsageResponse:
    """Current user's token budget for the active rate-limit window (see ``TokenUsageResponse``)."""
    return TokenUsageResponse.model_validate(await get_token_usage_summary(user))


@router.patch("", response_model=User)
async def update_user(
    request: UpdateUserRequest, user: User = Depends(get_current_user)
) -> User:
    """
    Update the authenticated user's profile.

    Args:
        request (UpdateUserRequest): New user attributes to set.
        user (User): Authenticated user injected by dependency.

    Returns:
        Updated user.
    """
    user.first_name = request.first_name
    user.last_name = request.last_name
    await user.save()
    return user


@router.post("/api-keys", response_model=CreateApiKeyResponse, status_code=201)
async def create_api_key(
    request: CreateApiKeyRequest,
    user: User = Depends(get_current_user),
) -> CreateApiKeyResponse:
    """
    Create a new opaque API key for programmatic access.

    The raw token is returned exactly once and never stored.  Store it securely.

    Args:
        request (CreateApiKeyRequest): Key name and optional expiry.
        user (User): Authenticated user injected by dependency.

    Returns:
        Key metadata and the raw token (shown once).
    """
    raw_token, key_hash = generate_api_key()
    api_key = await ApiKey.create(
        user_id=user.id,
        name=request.name,
        key_hash=key_hash,
        expires_at=request.expires_at,
    )
    return CreateApiKeyResponse(
        id=api_key.id,
        name=api_key.name,
        token=raw_token,
        expires_at=api_key.expires_at,
        created_at=api_key.timestamp,
    )


@router.get("/api-keys", response_model=list[ApiKeyItem])
async def list_api_keys(
    user: User = Depends(get_current_user),
) -> list[ApiKeyItem]:
    """
    List all API keys for the authenticated user (including revoked ones).

    Args:
        user (User): Authenticated user injected by dependency.

    Returns:
        List of key metadata (no raw tokens).
    """
    keys = await ApiKey.find_all(filter_dict={"user_id": user.id}, sort=[("timestamp", -1)])
    return [
        ApiKeyItem(
            id=k.id,
            name=k.name,
            expires_at=k.expires_at,
            revoked_at=k.revoked_at,
            last_used_at=k.last_used_at,
            created_at=k.timestamp,
        )
        for k in keys
    ]


@router.delete("/api-keys/{key_id}", status_code=204)
async def revoke_api_key(
    key_id: str,
    user: User = Depends(get_current_user),
) -> None:
    """
    Revoke an API key immediately.

    Args:
        key_id (str): ID of the key to revoke.
        user (User): Authenticated user injected by dependency.

    Raises:
        HTTPException: 404 if key not found; 403 if key belongs to another user.
    """
    try:
        oid = ObjectId(key_id)
    except Exception:
        raise HTTPException(status_code=404, detail="API key not found")
    result = await ApiKey.get_collection().find_one_and_update(
        {"_id": oid, "user_id": user.id},
        {"$set": {"revoked_at": datetime.now(timezone.utc)}},
    )
    if result is None:
        existing = await ApiKey.find_by_id(key_id)
        if not existing:
            raise HTTPException(status_code=404, detail="API key not found")
        raise HTTPException(status_code=403, detail="Not authorized to revoke this key")
