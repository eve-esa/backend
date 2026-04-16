import logging
from src.schemas.user import UpdateUserRequest, TokenUsageResponse
from src.database.models.user import User
from fastapi import APIRouter, Depends
from src.middlewares.auth import get_current_user
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
