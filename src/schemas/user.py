from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from src.database.models.user import User


class UpdateUserRequest(BaseModel):
    first_name: str
    last_name: str


class UserCreate(BaseModel):
    email: str
    password: str
    first_name: str | None = None
    last_name: str | None = None


class UserResponse(BaseModel):
    id: str
    email: str
    first_name: str | None = None
    last_name: str | None = None


class UserPublic(BaseModel):
    """What ``/users/me`` and ``PATCH /users`` return.

    Deliberately not the full ``User`` model: that one still carries
    ``password_hash`` (read by the Cognito migration Lambda) and
    ``activation_code``, neither of which any API caller needs to see.
    """

    id: str
    email: str
    first_name: str | None = None
    last_name: str | None = None
    approval_status: str | None = None
    rate_limit_group: str
    created_at: datetime


def to_user_public(user: "User") -> UserPublic:
    return UserPublic(
        id=user.id,
        email=user.email,
        first_name=user.first_name,
        last_name=user.last_name,
        approval_status=user.approval_status,
        rate_limit_group=user.rate_limit_group.value,
        created_at=user.timestamp,
    )


class TokenUsageResponse(BaseModel):
    """Token budget for the current billing period (config-driven rate limit)."""

    unlimited: bool
    rate_limit_group: str
    used_tokens: int
    max_tokens: int | None = None
    remaining_tokens: int | None = None
    used_ratio: float | None = None
    remaining_ratio: float | None = None
    period_start: datetime | None = None
    period_end: datetime | None = None
