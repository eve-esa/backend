from datetime import datetime
from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, StringConstraints, field_validator

if TYPE_CHECKING:
    from src.database.models.user import User


COUNTRY_MAX_LENGTH = 100
INSTITUTION_MAX_LENGTH = 200


class UpdateUserRequest(BaseModel):
    """Body of ``PATCH /users``.

    The names are required, as before. ``country`` and ``institution`` are
    optional profile fields: left out of the body, the stored value is kept;
    sent as an empty (or blank) string or null, the stored value is cleared.
    Both are stripped, and the length limit applies to the stripped value.
    """

    first_name: str
    last_name: str
    country: Annotated[
        str, StringConstraints(strip_whitespace=True, max_length=COUNTRY_MAX_LENGTH)
    ] | None = None
    institution: Annotated[
        str, StringConstraints(strip_whitespace=True, max_length=INSTITUTION_MAX_LENGTH)
    ] | None = None

    @field_validator("country", "institution")
    @classmethod
    def _blank_clears(cls, value: str | None) -> str | None:
        return value or None


class UserCreate(BaseModel):
    email: str
    password: str
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
    country: str | None = None
    institution: str | None = None
    approval_status: str | None = None
    rate_limit_group: str
    created_at: datetime


def to_user_public(user: "User") -> UserPublic:
    return UserPublic(
        id=user.id,
        email=user.email,
        first_name=user.first_name,
        last_name=user.last_name,
        country=user.country,
        institution=user.institution,
        approval_status=user.approval_status,
        rate_limit_group=user.rate_limit_group.value,
        created_at=user.timestamp,
    )


class StorageUsageResponse(BaseModel):
    """Object storage used by the current user, and the cap of their plan.

    ``used_gb`` and ``max_gb`` are GiB (1024^3), rounded to 2 decimals, for
    display. The byte fields are exact. ``unlimited`` is true when the plan
    has no cap. ``used_bytes`` is never negative: a counter below zero, which
    happens before the initial fill, is reported as 0.
    """

    unlimited: bool
    rate_limit_group: str
    used_bytes: int
    max_bytes: int | None = None
    remaining_bytes: int | None = None
    used_ratio: float | None = None
    used_gb: float
    max_gb: float | None = None


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
