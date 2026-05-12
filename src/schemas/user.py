from datetime import datetime

from pydantic import BaseModel


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
