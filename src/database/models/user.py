from datetime import datetime
from typing import Optional, ClassVar
from pydantic import Field, EmailStr, field_validator
from src.database.mongo_model import MongoModel
from src.schemas.rate_limit import RateLimitGroup, normalize_rate_limit_group


class User(MongoModel):
    """Base user model for the application."""

    email: EmailStr = Field(..., description="User's email address")
    password_hash: str = Field(..., description="Hashed password")
    first_name: Optional[str] = Field(default=None, description="User's first name")
    last_name: Optional[str] = Field(default=None, description="User's last name")
    is_active: bool = Field(
        default=False, description="Indicates if the user is active"
    )
    activation_code: Optional[str] = Field(
        default=None, description="6-character activation code for email verification"
    )
    rate_limit_group: RateLimitGroup = Field(
        default=RateLimitGroup.EVE_FREE,
        description="Rate limit group used to select token policy from config",
    )
    rate_limit_tokens_used: int = Field(
        default=0,
        ge=0,
        description="Tokens used in the current rate-limit period",
    )
    rate_limit_period_start: Optional[datetime] = Field(
        default=None, description="Current rate-limit period start timestamp"
    )
    rate_limit_period_end: Optional[datetime] = Field(
        default=None, description="Current rate-limit period end timestamp"
    )

    collection_name: ClassVar[str] = "users"

    @field_validator("rate_limit_group", mode="before")
    @classmethod
    def _normalize_rate_limit_group(cls, value: object) -> RateLimitGroup:
        return normalize_rate_limit_group(value)
