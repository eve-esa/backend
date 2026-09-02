from datetime import datetime
from typing import Optional, ClassVar
from pydantic import Field, EmailStr, field_validator
from src.database.mongo_model import MongoModel
from src.schemas.rate_limit import RateLimitGroup, normalize_rate_limit_group


class User(MongoModel):
    """Base user model for the application."""

    email: EmailStr = Field(..., description="User's email address")
    # Credentials are the identity provider's business now, so nothing in this
    # application writes these three any more. They stay on the model because
    # MongoModel.save() is a full replace_one of model_dump(): dropping a field
    # here erases it from every stored document on the next save, which fires on
    # every message. The prod migration Lambda still has to verify these hashes,
    # and rollback would be one-way without them. They go in the cleanup PR,
    # after the migration window closes, together with a one-off $unset sweep.
    password_hash: Optional[str] = Field(
        default=None, description="Legacy password hash, retained for migration only"
    )
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
    private_document_count: int = Field(
        default=0,
        ge=0,
        description="Number of private documents owned by the user",
    )

    collection_name: ClassVar[str] = "users"

    @field_validator("rate_limit_group", mode="before")
    @classmethod
    def _normalize_rate_limit_group(cls, value: object) -> RateLimitGroup:
        return normalize_rate_limit_group(value)
