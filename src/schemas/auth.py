from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str


class RefreshResponse(BaseModel):
    access_token: str


class RefreshRequest(BaseModel):
    refresh_token: str


class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    first_name: str | None = None
    last_name: str | None = None


class SignupResponse(BaseModel):
    id: str
    email: EmailStr
    first_name: str | None = None
    last_name: str | None = None


class ResendActivationRequest(BaseModel):
    email: str


class VerifyRequest(BaseModel):
    email: str
    activation_code: str


class CreateApiKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Label for this API key")
    expires_at: Optional[datetime] = Field(
        default=None,
        description=(
            "Optional expiry in ISO 8601 format. "
            "Recommended: full datetime with timezone, e.g. 2026-07-09T12:00:00Z. "
            "Omit for a non-expiring key."
        ),
        examples=["2026-07-09T12:00:00Z"],
    )

    @field_validator("expires_at")
    @classmethod
    def expires_at_must_be_future(cls, value: Optional[datetime]) -> Optional[datetime]:
        if value is None:
            return value
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        if value <= datetime.now(timezone.utc):
            raise ValueError("expires_at must be in the future")
        return value


class CreateApiKeyResponse(BaseModel):
    id: str
    name: str
    token: str
    expires_at: Optional[datetime] = None
    created_at: datetime


class ApiKeyItem(BaseModel):
    id: str
    name: str
    expires_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    created_at: datetime
