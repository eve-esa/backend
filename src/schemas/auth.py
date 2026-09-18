import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal, Optional

from pydantic import AfterValidator, BaseModel, Field, field_validator, model_validator

from src.config import API_KEY_MAX_LIFETIME_DAYS
from src.database.models.api_key import ApiKeyStatus

# Categories rejected from a key name: control (Cc), format/bidi-override (Cf),
# line/paragraph separators (Zl/Zp). Blocks the RLO trick and stray newlines
# without touching ordinary Unicode letters.
_DISALLOWED_NAME_CATEGORIES = {"Cc", "Cf", "Zl", "Zp"}


def as_utc(value: datetime) -> datetime:
    """Naive datetimes stored or received here are ours, so they are UTC."""
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


UtcDatetime = Annotated[datetime, AfterValidator(as_utc)]


class CreateApiKeyRequest(BaseModel):
    """Every field optional: an empty body, or no body at all, takes the defaults."""

    name: Optional[str] = Field(default=None, max_length=100)
    expires_in_days: Optional[int] = Field(default=None, ge=1, le=API_KEY_MAX_LIFETIME_DAYS, strict=True)
    # Kept for compatibility with clients that send an absolute timestamp.
    expires_at: Optional[datetime] = None

    @field_validator("name", mode="before")
    @classmethod
    def _clean_name(cls, value: object) -> Optional[str]:
        if value is None or not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped:
            return None
        if any(unicodedata.category(ch) in _DISALLOWED_NAME_CATEGORIES for ch in stripped):
            raise ValueError("name must not contain control or bidi-override characters")
        return stripped

    @field_validator("expires_at")
    @classmethod
    def _validate_expires_at(cls, value: Optional[datetime]) -> Optional[datetime]:
        if value is None:
            return None
        value = as_utc(value)
        now = datetime.now(timezone.utc)
        if value <= now:
            raise ValueError("expires_at must be in the future")
        if value > now + timedelta(days=API_KEY_MAX_LIFETIME_DAYS):
            raise ValueError(f"expires_at must be at most {API_KEY_MAX_LIFETIME_DAYS} days from now")
        return value

    @model_validator(mode="after")
    def _one_expiry_field(self) -> "CreateApiKeyRequest":
        fields = self.model_fields_set
        if "expires_at" in fields and "expires_in_days" in fields:
            raise ValueError("set either expires_in_days or expires_at, not both")
        return self

    def resolve_expires_at(self, *, now: datetime, default_days: int) -> Optional[datetime]:
        """Resolve the expiry the create request asked for.

        ``model_fields_set`` is the only way to tell "omitted" from "explicit
        null": both mean "no value" to a plain attribute read, but only the
        first falls back to the default.
        """
        fields = self.model_fields_set
        if "expires_at" in fields:
            return self.expires_at  # already normalised to UTC by the validator, or None (never)
        if "expires_in_days" in fields:
            if self.expires_in_days is None:
                return None
            return now + timedelta(days=self.expires_in_days)
        return now + timedelta(days=default_days) if default_days > 0 else None


class ApiKeyParent(BaseModel):
    """The key that created another key, resolved server side."""

    id: str
    name: str
    token_suffix: Optional[str] = None
    status: ApiKeyStatus


class ApiKeyItem(BaseModel):
    id: str
    name: str
    token_suffix: Optional[str] = None
    status: ApiKeyStatus
    created_at: UtcDatetime
    expires_at: Optional[UtcDatetime] = None
    revoked_at: Optional[UtcDatetime] = None
    last_used_at: Optional[UtcDatetime] = None
    created_via: Optional[Literal["oidc", "api_key"]] = None
    created_by_key_id: Optional[str] = None
    created_by: Optional[ApiKeyParent] = None
    is_current: bool = False


class CreateApiKeyResponse(ApiKeyItem):
    token: str
