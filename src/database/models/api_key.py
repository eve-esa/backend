from datetime import datetime, timezone
from typing import ClassVar, Literal, Optional

from pydantic import Field

from src.database.mongo_model import MongoModel

ApiKeyStatus = Literal["active", "expired", "revoked"]


class ApiKey(MongoModel):
    """SHA-256 hash API key record."""

    user_id: str = Field(..., description="Owner user ID")
    name: str = Field(..., description="Human-readable label for this key")
    key_hash: str = Field(..., description="SHA-256 hash of the raw token")
    # None on every row minted before this PR: legacy keys have no suffix to
    # show and no recorded provenance, and that is a fact, not a bug.
    token_suffix: Optional[str] = Field(
        default=None,
        description="Last 6 hex characters of the raw token, for display as eve_...a1b2c3",
    )
    created_by_key_id: Optional[str] = Field(
        default=None,
        description="ID of the API key that created this one; None for a session-created key or a legacy row",
    )
    created_via: Optional[Literal["oidc", "api_key"]] = Field(
        default=None,
        description="How the creator authenticated. None on legacy rows.",
    )
    expires_at: Optional[datetime] = Field(
        default=None, description="Expiry timestamp; None means the key never expires"
    )
    revoked_at: Optional[datetime] = Field(
        default=None, description="Set when the key has been revoked"
    )
    last_used_at: Optional[datetime] = Field(
        default=None, description="Timestamp of the most recent successful authentication"
    )

    collection_name: ClassVar[str] = "api_keys"

    def status_at(self, now: datetime) -> ApiKeyStatus:
        """The key's status as of ``now``. Revoked wins over expired."""
        if self.revoked_at is not None:
            return "revoked"
        if self.expires_at is not None:
            expires = (
                self.expires_at
                if self.expires_at.tzinfo is not None
                else self.expires_at.replace(tzinfo=timezone.utc)
            )
            if expires <= now:
                return "expired"
        return "active"

    @property
    def is_valid(self) -> bool:
        return self.status_at(datetime.now(timezone.utc)) == "active"
