from datetime import datetime, timezone
from typing import ClassVar, Optional

from pydantic import Field

from src.database.mongo_model import MongoModel


class ApiKey(MongoModel):
    """SHA-256 hash API key record."""

    user_id: str = Field(..., description="Owner user ID")
    name: str = Field(..., description="Human-readable label for this key")
    key_hash: str = Field(..., description="SHA-256 hash of the raw token")
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

    @property
    def is_valid(self) -> bool:
        if self.revoked_at is not None:
            return False
        if self.expires_at is not None:
            expires = (
                self.expires_at
                if self.expires_at.tzinfo is not None
                else self.expires_at.replace(tzinfo=timezone.utc)
            )
            if expires <= datetime.now(timezone.utc):
                return False
        return True
