from datetime import datetime, timezone
from typing import ClassVar, Optional

from pydantic import Field

from src.database.mongo_model import MongoModel


class ExternalIdentity(MongoModel):
    """One identity-provider account, bound to one EVE user.

    Its own collection rather than a field on ``users`` for two reasons. A user
    may end up with more than one provider account (a second pool, a migration,
    an enterprise IdP later), and the ``(issuer, subject)`` uniqueness that makes
    concurrent first sign-ins safe has to be a real unique index. DocumentDB
    supports a compound unique index on a top-level pair; it does not support
    one on array elements.

    ``subject`` is the provider's stable id for the account. It is the join key,
    never the email: emails get recycled, subjects do not.
    """

    user_id: str = Field(..., description="EVE user this identity resolves to")
    issuer: str = Field(..., description="Exact ``iss`` claim value of the provider")
    subject: str = Field(..., description="Provider's stable ``sub`` for the account")
    email: Optional[str] = Field(
        default=None,
        description="Lowercased address at link time, for support lookups only",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the identity was first linked",
    )
    last_seen_at: Optional[datetime] = Field(
        default=None, description="Most recent resolution of this identity"
    )

    collection_name: ClassVar[str] = "external_identities"
