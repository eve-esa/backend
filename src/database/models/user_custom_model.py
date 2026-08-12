from datetime import datetime, timezone
from typing import ClassVar, Optional

from pydantic import Field

from src.database.mongo_model import MongoModel


class UserCustomModel(MongoModel):
    """User-owned custom model metadata.

    API keys are envelope-encrypted (AES-256-GCM; see
    ``src.services.custom_model_cipher``) and stored as ``encrypted_key`` on this
    row. ``secret_arn`` is legacy: rows created before envelope encryption keep
    their API key in AWS Secrets Manager until
    ``python -m src.commands.migrate_custom_model_secrets`` re-encrypts them and
    clears the field. ``base_url`` and ``model_name`` are resolved from the
    provider catalog at runtime.
    """

    user_id: str = Field(..., description="Owner user ID")
    display_name: str = Field(..., description="User-facing label")
    provider_id: str = Field(..., description="Catalog provider identifier")
    catalog_model_id: str = Field(..., description="Catalog model identifier")
    model_name: str = Field(
        ..., description="Provider API model identifier (denormalized from catalog)"
    )
    base_url: Optional[str] = Field(
        default=None,
        description="Deprecated; ignored for catalog-backed models",
    )
    encrypted_key: Optional[str] = Field(
        default=None,
        description="Envelope-encrypted API key blob (AES-256-GCM; see custom_model_cipher)",
    )
    secret_arn: Optional[str] = Field(
        default=None,
        description="Legacy AWS Secrets Manager ARN; set only on un-migrated rows",
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp",
    )
    deleted_at: Optional[datetime] = Field(
        default=None, description="Soft delete timestamp"
    )

    collection_name: ClassVar[str] = "user_custom_models"
