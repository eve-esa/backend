from datetime import datetime, timezone
from typing import ClassVar, Optional

from pydantic import Field

from src.database.mongo_model import MongoModel


class UserCustomModel(MongoModel):
    """User-owned custom model metadata.

    API keys are stored in AWS Secrets Manager; only ``secret_arn`` is persisted here.
    ``base_url`` and ``model_name`` are resolved from the provider catalog at runtime.
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
        description="Deprecated legacy field; catalog-backed models ignore user values",
    )
    secret_arn: str = Field(..., description="AWS Secrets Manager ARN for the API key")

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
