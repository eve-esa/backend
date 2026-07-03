from datetime import datetime, timezone
from typing import ClassVar, Optional

from pydantic import Field

from src.database.mongo_model import MongoModel


class UserCustomModel(MongoModel):
    """User-owned custom OpenAI-compatible model metadata.

    API keys are stored in AWS Secrets Manager; only ``secret_arn`` is persisted here.
    """

    user_id: str = Field(..., description="Owner user ID")
    display_name: str = Field(..., description="User-facing label")
    model_name: str = Field(..., description="Provider model identifier")
    base_url: str = Field(..., description="OpenAI-compatible API base URL")
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
