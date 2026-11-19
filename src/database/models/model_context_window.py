from datetime import datetime, timezone
from typing import ClassVar

from pydantic import Field

from src.database.mongo_model import MongoModel


class ModelContextWindow(MongoModel):
    """Context window reported for one model on one endpoint.

    The same model name reports a different window on each deployment, so the
    key is the endpoint base URL plus the model name.
    """

    base_url: str = Field(..., description="OpenAI-compatible base URL, no trailing slash")
    model_name: str = Field(..., description="Model id sent to that endpoint")
    context_window: int = Field(..., description="Input window in tokens")
    source: str = Field(..., description="catalog or endpoint")
    fetched_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this window was last confirmed",
    )

    collection_name: ClassVar[str] = "model_context_windows"
