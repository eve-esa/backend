from typing import ClassVar, Optional

from pydantic import Field

from src.database.mongo_model import MongoModel


class CatalogPlatformModelDoc(MongoModel):
    """Platform-hosted LLM exposed in the model picker."""

    catalog_id: str = Field(..., description="Stable platform model identifier")
    llm_type: str = Field(..., description="LLM manager type (main, fallback, ...)")
    display_name: str = Field(..., description="User-facing label")
    description: Optional[str] = Field(default=None, description="Optional description")
    enabled: bool = Field(default=True, description="Whether the model is selectable")
    sort_order: int = Field(default=0, description="Display order (ascending)")

    collection_name: ClassVar[str] = "catalog_platform_models"
