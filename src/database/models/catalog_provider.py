from typing import ClassVar, List, Optional

from pydantic import BaseModel, Field

from src.database.mongo_model import MongoModel


class CatalogProviderModelEntry(BaseModel):
    """Provider model entry stored inside a catalog provider document."""

    catalog_model_id: str = Field(
        ..., description="Stable catalog identifier used by the API/UI"
    )
    display_name: str = Field(..., description="User-facing model label")
    model_name: str = Field(
        ..., description="Provider API model identifier sent to the upstream API"
    )
    context_window: Optional[int] = Field(
        default=None,
        description="Token window when the provider's /models does not report one",
    )
    enabled: bool = Field(default=True, description="Whether the model is selectable")


class CatalogProviderDoc(MongoModel):
    """BYOM provider definition with nested catalog models."""

    catalog_id: str = Field(..., description="Stable provider identifier")
    display_name: str = Field(..., description="User-facing provider label")
    base_url: str = Field(..., description="OpenAI-compatible API base URL")
    models: List[CatalogProviderModelEntry] = Field(default_factory=list)
    enabled: bool = Field(default=True, description="Whether the provider is exposed")
    sort_order: int = Field(default=0, description="Display order (ascending)")

    collection_name: ClassVar[str] = "catalog_providers"
