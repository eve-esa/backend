from typing import ClassVar, Literal, Optional

from pydantic import Field

from src.database.mongo_model import MongoModel

PublicCatalogKind = Literal["qdrant", "wiley", "satcom"]


class PublicCatalogDoc(MongoModel):
    """Named public collection exposed in the picker and retrieval allow-list."""

    name: str = Field(..., description="API id / Qdrant catalog name")
    alias: Optional[str] = Field(default=None, description="Optional UI label")
    description: Optional[str] = Field(default=None, description="Collection description")
    enabled: bool = Field(default=True, description="Whether the collection is listed")
    sort_order: int = Field(default=0, description="Display order (ascending)")
    visible_in_prod: bool = Field(
        default=True, description="Listed when IS_PROD is true"
    )
    visible_in_non_prod: bool = Field(
        default=True, description="Listed on dev/staging"
    )
    kind: PublicCatalogKind = Field(
        default="qdrant",
        description="qdrant (main cluster), wiley (MCP), or satcom (other cluster)",
    )
    applies_eve_filters: bool = Field(
        default=False,
        description="Apply EVE year/keyword client filters on search",
    )

    collection_name: ClassVar[str] = "public_catalog"
