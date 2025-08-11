from typing import ClassVar, Optional
from src.constants import DEFAULT_EMBEDDING_MODEL
from pydantic import Field
from src.database.mongo_model import MongoModel


class Collection(MongoModel):
    """Model for storing collection messages with role."""

    # User ID is optional because public collections don't have a user ID
    user_id: Optional[str] = Field(None, description="User ID")

    name: str = Field(..., description="Collection name")
    description: Optional[str] = Field(None, description="Collection description")
    embeddings_model: str = Field(
        default=DEFAULT_EMBEDDING_MODEL,
        min_length=1,
        description="Embedding model to use for the collection",
    )

    collection_name: ClassVar[str] = "collections"
