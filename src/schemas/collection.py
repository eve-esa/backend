from pydantic import BaseModel, Field, field_validator
from src.constants import DEFAULT_EMBEDDING_MODEL
from src.database.models.collection import Collection as CollectionModel


class CollectionRequest(BaseModel):
    embeddings_model: str = Field(
        default=DEFAULT_EMBEDDING_MODEL,
        min_length=1,
        description="Embedding model to use for the collection",
    )
    name: str = Field(
        min_length=1,
        description="Name of the collection",
    )
    description: str = Field(
        default="",
        description="Description of the collection",
    )

    @field_validator("embeddings_model")
    @classmethod
    def validate_embeddings_model(cls, v):
        if not v.strip():
            raise ValueError("Embeddings model cannot be empty or whitespace only")
        return v.strip()


class CollectionUpdate(BaseModel):
    name: str


class CollectionOut(CollectionModel):
    documents_count: int
    points_count: int
