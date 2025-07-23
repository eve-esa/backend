from pydantic import BaseModel, Field, validator
from src.constants import DEFAULT_EMBEDDING_MODEL


class CollectionRequest(BaseModel):
    embeddings_model: str = Field(
        default=DEFAULT_EMBEDDING_MODEL,
        min_length=1,
        description="Embedding model to use for the collection",
    )

    @validator("embeddings_model")
    def validate_embeddings_model(cls, v):
        if not v.strip():
            raise ValueError("Embeddings model cannot be empty or whitespace only")
        return v.strip()
