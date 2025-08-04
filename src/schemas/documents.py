from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from src.constants import (
    DEFAULT_QUERY,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_K,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)


class RetrieveRequest(BaseModel):
    query: str = Field(
        default=DEFAULT_QUERY, min_length=1, max_length=1000, description="Search query"
    )
    year: List[int] = Field(
        default=[],
        description="A list with two values [start_year, end_year] to filter by publication year.",
    )
    keywords: List[str] = Field(
        default=[],
        description="List of keywords to filter by title.",
    )
    embeddings_model: str = Field(
        default=DEFAULT_EMBEDDING_MODEL,
        min_length=1,
        description="Embedding model to use",
    )
    score_threshold: float = Field(
        default=DEFAULT_SCORE_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold",
    )
    k: int = Field(
        default=DEFAULT_K, ge=1, le=100, description="Number of documents to retrieve"
    )
    get_unique_docs: bool = Field(
        default=True, description="Whether to get unique documents"
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v.strip()


class DeleteRequest(BaseModel):
    embeddings_model: str = Field(
        default=DEFAULT_EMBEDDING_MODEL,
        min_length=1,
        description="Embedding model to use",
    )
    document_list: List[str] = Field(
        default=[], description="List of document IDs to delete"
    )

    @field_validator("document_list")
    @classmethod
    def validate_document_list(cls, v):
        if len(v) > 100:
            raise ValueError("Cannot delete more than 100 documents at once")
        return v


class AddDocumentRequest(BaseModel):
    embeddings_model: str = Field(
        default=DEFAULT_EMBEDDING_MODEL,
        min_length=1,
        description="Embedding model to use",
    )
    chunk_size: int = Field(
        default=DEFAULT_CHUNK_SIZE, ge=100, le=4096, description="Size of text chunks"
    )
    chunk_overlap: int = Field(
        default=DEFAULT_CHUNK_OVERLAP,
        ge=0,
        le=1024,
        description="Overlap between chunks",
    )
    metadata_urls: Optional[List[str]] = Field(
        default=None, description="Optional URLs for metadata"
    )
    metadata_names: Optional[List[str]] = Field(
        default=None, description="Optional names for metadata"
    )

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v, info):
        if info.data and "chunk_size" in info.data and v >= info.data["chunk_size"]:
            raise ValueError("Chunk overlap must be less than chunk size")
        return v

    @field_validator("metadata_urls", "metadata_names")
    @classmethod
    def validate_metadata_lists(cls, v):
        if v is not None and len(v) > 50:
            raise ValueError("Metadata lists cannot exceed 50 items")
        return v


class UpdateDocumentRequest(BaseModel):
    embeddings_model: str = Field(
        default=DEFAULT_EMBEDDING_MODEL,
        min_length=1,
        description="Embedding model to use",
    )
    source_name: str = Field(
        min_length=1, max_length=255, description="Name of the source document"
    )
    new_metadata: Optional[dict] = Field(
        default=None, description="New metadata to update"
    )

    @field_validator("source_name")
    @classmethod
    def validate_source_name(cls, v):
        if not v.strip():
            raise ValueError("Source name cannot be empty or whitespace only")
        return v.strip()

    @field_validator("new_metadata")
    @classmethod
    def validate_metadata(cls, v):
        if v is not None and len(v) > 20:
            raise ValueError("Metadata cannot have more than 20 key-value pairs")
        return v
