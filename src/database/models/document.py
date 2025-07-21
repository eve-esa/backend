from typing import ClassVar, Optional
from pydantic import Field
from src.database.mongo_model import MongoModel


class Document(MongoModel):
    """Model for storing user-uploaded documents belonging to a collection."""

    user_id: str = Field(..., description="User ID")
    collection_id: str = Field(..., description="Parent collection ID")

    # Display / identification fields
    name: str = Field(..., description="Display name of the document")

    # File details (optional because older records might not have them)
    filename: Optional[str] = Field(None, description="Original filename on upload")
    file_type: Optional[str] = Field(
        None, description="File type / extension (e.g. pdf, txt)"
    )
    source_url: Optional[str] = Field(None, description="Original source URL, if any")

    # Stats
    chunk_count: Optional[int] = Field(
        None, description="Number of chunks stored in vector DB"
    )
    file_size: Optional[int] = Field(None, description="File size in bytes")

    vector_ids: Optional[list[str]] = Field(
        None, description="Qdrant point IDs for this document's chunks"
    )

    collection_name: ClassVar[str] = "documents"
