from typing import Any, ClassVar, Dict, Optional

from pydantic import Field

from src.database.mongo_model import MongoModel


class Image(MongoModel):
    """Model for storing a user-uploaded image backed by S3/MinIO object storage."""

    user_id: str = Field(..., description="Owner user ID")
    key: str = Field(..., description="S3 object key (users/{user_id}/{uuid}.{ext})")
    filename: str = Field(..., description="Original filename on upload")
    content_type: str = Field(
        ..., description="Sniffed MIME type (e.g. image/png), never the client-declared one"
    )
    size_bytes: int = Field(..., description="Object size in bytes")

    conversation_id: Optional[str] = Field(
        None, description="Conversation the image was attached to, if any"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Optional free-form metadata"
    )

    collection_name: ClassVar[str] = "images"
