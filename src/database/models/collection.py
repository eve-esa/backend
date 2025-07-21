from typing import ClassVar
from pydantic import Field
from src.database.mongo_model import MongoModel


class Collection(MongoModel):
    """Model for storing collection messages with role."""

    user_id: str = Field(..., description="User ID")
    name: str = Field(..., description="Collection name")

    collection_name: ClassVar[str] = "collections"
