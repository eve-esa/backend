from typing import ClassVar
from pydantic import Field
from src.database.mongo_model import MongoModel


class Conversation(MongoModel):
    """Model for storing conversation messages with role."""

    user_id: str = Field(..., description="User ID")
    name: str = Field(..., description="Conversation name")
    summary: str | None = Field(
        default=None, description="Rolling summary of the conversation"
    )

    collection_name: ClassVar[str] = "conversations"
