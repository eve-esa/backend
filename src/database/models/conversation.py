from typing import ClassVar
from pydantic import Field
from src.database.mongo_model import MongoModel


class Conversation(MongoModel):
    """Model for storing conversation messages with role."""

    user_id: str = Field(..., description="User ID")
    name: str = Field(..., description="Conversation name")
    summary: str | None = Field(
        default=None,
        description="Conversation summary: the rolling summary, or the deep agent's own",
    )
    summary_first_message_id: str | None = Field(
        default=None, description="First message the agent summary covers"
    )
    summary_last_message_id: str | None = Field(
        default=None, description="Last message the agent summary covers"
    )

    collection_name: ClassVar[str] = "conversations"
