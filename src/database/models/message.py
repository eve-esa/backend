from typing import Any, ClassVar
from pydantic import Field
from src.database.mongo_model import MongoModel


class Message(MongoModel):
    """Model for storing individual messages."""

    conversation_id: str = Field(..., description="Conversation ID")
    input: str = Field(..., description="Message input")
    output: str = Field(..., description="Message output")
    feedback: str | None = Field(default=None, description="Feedback for the message")
    documents: Any = Field(
        default=None, description="Documents used to generate the answer"
    )
    use_rag: bool = Field(
        default=False, description="Whether the message was generated using RAG"
    )

    collection_name: ClassVar[str] = "messages"
