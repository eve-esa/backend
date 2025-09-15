from typing import Any, ClassVar, Dict
from pydantic import Field
from src.database.mongo_model import MongoModel
from typing import Optional


class Message(MongoModel):
    """Model for storing individual messages."""

    conversation_id: str = Field(..., description="Conversation ID")
    input: str = Field(..., description="Message input")
    output: str = Field(..., description="Message output")
    feedback: Optional[str] = Field(
        default=None, description="Feedback for the message"
    )
    feedback_reason: Optional[str] = Field(
        default=None, description="Reason for the feedback"
    )
    documents: Any = Field(
        default=None, description="Documents used to generate the answer"
    )
    use_rag: bool = Field(
        default=False, description="Whether the message was generated using RAG"
    )
    was_copied: bool = Field(
        default=False,
        description="Whether the message was copied from the previous message",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Metadata for the message"
    )

    collection_name: ClassVar[str] = "messages"
