from typing import ClassVar
from pydantic import Field
from src.database.mongo_model import MongoModel


class Message(MongoModel):
    """Model for storing individual messages."""

    conversation_id: str = Field(..., description="Conversation ID")
    input: str = Field(..., description="Message input")
    output: str = Field(..., description="Message output")
    feedback: str | None = Field(default=None, description="Feedback for the message")
    documents: list[str] = Field(
        default=[], description="Documents used to generate the answer"
    )
    results: list[dict] = Field(
        default=[],
        description="Results from the RAG, including the score and the document",
    )
    use_rag: bool = Field(
        default=False, description="Whether the message was generated using RAG"
    )

    collection_name: ClassVar[str] = "messages"
