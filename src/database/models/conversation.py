from typing import ClassVar
from pydantic import Field
from src.database.mongo_model import MongoModel


class Conversation(MongoModel):
    """Model for storing conversation messages with role."""

    user_id: str = Field(..., description="User ID")
    input: str = Field(..., description="Message content (prompt or response)")
    output: str = Field(..., description="Model output or response")
    feedback: str = Field(default="none", description="Feedback from the user")
    metadata: dict = Field(
        default_factory=dict, description="LLM metadata used to generate the response"
    )

    collection_name: ClassVar[str] = "conversations"

    @classmethod
    async def safe_create(cls, user_id: str, input: str, output: str, metadata: dict):
        """Create a conversation message safely that only logs error if it fails."""
        try:
            await cls.create(user_id=user_id, input=input, output=output, metadata=metadata)
        except Exception as e:
            cls.logger.error(f"Error saving conversation: {e}")
