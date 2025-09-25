from typing import Any, ClassVar, Dict
from pydantic import Field
from src.services.generate_answer import GenerationRequest
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
    request_input: GenerationRequest = Field(
        description="Request input for the message generation",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Metadata for the message"
    )

    collection_name: ClassVar[str] = "messages"

    def to_dict(self) -> Dict[str, Any]:
        """Convert Message to a Mongo-storable dict, persisting private attrs.

        Pydantic PrivateAttr fields on nested models (e.g., GenerationRequest.collection_ids)
        are not serialized by default. We inject them here so they are available on retry.
        """
        doc = super().to_dict()
        try:
            request_input_dict = doc.get("request_input")
            if isinstance(request_input_dict, dict):
                collection_ids = getattr(self.request_input, "collection_ids", [])
                request_input_dict["collection_ids"] = (
                    list(collection_ids) if collection_ids else []
                )
                doc["request_input"] = request_input_dict
        except Exception:
            pass
        return doc

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Rehydrate Message from Mongo dict and restore private attrs on nested models."""
        instance = super().from_dict(data)
        try:
            request_input_dict = (
                data.get("request_input") if isinstance(data, dict) else None
            )
            if isinstance(request_input_dict, dict):
                collection_ids = request_input_dict.get("collection_ids") or []
                if (
                    hasattr(instance, "request_input")
                    and instance.request_input is not None
                ):
                    try:
                        instance.request_input.collection_ids = list(collection_ids)
                    except Exception:
                        pass
        except Exception:
            pass
        return instance
