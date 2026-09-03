from typing import Any, ClassVar, Dict, List, Optional
from pydantic import Field
from src.schemas.generation_request import GenerationRequest
from src.database.mongo_model import MongoModel
import logging

logger = logging.getLogger(__name__)


class Message(MongoModel):
    """Model for storing individual messages."""

    conversation_id: str = Field(..., description="Conversation ID")
    input: str = Field(..., description="Message input")
    output: str = Field(..., description="Message output")
    stopped: Optional[bool] = Field(
        default=False, description="Whether the message was stopped"
    )
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
    request_input: Optional[GenerationRequest] = Field(
        default=None,
        description="Request input for the message generation",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Metadata for the message"
    )
    hallucination: Optional[Dict[str, Any]] = Field(
        default=None, description="Hallucination analysis data"
    )
    trace: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Full agentic execution trace: every LangGraph message, tool call, and result",
    )
    attachments: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Image attachments associated with the message"
    )
    artifact_ids: Optional[List[str]] = Field(
        default=None,
        description="Artifacts produced by MCP tool calls during this message's generation",
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
                private_map = getattr(
                    self.request_input, "private_collections_map", {}
                )
                request_input_dict["private_collections_map"] = (
                    dict(private_map) if private_map else {}
                )
                user_id = getattr(self.request_input, "user_id", None)
                if user_id:
                    request_input_dict["user_id"] = user_id
                doc["request_input"] = request_input_dict
        except Exception as e:
            logger.error(f"Error serializing request_input: {e}")
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
                        private_map = (
                            request_input_dict.get("private_collections_map") or {}
                        )
                        instance.request_input.private_collections_map = dict(
                            private_map
                        )
                        stored_user_id = request_input_dict.get("user_id")
                        if stored_user_id:
                            instance.request_input.user_id = stored_user_id
                    except Exception as e:
                        logger.error(f"Error deserializing request_input: {e}")
                        pass
        except Exception as e:
            logger.error(f"Error deserializing request_input: {e}")
            pass
        return instance
