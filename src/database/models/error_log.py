"""Model for storing error logs in MongoDB."""

from typing import Any, ClassVar, Dict, Optional
from pydantic import Field
from src.database.mongo_model import MongoModel
import logging

logger = logging.getLogger(__name__)


class ErrorLog(MongoModel):
    """Model for storing error logs from the application."""

    user_id: Optional[str] = Field(default=None, description="User ID associated with the error")
    conversation_id: Optional[str] = Field(
        default=None, description="Conversation ID associated with the error"
    )
    message_id: Optional[str] = Field(
        default=None, description="Message ID associated with the error"
    )
    logger_name: str = Field(
        default="src.services.generate_answer",
        description="Name of the logger/module where error occurred",
    )
    component: str = Field(
        ..., description="Component where error occurred (LLM, RETRIEVAL, etc.)"
    )
    error: Dict[str, Any] = Field(..., description="Original error information")
    error_type: str = Field(..., description="Type of error (e.g., 'TimeoutError', 'ValueError')")
    pipeline_stage: str = Field(
        ..., description="Pipeline stage where error occurred (re-querying, generation, etc.)"
    )
    description: str = Field(..., description="Human-readable description of the error")

    collection_name: ClassVar[str] = "error_logs"

