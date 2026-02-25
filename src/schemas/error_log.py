from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class FrontendErrorLogRequest(BaseModel):
    """Request payload for logging frontend errors."""

    error_message: str = Field(..., description="Error message")
    error_stack: Optional[str] = Field(
        default=None, description="Error stack trace"
    )
    error_type: str = Field(..., description="Type of error (e.g., 'TypeError', 'ReferenceError')")
    url: Optional[str] = Field(
        default=None, description="URL where the error occurred"
    )
    user_agent: Optional[str] = Field(
        default=None, description="User agent string"
    )
    component: Optional[str] = Field(
        default="FRONTEND", description="Component where error occurred"
    )
    description: Optional[str] = Field(
        default=None, description="Human-readable description of the error"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional error metadata"
    )

