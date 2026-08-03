from typing import Any, ClassVar, Dict, Literal, Optional

from pydantic import BaseModel, Field

from src.database.mongo_model import MongoModel


class ArtifactSource(BaseModel):
    """How an artifact came to exist: a direct user upload, or an MCP tool call."""

    type: Literal["mcp_tool", "upload"] = Field(
        ..., description="Origin of the artifact"
    )
    mcp_server: Optional[str] = Field(
        None, description="MCP server name that produced the artifact, if any"
    )
    tool_name: Optional[str] = Field(
        None, description="MCP tool name that produced the artifact, if any"
    )


class Artifact(MongoModel):
    """Model for a user-owned artifact backed by S3/MinIO object storage.

    Generalizes the former Image model to any artifact type: user uploads and
    MCP tool outputs alike, distinguished by `source.type`.
    """

    user_id: str = Field(..., description="Owner user ID")
    key: str = Field(..., description="S3 object key (users/{user_id}/...)")
    filename: str = Field(..., description="Original or tool-provided filename")
    content_type: str = Field(
        ...,
        description="Sniffed (upload) or tool-declared (mcp_tool) MIME type",
    )
    size_bytes: int = Field(..., description="Object size in bytes")
    source: ArtifactSource = Field(
        ..., description="Origin of the artifact: user upload or MCP tool call"
    )

    conversation_id: Optional[str] = Field(
        None, description="Conversation the artifact was attached to, if any"
    )
    message_id: Optional[str] = Field(
        None, description="Message the artifact was attached to, if any"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Optional free-form metadata"
    )

    collection_name: ClassVar[str] = "artifacts"
