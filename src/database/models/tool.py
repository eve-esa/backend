from typing import Any, ClassVar, Dict, List, Optional
from enum import Enum
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from src.database.mongo_model import MongoModel


class ToolTransport(str, Enum):
    """Supported transport types for MCP tools."""

    STREAMABLE_HTTP = "streamable_http"
    STDIO = "stdio"


class ToolType(str, Enum):
    """High-level tool types. Default is MCP to support Model Context Protocol tools."""

    MCP = "mcp"


class ToolConfig(BaseModel):
    """Configuration for connecting to an MCP tool server.

    All fields are optional to support both remote and local servers.
    """

    # Remote server options
    url: Optional[str] = Field(
        default=None, description="Remote MCP server URL (if using HTTP transport)"
    )
    transport: Optional[ToolTransport] = Field(
        default=None, description="Transport type: streamable_http or stdio"
    )
    headers: Optional[Dict[str, str]] = Field(
        default=None, description="HTTP headers for remote transport"
    )

    # Local/stdio server options
    command: Optional[str] = Field(
        default=None, description="Local command/binary to start an MCP stdio server"
    )
    args: Optional[List[str]] = Field(
        default=None, description="Arguments to pass to the local command"
    )
    env: Optional[Dict[str, str]] = Field(
        default=None, description="Environment variables for the MCP server process"
    )


class Tool(MongoModel):
    """Model for storing tool configuration and MCP connection metadata."""

    # Display / identification fields
    user_id: Optional[str] = Field(
        default=None, description="Owner user ID; None means globally managed tool"
    )
    name: str = Field(..., description="Tool name")
    provider: Optional[str] = Field(None, description="Provider or organization name")
    description: Optional[str] = Field(None, description="Tool description")
    type: ToolType = Field(default=ToolType.MCP, description="Tool type")

    # Operational flags
    enabled: bool = Field(default=False, description="Whether the tool is enabled")
    environment: Optional[List[str]] = Field(
        default=None,
        description="Environments where the tool is allowed (e.g. staging, prod)",
    )

    # Connection configuration
    config: ToolConfig = Field(..., description="MCP connection configuration")

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp",
    )
    deleted_at: Optional[datetime] = Field(
        default=None, description="Soft delete timestamp (if deleted)"
    )

    collection_name: ClassVar[str] = "tools"
