from typing import Any, ClassVar, Dict, List, Optional
from enum import Enum
from datetime import datetime, timezone
from pydantic import BaseModel, ConfigDict, Field
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


class MCPServerMetadata(BaseModel):
    """Descriptive metadata about an MCP server, written by the backoffice.

    ``citation`` is the server's ``CITATION.cff`` from eve-esa/mcp-tool-registry,
    parsed as-is (Citation File Format 1.2.0, hyphenated keys such as
    ``given-names``). It is kept as a plain dict so a save from this service
    round-trips it unchanged; ``src/services/mcp/attribution.py`` turns it into
    the public ``attribution`` payload.
    """

    model_config = ConfigDict(extra="allow")

    # Any, not Dict: a malformed value must not fail loading the whole row.
    citation: Optional[Any] = Field(
        default=None, description="CITATION.cff content (CFF 1.2.0) as stored"
    )


class MCPServer(MongoModel):
    """Model for storing MCP server configuration and connection metadata."""

    # Display / identification fields
    user_id: Optional[str] = Field(
        default=None, description="Owner user ID; None means globally managed tool"
    )
    name: str = Field(..., description="MCP server name")
    provider: Optional[str] = Field(None, description="Provider or organization name")
    description: Optional[str] = Field(None, description="MCP server description")
    type: ToolType = Field(default=ToolType.MCP, description="Server type")

    # Operational flags
    enabled: bool = Field(default=False, description="Whether the server is enabled")

    # Connection configuration
    config: ToolConfig = Field(..., description="MCP connection configuration")

    metadata: Optional[MCPServerMetadata] = Field(
        default=None, description="Descriptive metadata such as author attribution"
    )

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

    collection_name: ClassVar[str] = "mcp_servers"


# Backward compatibility alias for existing imports.
Tool = MCPServer
