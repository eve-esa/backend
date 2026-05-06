from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field
from src.database.models.mcp_server import MCPServer


TransportLiteral = Literal["streamable_http", "stdio"]


class MCPServerConfigRequest(BaseModel):
    """Request payload for MCP server configuration."""

    url: Optional[str] = Field(
        default=None, description="Remote MCP server URL (when using HTTP)"
    )
    transport: Optional[TransportLiteral] = Field(
        default=None, description="Transport type for MCP connection"
    )
    headers: Optional[Dict[str, str]] = Field(
        default=None, description="HTTP headers for remote transport"
    )
    command: Optional[str] = Field(
        default=None, description="Local command to start an MCP stdio server"
    )
    args: Optional[List[str]] = Field(
        default=None, description="Arguments for the local command"
    )
    env: Optional[Dict[str, str]] = Field(
        default=None, description="Environment variables for the MCP server"
    )


class MCPServerRequest(BaseModel):
    name: str = Field(..., min_length=1, description="MCP server name")
    provider: Optional[str] = Field(default=None, description="Provider name")
    description: Optional[str] = Field(default=None, description="Description")
    type: str = Field(default="mcp", description="Server type; default is mcp")
    enabled: bool = Field(default=False, description="Whether the server is enabled")
    environment: Optional[List[str]] = Field(
        default=None, description="Environments where the tool is allowed"
    )
    config: MCPServerConfigRequest = Field(
        ..., description="MCP connection configuration"
    )


class MCPServerUpdate(BaseModel):
    name: Optional[str] = None
    provider: Optional[str] = None
    description: Optional[str] = None
    type: Optional[str] = None
    enabled: Optional[bool] = None
    environment: Optional[List[str]] = None
    config: Optional[MCPServerConfigRequest] = None


class MCPServerDetail(MCPServer):
    tools: List[Dict[str, Any]] = Field(
        default_factory=list, description="Tools available on this MCP server"
    )


# Backward compatibility aliases for existing imports.
ToolConfigRequest = MCPServerConfigRequest
ToolRequest = MCPServerRequest
ToolUpdate = MCPServerUpdate
