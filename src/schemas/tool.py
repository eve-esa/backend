from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field


TransportLiteral = Literal["streamable_http", "stdio"]


class ToolConfigRequest(BaseModel):
    """Request payload for MCP tool configuration."""

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


class ToolRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Tool name")
    provider: Optional[str] = Field(default=None, description="Provider name")
    description: Optional[str] = Field(default=None, description="Description")
    type: str = Field(default="mcp", description="Tool type; default is mcp")
    enabled: bool = Field(default=False, description="Whether the tool is enabled")
    environment: Optional[List[str]] = Field(
        default=None, description="Environments where the tool is allowed"
    )
    config: ToolConfigRequest = Field(..., description="MCP connection configuration")


class ToolUpdate(BaseModel):
    name: Optional[str] = None
    provider: Optional[str] = None
    description: Optional[str] = None
    type: Optional[str] = None
    enabled: Optional[bool] = None
    environment: Optional[List[str]] = None
    config: Optional[ToolConfigRequest] = None
