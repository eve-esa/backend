from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.database.models.mcp_server import ToolType

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


class MCPServerPublicConfig(BaseModel):
    """Sanitized MCP server config returned to clients."""

    transport: Optional[TransportLiteral] = Field(
        default=None, description="Transport type for MCP connection"
    )
    url: Optional[str] = Field(
        default=None, description="Public MCP proxy URL for this server"
    )


class MCPServerAuthor(BaseModel):
    """One author from CITATION.cff: a person or an entity (organization, team)."""

    name: str = Field(..., description="Display name")
    given_names: Optional[str] = Field(default=None, description="Person's given names")
    family_names: Optional[str] = Field(default=None, description="Person's family names")
    alias: Optional[str] = Field(default=None, description="Alias, usually a GitHub handle")
    github_url: Optional[str] = Field(
        default=None, description="GitHub profile, when the alias is a valid GitHub handle"
    )
    affiliation: Optional[str] = Field(default=None, description="Affiliation")
    orcid: Optional[str] = Field(default=None, description="ORCID URL")
    website: Optional[str] = Field(default=None, description="Personal or entity website")


class MCPServerPreferredCitation(BaseModel):
    """The work users are asked to cite instead of the server itself, e.g. a paper."""

    title: str = Field(..., description="Title of the work")
    authors: List[str] = Field(default_factory=list, description="Author display names")
    year: Optional[str] = Field(default=None, description="Publication year")
    doi: Optional[str] = Field(default=None, description="DOI, without the resolver prefix")
    url: Optional[str] = Field(
        default=None, description="Link to the work; the DOI resolver when a DOI is set"
    )


class MCPServerAttribution(BaseModel):
    """Who wrote the server and how to cite it, derived from its CITATION.cff."""

    title: Optional[str] = Field(default=None, description="Title of the server")
    abstract: Optional[str] = Field(default=None, description="Short description")
    authors: List[MCPServerAuthor] = Field(default_factory=list, description="Authors")
    license: Optional[str] = Field(default=None, description="SPDX license identifier(s)")
    repository_url: Optional[str] = Field(default=None, description="Source code URL")
    preferred_citation: Optional[MCPServerPreferredCitation] = Field(
        default=None, description="Work to cite when using the server"
    )


class MCPServerCredentialFieldPublic(BaseModel):
    """One value the user fills for a toolkit. The upstream header stays in the backend."""

    name: str = Field(..., description="Field key, used in the request body")
    label: str = Field(..., description="Label to show next to the input")
    secret: bool = Field(..., description="Whether the input should be masked")
    help_text: Optional[str] = Field(default=None, description="Short help under the input")


class MCPServerUserCredentialsPublic(BaseModel):
    """What a toolkit asks the user for and whether this user has provided it.

    Never carries a stored value, a header name or the check configuration.
    """

    fields: List[MCPServerCredentialFieldPublic] = Field(..., description="Values to fill")
    required: bool = Field(
        default=False, description="Whether the toolkit cannot work without them"
    )
    help_url: Optional[str] = Field(default=None, description="Where to get the values")
    configured: bool = Field(..., description="Whether this user has saved values")
    hints: Dict[str, str] = Field(
        default_factory=dict, description="Masked hints for the non-secret fields"
    )
    verified_at: Optional[datetime] = Field(
        default=None, description="Last time the provider accepted the values"
    )
    updated_at: Optional[datetime] = Field(
        default=None, description="Last time the user saved the values"
    )


class ToolkitCredentialsRequest(BaseModel):
    """The user's values for a toolkit, keyed by field name."""

    model_config = ConfigDict(extra="forbid")

    values: Dict[str, str] = Field(..., max_length=5, description="Field values by name")


class MCPServerPublic(BaseModel):
    """Sanitized MCP server payload that never exposes internal URLs or headers."""

    id: Optional[str] = Field(default=None, description="MCP server identifier")
    timestamp: datetime = Field(description="Creation timestamp")
    name: str = Field(..., description="MCP server name")
    provider: Optional[str] = Field(default=None, description="Provider name")
    description: Optional[str] = Field(default=None, description="Description")
    type: ToolType = Field(default=ToolType.MCP, description="Server type")
    enabled: bool = Field(default=False, description="Whether the server is enabled")
    environment: Optional[List[str]] = Field(
        default=None, description="Environments where the tool is allowed"
    )
    config: MCPServerPublicConfig = Field(
        ..., description="Sanitized MCP connection configuration"
    )
    attribution: Optional[MCPServerAttribution] = Field(
        default=None,
        description="Authors and citation from the server's CITATION.cff, or null if none",
    )
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")
    deleted_at: Optional[datetime] = Field(
        default=None, description="Soft delete timestamp"
    )
    user_credentials: Optional[MCPServerUserCredentialsPublic] = Field(
        default=None,
        description="Values the user may provide for this toolkit, or null when it asks for none",
    )


class MCPServerPublicDetail(MCPServerPublic):
    tools: List[Dict[str, Any]] = Field(
        default_factory=list, description="Tools available on this MCP server"
    )
    # An empty tools list has two very different meanings, and until this field existed a client
    # could not tell them apart: a server that genuinely exposes nothing, and a server we failed to
    # reach. Both rendered as "No tools available", which is how six unreachable toolkits could look
    # exactly like six empty ones. None means the discovery call succeeded.
    tools_error: Optional[str] = Field(
        default=None,
        description=(
            "Why tool discovery failed, or null when it succeeded. When set, `tools` is empty "
            "because the server could not be queried, NOT because it exposes no tools."
        ),
    )


# Backward compatibility aliases for existing imports.
ToolConfigRequest = MCPServerConfigRequest
ToolRequest = MCPServerRequest
ToolUpdate = MCPServerUpdate
