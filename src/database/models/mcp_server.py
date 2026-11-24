from typing import Any, ClassVar, Dict, List, Literal, Optional
from enum import Enum
from datetime import datetime, timezone
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
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


# Headers the user must never set: they carry EVE identity, tracing or MCP session
# state, or are framing headers an upstream would misread. Compared lowercase.
RESERVED_CREDENTIAL_HEADERS = frozenset(
    {
        "authorization",
        "x-eve-token",
        "cookie",
        "host",
        "content-length",
        "content-type",
        "transfer-encoding",
        "connection",
        "te",
        "upgrade",
        "traceparent",
        "tracestate",
        "mcp-session-id",
        "mcp-protocol-version",
    }
)


def _require_https(value: Optional[str]) -> Optional[str]:
    """Reject any non-https URL, so a credential never travels in clear text."""
    if value is not None and not value.startswith("https://"):
        raise ValueError("must start with https://")
    return value


class UserCredentialField(BaseModel):
    """One value a user fills for a toolkit and the upstream header it becomes."""

    name: str = Field(..., pattern=r"^[a-z][a-z0-9_]{0,31}$")
    label: str = Field(..., min_length=1, max_length=64)
    header: str = Field(..., pattern=r"^[A-Za-z0-9-]{1,64}$")
    secret: bool = True
    help_text: Optional[str] = Field(default=None, max_length=200)

    @field_validator("header")
    @classmethod
    def _header_not_reserved(cls, value: str) -> str:
        if value.lower() in RESERVED_CREDENTIAL_HEADERS:
            raise ValueError(f"header {value!r} is reserved")
        return value


class UserCredentialCheck(BaseModel):
    """How to prove the values work before saving them."""

    type: Literal["oauth2_client_credentials"]
    token_url: str
    client_id_field: str
    client_secret_field: str

    _token_url_https = field_validator("token_url")(_require_https)


class UserCredentialSpec(BaseModel):
    """What an operator asks the user to provide for one toolkit."""

    fields: List[UserCredentialField] = Field(..., min_length=1, max_length=5)
    required: bool = False
    help_url: Optional[str] = None
    check: Optional[UserCredentialCheck] = None

    _help_url_https = field_validator("help_url")(_require_https)

    @model_validator(mode="after")
    def _consistent(self) -> "UserCredentialSpec":
        names = [f.name for f in self.fields]
        if len(set(names)) != len(names):
            raise ValueError("field names must be unique")
        headers = [f.header.lower() for f in self.fields]
        if len(set(headers)) != len(headers):
            raise ValueError("headers must be unique, ignoring case")
        if self.check is not None:
            for ref in (self.check.client_id_field, self.check.client_secret_field):
                if ref not in names:
                    raise ValueError(f"check refers to undeclared field {ref!r}")
        return self


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

    # Set by operators only (back office or Mongo), never by the user API:
    # whether this server receives the caller's EVE token as ``X-EVE-Token``.
    # ``None`` (field absent) falls back to the reserved retrieval names, see
    # ``src/services/mcp/user_token_scope.py``. A change reaches each worker
    # within ``MCP_TOOLS_CACHE_TTL`` (the proxy URL and tool caches).
    forward_user_token: Optional[bool] = Field(
        default=None,
        description="Forward the caller's EVE token as X-EVE-Token (retrieval only)",
    )

    metadata: Optional[MCPServerMetadata] = Field(
        default=None, description="Descriptive metadata such as author attribution"
    )

    # Set by operators only (back office or Mongo), never by the user API, and
    # effective on global rows only. Any, not UserCredentialSpec, like
    # ``MCPServerMetadata.citation``: a malformed operator value must not make the
    # row fail to load. The credential service parses it into a UserCredentialSpec.
    user_credentials: Optional[Any] = Field(
        default=None,
        description="Operator-declared credential fields each user may fill",
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
