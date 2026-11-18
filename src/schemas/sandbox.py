"""Response models for the conversation files and sandbox routes."""

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

SandboxStatus = Literal["disabled", "none", "idle", "busy"]


class SandboxRead(BaseModel):
    """The conversation's sandbox as the pool records it.

    Read from the pool map only: the provider is not asked, so a sandbox the
    provider already ended still reads ``idle`` until the reaper drops it.
    """

    status: SandboxStatus = Field(
        description=(
            "disabled: SANDBOX_PROVIDER is unset or none. none: no sandbox for "
            "this conversation. idle: no turn is using it. busy: a turn is."
        )
    )
    provider: Optional[str] = None
    sandbox_id: Optional[str] = None
    inflight: int = Field(default=0, description="Turns using the sandbox now")
    created_at: Optional[datetime] = Field(
        default=None, description="AgentCore only. OpenSandbox does not store it"
    )
    expires_at: Optional[datetime] = Field(
        default=None,
        description="AgentCore hard cap, created_at plus max_lifetime_s",
    )
    idle_deadline: Optional[datetime] = Field(
        default=None, description="When the reaper stops the sandbox"
    )
    skills_digest: Optional[str] = None


class FileEntry(BaseModel):
    path: str = Field(description="Relative to the listed root")
    size: int
    modified_at: Optional[datetime] = None


class FileListing(BaseModel):
    items: List[FileEntry]
    truncated: bool = Field(
        description="More entries exist than were returned, or the scan stopped early"
    )
