from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class SkillFilePublic(BaseModel):
    """One file of a skill, without its bytes."""

    path: str = Field(..., description="Path relative to the skill folder")
    size: int = Field(..., ge=0, description="File size in bytes")


class SkillPublic(BaseModel):
    """A skill the caller may see. File contents stay in Mongo."""

    id: Optional[str] = Field(default=None, description="Skill identifier")
    timestamp: datetime = Field(description="Creation timestamp")
    name: str = Field(..., description="Skill name")
    description: str = Field(..., description="Skill description")
    enabled: bool = Field(default=False, description="Whether the skill can be requested")
    requires_mcp_servers: List[str] = Field(
        default_factory=list,
        description="MCP server names this skill expects",
    )
    files: List[SkillFilePublic] = Field(
        default_factory=list, description="Skill files, path and size only"
    )
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")
    deleted_at: Optional[datetime] = Field(
        default=None, description="Soft delete timestamp"
    )
