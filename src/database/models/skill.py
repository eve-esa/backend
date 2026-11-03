from datetime import datetime, timezone
from typing import ClassVar, Dict, List, Optional

from pydantic import BaseModel, Field

from src.database.mongo_model import MongoModel


class SkillFile(BaseModel):
    """One file of a skill, stored as its exact bytes.

    ``path`` is relative to the skill folder (``scripts/roll.py``). ``content``
    is BSON BinData subtype 0, which PyMongo reads back as ``bytes``.
    """

    path: str = Field(..., description="Path relative to the skill folder")
    content: bytes = Field(..., description="File bytes")


class Skill(MongoModel):
    """A skill document. ``SKILL.md`` inside ``files`` is the source of truth.

    ``name`` and ``description`` copy the frontmatter so a listing does not
    have to parse files. The loader rejects a row whose copies differ.
    """

    user_id: Optional[str] = Field(
        default=None, description="Owner user ID; None means a global skill"
    )
    name: str = Field(..., description="Skill name; equals the SKILL.md frontmatter name")
    description: str = Field(
        ..., description="Skill description; equals the SKILL.md frontmatter description"
    )
    enabled: bool = Field(default=False, description="Whether the skill can be requested")
    requires_mcp_servers: List[str] = Field(
        default_factory=list,
        description="MCP server names this skill expects. Stored, not enforced yet.",
    )
    source: Optional[Dict[str, str]] = Field(
        default=None, description="Origin of an imported skill: repo, path, commit"
    )
    files: List[SkillFile] = Field(default_factory=list, description="Skill file tree")
    checksum: Optional[str] = Field(
        default=None, description="sha256 over the sorted (path, content) pairs"
    )
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

    collection_name: ClassVar[str] = "skills"
