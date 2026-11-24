from datetime import datetime, timezone
from typing import ClassVar, Dict, Optional

from pydantic import Field

from src.database.mongo_model import MongoModel


class MCPServerCredential(MongoModel):
    """A user's own credentials for one toolkit.

    One row per user and toolkit. The values are envelope-encrypted with the
    ``byok`` KMS key, so a database read alone never yields a secret.
    """

    user_id: str = Field(..., description="Owner user ID")
    server_id: str = Field(..., description="ID of the global mcp_servers row")
    encrypted_values: str = Field(
        ..., description="Envelope blob of the field values, never plaintext"
    )
    hints: Dict[str, str] = Field(
        default_factory=dict,
        description="Masked hints for the non-secret fields, safe to show back",
    )
    verified_at: Optional[datetime] = Field(
        default=None, description="Last time the declared check passed"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp",
    )

    collection_name: ClassVar[str] = "mcp_server_credentials"
