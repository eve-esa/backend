"""Request and response shapes for ``POST /bug-reports``.

The request is multipart: ``description`` and ``context`` are form fields,
``context`` carries JSON text validated by :class:`BugReportContext`, and
``screenshot`` is an optional file part.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

DESCRIPTION_MAX_CHARS = 4000
# Upper bound on the raw context JSON text, well above what the fields allow.
CONTEXT_MAX_CHARS = 64 * 1024
CONSOLE_ERRORS_MAX_ITEMS = 50
CONSOLE_ERROR_MAX_CHARS = 2000


class BugReportViewport(BaseModel):
    """Browser viewport in CSS pixels."""

    width: int = Field(..., ge=0, le=100_000)
    height: int = Field(..., ge=0, le=100_000)


class BugReportContext(BaseModel):
    """What the browser knew when the report was filed.

    Every field is optional: with telemetry off the session, replay and trace
    ids are absent, and a report from outside a conversation has no
    conversation or message id. Unknown keys are dropped. Console errors longer
    than the cap are clipped on store, not rejected.
    """

    model_config = ConfigDict(extra="ignore")

    session_id: Optional[str] = Field(default=None, max_length=128, description="rum.sessionId")
    replay_url: Optional[str] = Field(default=None, max_length=2048, description="Session replay deep link")
    trace_id: Optional[str] = Field(default=None, max_length=64, description="Last trace id the page saw")
    conversation_id: Optional[str] = Field(default=None, max_length=64)
    message_id: Optional[str] = Field(default=None, max_length=64)
    app_version: Optional[str] = Field(default=None, max_length=128)
    app_commit: Optional[str] = Field(default=None, max_length=64)
    environment: Optional[str] = Field(default=None, max_length=64)
    user_agent: Optional[str] = Field(default=None, max_length=1024)
    viewport: Optional[BugReportViewport] = None
    path: Optional[str] = Field(default=None, max_length=2048, description="Page path when filed")
    console_errors: List[str] = Field(
        default_factory=list,
        max_length=CONSOLE_ERRORS_MAX_ITEMS,
        description="Most recent console errors, oldest first",
    )
    privacy_mode: Optional[str] = Field(
        default=None, max_length=32, description="Replay privacy mode: off, mask, on_demand, clear"
    )


class BugReportCreatedResponse(BaseModel):
    """201 answer to ``POST /bug-reports``."""

    id: str
    created_at: datetime
    screenshot: bool = Field(..., description="True when a screenshot was stored with the report")
