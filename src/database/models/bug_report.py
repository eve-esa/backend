"""A bug report filed from the chat: what the user wrote plus the telemetry
context the browser had at the time (session, replay, last trace, ids).

Every string is redacted before it is stored. The screenshot bytes live in
object storage under ``bug-reports/{user_id}/{id}.{ext}``; only the key and the
sniffed type are kept here.
"""

from typing import Any, ClassVar, Dict, Optional

from pydantic import BaseModel, Field

from src.database.mongo_model import MongoModel


class BugReportScreenshot(BaseModel):
    """Where the screenshot is stored and what it is."""

    key: str = Field(..., description="Object key: bug-reports/{user_id}/{id}.{ext}")
    content_type: str = Field(..., description="Sniffed MIME type, image/png or image/jpeg")
    size_bytes: int = Field(..., description="Object size in bytes")


class BugReport(MongoModel):
    """One user bug report, collection ``bug_reports``."""

    user_id: str = Field(..., description="Author user ID")
    description: str = Field(..., description="What the user wrote, redacted")
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Browser context sent with the report (BugReportContext), redacted",
    )
    screenshot: Optional[BugReportScreenshot] = Field(
        default=None, description="Stored screenshot, None when the report has none"
    )
    request_trace_id: Optional[str] = Field(
        default=None,
        description="Trace id of the POST /bug-reports request, None with telemetry off",
    )

    collection_name: ClassVar[str] = "bug_reports"
