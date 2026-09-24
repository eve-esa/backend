"""A bug report filed from the chat: what the user wrote plus the telemetry
context the browser had at the time (session, replay, last trace, ids), plus
a server side snapshot of the conversation it points at.

Every string is redacted before it is stored. Reports filed before the
screenshot was dropped may still carry a ``screenshot`` subdocument; it is
ignored on read.
"""

from typing import Any, ClassVar, Dict, Optional

from pydantic import BaseModel, Field

from src.database.mongo_model import MongoModel


class BugReport(MongoModel):
    """One user bug report, collection ``bug_reports``."""

    user_id: str = Field(..., description="Author user ID")
    description: str = Field(..., description="What the user wrote, redacted")
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Browser context sent with the report (BugReportContext), redacted",
    )
    conversation: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Snapshot of the whole conversation read from Mongo at filing "
        "time (src/services/bug_report_snapshot.py), redacted and capped; None "
        "when the report names no conversation",
    )
    request_trace_id: Optional[str] = Field(
        default=None,
        description="Trace id of the POST /bug-reports request, None with telemetry off",
    )

    collection_name: ClassVar[str] = "bug_reports"
