"""Bug reports filed from the chat: validate, throttle, store, announce.

Routes in ``src/routers/bug_report.py`` stay thin; this module holds the rate
limit (same Mongo time window count as ``src/services/api_keys.py``, no Redis),
the screenshot checks, the storage key layout and the ``bug_report.created``
log event.

Nothing here logs the description or the screenshot. The log event carries
ids and links only; the description stays in Mongo, redacted.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from bson import ObjectId
from fastapi import HTTPException, UploadFile

from src.config import BUG_REPORT_MAX_PER_HOUR, BUG_REPORT_SCREENSHOT_MAX_BYTES
from src.database.models.bug_report import BugReport, BugReportScreenshot
from src.database.models.user import User
from src.observability import deployment_environment
from src.observability.context import format_trace_id
from src.schemas.bug_report import (
    CONSOLE_ERROR_MAX_CHARS,
    BugReportContext,
    BugReportCreatedResponse,
)
from src.services.storage import (
    ARTIFACT_TYPE_CONTENT_TYPES,
    sniff_artifact_type,
    storage_service,
)
from src.utils.redaction import redact_secrets, redact_value

logger = logging.getLogger(__name__)
# The event HyperDX searches and alerts on. Its own logger name so a level
# override on this module's diagnostics never mutes it.
event_logger = logging.getLogger("eve.bug_report")

EVENT_CREATED = "bug_report.created"
SCREENSHOT_PREFIX = "bug-reports"
# Image only allowlist, a subset of the artifact type keys sniffed by
# sniff_artifact_type. The key doubles as the object extension.
SCREENSHOT_ALLOWED_TYPES = ("png", "jpeg")
_WINDOW = timedelta(hours=1)

# Attribute names on the log record and the request span.
ATTR_BUG_REPORT_ID = "eve.bug_report.id"
ATTR_RUM_SESSION_ID = "rum.sessionId"
ATTR_REPLAY_URL = "eve.replay_url"
ATTR_CONVERSATION_ID = "gen_ai.conversation.id"
ATTR_MESSAGE_ID = "eve.message_id"
ATTR_USER_ID = "user.id"
ATTR_ENVIRONMENT = "deployment.environment.name"


def _throttled_detail() -> dict:
    return {
        "code": "bug_report_rate_limited",
        "message": "Too many bug reports sent recently. Try again later.",
        "limit": BUG_REPORT_MAX_PER_HOUR,
    }


def _throttled() -> HTTPException:
    return HTTPException(
        status_code=429,
        detail=_throttled_detail(),
        headers={"Retry-After": str(int(_WINDOW.total_seconds()))},
    )


def _window_filter(user_id: str, now: datetime) -> dict:
    return {"user_id": user_id, "timestamp": {"$gte": now - _WINDOW}}


async def enforce_rate_limit(user_id: str, now: Optional[datetime] = None) -> None:
    """Refuse with 429 once the user has filed the cap within the last hour.

    Stored reports are the counter: a request refused by validation never
    counts, and nothing needs expiring. <=0 disables the throttle.
    """
    if BUG_REPORT_MAX_PER_HOUR <= 0:
        return
    now = now or datetime.now(timezone.utc)
    recent = await BugReport.count_documents(_window_filter(user_id, now))
    if recent >= BUG_REPORT_MAX_PER_HOUR:
        raise _throttled()


async def _is_among_first_in_window(user_id: str, report_id: str, now: datetime) -> bool:
    """True if ``report_id`` is one of the first ``cap`` reports of the window.

    The count before the insert lets a burst of concurrent requests all pass;
    this recount after the insert ranks the window by ``_id`` and drops the
    rows past the cap, the same approach as ``_is_among_first_active`` in
    ``api_keys``. It narrows the race, it does not close it: ``_id`` is minted
    before the insert commits, so a row with a lower id that becomes visible
    after a higher one has already passed can let a burst overshoot by one or
    two. Fine for a throttle whose job is to stop a flood.
    """
    cursor = (
        BugReport.get_collection()
        .find(_window_filter(user_id, now), {"_id": 1})
        .sort("_id", 1)
        .limit(BUG_REPORT_MAX_PER_HOUR)
    )
    survivors = {str(doc["_id"]) async for doc in cursor}
    return report_id in survivors


async def read_screenshot(upload: Optional[UploadFile]) -> Optional[Tuple[bytes, str]]:
    """Read and validate the optional screenshot part.

    The declared Content-Type is ignored: the type is sniffed from the magic
    bytes and must be PNG or JPEG. An empty part counts as no screenshot.

    Returns:
        ``(data, type_key)`` or None when no screenshot was sent.

    Raises:
        HTTPException: 413 above the byte cap, 415 for any other type.
    """
    if upload is None:
        return None
    data = await upload.read(BUG_REPORT_SCREENSHOT_MAX_BYTES + 1)
    if not data:
        return None
    if len(data) > BUG_REPORT_SCREENSHOT_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Screenshot exceeds {BUG_REPORT_SCREENSHOT_MAX_BYTES} bytes",
        )
    type_key = sniff_artifact_type(data[:16], upload.filename, data)
    if type_key not in SCREENSHOT_ALLOWED_TYPES:
        raise HTTPException(
            status_code=415, detail="Screenshot must be a PNG or JPEG image"
        )
    return data, type_key


def screenshot_key(user_id: str, report_id: str, type_key: str) -> str:
    """``bug-reports/{user_id}/{report_id}.{ext}``, outside the ``users/`` artifact tree."""
    return f"{SCREENSHOT_PREFIX}/{user_id}/{report_id}.{type_key}"


def _redacted_context(context: BugReportContext) -> dict:
    """Every context field, strings redacted, console errors clipped."""
    data = context.model_dump()
    data["console_errors"] = [
        entry[:CONSOLE_ERROR_MAX_CHARS] for entry in data.get("console_errors") or []
    ]
    return redact_value(data)


def _current_span():
    try:
        from opentelemetry import trace

        return trace.get_current_span()
    except Exception:  # pragma: no cover - defensive
        return None


def _event_attributes(report: BugReport, environment: str) -> dict:
    """Log record extras for ``bug_report.created``; absent values are left out.

    Ids and links only, never the description: this record leaves the process.
    """
    context = report.context or {}
    candidates = {
        ATTR_BUG_REPORT_ID: report.id,
        ATTR_RUM_SESSION_ID: context.get("session_id"),
        ATTR_REPLAY_URL: context.get("replay_url"),
        ATTR_CONVERSATION_ID: context.get("conversation_id"),
        ATTR_MESSAGE_ID: context.get("message_id"),
        ATTR_USER_ID: report.user_id,
        ATTR_ENVIRONMENT: environment,
    }
    attributes = {k: v for k, v in candidates.items() if v not in (None, "")}
    attributes["event.name"] = EVENT_CREATED
    return attributes


def emit_created_event(report: BugReport) -> None:
    """WARNING ``bug_report.created`` in the caller's context, so inside the
    request span when telemetry is on, and a plain WARNING line when it is off.

    The ids also go on the request span, so the trace is found from the report.
    """
    attributes = _event_attributes(report, deployment_environment())
    span = _current_span()
    if span is not None and span.is_recording():
        try:
            span.set_attribute(ATTR_BUG_REPORT_ID, report.id)
            span.set_attribute(ATTR_USER_ID, report.user_id)
        except Exception:  # pragma: no cover - telemetry never breaks a request
            pass
    event_logger.warning(
        "%s id=%s user_id=%s screenshot=%s",
        EVENT_CREATED,
        report.id,
        report.user_id,
        report.screenshot is not None,
        extra=attributes,
    )


async def create_bug_report(
    user: User,
    description: str,
    context: BugReportContext,
    screenshot: Optional[Tuple[bytes, str]],
) -> BugReportCreatedResponse:
    """Throttle, store the report, then its screenshot, then announce it.

    Optimistic insert then verify, as in ``api_keys.create_api_key``: the row is
    inserted, the window recounted, and a row past the cap deleted before
    anything is uploaded, so a refused report never leaves an object behind.
    A storage failure keeps the report and answers ``screenshot: false``: the
    description is the part the user cannot easily send again.
    """
    now = datetime.now(timezone.utc)
    await enforce_rate_limit(user.id, now)

    span = _current_span()
    request_trace_id = format_trace_id(span.get_span_context()) if span is not None else None

    report = BugReport(
        user_id=user.id,
        description=redact_secrets(description),
        context=_redacted_context(context),
        request_trace_id=request_trace_id,
        timestamp=now,
    )
    oid = ObjectId()
    doc = report.to_dict()
    doc["_id"] = oid
    await BugReport.get_collection().insert_one(doc)
    report.id = str(oid)

    if BUG_REPORT_MAX_PER_HOUR > 0 and not await _is_among_first_in_window(
        user.id, report.id, now
    ):
        await BugReport.get_collection().delete_one({"_id": oid})
        raise _throttled()

    if screenshot is not None:
        data, type_key = screenshot
        key = screenshot_key(user.id, report.id, type_key)
        content_type = ARTIFACT_TYPE_CONTENT_TYPES[type_key]
        try:
            await storage_service.put_object(key, data, content_type)
        except Exception as exc:
            logger.error(
                "bug_report.screenshot_store_failed id=%s error=%s",
                report.id,
                type(exc).__name__,
            )
        else:
            report.screenshot = BugReportScreenshot(
                key=key, content_type=content_type, size_bytes=len(data)
            )
            await BugReport.get_collection().update_one(
                {"_id": oid}, {"$set": {"screenshot": report.screenshot.model_dump()}}
            )

    emit_created_event(report)
    return BugReportCreatedResponse(
        id=report.id, created_at=report.timestamp, screenshot=report.screenshot is not None
    )


async def get_owned_report(report_id: str, user: User) -> BugReport:
    """The report if ``user`` wrote it, otherwise 404.

    One query filtered by author: a missing id and somebody else's id answer
    the same, so the route is not an oracle for which ids exist.
    """
    try:
        oid = ObjectId(report_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Bug report not found")
    doc = await BugReport.get_collection().find_one({"_id": oid, "user_id": user.id})
    if doc is None:
        raise HTTPException(status_code=404, detail="Bug report not found")
    return BugReport.from_dict(doc)
