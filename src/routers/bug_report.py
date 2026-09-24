"""Bug reports from the chat: file one, and read back its screenshot.

``POST /bug-reports`` is multipart: ``description`` (text), ``context`` (JSON
text, see :class:`src.schemas.bug_report.BugReportContext`) and an optional
``screenshot`` (PNG or JPEG). The conversation named by
``context.conversation_id`` is snapshotted server side from Mongo, see
``src/services/bug_report_snapshot.py``. ``GET /bug-reports/{id}`` reads the
whole report back. Logic lives in ``src/services/bug_reports.py``.
"""

import logging
import re
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Path, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from src.database.models.user import User
from src.middlewares.auth import get_current_user
from src.routers.artifact import INLINE_CONTENT_TYPES
from src.schemas.bug_report import (
    CONTEXT_MAX_CHARS,
    DESCRIPTION_MAX_CHARS,
    BugReportContext,
    BugReportCreatedResponse,
    BugReportDetail,
    BugReportScreenshotInfo,
)
from src.services import bug_reports
from src.services.storage import ObjectNotFoundError

router = APIRouter()
logger = logging.getLogger(__name__)


def _field_error(field: str, message: str, value: object = None) -> RequestValidationError:
    """A 422 shaped like FastAPI's own, pointed at one form field."""
    return RequestValidationError(
        [{"type": "value_error", "loc": ("body", field), "msg": message, "input": value}]
    )


def _parse_context(raw: str) -> BugReportContext:
    try:
        return BugReportContext.model_validate_json(raw)
    except ValidationError as exc:
        errors = []
        for err in exc.errors(include_url=False, include_input=False):
            err = dict(err)
            err["loc"] = ("body", "context", *err.get("loc", ()))
            # The ctx of a JSON error can hold an exception object FastAPI
            # cannot serialize; the message already says what went wrong.
            err.pop("ctx", None)
            errors.append(err)
        raise RequestValidationError(errors)


@router.post("/bug-reports", status_code=201, response_model=BugReportCreatedResponse)
async def create_bug_report(
    description: Annotated[
        str,
        Form(
            min_length=1,
            max_length=DESCRIPTION_MAX_CHARS,
            description="What went wrong, 1 to 4000 characters",
        ),
    ],
    context: Annotated[
        str,
        Form(
            max_length=CONTEXT_MAX_CHARS,
            description="JSON text: session_id, replay_url, trace_id, conversation_id, "
            "message_id, app_version, app_commit, environment, user_agent, "
            "viewport {width, height}, path, console_errors, privacy_mode",
        ),
    ],
    screenshot: Annotated[
        Optional[UploadFile],
        File(description="Optional PNG or JPEG, at most 1 MB"),
    ] = None,
    requesting_user: User = Depends(get_current_user),
) -> BugReportCreatedResponse:
    """
    File a bug report with its browser context and an optional screenshot.

    The whole conversation named by ``context.conversation_id`` is read from
    Mongo and stored with the report, capped at 2 MB of JSON (oldest outputs
    cut first). Every string is redacted (credentials, API keys, emails)
    before it is stored in ``bug_reports``. The screenshot type is sniffed from its bytes,
    never taken from the declared Content-Type. On success a WARNING log event
    ``bug_report.created`` is emitted inside the request span.

    Args:
        description (str): What the user wrote, 1 to 4000 characters.
        context (str): Browser context as JSON text.
        screenshot (UploadFile | None): Optional PNG or JPEG screenshot.
        requesting_user (User): Authenticated user injected by dependency.

    Returns:
        BugReportCreatedResponse: The report id, its creation time and whether
        a screenshot was stored.

    Raises:
        HTTPException: 401 without a valid credential; 404 when
        ``context.conversation_id`` is missing or not the caller's (nothing is
        stored); 413 for a screenshot
        above the cap; 415 for a screenshot that is not PNG or JPEG; 422 for
        an empty or too long description or invalid context JSON; 429 with
        ``detail.code == "bug_report_rate_limited"`` past 5 reports per hour.
    """
    if not description.strip():
        raise _field_error("description", "Description must not be blank")
    parsed_context = _parse_context(context)
    image = await bug_reports.read_screenshot(screenshot)
    return await bug_reports.create_bug_report(
        requesting_user, description, parsed_context, image
    )


@router.get("/bug-reports/{report_id}", response_model=BugReportDetail)
async def get_bug_report(
    report_id: str = Path(..., description="Bug report ID"),
    requesting_user: User = Depends(get_current_user),
) -> BugReportDetail:
    """
    Read a whole bug report, to its author only.

    This is the read path for the planned Asana automation: the report plus
    the server side conversation snapshot, so a ticket never has to ask the
    user for the conversation. The automation will read it with an internal
    credential that does not exist yet; today only the author can.

    Args:
        report_id (str): Bug report identifier.
        requesting_user (User): Authenticated user injected by dependency.

    Returns:
        BugReportDetail: Description, browser context, conversation snapshot,
        screenshot metadata and the trace id of the filing request.

    Raises:
        HTTPException: 401 without a valid credential; 404 when the report
        does not exist or belongs to someone else.
    """
    report = await bug_reports.get_owned_report(report_id, requesting_user)
    screenshot = (
        BugReportScreenshotInfo(
            content_type=report.screenshot.content_type,
            size_bytes=report.screenshot.size_bytes,
        )
        if report.screenshot is not None
        else None
    )
    return BugReportDetail(
        id=report.id,
        user_id=report.user_id,
        created_at=report.timestamp,
        description=report.description,
        context=report.context,
        conversation=report.conversation,
        screenshot=screenshot,
        request_trace_id=report.request_trace_id,
    )


@router.get("/bug-reports/{report_id}/screenshot")
async def get_bug_report_screenshot(
    report_id: str = Path(..., description="Bug report ID"),
    requesting_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """
    Stream the screenshot of a bug report, to its author only.

    Args:
        report_id (str): Bug report identifier.
        requesting_user (User): Authenticated user injected by dependency.

    Returns:
        StreamingResponse: The image, served inline.

    Raises:
        HTTPException: 404 when the report does not exist, belongs to someone
        else, has no screenshot, or its bytes are gone from storage.
    """
    report = await bug_reports.get_owned_report(report_id, requesting_user)
    if report.screenshot is None:
        raise HTTPException(status_code=404, detail="Bug report has no screenshot")
    try:
        response = await bug_reports.storage_service.get_object(report.screenshot.key)
    except ObjectNotFoundError:
        logger.warning("Bug report %s has no object at its screenshot key", report.id)
        raise HTTPException(status_code=404, detail="Screenshot is no longer available")

    content_type = report.screenshot.content_type
    disposition = "inline" if content_type in INLINE_CONTENT_TYPES else "attachment"
    safe_id = re.sub(r"[^0-9a-f]", "", report.id)
    ext = report.screenshot.key.rsplit(".", 1)[-1]
    headers = {
        "Cache-Control": "private, no-cache",
        "Vary": "Authorization",
        "X-Content-Type-Options": "nosniff",
        "Content-Disposition": f'{disposition}; filename="bug-report-{safe_id}.{ext}"',
    }
    content_length = response.get("ContentLength", report.screenshot.size_bytes)
    if content_length is not None:
        headers["Content-Length"] = str(content_length)
    return StreamingResponse(
        bug_reports.storage_service.stream_body(response["Body"]),
        media_type=content_type,
        headers=headers,
    )
