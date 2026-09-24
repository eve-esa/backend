"""Bug reports from the chat: file one, and read it back.

``POST /bug-reports`` is multipart: ``description`` (text) and ``context``
(JSON text, see :class:`src.schemas.bug_report.BugReportContext`). Any other
part, a ``screenshot`` file included, is ignored: the session replay linked by
``context.replay_url`` replaced the screenshot. The conversation named by
``context.conversation_id`` is snapshotted server side from Mongo, see
``src/services/bug_report_snapshot.py``. ``GET /bug-reports/{id}`` reads the
whole report back. Logic lives in ``src/services/bug_reports.py``.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Form, Path
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from src.database.models.user import User
from src.middlewares.auth import get_current_user
from src.schemas.bug_report import (
    CONTEXT_MAX_CHARS,
    DESCRIPTION_MAX_CHARS,
    BugReportContext,
    BugReportCreatedResponse,
    BugReportDetail,
)
from src.services import bug_reports

router = APIRouter()


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
    requesting_user: User = Depends(get_current_user),
) -> BugReportCreatedResponse:
    """
    File a bug report with its browser context.

    The whole conversation named by ``context.conversation_id`` is read from
    Mongo and stored with the report, capped at 2 MB of JSON (oldest outputs
    cut first). Every string is redacted (credentials, API keys, emails)
    before it is stored in ``bug_reports``. A ``screenshot`` part, still sent
    by older frontends, is ignored like any other unknown part. On success a
    WARNING log event ``bug_report.created`` is emitted inside the request span.

    Args:
        description (str): What the user wrote, 1 to 4000 characters.
        context (str): Browser context as JSON text.
        requesting_user (User): Authenticated user injected by dependency.

    Returns:
        BugReportCreatedResponse: The report id and its creation time.

    Raises:
        HTTPException: 401 without a valid credential; 404 when
        ``context.conversation_id`` is missing or not the caller's (nothing is
        stored); 422 for
        an empty or too long description or invalid context JSON; 429 with
        ``detail.code == "bug_report_rate_limited"`` past
        ``BUG_REPORT_MAX_PER_HOUR`` reports per hour (default 5), only while
        ``FEATURE_BUG_REPORT_RATE_LIMIT`` is on.
    """
    if not description.strip():
        raise _field_error("description", "Description must not be blank")
    parsed_context = _parse_context(context)
    return await bug_reports.create_bug_report(requesting_user, description, parsed_context)


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
        BugReportDetail: Description, browser context, conversation snapshot
        and the trace id of the filing request.

    Raises:
        HTTPException: 401 without a valid credential; 404 when the report
        does not exist or belongs to someone else.
    """
    report = await bug_reports.get_owned_report(report_id, requesting_user)
    return BugReportDetail(
        id=report.id,
        user_id=report.user_id,
        created_at=report.timestamp,
        description=report.description,
        context=report.context,
        conversation=report.conversation,
        request_trace_id=report.request_trace_id,
    )

