"""Bug reports read by the observability Asana sink.

The sink runs on the observability host, receives the HyperDX "Bug reports"
alert, and needs the whole report to build a ticket. ``GET /bug-reports/{id}``
answers only the author, so this route reads any report for a caller holding
``INTERNAL_API_SECRET``.

Same guard as src/routers/internal_notifications.py. The edge blocks
``/api/internal/*``, so only a caller inside the VPC reaches this, and the
shared secret is the second wall. Unset secret means closed (503): a route that
hands out conversation snapshots must not open because a variable is missing.
"""

import logging
import secrets
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Path

from src.config import INTERNAL_API_SECRET
from src.observability import deployment_environment
from src.schemas.bug_report import InternalBugReportDetail
from src.services import bug_reports

router = APIRouter(prefix="/internal/bug-reports")
logger = logging.getLogger(__name__)


def _assert_shared_secret(provided: Optional[str]) -> None:
    if not INTERNAL_API_SECRET:
        raise HTTPException(status_code=503, detail="Internal bug report reads are not configured")
    if not provided or not secrets.compare_digest(provided, INTERNAL_API_SECRET):
        raise HTTPException(status_code=403, detail="Forbidden")


@router.get("/{report_id}", response_model=InternalBugReportDetail)
async def read_bug_report(
    report_id: str = Path(..., description="Bug report ID"),
    x_internal_secret: Optional[str] = Header(default=None),
) -> InternalBugReportDetail:
    """
    Read a whole bug report for the observability sink, whoever wrote it.

    Same body as ``GET /bug-reports/{id}`` plus ``trace_id`` and
    ``conversation_id`` from the stored context and the backend's
    ``environment``. Logs ``bug_report.read_internal id=<id>`` at INFO.

    Raises:
        HTTPException: 503 when ``INTERNAL_API_SECRET`` is unset; 403 when the
        ``X-Internal-Secret`` header is absent or wrong; 404 when the report
        does not exist or the id is malformed.
    """
    _assert_shared_secret(x_internal_secret)
    report = await bug_reports.get_report(report_id)
    # Logged after the read: the id is a valid ObjectId by now, never raw input.
    logger.info("bug_report.read_internal id=%s", report.id)
    context = report.context or {}
    return InternalBugReportDetail(
        id=report.id,
        user_id=report.user_id,
        created_at=report.timestamp,
        description=report.description,
        context=report.context,
        conversation=report.conversation,
        request_trace_id=report.request_trace_id,
        trace_id=context.get("trace_id"),
        conversation_id=context.get("conversation_id"),
        environment=deployment_environment(),
    )
