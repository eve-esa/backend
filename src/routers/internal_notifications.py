"""Mail the application owes a person, triggered by something outside it.

One consumer today: the back office, which flips an account to "approved" in
Mongo and then asks here for the person to be told. Approving is a database
write the back office already owns; composing the message and knowing which
transport is configured is this application's job, and this endpoint is the
seam between the two.

Same shape and same guard as src/routers/migration.py. The edge blocks
``/api/internal/*``, so only a caller already inside the VPC reaches this, and
the shared secret is the second wall rather than the only one. Unset secret
means closed: an endpoint that sends mail on request must not be reachable
because somebody forgot to set a variable.

The status is re-read here rather than trusted from the caller. A "your account
is ready" mail to somebody who is still in the queue is worse than no mail, so
the endpoint refuses unless the row actually says approved.
"""

import logging
import secrets
from typing import Optional

from bson import ObjectId
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from src.config import INTERNAL_API_SECRET
from src.database.models.user import User
from src.services.account_notifications import notify_account_approved
from src.services.approval import APPROVAL_APPROVED

router = APIRouter(prefix="/internal/notifications")
logger = logging.getLogger(__name__)


def _assert_shared_secret(provided: Optional[str]) -> None:
    if not INTERNAL_API_SECRET:
        raise HTTPException(
            status_code=503, detail="Internal notifications are not configured"
        )
    if not provided or not secrets.compare_digest(provided, INTERNAL_API_SECRET):
        raise HTTPException(status_code=403, detail="Forbidden")


class AccountApprovedRequest(BaseModel):
    user_id: str


class NotificationSentResponse(BaseModel):
    sent: bool


@router.post("/account-approved", response_model=NotificationSentResponse)
async def account_approved(
    request: AccountApprovedRequest,
    x_internal_secret: Optional[str] = Header(default=None),
) -> NotificationSentResponse:
    """Tell an approved account that it can sign in now."""
    _assert_shared_secret(x_internal_secret)

    # A malformed id is the same answer as an id nobody has: not found. Letting
    # ObjectId raise here would turn a typo into a 500.
    if not ObjectId.is_valid(request.user_id):
        raise HTTPException(status_code=404, detail="Unknown user")

    user = await User.find_by_id(request.user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="Unknown user")

    if user.approval_status != APPROVAL_APPROVED:
        raise HTTPException(
            status_code=409, detail="Account is not approved"
        )

    try:
        await notify_account_approved(user.id, user.email)
    except Exception:
        # Already logged with a stack trace one layer down. The caller only
        # needs to know it may retry, which 502 says and 500 does not.
        raise HTTPException(status_code=502, detail="Mail delivery failed")

    return NotificationSentResponse(sent=True)
