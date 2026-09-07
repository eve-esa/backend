"""Soft cap on self sign-up: the first N accounts get in, the rest wait.

Two seams, and there is no third. ``next_approval_status`` runs once per
account, at the single point that provisions one (``src/services/identity.py``),
and ``assert_user_approved`` runs on every authenticated request, at the single
principal resolver (``src/middlewares/auth.py``). Neither knows or cares which
identity provider the caller signed in with.

Approving an account is the back office's job, not this application's: nothing
here ever writes "approved" onto an existing row.
"""

from __future__ import annotations

from bson import ObjectId

from src.config import SIGNUP_AUTO_APPROVE_LIMIT
from src.database.models.user import User

APPROVAL_APPROVED = "approved"
APPROVAL_PENDING = "pending"

# One body for every door that meets a pending account: the REST routes and the
# two ASGI proxy dispatchers, which cannot raise HTTPException. Written down
# once so a client can branch on ``code`` whichever way it came in.
PENDING_APPROVAL_DETAIL = {
    "code": "pending_approval",
    "message": "Your account is awaiting approval",
}


class ApprovalPending(Exception):
    """The caller is authenticated and known, but not allowed in yet.

    Deliberately not a ``PermissionError``: the existing handlers turn that into
    a 401, and a 401 sends the browser back through a sign-in that would
    succeed and land on the same wall, forever.
    """


async def next_approval_status() -> str:
    """The status to stamp on an account being provisioned right now.

    Only rows that read "approved" hold a seat, so approving somebody by hand
    consumes one and deleting an account frees one, with no counter to keep in
    sync. Two workers provisioning at the same instant can both see the same
    count and both approve, so the overshoot is bounded by how many workers are
    provisioning at once. That is acceptable for a soft limit; an atomic counter
    document is the upgrade if an exact count is ever needed.
    """
    if SIGNUP_AUTO_APPROVE_LIMIT <= 0:
        return APPROVAL_APPROVED

    approved = await User.count_documents({"approval_status": APPROVAL_APPROVED})
    if approved < SIGNUP_AUTO_APPROVE_LIMIT:
        return APPROVAL_APPROVED
    return APPROVAL_PENDING


async def assert_user_approved(user_id: str) -> None:
    """Raise :class:`ApprovalPending` if this user is still waiting.

    Projected read: this runs on every authenticated request, and the rest of
    the document is not needed to answer the question.
    """
    doc = await User.get_collection().find_one(
        {"_id": ObjectId(user_id)},
        projection={"approval_status": 1},
    )
    if doc is not None and doc.get("approval_status") == APPROVAL_PENDING:
        raise ApprovalPending("Account is awaiting approval")


def assert_approved_doc(user: User) -> None:
    """Same rule as :func:`assert_user_approved`, on an already loaded user."""
    if user.approval_status == APPROVAL_PENDING:
        raise ApprovalPending("Account is awaiting approval")
