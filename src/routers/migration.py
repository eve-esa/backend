"""TEMPORARY. Delete this module when the production migration window closes.

It exists for exactly one consumer: the Cognito Migrate-user Lambda, which runs
during the cutover so an existing user signs in with the password they already
have and never learns a migration happened. The Lambda cannot read Mongo, so it
asks this endpoint, and this endpoint is the last place in the application that
knows what a password is.

It is scheduled for deletion in the cleanup PR, together with the Lambda, the
shared secret, and the ``password_hash``/``is_active``/``activation_code``
fields it reads. Nothing else may start depending on it.

Two Cognito trigger sources are served by the same call:

* ``UserMigration_Authentication`` sends email and password, and the password
  must verify;
* ``UserMigration_ForgotPassword`` sends the email alone, because there is no
  password to check at that point in a reset.

The password never reaches a log line, an exception message, or a trace.
"""

import hashlib
import logging
import secrets
import time
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, EmailStr

from src.config import MIGRATION_SHARED_SECRET
from src.database.models.user import User

router = APIRouter(prefix="/internal/migration")
logger = logging.getLogger(__name__)

# In-process, per-email, best effort. The endpoint is already behind a shared
# secret; this is the second wall, so a leaked secret does not turn the legacy
# hashes into an offline-speed online oracle. Per process rather than shared:
# a Redis dependency for a component with a scheduled deletion date is not worth
# the coupling, and two workers only double a deliberately small number.
_MAX_ATTEMPTS_PER_WINDOW = 10
_ATTEMPT_WINDOW_SECONDS = 300.0
_MAX_TRACKED_EMAILS = 10_000
_attempts: dict[str, list[float]] = {}


def hash_password(password: str) -> str:
    """Reproduce the legacy password hash, for comparison only.

    Unsalted SHA-256, which is exactly why the application stopped owning
    passwords. It lives here rather than in src/services/utils.py so that the
    last caller and the last implementation are deleted in the same commit, and
    so that nothing new can import it by accident.
    """
    return hashlib.sha256(password.encode()).hexdigest()


def _reset_attempts_for_tests() -> None:
    _attempts.clear()


def _register_attempt(email: str) -> None:
    """Record an attempt for ``email``; raise 429 once the window is full."""
    now = time.monotonic()
    recent = [at for at in _attempts.get(email, []) if now - at < _ATTEMPT_WINDOW_SECONDS]
    if len(recent) >= _MAX_ATTEMPTS_PER_WINDOW:
        _attempts[email] = recent
        raise HTTPException(status_code=429, detail="Too many attempts")
    recent.append(now)
    if email not in _attempts and len(_attempts) >= _MAX_TRACKED_EMAILS:
        _attempts.pop(next(iter(_attempts)), None)
    _attempts[email] = recent


def _assert_shared_secret(provided: Optional[str]) -> None:
    if not MIGRATION_SHARED_SECRET:
        # Unconfigured means closed. An endpoint that reads password hashes must
        # not be reachable because somebody forgot to set a variable.
        raise HTTPException(status_code=503, detail="Migration endpoint is not configured")
    if not provided or not secrets.compare_digest(provided, MIGRATION_SHARED_SECRET):
        raise HTTPException(status_code=403, detail="Forbidden")


class VerifyCredentialsRequest(BaseModel):
    email: EmailStr
    # Absent for the ForgotPassword trigger, which has no password to check.
    password: Optional[str] = None


class MigrationProfileResponse(BaseModel):
    email: EmailStr
    # A string, not a boolean: this is copied straight into the Cognito user
    # attributes, and Cognito's email_verified is the string "true"/"false".
    # Get it wrong and the account migrates unverified, which then fails the
    # backend's fail-closed link rule and provisions a second, empty account.
    email_verified: str = "true"
    first_name: Optional[str] = None
    last_name: Optional[str] = None


@router.post("/verify-credentials", response_model=MigrationProfileResponse)
async def verify_credentials(
    request: VerifyCredentialsRequest,
    x_migration_secret: Optional[str] = Header(default=None),
) -> MigrationProfileResponse:
    """Confirm a legacy account, and its password when one is supplied.

    Returns the profile the Migrate-user Lambda copies into the new pool.
    """
    _assert_shared_secret(x_migration_secret)

    email = request.email.strip().lower()
    _register_attempt(email)

    user = await User.find_one({"email": email})
    if user is None:
        logger.info("Migration lookup: no legacy account for %s", email)
        raise HTTPException(status_code=404, detail="No legacy account")

    if request.password is not None:
        if not user.password_hash:
            logger.info("Migration lookup: %s has no legacy password hash", email)
            raise HTTPException(status_code=404, detail="No legacy password")
        if not secrets.compare_digest(
            user.password_hash, hash_password(request.password)
        ):
            logger.info("Migration lookup: password mismatch for %s", email)
            raise HTTPException(status_code=401, detail="Invalid credentials")

    return MigrationProfileResponse(
        email=email,
        first_name=user.first_name,
        last_name=user.last_name,
    )
