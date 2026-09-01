"""Map a verified provider token to the stable EVE user id.

The identity provider owns who someone is; this module owns which EVE row that
person has always been. The join key is ``(issuer, subject)``, never the email,
because subjects are stable and addresses get recycled.

Three decisions are load-bearing and are spelled out where they are made:

* ``email_verified`` fails closed. Cognito sends the string "true", Keycloak
  sends the boolean; anything else is read as not verified rather than guessed.
* an existing EVE account is adopted by a first sign-in only when it has no
  identity of its own yet. A recycled address must not hand a stranger the
  account of whoever held it before.
* the identity row is written before the user row. The unique index is the
  arbiter, so two workers racing on the same first sign-in produce one account.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

from bson import ObjectId

from src.config import AUTH_LINK_BY_VERIFIED_EMAIL
from src.database.models.external_identity import ExternalIdentity
from src.database.models.user import User
from src.services.oidc import fetch_userinfo

logger = logging.getLogger(__name__)

# Short on purpose. The cache exists to keep a burst of requests in one
# conversation off Mongo, not to hold a session. Sixty seconds is also the
# correction window: revoking access fleet-wide is a TTL lapse or a rolling
# restart, never a cache poke, and the ADR says so.
IDENTITY_CACHE_TTL_SECONDS = 60.0
# Bounded so a stream of unknown subjects cannot grow the process heap.
IDENTITY_CACHE_MAX_ENTRIES = 10_000

_identity_cache: dict[tuple[str, str], tuple[float, str]] = {}
_identity_locks: dict[tuple[str, str], asyncio.Lock] = {}
_registry_lock = asyncio.Lock()


def clear_identity_cache() -> None:
    """Reset the in-process cache (used by tests and hot reload)."""
    _identity_cache.clear()
    _identity_locks.clear()


def normalize_email(value: Any) -> Optional[str]:
    """Lowercase and strip an address, or return ``None`` if there isn't one.

    Applied on every read and every write. Providers are inconsistent about
    case, and a lookup that misses because of it provisions a duplicate account.
    """
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized or None


def normalize_email_verified(value: Any) -> bool:
    """Read the two shapes providers actually send, and fail closed otherwise.

    Cognito sends the string "true"/"false", Keycloak sends a boolean. Anything
    else, including ``None`` and a missing claim, reads as not verified: this
    value decides whether a stranger may adopt an existing account.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return False


def _cache_get(key: tuple[str, str]) -> Optional[str]:
    entry = _identity_cache.get(key)
    if entry is None:
        return None
    if time.monotonic() >= entry[0]:
        _identity_cache.pop(key, None)
        return None
    return entry[1]


def _cache_put(key: tuple[str, str], user_id: str) -> None:
    if len(_identity_cache) >= IDENTITY_CACHE_MAX_ENTRIES:
        # Insertion-ordered dict: the oldest entry is the first one.
        _identity_cache.pop(next(iter(_identity_cache)), None)
    _identity_cache[key] = (time.monotonic() + IDENTITY_CACHE_TTL_SECONDS, user_id)


async def _get_lock(key: tuple[str, str]) -> asyncio.Lock:
    if key not in _identity_locks:
        async with _registry_lock:
            if key not in _identity_locks:
                if len(_identity_locks) >= IDENTITY_CACHE_MAX_ENTRIES:
                    _identity_locks.pop(next(iter(_identity_locks)), None)
                _identity_locks[key] = asyncio.Lock()
    return _identity_locks[key]


async def _stamp_last_seen(identity_id: str) -> None:
    """Record the sighting without a full-document replace.

    ``MongoModel.save()`` is a ``replace_one``; using it here would rewrite the
    whole row for a timestamp and clobber a concurrent writer.
    """
    try:
        await ExternalIdentity.get_collection().update_one(
            {"_id": ObjectId(identity_id)},
            {"$set": {"last_seen_at": datetime.now(timezone.utc)}},
        )
    except Exception:  # pragma: no cover - telemetry must never fail a request
        logger.warning("Could not stamp last_seen_at on identity %s", identity_id)


async def _claim_identity(
    *, issuer: str, subject: str, email: Optional[str], user_id: str
) -> str:
    """Insert the identity row and return the user id that actually won.

    Written before the user row on purpose. A duplicate key means another worker
    got there first for the same ``(issuer, subject)``, and the answer is to
    adopt its user id rather than create a second account for the same person.
    """
    identity = ExternalIdentity(
        user_id=user_id,
        issuer=issuer,
        subject=subject,
        email=email,
        last_seen_at=datetime.now(timezone.utc),
    )
    try:
        await identity.save()
    except ValueError:
        winner = await ExternalIdentity.find_one(
            {"issuer": issuer, "subject": subject}
        )
        if winner is None:
            raise
        return winner.user_id
    return user_id


async def _insert_user(user_id: ObjectId, email: str, profile: dict[str, Any]) -> None:
    """Create the EVE row under an id the identity row already points at."""
    user = User(
        email=email,
        first_name=profile.get("given_name") or None,
        last_name=profile.get("family_name") or None,
    )
    document = user.to_dict()
    document["_id"] = user_id
    await User.get_collection().insert_one(document)


async def _link_or_provision(issuer: str, subject: str, token: str) -> str:
    profile = await fetch_userinfo(token)
    email = normalize_email(profile.get("email"))
    verified = normalize_email_verified(profile.get("email_verified"))

    if not email:
        raise PermissionError("Provider returned no email for this account")

    if verified and AUTH_LINK_BY_VERIFIED_EMAIL:
        candidate = await User.find_one({"email": email})
        if candidate is not None:
            existing = await ExternalIdentity.get_collection().find_one(
                {"user_id": candidate.id}
            )
            if existing is not None:
                # The account is already somebody's. Silently re-pointing it at
                # the new subject is exactly the recycled-address takeover this
                # rule exists to stop, so refuse and leave a trail.
                logger.warning(
                    "Refusing to link issuer=%s subject=%s to user %s: "
                    "that account already has an external identity",
                    issuer,
                    subject,
                    candidate.id,
                )
                raise PermissionError(
                    "This account is already linked to a different identity"
                )
            return await _claim_identity(
                issuer=issuer, subject=subject, email=email, user_id=candidate.id
            )

    # No account to adopt: provision one. The id is minted here so the identity
    # row can be written first and still point somewhere.
    new_user_id = ObjectId()
    winner = await _claim_identity(
        issuer=issuer, subject=subject, email=email, user_id=str(new_user_id)
    )
    if winner != str(new_user_id):
        return winner
    await _insert_user(new_user_id, email, profile)
    return winner


async def resolve_user_id(claims: dict[str, Any], token: str) -> str:
    """Resolve verified token claims to the EVE user id, linking on first sight.

    Raises ``PermissionError`` when the token cannot be attached to an account.
    """
    issuer = claims.get("iss")
    subject = claims.get("sub")
    if not isinstance(issuer, str) or not isinstance(subject, str):
        raise PermissionError("Token has no issuer or subject")

    key = (issuer, subject)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    lock = await _get_lock(key)
    async with lock:
        cached = _cache_get(key)
        if cached is not None:
            return cached

        identity = await ExternalIdentity.find_one(
            {"issuer": issuer, "subject": subject}
        )
        if identity is not None:
            await _stamp_last_seen(identity.id)
            _cache_put(key, identity.user_id)
            return identity.user_id

        user_id = await _link_or_provision(issuer, subject, token)
        _cache_put(key, user_id)
        return user_id
