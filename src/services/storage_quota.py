"""Per-user object storage quota.

``storage_bytes_used`` on the user is the bytes currently stored under
``users/{id}/``. Every write adds its bytes with ``$inc`` before the write,
and every delete subtracts them. A full ``save()`` of the user is never used:
it would write a stale copy over a concurrent increment.

Counting is always on. ``FEATURE_STORAGE_QUOTA`` only turns refusals and the
notice on. A plan with no ``*_STORAGE_GB`` variable, and the ``unlimited``
plan, have no cap. ``0`` is no cap too, the same as a token plan with no
``max_tokens``.

The unit is GiB (1024^3), the unit AWS uses for S3 billing.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import HTTPException
from pymongo import ReturnDocument

from src import config
from src.database.models.user import User
from src.database.mongo import async_mongo_manager
from src.services.storage import storage_service
from src.schemas.rate_limit import (
    RateLimitGroup,
    apply_yaml_aliases,
    normalize_rate_limit_group,
)

logger = logging.getLogger(__name__)

GIB = 1024**3

STORAGE_QUOTA_CODE = "storage_quota_exceeded"
STORAGE_QUOTA_MESSAGE = (
    "Storage is full. Delete files or conversations to free space."
)

# The env prefix is the plan id, with the same names as the token budgets.
_STORAGE_ENV = {
    RateLimitGroup.FREE: "FREE_STORAGE_GB",
    RateLimitGroup.PRO: "PRO_STORAGE_GB",
    RateLimitGroup.PRO_PLUS: "PRO_PLUS_STORAGE_GB",
    RateLimitGroup.ULTRA: "ULTRA_STORAGE_GB",
}


def displayed_bytes(value: Any) -> int:
    """The counter as a client sees it: never negative."""
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


def gib(nbytes: int) -> float:
    """``nbytes`` in GiB, rounded to 2 decimals."""
    return round(nbytes / GIB, 2)


def _oid(user_id: str) -> Optional[ObjectId]:
    try:
        return ObjectId(user_id)
    except (InvalidId, TypeError):
        return None


def _settings() -> dict[str, Any]:
    raw = config.config.get("token_rate_limit", default={})
    return raw if isinstance(raw, dict) else {}


def _str_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def resolve_storage_group(user: User) -> RateLimitGroup:
    """The user's plan, resolved the same way as the token budget.

    The token feature flag is ignored: storage has its own switch. Aliases
    and the default plan still come from the token configuration, which is
    where those names are defined.
    """
    settings = _settings()
    default_group = normalize_rate_limit_group(
        settings.get("default_group", RateLimitGroup.FREE.value)
    )
    raw = str(user.rate_limit_group.value)
    if not raw.strip():
        raw = default_group.value
    return normalize_rate_limit_group(apply_yaml_aliases(raw, _str_dict(settings.get("aliases"))))


def _configured_cap(group: RateLimitGroup) -> tuple[bool, Optional[int]]:
    """``(variable was set, cap in bytes)``.

    A missing variable is ``(False, None)`` so the caller can fall back to
    the default plan. ``0`` or a bad value is ``(True, None)``: the plan was
    configured, and it has no cap. A negative number is ignored the same way.
    """
    name = _STORAGE_ENV.get(group)
    if name is None:
        return False, None
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return False, None
    try:
        gigabytes = int(raw.strip())
    except ValueError:
        logger.warning("Ignoring invalid integer value for %s.", name)
        return True, None
    if gigabytes <= 0:
        return True, None
    return True, gigabytes * GIB


def quota_bytes_for(user: User) -> Optional[int]:
    """The user's cap in bytes, or ``None`` when the plan has no cap.

    ``unlimited`` never has a cap. A plan with no variable of its own uses
    the default plan's cap. A plan whose variable is set, including to 0,
    does not fall back.
    """
    group = resolve_storage_group(user)
    if group is RateLimitGroup.UNLIMITED:
        return None
    configured, cap = _configured_cap(group)
    if configured:
        return cap
    default_group = normalize_rate_limit_group(
        _settings().get("default_group", RateLimitGroup.FREE.value)
    )
    if default_group is group or default_group is RateLimitGroup.UNLIMITED:
        return None
    _configured, fallback = _configured_cap(default_group)
    return fallback


def refusals_enabled() -> bool:
    """True when a write that would pass the cap is refused and signalled."""
    return config.FEATURE_STORAGE_QUOTA


def notice_frame(used_bytes: int, max_bytes: int) -> dict[str, Any]:
    """The legacy SSE frame for one refusal. Sent at most once per turn."""
    return {
        "type": "notice",
        "code": STORAGE_QUOTA_CODE,
        "used_bytes": displayed_bytes(used_bytes),
        "max_bytes": max_bytes,
    }


def storage_quota_http(used_bytes: int, max_bytes: int, needed_bytes: int) -> HTTPException:
    """403 for ``POST /artifacts``. No ``Retry-After``: waiting does not free space."""
    return HTTPException(
        status_code=403,
        detail={
            "code": STORAGE_QUOTA_CODE,
            "message": STORAGE_QUOTA_MESSAGE,
            "used_bytes": displayed_bytes(used_bytes),
            "max_bytes": max_bytes,
            "needed_bytes": needed_bytes,
        },
    )


async def _inc(user_id: str, nbytes: int) -> Optional[int]:
    """Add ``nbytes`` (negative to release). The stored value afterwards, or None."""
    oid = _oid(user_id)
    if oid is None or nbytes == 0:
        return None
    doc = await User.get_collection().find_one_and_update(
        {"_id": oid},
        {"$inc": {"storage_bytes_used": int(nbytes)}},
        projection={"storage_bytes_used": 1},
        return_document=ReturnDocument.AFTER,
    )
    if doc is None:
        return None
    return int(doc.get("storage_bytes_used") or 0)


def adjust_storage_sync(user_id: str, nbytes: int) -> None:
    """The sync tool-result write has no event loop it can await on.

    Uses the pymongo client Motor already holds. A missing connection or a
    user id that is not an ObjectId is skipped: the write still happens, and
    the reconcile command corrects the counter.
    """
    if nbytes == 0:
        return
    oid = _oid(user_id)
    database = async_mongo_manager.database
    if oid is None or database is None:
        return
    try:
        database.delegate["users"].update_one(
            {"_id": oid},
            {"$inc": {"storage_bytes_used": int(nbytes)}},
        )
    except Exception:
        logger.warning("storage.count_failed", exc_info=True)


async def count_storage(user_id: str, nbytes: int) -> None:
    """Add ``nbytes`` with no cap. For writes that must not be refused."""
    if nbytes <= 0:
        return
    await _inc(user_id, nbytes)


async def release_storage(user_id: str, nbytes: int) -> None:
    """Give ``nbytes`` back after a delete, a failed write, or a shrink.

    The stored value may go negative when the file was never counted. Reads
    clamp that at 0, and the initial fill corrects it.
    """
    if nbytes <= 0:
        return
    await _inc(user_id, -int(nbytes))


async def reserve_storage(
    user_id: str, nbytes: int, cap: Optional[int]
) -> tuple[bool, int]:
    """Claim ``nbytes`` before a write.

    Returns ``(accepted, used_bytes)``. ``accepted`` is false only when
    refusals are on, ``cap`` is set, and the write would pass it. The counter
    is unchanged then, and ``used_bytes`` is what is stored now. Otherwise the
    bytes are counted and ``used_bytes`` is the counter afterwards.

    A user id that is not an ObjectId cannot be charged. The write is allowed:
    production ids are ObjectIds, and a test id must not fail the turn.

    A missing counter counts as 0, but only when ``nbytes`` itself fits in the
    cap. Matching every missing field would let one oversized write through.
    """
    if nbytes <= 0:
        return True, 0
    oid = _oid(user_id)
    if oid is None:
        return True, 0
    if not refusals_enabled() or cap is None:
        used = await _inc(user_id, nbytes)
        return True, displayed_bytes(used)

    room = cap - nbytes
    clauses: list[dict[str, Any]] = [{"storage_bytes_used": {"$lte": room}}]
    if nbytes <= cap:
        clauses.append({"storage_bytes_used": {"$exists": False}})
    doc = await User.get_collection().find_one_and_update(
        {"_id": oid, "$or": clauses},
        {"$inc": {"storage_bytes_used": int(nbytes)}},
        projection={"storage_bytes_used": 1},
        return_document=ReturnDocument.AFTER,
    )
    if doc is not None:
        return True, displayed_bytes(doc.get("storage_bytes_used"))
    current = await User.get_collection().find_one(
        {"_id": oid}, projection={"storage_bytes_used": 1}
    )
    if current is None:
        # No row to charge. Don't block the write on a missing user.
        return True, 0
    return False, displayed_bytes(current.get("storage_bytes_used"))


async def mark_storage_notice(user_id: str) -> None:
    """Remember a refusal that happened after the client's stream had closed.

    The next turn sends the notice and clears the flag. While the user is at
    the cap, the turn sends it anyway, so a missed clear still arrives.
    """
    if not refusals_enabled():
        return
    oid = _oid(user_id)
    if oid is None:
        return
    await User.get_collection().update_one(
        {"_id": oid}, {"$set": {"storage_notice_pending": True}}
    )


async def opening_storage_notice(user_id: str) -> Optional[dict[str, Any]]:
    """The notice for a turn that starts with the user full, or with one pending.

    ``None`` when refusals are off, the plan has no cap, or there is room and
    nothing is pending. A pending flag is cleared here, including when the
    plan no longer has a cap.
    """
    if not refusals_enabled():
        return None
    oid = _oid(user_id)
    if oid is None:
        return None
    user = await User.find_by_id(user_id)
    if user is None:
        return None
    cap = quota_bytes_for(user)
    used = displayed_bytes(user.storage_bytes_used)
    pending = bool(user.storage_notice_pending)
    if pending:
        await User.get_collection().update_one(
            {"_id": oid}, {"$set": {"storage_notice_pending": False}}
        )
    if cap is None:
        return None
    if not pending and used < cap:
        return None
    return notice_frame(used, cap)


async def read_storage_listing(user_id: str) -> tuple[int, int]:
    """``(counter_read, listed_bytes)`` for ``users/{id}/``. Does not write.

    The counter is read before the listing. A write during the listing is
    then kept by applying ``listed - counter_read`` with ``$inc``.
    """
    oid = _oid(user_id)
    if oid is None:
        raise ValueError("not a user id")
    doc = await User.get_collection().find_one(
        {"_id": oid}, projection={"storage_bytes_used": 1}
    )
    if doc is None:
        raise ValueError("no user")
    counter = int(doc.get("storage_bytes_used") or 0)
    listed = await storage_service.sum_prefix(f"users/{user_id}/")
    return counter, listed


async def apply_storage_listing_delta(user_id: str, counter_read: int, listed: int) -> None:
    """Add ``listed - counter_read`` to the counter. A zero difference writes nothing."""
    oid = _oid(user_id)
    if oid is None:
        raise ValueError("not a user id")
    delta = listed - counter_read
    if delta == 0:
        return
    await User.get_collection().update_one(
        {"_id": oid},
        {"$inc": {"storage_bytes_used": delta}},
    )


async def get_storage_usage_summary(user: User) -> dict[str, Any]:
    """The usage payload for ``GET /users/me/storage-usage``."""
    group = resolve_storage_group(user)
    used = displayed_bytes(user.storage_bytes_used)
    cap = quota_bytes_for(user)
    if cap is None:
        return {
            "unlimited": True,
            "rate_limit_group": group.value,
            "used_bytes": used,
            "max_bytes": None,
            "remaining_bytes": None,
            "used_ratio": None,
            "used_gb": gib(used),
            "max_gb": None,
        }
    remaining = max(cap - used, 0)
    return {
        "unlimited": False,
        "rate_limit_group": group.value,
        "used_bytes": used,
        "max_bytes": cap,
        "remaining_bytes": remaining,
        "used_ratio": min(used / cap, 1.0),
        "used_gb": gib(used),
        "max_gb": gib(cap),
    }
