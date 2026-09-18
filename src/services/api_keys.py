"""Self-service API key management: create, list, revoke.

Routes in ``src/routers/user.py`` stay thin; this module holds the algorithms
(clamp, throttle, cap, cascade revoke) and the audit trail. See
``/private/tmp/.../design-backend.md`` section 3 for the design this
implements; the plan file's API contract wins on any conflict.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from bson import ObjectId
from fastapi import HTTPException

from src.config import (
    API_KEY_CREATE_MAX_PER_HOUR,
    API_KEY_DEFAULT_EXPIRES_IN_DAYS,
    API_KEY_MAX_ACTIVE_PER_USER,
)
from src.database.models.api_key import ApiKey
from src.middlewares.auth import AUTH_TYPE_API_KEY, AuthContext
from src.schemas.auth import ApiKeyItem, ApiKeyParent, CreateApiKeyRequest, CreateApiKeyResponse
from src.services.auth import generate_api_key

logger = logging.getLogger(__name__)
# IDs, enums and timestamps only: never a key name, a raw token, a hash or a
# suffix. Same values go in ``extra={"audit": {...}}`` for structured sinks.
audit = logging.getLogger("eve.audit")

_LIST_MAX = 200


def _limit_reached_detail() -> dict:
    return {
        "code": "api_key_limit_reached",
        "message": (
            f"You already have {API_KEY_MAX_ACTIVE_PER_USER} active API keys. "
            "Revoke one before creating another."
        ),
        "limit": API_KEY_MAX_ACTIVE_PER_USER,
    }


def _throttled_detail() -> dict:
    return {
        "code": "api_key_create_rate_limited",
        "message": "Too many API keys created recently. Try again later.",
    }


def _as_utc(value: Optional[datetime]) -> Optional[datetime]:
    """The Motor client is not tz_aware, so a value loaded from Mongo comes
    back naive even though it was stored as UTC. Normalise before comparing.
    """
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=timezone.utc)


def _audit_create_refused(
    user_id: str, reason: str, *, created_by_key_id: Optional[str] = None
) -> None:
    audit.info(
        "api_key.create_refused user_id=%s reason=%s created_by_key_id=%s",
        user_id,
        reason,
        created_by_key_id,
        extra={
            "audit": {
                "event": "api_key.create_refused",
                "user_id": user_id,
                "reason": reason,
                "created_by_key_id": created_by_key_id,
            }
        },
    )


async def _count_active(user_id: str) -> int:
    """Active means neither revoked nor expired: both must be excluded from the cap."""
    now = datetime.now(timezone.utc)
    return await ApiKey.count_documents(
        {
            "user_id": user_id,
            "revoked_at": None,
            "$or": [{"expires_at": None}, {"expires_at": {"$gt": now}}],
        }
    )


async def _is_among_first_active(user_id: str, key_id: str, cap: int) -> bool:
    """True if ``key_id`` is one of the first ``cap`` active keys for the
    user, ordered by ``_id`` (insertion order). Every concurrent caller runs
    this same query and reaches the same verdict, so the rows that survive a
    burst of concurrent creates are always the earliest ``cap`` of them
    instead of whichever request happened to run its recount last.
    """
    if cap <= 0:
        return False
    now = datetime.now(timezone.utc)
    cursor = (
        ApiKey.get_collection()
        .find(
            {
                "user_id": user_id,
                "revoked_at": None,
                "$or": [{"expires_at": None}, {"expires_at": {"$gt": now}}],
            },
            {"_id": 1},
        )
        .sort("_id", 1)
        .limit(cap)
    )
    survivor_ids = {str(doc["_id"]) async for doc in cursor}
    return key_id in survivor_ids


async def create_api_key(
    request: Optional[CreateApiKeyRequest], auth: AuthContext
) -> CreateApiKeyResponse:
    """Create a key, clamped to the parent's expiry and to the user's cap.

    Optimistic insert then verify: no Mongo transactions are available on
    every deployment target, so the key is inserted first and the cap and
    parent checks run again afterwards. A survivor that loses either check is
    deleted before the response goes out, so the table itself never holds more
    than the cap and never holds a child of an already-revoked parent for
    longer than one round trip.
    """
    user = auth.user
    req = request or CreateApiKeyRequest()
    now = datetime.now(timezone.utc)

    parent_key_id = auth.principal.api_key_id
    created_via = AUTH_TYPE_API_KEY if parent_key_id else auth.principal.auth_type
    parent: Optional[ApiKey] = None
    if parent_key_id:
        parent = await ApiKey.find_by_id(parent_key_id)
        # The calling key was valid at auth time but is gone or revoked now:
        # answer the same as an invalid credential, not a 500.
        if parent is None or parent.user_id != user.id or parent.revoked_at is not None:
            _audit_create_refused(user.id, "parent_revoked", created_by_key_id=parent_key_id)
            raise HTTPException(status_code=401, detail="Invalid or revoked API key")

    expires_at = req.resolve_expires_at(now=now, default_days=API_KEY_DEFAULT_EXPIRES_IN_DAYS)
    parent_expires_at = _as_utc(parent.expires_at) if parent is not None else None
    if parent_expires_at is not None:
        expires_at = parent_expires_at if expires_at is None else min(expires_at, parent_expires_at)

    if API_KEY_CREATE_MAX_PER_HOUR > 0:
        hour_ago = now - timedelta(hours=1)
        recent = await ApiKey.count_documents(
            {"user_id": user.id, "timestamp": {"$gte": hour_ago}}
        )
        if recent >= API_KEY_CREATE_MAX_PER_HOUR:
            _audit_create_refused(user.id, "throttled")
            raise HTTPException(
                status_code=429,
                detail=_throttled_detail(),
                headers={"Retry-After": "3600"},
            )

    if API_KEY_MAX_ACTIVE_PER_USER <= 0 or await _count_active(user.id) >= API_KEY_MAX_ACTIVE_PER_USER:
        _audit_create_refused(user.id, "limit_reached")
        raise HTTPException(status_code=409, detail=_limit_reached_detail())

    name = req.name or f"Key created {now:%Y-%m-%d %H:%M} UTC"
    raw_token, key_hash = generate_api_key()
    api_key = await ApiKey.create(
        user_id=user.id,
        name=name,
        key_hash=key_hash,
        expires_at=expires_at,
        token_suffix=raw_token[-6:],
        created_by_key_id=parent_key_id,
        created_via=created_via,
    )

    # Verify again post-insert: a concurrent create or a concurrent revoke of
    # the parent may have invalidated this insert between the checks above and
    # now. A plain recount is not enough: every concurrent inserter would see
    # itself as "one of the over-cap ones" and delete its own row, so a burst
    # of concurrent creates could spuriously leave zero survivors even though
    # the cap was never reached. Instead, rank the currently active keys by
    # _id (insertion order) and keep only this row if it is among the first
    # `cap` of them; everyone agrees on the same ranking, so exactly
    # min(active count, cap) rows survive and the earliest inserts win.
    parent_still_active = True
    if parent is not None:
        fresh_parent = await ApiKey.find_by_id(parent.id)
        parent_still_active = fresh_parent is not None and fresh_parent.revoked_at is None

    within_cap = await _is_among_first_active(user.id, api_key.id, API_KEY_MAX_ACTIVE_PER_USER)

    if not parent_still_active or not within_cap:
        await ApiKey.get_collection().delete_one({"_id": ObjectId(api_key.id)})
        if not parent_still_active:
            _audit_create_refused(user.id, "parent_revoked", created_by_key_id=parent_key_id)
            raise HTTPException(status_code=401, detail="Invalid or revoked API key")
        _audit_create_refused(user.id, "limit_reached")
        raise HTTPException(status_code=409, detail=_limit_reached_detail())

    expires_at_iso = expires_at.isoformat() if expires_at else None
    audit.info(
        "api_key.created user_id=%s key_id=%s created_by_key_id=%s created_via=%s expires_at=%s",
        user.id,
        api_key.id,
        parent_key_id,
        created_via,
        expires_at_iso,
        extra={
            "audit": {
                "event": "api_key.created",
                "user_id": user.id,
                "key_id": api_key.id,
                "created_by_key_id": parent_key_id,
                "created_via": created_via,
                "expires_at": expires_at_iso,
            }
        },
    )

    parent_summary = _parent_summary(parent, now) if parent is not None else None
    return CreateApiKeyResponse(
        id=api_key.id,
        name=api_key.name,
        token=raw_token,
        token_suffix=api_key.token_suffix,
        status=api_key.status_at(now),
        created_at=api_key.timestamp,
        expires_at=api_key.expires_at,
        revoked_at=None,
        last_used_at=None,
        created_via=created_via,
        created_by_key_id=parent_key_id,
        created_by=parent_summary,
        is_current=False,
    )


def _parent_summary(parent: ApiKey, now: datetime) -> ApiKeyParent:
    return ApiKeyParent(
        id=parent.id, name=parent.name, token_suffix=parent.token_suffix, status=parent.status_at(now)
    )


async def list_api_keys(auth: AuthContext, *, include_revoked: bool) -> list[ApiKeyItem]:
    """One query for the page, plus at most one ``$in`` for parents not on it."""
    user = auth.user
    now = datetime.now(timezone.utc)

    query: dict = {"user_id": user.id}
    if not include_revoked:
        query["revoked_at"] = None
    docs = (
        await ApiKey.get_collection()
        .find(query)
        .sort("timestamp", -1)
        .limit(_LIST_MAX)
        .to_list(length=_LIST_MAX)
    )
    keys = [ApiKey.from_dict(doc) for doc in docs]
    by_id = {k.id: k for k in keys}

    missing_parent_ids = {
        k.created_by_key_id for k in keys if k.created_by_key_id and k.created_by_key_id not in by_id
    }
    parents: dict[str, ApiKey] = {}
    if missing_parent_ids:
        parent_docs = await ApiKey.get_collection().find(
            {"_id": {"$in": [ObjectId(pid) for pid in missing_parent_ids]}, "user_id": user.id}
        ).to_list(length=None)
        parents = {p.id: p for p in (ApiKey.from_dict(d) for d in parent_docs)}

    current_key_id = auth.principal.api_key_id
    items = []
    for k in keys:
        parent_summary = None
        if k.created_by_key_id:
            parent = by_id.get(k.created_by_key_id) or parents.get(k.created_by_key_id)
            if parent is not None:
                parent_summary = _parent_summary(parent, now)
        items.append(
            ApiKeyItem(
                id=k.id,
                name=k.name,
                token_suffix=k.token_suffix,
                status=k.status_at(now),
                created_at=k.timestamp,
                expires_at=k.expires_at,
                revoked_at=k.revoked_at,
                last_used_at=k.last_used_at,
                created_via=k.created_via,
                created_by_key_id=k.created_by_key_id,
                created_by=parent_summary,
                is_current=(k.id == current_key_id),
            )
        )
    return items


async def _load_user_keys(user_id: str) -> dict[str, dict]:
    rows = await ApiKey.get_collection().find(
        {"user_id": user_id}, projection={"created_by_key_id": 1, "revoked_at": 1}
    ).to_list(length=None)
    return {str(r["_id"]): r for r in rows}


def _bfs_descendants(rows: dict[str, dict], root_id: str) -> set[str]:
    """The root plus every key transitively created by it, walking revoked nodes too."""
    children_by_parent: dict[str, list[str]] = {}
    for key_id, row in rows.items():
        parent_id = row.get("created_by_key_id")
        if parent_id:
            children_by_parent.setdefault(parent_id, []).append(key_id)

    visited: set[str] = set()
    queue = [root_id]
    while queue:
        current = queue.pop()
        if current in visited:
            continue
        visited.add(current)
        queue.extend(children_by_parent.get(current, []))
    return visited


async def revoke_api_key(key_id: str, auth: AuthContext) -> None:
    """Revoke a key and every key it (transitively) created.

    One query filtered by ``user_id`` and one answer for "not found": a
    missing id and somebody else's id look identical from here, so revoking a
    foreign key is not an oracle for its existence.

    The cascade runs twice: the second pass catches a child whose create
    request raced this revoke and landed after the first pass already walked
    the tree.
    """
    user = auth.user
    try:
        target_oid = ObjectId(key_id)
    except Exception:
        raise HTTPException(status_code=404, detail="API key not found")
    target_id = str(target_oid)

    now = datetime.now(timezone.utc)
    cascade_ids: set[str] = set()
    newly_revoked = 0
    already_revoked = False

    for pass_index in range(2):
        rows = await _load_user_keys(user.id)
        if target_id not in rows:
            if pass_index == 0:
                raise HTTPException(status_code=404, detail="API key not found")
            break
        if pass_index == 0:
            already_revoked = rows[target_id]["revoked_at"] is not None

        ids = _bfs_descendants(rows, target_id)
        cascade_ids |= ids
        result = await ApiKey.get_collection().update_many(
            {"_id": {"$in": [ObjectId(i) for i in ids]}, "user_id": user.id, "revoked_at": None},
            {"$set": {"revoked_at": now}},
        )
        newly_revoked += result.modified_count

    audit.info(
        "api_key.revoked user_id=%s key_id=%s actor_key_id=%s actor_via=%s "
        "cascade_ids=%s newly_revoked=%s already_revoked=%s",
        user.id,
        target_id,
        auth.principal.api_key_id,
        auth.principal.auth_type,
        sorted(cascade_ids),
        newly_revoked,
        already_revoked,
        extra={
            "audit": {
                "event": "api_key.revoked",
                "user_id": user.id,
                "key_id": target_id,
                "actor_key_id": auth.principal.api_key_id,
                "actor_via": auth.principal.auth_type,
                "cascade_ids": sorted(cascade_ids),
                "newly_revoked": newly_revoked,
                "already_revoked": already_revoked,
            }
        },
    )
