from __future__ import annotations

import calendar
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from bson import ObjectId
from fastapi import HTTPException
from pymongo import ReturnDocument

from src.config import config
from src.database.models.user import User
from src.schemas.rate_limit import (
    RateLimitGroup,
    apply_yaml_aliases,
    normalize_rate_limit_group,
)
from src.utils.helpers import str_token_counter


def _rate_limit_settings() -> Dict[str, Any]:
    raw = config.get("token_rate_limit", default={})
    return raw if isinstance(raw, dict) else {}


def _add_months(dt: datetime, months: int) -> datetime:
    month_index = (dt.month - 1) + months
    year = dt.year + (month_index // 12)
    month = (month_index % 12) + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


def _str_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _to_utc_datetime(value: Any) -> Optional[datetime]:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _resolve_policy_for_user(user: User) -> Tuple[Optional[RateLimitGroup], Optional[Dict[str, Any]]]:
    settings = _rate_limit_settings()
    if not settings.get("enabled", False):
        return None, None

    groups = _str_dict(settings.get("groups"))
    yaml_aliases = _str_dict(settings.get("aliases"))

    default_group = normalize_rate_limit_group(
        settings.get("default_group", RateLimitGroup.EVE_FREE.value)
    )

    raw = str(user.rate_limit_group.value)
    if not raw.strip():
        raw = default_group.value
    after_yaml = apply_yaml_aliases(raw, yaml_aliases)
    canonical = normalize_rate_limit_group(after_yaml)

    policy = groups.get(canonical.value)
    if isinstance(policy, dict):
        return canonical, policy

    fallback = groups.get(default_group.value)
    if isinstance(fallback, dict):
        return default_group, fallback

    return None, None


async def _ensure_active_window(user: User, policy: Dict[str, Any]) -> None:
    """Normalise tz in memory only, roll the period over with a compare-and-set.

    The Motor client is not ``tz_aware``, so datetimes load naive while
    ``_to_utc_datetime`` returns aware ones: comparing them is always
    unequal. That used to trip a ``user.save()`` (a full ``replace_one`` of
    the document) on every single request, which raced and clobbered the
    atomic ``$inc`` in ``consume_tokens_for_user`` and any other field a
    concurrent write touched. Normalisation here never writes; only an
    actual rollover does, and it does it with a filter keyed on the exact
    ``rate_limit_period_end`` this call loaded, so two concurrent requests
    that both see an expired window only let one of them win the roll.
    """
    period_months = max(int(policy.get("period_months", 1) or 1), 1)
    now = datetime.now(timezone.utc)

    raw_end = user.rate_limit_period_end
    start = _to_utc_datetime(user.rate_limit_period_start)
    end = _to_utc_datetime(raw_end)
    used = int(user.rate_limit_tokens_used or 0)

    user.rate_limit_period_start = start
    user.rate_limit_period_end = end

    if start is None or end is None or used < 0 or now >= end:
        new_start = now
        new_end = _add_months(now, period_months)
        result = await User.get_collection().update_one(
            {"_id": ObjectId(user.id), "rate_limit_period_end": raw_end},
            {
                "$set": {
                    "rate_limit_period_start": new_start,
                    "rate_limit_period_end": new_end,
                    "rate_limit_tokens_used": 0,
                }
            },
        )
        if result.matched_count:
            user.rate_limit_period_start = new_start
            user.rate_limit_period_end = new_end
            user.rate_limit_tokens_used = 0
        else:
            # Another request rolled the window over first: reload what it
            # wrote instead of trusting the stale copy still in memory.
            doc = await User.get_collection().find_one(
                {"_id": ObjectId(user.id)},
                projection={
                    "rate_limit_period_start": 1,
                    "rate_limit_period_end": 1,
                    "rate_limit_tokens_used": 1,
                },
            )
            if doc:
                user.rate_limit_period_start = _to_utc_datetime(doc.get("rate_limit_period_start"))
                user.rate_limit_period_end = _to_utc_datetime(doc.get("rate_limit_period_end"))
                user.rate_limit_tokens_used = int(doc.get("rate_limit_tokens_used") or 0)


def _policy_cap_tokens(policy: Dict[str, Any]) -> int:
    """Non-negative configured cap from a group policy dict (0 means no numeric cap)."""
    return max(int(policy.get("max_tokens", 0) or 0), 0)


def _token_usage_dict(
    *,
    unlimited: bool,
    rate_limit_group: str,
    used_tokens: int,
    max_tokens: Optional[int],
    remaining_tokens: Optional[int],
    used_ratio: Optional[float],
    remaining_ratio: Optional[float],
    user: User,
) -> Dict[str, Any]:
    return {
        "unlimited": unlimited,
        "rate_limit_group": rate_limit_group,
        "used_tokens": used_tokens,
        "max_tokens": max_tokens,
        "remaining_tokens": remaining_tokens,
        "used_ratio": used_ratio,
        "remaining_ratio": remaining_ratio,
        "period_start": _to_utc_datetime(user.rate_limit_period_start),
        "period_end": _to_utc_datetime(user.rate_limit_period_end),
    }


def count_tokens_for_texts(*texts: str) -> int:
    """Count the tokens the budget charges for one exchange.

    Scope of the cap, decided for the 0.1.0 opening and deliberately narrow:

    * Only what callers pass here is counted, and every call site passes exactly
      the user query and the final answer text. Retrieved context, the system
      prompt and the conversation history sent to the model are NOT counted, so
      the number is always smaller than what the provider actually processes.
      The cap bounds how much answer text a user can make the service produce,
      it is not a proxy for model cost or for infrastructure load.
    * There is one window and one cap per group: ``max_tokens`` over
      ``period_months`` months (minimum one month, see ``_ensure_active_window``).
      A monthly ceiling has no burst control, so the whole budget can be spent in
      a single day.
    * A daily or burst cap would need a second window key on the policy plus a
      second check in ``_ensure_active_window`` and ``enforce_token_budget_or_raise``,
      i.e. a schema change. That is out of scope for this sprint. The only lever
      that exists today is a smaller ``max_tokens``.
    """
    return sum(str_token_counter(text or "") for text in texts if text is not None)


@dataclass(frozen=True)
class TokenBudgetExceeded:
    """A resolved, exhausted budget: enough to build both the 429 body and the retry hint."""

    group: str
    max_tokens: int
    used_tokens: int
    reset_at: Optional[datetime]

    @property
    def message(self) -> str:
        detail = (
            f"Token budget exceeded for group '{self.group}'. "
            f"Limit is {self.max_tokens} tokens per period."
        )
        if self.reset_at is not None:
            detail += f" Resets at {self.reset_at.astimezone(timezone.utc).isoformat()}."
        return detail

    def retry_after_seconds(self, now: Optional[datetime] = None) -> Optional[int]:
        if self.reset_at is None:
            return None
        now = now or datetime.now(timezone.utc)
        return max(1, math.ceil((self.reset_at - now).total_seconds()))


def _retry_after_headers(exceeded: TokenBudgetExceeded) -> Optional[Dict[str, str]]:
    retry_after = exceeded.retry_after_seconds()
    return {"Retry-After": str(retry_after)} if retry_after is not None else None


async def check_token_budget(user: User) -> Optional[TokenBudgetExceeded]:
    """Resolve today's policy and window, and say whether the user is over the cap.

    Read-only from the caller's point of view: ``_ensure_active_window`` may
    roll an expired period over (a write, but never a full-document save),
    nothing else here writes. Callers that must charge tokens still call
    ``consume_tokens_for_user`` afterwards.
    """
    group, policy = _resolve_policy_for_user(user)
    if not policy or group is None:
        return None

    await _ensure_active_window(user, policy)
    max_tokens = _policy_cap_tokens(policy)
    used = int(user.rate_limit_tokens_used or 0)

    if max_tokens <= 0 or used < max_tokens:
        return None

    return TokenBudgetExceeded(
        group=group.value,
        max_tokens=max_tokens,
        used_tokens=used,
        reset_at=_to_utc_datetime(user.rate_limit_period_end),
    )


async def enforce_token_budget_or_raise(user: User) -> None:
    """Same 429 detail text the eleven chat call sites have always raised, now with Retry-After."""
    exceeded = await check_token_budget(user)
    if exceeded is None:
        return
    raise HTTPException(
        status_code=429,
        detail=exceeded.message,
        headers=_retry_after_headers(exceeded),
    )


async def consume_tokens_for_user(user: User, token_count: int) -> None:
    """Charge tokens with an atomic ``$inc``, never a full-document ``save()``.

    A ``save()`` here would race and silently undo any concurrent ``$inc``
    elsewhere on the same document (``private_document_count``) or a
    back-office edit of ``rate_limit_group`` / ``approval_status`` made
    between this function's read and its write.
    """
    if token_count <= 0:
        return

    _group, policy = _resolve_policy_for_user(user)
    if not policy:
        return

    await _ensure_active_window(user, policy)
    doc = await User.get_collection().find_one_and_update(
        {"_id": ObjectId(user.id)},
        {"$inc": {"rate_limit_tokens_used": int(token_count)}},
        projection={"rate_limit_tokens_used": 1},
        return_document=ReturnDocument.AFTER,
    )
    if doc:
        user.rate_limit_tokens_used = int(doc.get("rate_limit_tokens_used") or 0)


async def reserve_token_budget(
    user: User, estimated_tokens: int
) -> Tuple[Optional[TokenBudgetExceeded], int]:
    """Atomically claim ``estimated_tokens`` against the cap before any upstream call.

    ``check_token_budget`` on its own is a read, and a caller that forwards to
    an upstream only after reading it is check-then-act: every request in a
    concurrent batch can read the same stale ``used`` value and all get
    admitted, so the cap is bypassed by however many requests the caller fires
    at once, not just by the single soft-cap crossing the design accepts. This
    folds the check and the claim into one conditional ``$inc``: the filter
    only matches while ``used`` is still under the cap, so under concurrency
    each successful reservation is what the next one's filter sees, and at
    most one reservation can land once the cap is already met (mirroring the
    accepted single-request soft cap instead of multiplying it by whatever
    concurrency the caller chooses).

    Returns ``(None, 0)`` when no policy applies (nothing to reserve),
    ``(TokenBudgetExceeded, 0)`` when the cap is already met (nothing
    reserved, caller must not proceed), or ``(None, reserved)`` on a
    successful claim. The caller must always settle ``reserved`` afterwards
    with ``settle_reserved_tokens``, refunding it in full on any no-charge
    path.
    """
    group, policy = _resolve_policy_for_user(user)
    if not policy or group is None:
        return None, 0

    await _ensure_active_window(user, policy)
    max_tokens = _policy_cap_tokens(policy)
    if max_tokens <= 0:
        return None, 0

    used = int(user.rate_limit_tokens_used or 0)
    if used >= max_tokens:
        return (
            TokenBudgetExceeded(
                group=group.value,
                max_tokens=max_tokens,
                used_tokens=used,
                reset_at=_to_utc_datetime(user.rate_limit_period_end),
            ),
            0,
        )

    reserve = max(int(estimated_tokens), 1)
    doc = await User.get_collection().find_one_and_update(
        {
            "_id": ObjectId(user.id),
            "rate_limit_period_end": user.rate_limit_period_end,
            "rate_limit_tokens_used": {"$lt": max_tokens},
        },
        {"$inc": {"rate_limit_tokens_used": reserve}},
        projection={"rate_limit_tokens_used": 1},
        return_document=ReturnDocument.AFTER,
    )
    if doc is None:
        # Either a concurrent reservation already pushed `used` to the cap, or
        # the window rolled over between the read above and here (the period
        # filter stopped matching): refuse rather than reserve against a stale
        # window. Reload so the 429 body reports the real current state.
        refreshed = await User.get_collection().find_one(
            {"_id": ObjectId(user.id)},
            projection={"rate_limit_tokens_used": 1, "rate_limit_period_end": 1},
        )
        used_now = int((refreshed or {}).get("rate_limit_tokens_used") or used)
        reset_at = _to_utc_datetime((refreshed or {}).get("rate_limit_period_end")) or _to_utc_datetime(
            user.rate_limit_period_end
        )
        return (
            TokenBudgetExceeded(
                group=group.value,
                max_tokens=max_tokens,
                used_tokens=used_now,
                reset_at=reset_at,
            ),
            0,
        )

    user.rate_limit_tokens_used = int(doc.get("rate_limit_tokens_used") or 0)
    return None, reserve


async def settle_reserved_tokens(user: User, reserved: int, actual: Optional[int]) -> None:
    """Reconcile a ``reserve_token_budget`` claim with what the request really cost.

    ``actual`` is ``None`` on every no-charge path (upstream 4xx/5xx, a
    connection error, or the budget already refused the request): the whole
    reservation is refunded. Otherwise the gap between the estimate reserved
    upfront and the real ``_tokens_to_bill`` amount is applied as a signed
    ``$inc`` (negative to refund, positive to top up), never a re-read of the
    counter that could lose a concurrent write.
    """
    if reserved <= 0:
        return

    delta = (actual if actual is not None else 0) - reserved
    if delta == 0:
        return

    doc = await User.get_collection().find_one_and_update(
        {"_id": ObjectId(user.id)},
        {"$inc": {"rate_limit_tokens_used": delta}},
        projection={"rate_limit_tokens_used": 1},
        return_document=ReturnDocument.AFTER,
    )
    if doc:
        user.rate_limit_tokens_used = int(doc.get("rate_limit_tokens_used") or 0)


async def get_token_usage_summary(user: User) -> Dict[str, Any]:
    """Token budget snapshot; same policy resolution as enforce/consume (see ``_resolve_policy_for_user``)."""
    group, policy = _resolve_policy_for_user(user)

    if not policy or group is None:
        return _token_usage_dict(
            unlimited=True,
            rate_limit_group=str(user.rate_limit_group.value),
            used_tokens=int(user.rate_limit_tokens_used or 0),
            max_tokens=None,
            remaining_tokens=None,
            used_ratio=None,
            remaining_ratio=None,
            user=user,
        )

    await _ensure_active_window(user, policy)
    cap = _policy_cap_tokens(policy)
    used = int(user.rate_limit_tokens_used or 0)

    if cap <= 0:
        return _token_usage_dict(
            unlimited=True,
            rate_limit_group=str(group.value),
            used_tokens=used,
            max_tokens=None,
            remaining_tokens=None,
            used_ratio=None,
            remaining_ratio=None,
            user=user,
        )

    remaining = max(cap - used, 0)
    used_ratio = min(max(used / cap, 0.0), 1.0)
    remaining_ratio = min(max(remaining / cap, 0.0), 1.0)

    return _token_usage_dict(
        unlimited=False,
        rate_limit_group=str(group.value),
        used_tokens=used,
        max_tokens=cap,
        remaining_tokens=remaining,
        used_ratio=float(used_ratio),
        remaining_ratio=float(remaining_ratio),
        user=user,
    )
