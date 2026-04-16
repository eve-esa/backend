from __future__ import annotations

import calendar
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from fastapi import HTTPException

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
    period_months = max(int(policy.get("period_months", 1) or 1), 1)
    now = datetime.now(timezone.utc)

    start = _to_utc_datetime(user.rate_limit_period_start)
    end = _to_utc_datetime(user.rate_limit_period_end)
    used = int(user.rate_limit_tokens_used or 0)
    needs_normalization_save = False

    if user.rate_limit_period_start is not None and start != user.rate_limit_period_start:
        user.rate_limit_period_start = start
        needs_normalization_save = True
    if user.rate_limit_period_end is not None and end != user.rate_limit_period_end:
        user.rate_limit_period_end = end
        needs_normalization_save = True

    if (
        start is None
        or end is None
        or used < 0
        or now >= end
    ):
        user.rate_limit_period_start = now
        user.rate_limit_period_end = _add_months(now, period_months)
        user.rate_limit_tokens_used = 0
        await user.save()
    elif needs_normalization_save:
        await user.save()


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
    return sum(str_token_counter(text or "") for text in texts if text is not None)


async def enforce_token_budget_or_raise(user: User) -> None:
    group, policy = _resolve_policy_for_user(user)
    if not policy or group is None:
        return

    await _ensure_active_window(user, policy)
    max_tokens = _policy_cap_tokens(policy)
    used = int(user.rate_limit_tokens_used or 0)

    if max_tokens <= 0:
        return

    if used >= max_tokens:
        reset_at = _to_utc_datetime(user.rate_limit_period_end)
        reset_str = (
            reset_at.astimezone(timezone.utc).isoformat()
            if isinstance(reset_at, datetime)
            else None
        )
        detail = (
            f"Token budget exceeded for group '{group.value}'. "
            f"Limit is {max_tokens} tokens per period."
        )
        if reset_str:
            detail += f" Resets at {reset_str}."
        raise HTTPException(status_code=429, detail=detail)


async def consume_tokens_for_user(user: User, token_count: int) -> None:
    if token_count <= 0:
        return

    _group, policy = _resolve_policy_for_user(user)
    if not policy:
        return

    await _ensure_active_window(user, policy)
    user.rate_limit_tokens_used = int(user.rate_limit_tokens_used or 0) + int(token_count)
    await user.save()


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
