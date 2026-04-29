from enum import Enum
from typing import Any, Mapping


class RateLimitGroup(str, Enum):
    """Canonical EVE rate-limit tiers (stored on user and used as config keys)."""

    EVE_FREE = "eve_free"
    EVE_STANDARD = "eve_standard"
    EVE_ADVANCED = "eve_advanced"
    EVE_ENTERPRISE = "eve_enterprise"


# Short / legacy / marketing labels → canonical enum (code defaults; YAML may extend).
_GROUP_ALIASES: dict[str, RateLimitGroup] = {
    "default": RateLimitGroup.EVE_FREE,
    "free": RateLimitGroup.EVE_FREE,
    "pro": RateLimitGroup.EVE_STANDARD,
    "pro+": RateLimitGroup.EVE_ADVANCED,
    "ultra": RateLimitGroup.EVE_ENTERPRISE,
}


def _alias_lookup(key: str) -> RateLimitGroup | None:
    k = key.strip().lower()
    return _GROUP_ALIASES.get(k)


def normalize_rate_limit_group(value: object) -> RateLimitGroup:
    """Map storage/config strings to a canonical group; unknown → EVE_FREE."""
    s = str(value or "").strip()
    if not s:
        return RateLimitGroup.EVE_FREE
    lowered = s.lower()
    mapped = _alias_lookup(lowered)
    if mapped is not None:
        return mapped
    try:
        return RateLimitGroup(lowered)
    except ValueError:
        return RateLimitGroup.EVE_FREE


def apply_yaml_aliases(raw: str, yaml_aliases: Mapping[str, Any] | None) -> str:
    """Resolve config `token_rate_limit.aliases` (exact or case-insensitive key) → string for normalize."""
    if not yaml_aliases:
        return raw
    stripped = raw.strip()
    if stripped in yaml_aliases:
        return str(yaml_aliases[stripped])
    lower = stripped.lower()
    for key, target in yaml_aliases.items():
        if str(key).lower() == lower:
            return str(target)
    return raw
