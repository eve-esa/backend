from enum import Enum
from typing import Any, Mapping


class RateLimitGroup(str, Enum):
    """Token plans. The value is the same word as the env prefix.

    ``FREE_TOKENS`` is ``free``, ``PRO_TOKENS`` is ``pro``,
    ``PRO_PLUS_TOKENS`` is ``pro_plus``, ``ULTRA_TOKENS`` is ``ultra``.
    ``unlimited`` has no env var and no cap: it never reads a policy from config.
    """

    FREE = "free"
    PRO = "pro"
    PRO_PLUS = "pro_plus"
    ULTRA = "ultra"
    UNLIMITED = "unlimited"


# Older rows and configs still use these. Read them as the plan above.
# Do not write them back: the next full document replace stores the enum value.
_LEGACY_GROUP_NAMES: dict[str, RateLimitGroup] = {
    "default": RateLimitGroup.FREE,
    "eve_free": RateLimitGroup.FREE,
    "eve_standard": RateLimitGroup.PRO,
    "eve_advanced": RateLimitGroup.PRO_PLUS,
    "eve_enterprise": RateLimitGroup.ULTRA,
    "pro+": RateLimitGroup.PRO_PLUS,
}


def config_keys_for_group(group: RateLimitGroup) -> tuple[str, ...]:
    """Canonical key first, then the old ``eve_*`` config key for that plan."""
    legacy = tuple(
        name
        for name, target in _LEGACY_GROUP_NAMES.items()
        if target is group and name.startswith("eve_")
    )
    return (group.value, *legacy)


def _alias_lookup(key: str) -> RateLimitGroup | None:
    return _LEGACY_GROUP_NAMES.get(key.strip().lower())


def normalize_rate_limit_group(value: object) -> RateLimitGroup:
    """Map storage or config strings to a plan. Unknown values become ``free``."""
    s = str(value or "").strip()
    if not s:
        return RateLimitGroup.FREE
    lowered = s.lower()
    mapped = _alias_lookup(lowered)
    if mapped is not None:
        return mapped
    try:
        return RateLimitGroup(lowered)
    except ValueError:
        return RateLimitGroup.FREE


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
