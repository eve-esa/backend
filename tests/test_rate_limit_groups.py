"""Plan ids match FREE / PRO / PRO_PLUS / ULTRA. Old eve_* strings still resolve."""

import pytest

from src.database.models.user import User
from src.schemas.rate_limit import RateLimitGroup, normalize_rate_limit_group
from src.services.token_rate_limiter import _resolve_policy_for_user

_PLANS = {
    "free": (RateLimitGroup.FREE, 1),
    "pro": (RateLimitGroup.PRO, 2),
    "pro_plus": (RateLimitGroup.PRO_PLUS, 3),
    "ultra": (RateLimitGroup.ULTRA, 4),
}

_LEGACY = {
    "eve_free": RateLimitGroup.FREE,
    "default": RateLimitGroup.FREE,
    "eve_standard": RateLimitGroup.PRO,
    "eve_advanced": RateLimitGroup.PRO_PLUS,
    "pro+": RateLimitGroup.PRO_PLUS,
    "eve_enterprise": RateLimitGroup.ULTRA,
}


@pytest.mark.no_db
@pytest.mark.parametrize("raw,expected", list(_PLANS.items()))
def test_plan_name_matches_env_prefix(raw, expected):
    group, _cap = expected
    assert normalize_rate_limit_group(raw) is group


@pytest.mark.no_db
@pytest.mark.parametrize("raw,expected", list(_LEGACY.items()))
def test_legacy_plan_name_resolves(raw, expected):
    assert normalize_rate_limit_group(raw) is expected


@pytest.mark.no_db
def test_unknown_plan_name_is_free():
    assert normalize_rate_limit_group("nope") is RateLimitGroup.FREE
    assert normalize_rate_limit_group("") is RateLimitGroup.FREE


@pytest.mark.no_db
def test_stored_legacy_value_loads_as_the_new_plan():
    user = User(email="plan@example.com", rate_limit_group="eve_standard")
    assert user.rate_limit_group is RateLimitGroup.PRO


@pytest.mark.no_db
def test_each_env_plan_uses_its_own_cap(monkeypatch):
    settings = {
        "enabled": True,
        "default_group": "free",
        "groups": {name: {"max_tokens": cap} for name, (_group, cap) in _PLANS.items()},
    }
    monkeypatch.setattr(
        "src.services.token_rate_limiter._rate_limit_settings", lambda: settings
    )

    for name, (group, cap) in _PLANS.items():
        user = User(email=f"{name}@example.com", rate_limit_group=name)
        resolved, policy = _resolve_policy_for_user(user)
        assert resolved is group
        assert policy["max_tokens"] == cap


@pytest.mark.no_db
def test_legacy_config_key_still_selects_that_plan(monkeypatch):
    monkeypatch.setattr(
        "src.services.token_rate_limiter._rate_limit_settings",
        lambda: {
            "enabled": True,
            "default_group": "eve_free",
            "groups": {"eve_advanced": {"max_tokens": 50}},
        },
    )
    user = User(email="legacy@example.com", rate_limit_group="eve_advanced")
    resolved, policy = _resolve_policy_for_user(user)
    assert resolved is RateLimitGroup.PRO_PLUS
    assert policy["max_tokens"] == 50
