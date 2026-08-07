import importlib

import pytest

from src.config import Config, getenv_or


TOKEN_LIMIT_ENV_KEYS = (
    "TOKEN_RATE_LIMIT_ENABLED",
    "TOKEN_RATE_LIMIT_DEFAULT_GROUP",
    "TOKEN_RATE_LIMIT_ALIASES",
    "TOKEN_RATE_LIMIT_GROUPS",
    "FREE_TOKENS",
    "FREE_PERIOD_MONTHS",
    "PRO_TOKENS",
    "PRO_PERIOD_MONTHS",
    "PRO_PLUS_TOKENS",
    "PRO_PLUS_PERIOD_MONTHS",
    "ULTRA_TOKENS",
    "ULTRA_PERIOD_MONTHS",
)


def _clear_token_limit_env(monkeypatch):
    for key in TOKEN_LIMIT_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_token_rate_limit_groups_not_required_in_yaml(tmp_path, monkeypatch):
    _clear_token_limit_env(monkeypatch)

    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "token_rate_limit:",
                "  enabled: true",
                '  default_group: "eve_free"',
            ]
        ),
        encoding="utf-8",
    )

    cfg = Config(str(config_file))
    token_cfg = cfg.get("token_rate_limit", default={})

    assert token_cfg["enabled"] is True
    assert token_cfg["default_group"] == "eve_free"
    assert "groups" not in token_cfg


def test_token_rate_limit_env_overrides_yaml(tmp_path, monkeypatch):
    _clear_token_limit_env(monkeypatch)
    monkeypatch.setenv("TOKEN_RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("TOKEN_RATE_LIMIT_DEFAULT_GROUP", "eve_enterprise")
    monkeypatch.setenv("TOKEN_RATE_LIMIT_ALIASES", '{"vip":"eve_enterprise"}')
    monkeypatch.setenv(
        "TOKEN_RATE_LIMIT_GROUPS",
        '{"eve_enterprise":{"label":"VIP","max_tokens":300000,"period_months":1}}',
    )

    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "token_rate_limit:",
                "  enabled: true",
                '  default_group: "eve_free"',
                "  aliases:",
                '    free: "eve_free"',
            ]
        ),
        encoding="utf-8",
    )

    cfg = Config(str(config_file))
    token_cfg = cfg.get("token_rate_limit", default={})

    assert token_cfg["enabled"] is False
    assert token_cfg["default_group"] == "eve_enterprise"
    assert token_cfg["aliases"] == {"vip": "eve_enterprise"}
    assert token_cfg["groups"]["eve_enterprise"]["max_tokens"] == 300000


def test_token_rate_limit_flat_group_env_overrides(tmp_path, monkeypatch):
    _clear_token_limit_env(monkeypatch)
    monkeypatch.setenv(
        "TOKEN_RATE_LIMIT_GROUPS",
        '{"eve_free":{"label":"Free","max_tokens":100,"period_months":12}}',
    )
    monkeypatch.setenv("FREE_TOKENS", "1234")
    monkeypatch.setenv("FREE_PERIOD_MONTHS", "2")
    monkeypatch.setenv("PRO_PLUS_TOKENS", "9999")
    monkeypatch.setenv("PRO_PLUS_PERIOD_MONTHS", "3")

    config_file = tmp_path / "config.yaml"
    config_file.write_text("token_rate_limit:\n  enabled: true\n", encoding="utf-8")

    cfg = Config(str(config_file))
    token_cfg = cfg.get("token_rate_limit", default={})

    assert token_cfg["groups"]["eve_free"]["max_tokens"] == 1234
    assert token_cfg["groups"]["eve_free"]["period_months"] == 2
    assert token_cfg["groups"]["eve_advanced"]["max_tokens"] == 9999
    assert token_cfg["groups"]["eve_advanced"]["period_months"] == 3


# ── Blank-vs-absent env values ────────────────────────────────────────────────
# The infrastructure seeds intentionally-unused Secrets Manager secrets with a
# single space, because ECS cannot resolve a secret that has no version
# (infra/docs/RUNBOOK.md). `os.getenv(name, default)` returns that space, so the
# default never applies -- which is how a "disabled" JSC provider ended up
# enabled but keyless in production.


def test_getenv_or_treats_blank_as_absent(monkeypatch):
    monkeypatch.setenv("EVE_TEST_BLANK", " ")
    assert getenv_or("EVE_TEST_BLANK", "fallback") == "fallback"

    monkeypatch.setenv("EVE_TEST_BLANK", "")
    assert getenv_or("EVE_TEST_BLANK", "fallback") == "fallback"

    monkeypatch.delenv("EVE_TEST_BLANK", raising=False)
    assert getenv_or("EVE_TEST_BLANK", "fallback") == "fallback"


def test_getenv_or_strips_and_keeps_real_values(monkeypatch):
    monkeypatch.setenv("EVE_TEST_BLANK", "  real-value  ")
    assert getenv_or("EVE_TEST_BLANK", "fallback") == "real-value"


def test_placeholder_secret_leaves_jsc_key_empty(monkeypatch):
    """The whole point: a placeholder must read as unconfigured, not as a key.

    Exercises the real config module rather than monkeypatching the constant,
    which is what let this class of bug through the existing proxy tests.
    """
    monkeypatch.setenv("EVE_JSC_API_KEY", " ")
    reloaded = importlib.reload(importlib.import_module("src.config"))
    try:
        assert reloaded.EVE_JSC_API_KEY == ""
    finally:
        monkeypatch.delenv("EVE_JSC_API_KEY", raising=False)
        importlib.reload(reloaded)


# ── Environment identity ──────────────────────────────────────────────────────
# Terraform has always known dev/staging/prod (var.environment), but the value used to reach
# the container collapsed into a boolean IS_PROD, so dev and staging were indistinguishable
# from inside and /health could only answer "non-prod". The tri-state now travels intact and
# IS_PROD is derived from it.


def _reload_config(monkeypatch, **env):
    # src/config.py calls load_dotenv(override=True) at import, so a developer's local .env
    # wins over anything monkeypatch sets. Neutralise it for the reload, or these tests would
    # assert against whatever happens to be on the machine rather than against the code.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    for key in ("APP_ENVIRONMENT", "IS_PROD"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return importlib.reload(importlib.import_module("src.config"))


@pytest.mark.parametrize(
    "environment,expected_is_prod",
    [("dev", False), ("staging", False), ("prod", True)],
)
def test_is_prod_is_derived_from_app_environment(monkeypatch, environment, expected_is_prod):
    cfg = _reload_config(monkeypatch, APP_ENVIRONMENT=environment)
    try:
        assert cfg.APP_ENVIRONMENT == environment
        assert cfg.IS_PROD is expected_is_prod
    finally:
        _reload_config(monkeypatch)


def test_unknown_environment_does_not_read_as_production(monkeypatch):
    """A typo must not hand a container production behaviour."""
    cfg = _reload_config(monkeypatch, APP_ENVIRONMENT="Production")
    try:
        assert cfg.IS_PROD is False
    finally:
        _reload_config(monkeypatch)


def test_legacy_is_prod_still_honoured_during_the_rename(monkeypatch):
    """infra and code deploy independently; the fallback covers the window between them."""
    cfg = _reload_config(monkeypatch, IS_PROD="true")
    try:
        assert cfg.IS_PROD is True
        assert cfg.APP_ENVIRONMENT == "prod"
    finally:
        _reload_config(monkeypatch)


def test_app_environment_wins_over_legacy_is_prod(monkeypatch):
    cfg = _reload_config(monkeypatch, APP_ENVIRONMENT="staging", IS_PROD="true")
    try:
        assert cfg.APP_ENVIRONMENT == "staging"
        assert cfg.IS_PROD is False
    finally:
        _reload_config(monkeypatch)
