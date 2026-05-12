from src.config import Config


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
