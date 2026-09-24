"""Redaction table for src.utils.redaction, /log-error storage and log levels."""

import logging

import pytest

from src.config import NOISY_LOGGERS, configure_logging, resolve_log_level
from src.database.models.error_log import ErrorLog
from src.utils.redaction import (
    REDACTED,
    REDACTED_EMAIL,
    is_secret_key,
    redact_secrets,
    redact_value,
)
from tests.utils.cleaner import cleanup_models
from tests.utils.utils import create_test_user_and_token

JWT = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJzdWIiOiIxMjM0NTY3ODkwIn0.c2lnbmF0dXJlX3ZhbHVl"
)
EVE_KEY = "eve_" + "0123456789abcdef" * 4
RPA_KEY = "rpa_ABCDEF0123456789GHIJKLMNOPQRSTUV0123456789"
OPAQUE = "s3cr3tValue0987"

STRING_CASES = [
    # (id, input, expected, secret that must be gone)
    (
        "authorization_bearer_header",
        f"Authorization: Bearer {OPAQUE}",
        f"Authorization: Bearer {REDACTED}",
        OPAQUE,
    ),
    (
        "bearer_jwt",
        f"Bearer {JWT}",
        f"Bearer {REDACTED}",
        "eyJ",
    ),
    (
        "basic_header",
        "Authorization: Basic dXNlcjpwYXNzd29yZA==",
        f"Authorization: Basic {REDACTED}",
        "dXNlcjpwYXNzd29yZA",
    ),
    (
        "query_api_key",
        f"https://api.example.com/v1/search?q=mars&api_key={OPAQUE}&page=2",
        f"https://api.example.com/v1/search?q=mars&api_key={REDACTED}&page=2",
        OPAQUE,
    ),
    (
        "query_token_first_param",
        f"/callback?token={OPAQUE}",
        f"/callback?token={REDACTED}",
        OPAQUE,
    ),
    (
        "query_access_token_and_secret",
        f"https://x.test/?access_token={OPAQUE}&client_secret={OPAQUE}#frag",
        f"https://x.test/?access_token={REDACTED}&client_secret={REDACTED}#frag",
        OPAQUE,
    ),
    (
        "query_password",
        f"https://x.test/login?user=a&password={OPAQUE}",
        f"https://x.test/login?user=a&password={REDACTED}",
        OPAQUE,
    ),
    (
        "eve_api_key",
        f"key rejected: {EVE_KEY}",
        f"key rejected: {REDACTED}",
        EVE_KEY,
    ),
    (
        "rpa_runpod_key",
        f"upstream said {RPA_KEY} is invalid",
        f"upstream said {REDACTED} is invalid",
        RPA_KEY,
    ),
    (
        "email",
        "user jane.doe+eve@esa.example.org not approved",
        f"user {REDACTED_EMAIL} not approved",
        "jane.doe",
    ),
    (
        "url_credentials",
        f"mongodb://writer:{OPAQUE}@docdb.local:27017/eve",
        f"mongodb://{REDACTED}@docdb.local:27017/eve",
        OPAQUE,
    ),
    (
        "json_assignment",
        f'{{"api_key": "{OPAQUE}", "model": "eve"}}',
        f'{{"api_key": "{REDACTED}", "model": "eve"}}',
        OPAQUE,
    ),
]


@pytest.mark.no_db
@pytest.mark.parametrize(
    "raw,expected,secret",
    [case[1:] for case in STRING_CASES],
    ids=[case[0] for case in STRING_CASES],
)
def test_redact_secrets_table(raw, expected, secret):
    out = redact_secrets(raw)
    assert out == expected
    assert secret not in out


@pytest.mark.no_db
@pytest.mark.parametrize(
    "text",
    [
        "max_tokens=512 exceeded",
        "tier eve_free and eve_jsc are fine",
        "basic setup failed",
        "GET /conversations?page=2&limit=20",
        "",
    ],
)
def test_redact_secrets_leaves_harmless_text(text):
    assert redact_secrets(text) == text


@pytest.mark.no_db
def test_redact_secrets_passes_non_strings_through():
    assert redact_secrets(None) is None


@pytest.mark.no_db
def test_redact_value_nested_dict_list_and_exception():
    payload = {
        "route": "/chat",
        "count": 3,
        "ok": True,
        "nested": {
            "headers": {"Authorization": f"Bearer {OPAQUE}", "accept": "json"},
            "apiKey": OPAQUE,
            "owner": "someone@example.com",
            "deeper": {"refresh_token": OPAQUE, "note": f"sent {EVE_KEY}"},
        },
        "items": [f"token={OPAQUE}", {"password": OPAQUE}, 7],
        "pair": (f"Bearer {OPAQUE}", None),
        "error": RuntimeError(f"upstream 401 for api_key={OPAQUE}"),
    }

    out = redact_value(payload)

    assert OPAQUE not in repr(out)
    assert EVE_KEY not in repr(out)
    assert out["route"] == "/chat"
    assert out["count"] == 3 and out["ok"] is True
    assert out["nested"]["headers"]["Authorization"] == REDACTED
    assert out["nested"]["headers"]["accept"] == "json"
    assert out["nested"]["apiKey"] == REDACTED
    assert out["nested"]["owner"] == REDACTED_EMAIL
    assert out["nested"]["deeper"]["refresh_token"] == REDACTED
    assert out["nested"]["deeper"]["note"] == f"sent {REDACTED}"
    assert out["items"] == [f"token={REDACTED}", {"password": REDACTED}, 7]
    assert out["pair"] == [f"Bearer {REDACTED}", None]
    assert out["error"] == {
        "type": "RuntimeError",
        "message": f"upstream 401 for api_key={REDACTED}",
        "args": [f"upstream 401 for api_key={REDACTED}"],
    }


@pytest.mark.no_db
def test_redact_value_list_at_top_level():
    assert redact_value([f"Bearer {OPAQUE}", "a@b.io", 1]) == [
        f"Bearer {REDACTED}",
        REDACTED_EMAIL,
        1,
    ]


@pytest.mark.no_db
def test_redact_value_exception_with_secret():
    out = redact_value(ValueError(f"bad key {RPA_KEY}"))
    assert out["type"] == "ValueError"
    assert out["message"] == f"bad key {REDACTED}"
    assert RPA_KEY not in repr(out)


@pytest.mark.no_db
def test_redact_value_clips_strings_when_asked():
    assert redact_value("x" * 50, max_len=10) == "x" * 10
    assert redact_value("x" * 50) == "x" * 50


@pytest.mark.no_db
@pytest.mark.parametrize(
    "key,secret",
    [
        ("authorization", True),
        ("Authorization", True),
        ("x-api-key", True),
        ("apiKey", True),
        ("access_token", True),
        ("clientSecret", True),
        ("aws_secret_access_key", True),
        ("set-cookie", True),
        ("max_tokens", False),
        ("token_count", False),
        ("key_id", False),
        ("model", False),
    ],
)
def test_is_secret_key(key, secret):
    assert is_secret_key(key) is secret


@pytest.mark.no_db
def test_error_logger_reexports_the_single_implementation():
    from src.utils import error_logger, redaction

    assert error_logger.redact_secrets is redaction.redact_secrets
    assert error_logger.redact_value is redaction.redact_value


@pytest.mark.asyncio
async def test_log_error_stores_redacted_stack_url_and_metadata(async_client):
    user, token = await create_test_user_and_token()
    try:
        payload = {
            "error_message": f"fetch failed with Bearer {OPAQUE}",
            "error_type": "TypeError",
            "error_stack": (
                f"Error: 401 for {EVE_KEY}\n"
                f"    at fetch (https://app.test/main.js?token={OPAQUE}:10:5)"
            ),
            "url": f"https://app.test/chat?api_key={OPAQUE}&tab=2",
            "metadata": {
                "user": "jane.doe@example.com",
                "request": {
                    "headers": {"Authorization": f"Bearer {OPAQUE}"},
                    "attempts": [f"password={OPAQUE}", 2],
                },
                "status": 401,
            },
        }

        response = await async_client.post(
            "/log-error",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200

        stored = await ErrorLog.find_by_id(response.json()["id"])
        assert stored is not None
        try:
            blob = repr(stored.error) + stored.description
            for secret in (OPAQUE, EVE_KEY, "jane.doe"):
                assert secret not in blob

            assert stored.error["url"] == (
                f"https://app.test/chat?api_key={REDACTED}&tab=2"
            )
            # The query value runs to the next separator, so the line suffix
            # after the token goes too: losing ":10:5)" beats leaking a secret.
            assert stored.error["stack"] == (
                f"Error: 401 for {REDACTED}\n"
                f"    at fetch (https://app.test/main.js?token={REDACTED}"
            )
            assert stored.error["metadata"] == {
                "user": REDACTED_EMAIL,
                "request": {
                    "headers": {"Authorization": REDACTED},
                    "attempts": [f"password={REDACTED}", 2],
                },
                "status": 401,
            }
            assert stored.description == f"fetch failed with Bearer {REDACTED}"
        finally:
            await cleanup_models([stored])
    finally:
        await cleanup_models([user])


@pytest.fixture
def _restore_log_levels():
    names = ("",) + NOISY_LOGGERS
    saved = {name: logging.getLogger(name).level for name in names}
    yield
    for name, level in saved.items():
        logging.getLogger(name).setLevel(level)


@pytest.mark.no_db
@pytest.mark.parametrize(
    "env,expected",
    [
        (None, logging.INFO),
        ("", logging.INFO),
        ("debug", logging.DEBUG),
        ("WARNING", logging.WARNING),
        ("10", logging.DEBUG),
        ("nonsense", logging.INFO),
    ],
)
def test_configure_logging_reads_log_level(
    monkeypatch, _restore_log_levels, env, expected
):
    if env is None:
        monkeypatch.delenv("LOG_LEVEL", raising=False)
    else:
        monkeypatch.setenv("LOG_LEVEL", env)

    assert resolve_log_level() == expected
    configure_logging()

    assert logging.getLogger().level == expected
    for name in NOISY_LOGGERS:
        assert logging.getLogger(name).level == logging.WARNING, name


@pytest.mark.no_db
def test_configure_logging_quiets_client_libraries_even_at_debug(
    monkeypatch, _restore_log_levels
):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    configure_logging()

    assert logging.getLogger().level == logging.DEBUG
    assert set(NOISY_LOGGERS) == {
        "httpx",
        "httpcore",
        "urllib3",
        "pymongo",
        "botocore",
        "boto3",
        "openai",
        "mcp",
    }
    for name in NOISY_LOGGERS:
        assert not logging.getLogger(name).isEnabledFor(logging.INFO), name
        assert logging.getLogger(name).isEnabledFor(logging.WARNING), name
    assert logging.getLogger("src.anything").isEnabledFor(logging.DEBUG)
