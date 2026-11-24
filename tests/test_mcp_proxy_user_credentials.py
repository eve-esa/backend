"""A user's own toolkit credentials reach the upstream MCP server on tools/call only.

The proxy reads the operator's credential spec from the resolved row, asks
``headers_for`` for the caller's stored values and publishes them for the one
proxied call through a contextvar that ``_DynamicBearerAuth`` reads. A missing
required credential or an unreadable one answers the caller with an MCP tool
error and never calls upstream. No value ever reaches a log or a usage row.
"""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from bson import ObjectId

from src.middlewares.auth import AUTH_TYPE_API_KEY, Principal
from src.routers import mcp_proxy
from src.routers.mcp_proxy import (
    MCPProxyDispatcher,
    _DynamicBearerAuth,
    _upstream_credential_headers_var,
    _user_eve_token_var,
)
from src.services.mcp.user_credentials import CredentialsUnreadable

pytestmark = pytest.mark.no_db

_PROXY = "src.routers.mcp_proxy"
_TOKEN = "eve_caller-token-value"
_USER_A = "6ac1aaaaaaaaaaaaaaaaaaaa"
_USER_B = "6ac1bbbbbbbbbbbbbbbbbbbb"
_SERVER_ID = ObjectId("6ac1cccccccccccccccccccc")
_SECRET = "s3cr3t-value-never-logged"
_CLIENT = "client-id-value-never-logged"
_HEADERS = {"X-Client-Id": _CLIENT, "X-Client-Secret": _SECRET}

_MISSING = "This toolkit needs your own credentials. Add them in Settings, Toolkits, then try again."
_UNREADABLE = (
    "Your saved credentials for this toolkit could not be read. Add them again in Settings, Toolkits."
)
_UNAVAILABLE = (
    "Your saved credentials for this toolkit could not be read right now. Try again in a moment."
)


def _spec(required: bool = False) -> dict:
    return {
        "fields": [
            {"name": "client_id", "label": "Client id", "header": "X-Client-Id", "secret": False},
            {"name": "client_secret", "label": "Client secret", "header": "X-Client-Secret"},
        ],
        "required": required,
    }


def _row(name: str = "toolkit", *, user_id=None, spec: dict | None = None, with_spec=True) -> dict:
    row = {"_id": _SERVER_ID, "name": name, "user_id": user_id, "config": {"url": f"https://{name}/mcp"}}
    if with_spec:
        row["user_credentials"] = spec if spec is not None else _spec()
    return row


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    monkeypatch.setattr("src.config.FEATURE_TOOLKIT_CREDENTIALS", True)
    mcp_proxy._server_url_cache.clear()
    yield
    mcp_proxy._server_url_cache.clear()


def _rpc(method: str, request_id=7) -> bytes:
    params = {"name": "search", "arguments": {"q": "x"}} if method == "tools/call" else {}
    return json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}).encode()


async def _call(row, *, method="tools/call", headers_for=None, user_id=_USER_A, request_id=7):
    """Drive one request through the dispatcher.

    Returns (sent messages, upstream requests seen by the real auth flow, track mock,
    headers_for mock). Each upstream entry holds the outgoing httpx headers.
    """
    upstream: list[dict] = []

    async def proxy_app(scope, receive, send):
        provider = MagicMock()
        provider.get_token = AsyncMock(return_value="cognito-m2m")
        flow = _DynamicBearerAuth(provider).async_auth_flow(
            httpx.Request("POST", "https://agentcore.example/mcp")
        )
        request = await flow.asend(None)
        upstream.append({k.lower(): v for k, v in request.headers.items()})
        await send({"type": "http.response.start", "status": 200, "headers": []})
        body = {"jsonrpc": "2.0", "id": request_id, "result": {"content": [], "isError": False}}
        await send({"type": "http.response.body", "body": json.dumps(body).encode()})

    collection = MagicMock()
    collection.find_one = AsyncMock(return_value=row)
    body = _rpc(method, request_id)
    sent: list = []

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": f"/mcp/{row['name'] if row else 'toolkit'}/mcp",
        "headers": [(b"authorization", f"Bearer {_TOKEN}".encode())],
    }
    principal = Principal(user_id, AUTH_TYPE_API_KEY, "key-1")
    hf = headers_for if headers_for is not None else AsyncMock(return_value=dict(_HEADERS))
    with patch(
        f"{_PROXY}.resolve_principal_from_bearer_token", AsyncMock(return_value=principal)
    ), patch(f"{_PROXY}.get_collection", return_value=collection), patch(
        f"{_PROXY}.get_cognito_token_provider", return_value=MagicMock()
    ), patch(f"{_PROXY}.build_proxy_app", AsyncMock(return_value=proxy_app)), patch(
        f"{_PROXY}.check_or_raise", AsyncMock()
    ), patch(f"{_PROXY}.headers_for", hf), patch(
        f"{_PROXY}.track_usage", new_callable=AsyncMock
    ) as track:
        await MCPProxyDispatcher(main_app=None)(scope, receive, send)
    return sent, upstream, track, hf


def _tool_error(sent) -> dict:
    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 200
    assert [b"content-type", b"application/json"] in sent[0]["headers"]
    return json.loads(sent[1]["body"])


# ── auth flow ──────────────────────────────────────────────────────────────────


async def test_auth_flow_adds_credential_headers_and_keeps_authorization_and_eve_token():
    token_reset = _user_eve_token_var.set("caller-jwt")
    creds_reset = _upstream_credential_headers_var.set(
        {**_HEADERS, "authorization": "evil", "X-EVE-Token": "evil"}
    )
    try:
        provider = MagicMock()
        provider.get_token = AsyncMock(return_value="cognito-m2m")
        flow = _DynamicBearerAuth(provider).async_auth_flow(
            httpx.Request("POST", "https://agentcore.example/mcp")
        )
        sent = await flow.asend(None)
        assert sent.headers["Authorization"] == "Bearer cognito-m2m"
        assert sent.headers["X-EVE-Token"] == "caller-jwt"
        assert sent.headers["X-Client-Id"] == _CLIENT
        assert sent.headers["X-Client-Secret"] == _SECRET
        with pytest.raises(StopAsyncIteration):
            await flow.asend(httpx.Response(200))
    finally:
        _upstream_credential_headers_var.reset(creds_reset)
        _user_eve_token_var.reset(token_reset)


async def test_auth_flow_without_credentials_adds_nothing():
    provider = MagicMock()
    provider.get_token = AsyncMock(return_value="cognito-m2m")
    flow = _DynamicBearerAuth(provider).async_auth_flow(
        httpx.Request("POST", "https://agentcore.example/mcp")
    )
    sent = await flow.asend(None)
    assert "x-client-secret" not in {k.lower() for k in sent.headers}


# ── dispatcher ─────────────────────────────────────────────────────────────────


async def test_tools_call_with_stored_credentials_reaches_upstream_with_them():
    sent, upstream, track, hf = await _call(_row())
    assert sent[0]["status"] == 200
    assert upstream[0]["x-client-id"] == _CLIENT
    assert upstream[0]["x-client-secret"] == _SECRET
    assert upstream[0]["authorization"] == "Bearer cognito-m2m"
    kwargs = hf.await_args.kwargs
    assert kwargs["user_id"] == _USER_A and kwargs["server_id"] == str(_SERVER_ID)
    assert track.await_args.kwargs["credential_source"] == "user"
    assert track.await_args.kwargs["outcome"] == "success"


@pytest.mark.parametrize("method", ["tools/list", "initialize"])
async def test_housekeeping_hops_carry_no_credentials(method):
    sent, upstream, track, hf = await _call(_row(), method=method)
    assert sent[0]["status"] == 200
    assert "x-client-secret" not in upstream[0]
    hf.assert_not_awaited()
    track.assert_not_awaited()


async def test_required_and_missing_answers_a_tool_error_without_calling_upstream():
    row = _row(spec=_spec(required=True))
    sent, upstream, track, _ = await _call(row, headers_for=AsyncMock(return_value=None))
    assert upstream == []
    assert _tool_error(sent) == {
        "jsonrpc": "2.0",
        "id": 7,
        "result": {"content": [{"type": "text", "text": _MISSING}], "isError": True},
    }
    usage = track.await_args.kwargs
    assert usage["outcome"] == "error" and usage["is_error"] is True
    assert usage["operation"] == "tools/call" and usage["tool_name"] == "search"
    assert usage["credential_source"] == "user"


async def test_optional_and_missing_forwards_on_the_platform_credentials():
    sent, upstream, track, _ = await _call(_row(), headers_for=AsyncMock(return_value=None))
    assert sent[0]["status"] == 200
    assert len(upstream) == 1 and "x-client-secret" not in upstream[0]
    assert track.await_args.kwargs["credential_source"] == "platform"


async def test_unreadable_credentials_answer_their_tool_error(caplog):
    caplog.set_level(logging.DEBUG)
    hf = AsyncMock(side_effect=CredentialsUnreadable("stored credentials could not be read"))
    sent, upstream, track, _ = await _call(_row(), headers_for=hf, request_id="abc")
    assert upstream == []
    body = _tool_error(sent)
    assert body["id"] == "abc"
    assert body["result"] == {"content": [{"type": "text", "text": _UNREADABLE}], "isError": True}
    assert track.await_args.kwargs["outcome"] == "error"


async def test_a_kms_outage_answers_try_again_and_logs_the_class_only(caplog):
    caplog.set_level(logging.DEBUG)
    hf = AsyncMock(side_effect=RuntimeError(f"kms said no for {_SECRET}"))
    sent, upstream, track, _ = await _call(_row(spec=_spec(required=False)), headers_for=hf)
    assert upstream == []
    assert _tool_error(sent)["result"]["content"][0]["text"] == _UNAVAILABLE
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors and "RuntimeError" in errors[0].getMessage()
    assert all(r.exc_info is None for r in errors)
    rendered = " ".join(r.getMessage() for r in caplog.records)
    assert _SECRET not in rendered
    assert track.await_args.kwargs["outcome"] == "error"


async def test_no_leak_into_the_next_request():
    await _call(_row())
    assert _upstream_credential_headers_var.get() is None
    # Another user, a toolkit with no spec: nothing from the first call survives.
    sent, upstream, track, hf = await _call(_row("other", with_spec=False), user_id=_USER_B)
    assert "x-client-secret" not in upstream[0]
    hf.assert_not_awaited()
    assert track.await_args.kwargs["credential_source"] is None


async def test_flag_off_injects_nothing(monkeypatch):
    monkeypatch.setattr("src.config.FEATURE_TOOLKIT_CREDENTIALS", False)
    sent, upstream, track, hf = await _call(_row(spec=_spec(required=True)))
    assert sent[0]["status"] == 200
    assert "x-client-secret" not in upstream[0]
    hf.assert_not_awaited()
    assert track.await_args.kwargs["credential_source"] is None


async def test_a_user_owned_row_with_the_same_name_injects_nothing():
    row = _row(user_id=_USER_A, spec=_spec(required=True))
    sent, upstream, track, hf = await _call(row)
    assert sent[0]["status"] == 200
    assert "x-client-secret" not in upstream[0]
    hf.assert_not_awaited()
    assert track.await_args.kwargs["credential_source"] is None


async def test_the_spec_survives_the_url_cache():
    await _call(_row())
    sent, upstream, _track, hf = await _call(None)
    assert upstream[0]["x-client-secret"] == _SECRET
    assert hf.await_args.kwargs["server_id"] == str(_SERVER_ID)


async def test_no_value_in_logs_or_usage_rows(caplog):
    caplog.set_level(logging.DEBUG)
    _sent, _upstream, track, _ = await _call(_row())
    rendered = " ".join(r.getMessage() for r in caplog.records)
    assert _SECRET not in rendered and _CLIENT not in rendered
    stored = json.dumps(track.await_args.kwargs, default=str)
    assert _SECRET not in stored and _CLIENT not in stored


async def test_track_usage_stores_the_credential_source():
    from src.services.mcp import usage

    collection = MagicMock()
    collection.insert_one = AsyncMock(return_value=MagicMock(inserted_id="x"))
    with patch.object(usage, "get_collection", return_value=collection):
        await usage.track_usage(
            user_id=_USER_A, caller_type="api_key", server_name="toolkit",
            operation="tools/call", credential_source="user",
        )
    assert collection.insert_one.await_args.args[0]["credential_source"] == "user"
