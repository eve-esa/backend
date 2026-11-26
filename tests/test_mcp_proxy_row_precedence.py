"""A user row and a global row with one name: the proxy resolves the caller's row.

Without a sort ``find_one`` returns whichever row the store reads first, so a
turn could route a global server's calls to the user's URL or the reverse.
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from src.database.mongo import get_collection
from src.middlewares.auth import AUTH_TYPE_API_KEY, Principal
from src.routers import mcp_proxy
from src.routers.mcp_proxy import MCPProxyDispatcher, _DynamicBearerAuth

_PROXY = "src.routers.mcp_proxy"
_NAME = "row-precedence-test"
_USER_ID = "6ac1bbbbbbbbbbbbbbbbbbbb"


async def _resolved_url(user_id: str) -> str:
    built: list = []

    async def _build(url, _provider):
        built.append(url)
        return object()

    principal = Principal(user_id, AUTH_TYPE_API_KEY, "key-1")
    scope = {"headers": [(b"authorization", b"Bearer eve_test-token")]}
    with patch(
        f"{_PROXY}.resolve_principal_from_bearer_token", AsyncMock(return_value=principal)
    ), patch(f"{_PROXY}.get_cognito_token_provider", return_value=MagicMock()), patch(
        f"{_PROXY}.build_proxy_app", _build
    ):
        await MCPProxyDispatcher(main_app=None)._resolve_proxy(scope, _NAME)
    return built[0]


async def test_the_callers_row_wins_over_the_global_row_in_either_insert_order():
    collection = get_collection("mcp_servers")
    rows = {
        "global": {"name": _NAME, "enabled": True, "user_id": None,
                   "config": {"url": "https://global.invalid/mcp"}},
        "user": {"name": _NAME, "enabled": True, "user_id": _USER_ID,
                 "config": {"url": "https://user.invalid/mcp"}},
    }
    try:
        for order in (("global", "user"), ("user", "global")):
            await collection.delete_many({"name": _NAME})
            mcp_proxy._server_url_cache.clear()
            for key in order:
                await collection.insert_one(dict(rows[key]))

            assert await _resolved_url(_USER_ID) == "https://user.invalid/mcp"
            mcp_proxy._server_url_cache.clear()
            # Another user sees only the global row.
            assert await _resolved_url("6ac1cccccccccccccccccccc") == "https://global.invalid/mcp"
    finally:
        await collection.delete_many({"name": _NAME})
        mcp_proxy._server_url_cache.clear()


_SPEC = {
    "fields": [{"name": "client_secret", "label": "Client secret", "header": "X-Client-Secret"}],
    "required": True,
}


async def _tools_call(user_id: str) -> tuple[list, list, AsyncMock, AsyncMock]:
    """One tools/call through the dispatcher against the real ``mcp_servers`` store."""
    upstream: list[dict] = []
    sent: list = []

    async def proxy_app(scope, receive, send):
        provider = MagicMock()
        provider.get_token = AsyncMock(return_value="cognito-m2m")
        flow = _DynamicBearerAuth(provider).async_auth_flow(
            httpx.Request("POST", "https://agentcore.example/mcp")
        )
        request = await flow.asend(None)
        upstream.append({k.lower(): v for k, v in request.headers.items()})
        await send({"type": "http.response.start", "status": 200, "headers": []})
        body = {"jsonrpc": "2.0", "id": 1, "result": {"content": [], "isError": False}}
        await send({"type": "http.response.body", "body": json.dumps(body).encode()})

    body = json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "search"}}
    ).encode()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http", "method": "POST", "path": f"/mcp/{_NAME}/mcp",
        "headers": [(b"authorization", b"Bearer eve_test-token")],
    }
    principal = Principal(user_id, AUTH_TYPE_API_KEY, "key-1")
    hf = AsyncMock(return_value={"X-Client-Secret": "never-sent-secret"})
    with patch(
        f"{_PROXY}.resolve_principal_from_bearer_token", AsyncMock(return_value=principal)
    ), patch(f"{_PROXY}.get_cognito_token_provider", return_value=MagicMock()), patch(
        f"{_PROXY}.build_proxy_app", AsyncMock(return_value=proxy_app)
    ), patch(f"{_PROXY}.check_or_raise", AsyncMock()), patch(
        f"{_PROXY}.headers_for", hf
    ), patch(f"{_PROXY}.track_usage", new_callable=AsyncMock) as track:
        await MCPProxyDispatcher(main_app=None)(scope, receive, send)
    return sent, upstream, track, hf


async def test_a_same_name_user_row_shadows_the_global_spec_and_gets_no_credentials(
    monkeypatch,
):
    monkeypatch.setattr("src.config.FEATURE_TOOLKIT_CREDENTIALS", True)
    collection = get_collection("mcp_servers")
    try:
        for order in ((True, False), (False, True)):
            await collection.delete_many({"name": _NAME})
            mcp_proxy._server_url_cache.clear()
            for is_global in order:
                row = {"name": _NAME, "enabled": True, "deleted_at": None}
                if is_global:
                    row |= {"user_id": None, "config": {"url": "https://global.invalid/mcp"},
                            "user_credentials": _SPEC}
                else:
                    row |= {"user_id": _USER_ID, "config": {"url": "https://user.invalid/mcp"}}
                await collection.insert_one(row)

            sent, upstream, track, hf = await _tools_call(_USER_ID)
            assert sent[0]["status"] == 200
            assert len(upstream) == 1 and "x-client-secret" not in upstream[0]
            hf.assert_not_awaited()
            assert track.await_args.kwargs["credential_source"] is None
    finally:
        await collection.delete_many({"name": _NAME})
        mcp_proxy._server_url_cache.clear()


async def test_a_soft_deleted_global_row_with_a_spec_is_not_resolved(monkeypatch):
    monkeypatch.setattr("src.config.FEATURE_TOOLKIT_CREDENTIALS", True)
    collection = get_collection("mcp_servers")
    try:
        await collection.delete_many({"name": _NAME})
        mcp_proxy._server_url_cache.clear()
        await collection.insert_one(
            {"name": _NAME, "enabled": True, "user_id": None, "deleted_at": datetime(2026, 11, 20, tzinfo=timezone.utc),
             "config": {"url": "https://global.invalid/mcp"}, "user_credentials": _SPEC}
        )
        sent, upstream, track, hf = await _tools_call(_USER_ID)
        assert sent[0]["status"] == 404
        assert upstream == []
        hf.assert_not_awaited()
        track.assert_not_awaited()
    finally:
        await collection.delete_many({"name": _NAME})
        mcp_proxy._server_url_cache.clear()
