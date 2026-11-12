"""The caller's EVE token reaches only the MCP servers that need it.

The token is the user's own credential: it can mint ``eve_`` API keys. Only the
retrieval server calls back into EVE as the user (``POST /retrieve``), so only
rows flagged ``forward_user_token`` receive it as ``X-EVE-Token``. Global rows
named ``eve_retrieval`` or ``eve_retrieval_<env>`` count as flagged while the
field is absent, so existing rows keep working until the field is written.
"""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.database.models.mcp_server import MCPServer, ToolConfig, ToolTransport
from src.middlewares.auth import AUTH_TYPE_API_KEY, Principal
from src.routers import mcp_proxy
from src.routers.mcp_proxy import MCPProxyDispatcher, _user_eve_token_var
from src.services.agents.core.runner import _load_mcp_tools_for_servers
from src.services.mcp import user_token_scope
from src.services.mcp.tool_cache import clear_mcp_tool_cache
from src.services.mcp.user_token_scope import forwards_user_token

pytestmark = pytest.mark.no_db

_LOADER = "src.services.mcp.tool_loader"
_PROXY = "src.routers.mcp_proxy"
_TOKEN = "eve_secret-user-token-value"
_USER_ID = "6ac1aaaaaaaaaaaaaaaaaaaa"


@pytest.fixture(autouse=True)
def _clean_state():
    clear_mcp_tool_cache()
    mcp_proxy._server_url_cache.clear()
    user_token_scope._announced.clear()
    yield
    clear_mcp_tool_cache()
    mcp_proxy._server_url_cache.clear()
    user_token_scope._announced.clear()


# ── policy ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "user_id", "flag", "expected"),
    [
        ("eve_retrieval", None, None, True),
        ("eve_retrieval_dev", None, None, True),
        ("eve_retrieval_staging", None, None, True),
        ("eve_retrieval_prod", None, None, True),
        ("eve_retrieval", None, False, False),
        ("eve_retrieval_evil", None, None, False),
        ("effis", None, None, False),
        ("effis", None, False, False),
        ("custom_retrieval", None, True, True),
        # A user can register a row with the reserved name: no fallback for it.
        ("eve_retrieval", _USER_ID, None, False),
    ],
)
def test_forwards_user_token_policy(name, user_id, flag, expected):
    assert forwards_user_token(name=name, user_id=user_id, forward_user_token=flag) is expected


def test_model_field_absent_is_none_and_round_trips():
    srv = MCPServer(name="effis", config=ToolConfig(url="https://x/mcp"))
    assert srv.forward_user_token is None
    srv = MCPServer(
        name="custom", config=ToolConfig(url="https://x/mcp"), forward_user_token=True
    )
    assert srv.forward_user_token is True


def test_request_schemas_cannot_set_the_flag():
    """A user-registered row must never opt itself into the user's token."""
    from src.schemas.mcp_server import MCPServerRequest, MCPServerUpdate

    assert "forward_user_token" not in MCPServerRequest.model_fields
    assert "forward_user_token" not in MCPServerUpdate.model_fields


# ── tool_loader direct path ────────────────────────────────────────────────────


def _srv(name: str, *, flag=None, user_id=None) -> MCPServer:
    return MCPServer(
        name=name,
        user_id=user_id,
        config=ToolConfig(url=f"https://{name}.example/mcp", transport=ToolTransport.STREAMABLE_HTTP),
        forward_user_token=flag,
    )


async def _direct_connections(servers) -> dict:
    captured: dict = {}

    def make_client(connections, **kwargs):
        captured.update(connections)
        client = MagicMock()
        client.get_tools = AsyncMock(return_value=[])
        return client

    with patch(f"{_LOADER}._mcp_adapters_available", True), patch(
        f"{_LOADER}.MultiServerMCPClient", side_effect=make_client
    ), patch(f"{_LOADER}.backend_mcp_proxy_url", return_value=None), patch(
        f"{_LOADER}.get_cognito_token_provider", return_value=None
    ), patch(f"{_LOADER}.LatencyInterceptor", return_value=MagicMock()), patch(
        f"{_LOADER}.ErrorLoggingInterceptor", return_value=MagicMock()
    ), patch(f"{_LOADER}.persist_policy_event", new_callable=AsyncMock):
        await _load_mcp_tools_for_servers(servers, mcp_proxy_bearer_token=_TOKEN)
    return captured


def _has_eve_token(headers: dict) -> bool:
    return any(k.lower() == "x-eve-token" for k in headers)


async def test_direct_path_non_retrieval_server_gets_no_user_token():
    conns = await _direct_connections([_srv("effis"), _srv("serpapi", flag=False)])
    assert not _has_eve_token(conns["effis"]["headers"])
    assert not _has_eve_token(conns["serpapi"]["headers"])
    assert _TOKEN not in json.dumps(conns)


async def test_direct_path_retrieval_gets_user_token_by_flag_and_by_reserved_name():
    conns = await _direct_connections(
        [_srv("custom_retrieval", flag=True), _srv("eve_retrieval_dev"), _srv("eve_retrieval")]
    )
    for name in ("custom_retrieval", "eve_retrieval_dev", "eve_retrieval"):
        assert conns[name]["headers"]["X-EVE-Token"] == _TOKEN


async def test_direct_path_user_row_with_reserved_name_gets_no_user_token():
    conns = await _direct_connections([_srv("eve_retrieval", user_id=_USER_ID)])
    assert not _has_eve_token(conns["eve_retrieval"]["headers"])


# ── proxy dispatcher ───────────────────────────────────────────────────────────


async def _proxy_call(row: dict | None, server_name: str):
    """Drive one ``tools/list`` through ``MCPProxyDispatcher`` and return the
    token the upstream auth would forward (``_user_eve_token_var`` at call time)."""
    seen: dict = {}

    async def proxy_app(scope, receive, send):
        seen["token"] = _user_eve_token_var.get()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    collection = MagicMock()
    collection.find_one = AsyncMock(return_value=row)
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).encode()
    sent: list = []

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": f"/mcp/{server_name}/mcp",
        "headers": [(b"authorization", f"Bearer {_TOKEN}".encode())],
    }
    principal = Principal(_USER_ID, AUTH_TYPE_API_KEY, "key-1")
    with patch(
        f"{_PROXY}.resolve_principal_from_bearer_token", AsyncMock(return_value=principal)
    ), patch(f"{_PROXY}.get_collection", return_value=collection), patch(
        f"{_PROXY}.get_cognito_token_provider", return_value=MagicMock()
    ), patch(f"{_PROXY}.build_proxy_app", AsyncMock(return_value=proxy_app)), patch(
        f"{_PROXY}.track_usage", new_callable=AsyncMock
    ):
        await MCPProxyDispatcher(main_app=None)(scope, receive, send)
    assert sent[0]["status"] == 200
    return seen["token"]


def _row(name: str, **extra) -> dict:
    return {"name": name, "user_id": None, "config": {"url": f"https://{name}/mcp"}, **extra}


async def test_proxy_non_retrieval_server_gets_no_user_token():
    assert await _proxy_call(_row("effis"), "effis") is None


async def test_proxy_retrieval_gets_user_token_with_flag_set():
    token = await _proxy_call(_row("custom", forward_user_token=True), "custom")
    assert token == _TOKEN


async def test_proxy_retrieval_gets_user_token_with_flag_absent_and_reserved_name():
    assert await _proxy_call(_row("eve_retrieval"), "eve_retrieval") == _TOKEN


async def test_proxy_explicit_false_wins_over_reserved_name():
    row = _row("eve_retrieval", forward_user_token=False)
    assert await _proxy_call(row, "eve_retrieval") is None


async def test_proxy_user_row_with_reserved_name_gets_no_user_token():
    row = _row("eve_retrieval", user_id=_USER_ID)
    assert await _proxy_call(row, "eve_retrieval") is None


async def test_proxy_cached_resolution_keeps_the_decision():
    assert await _proxy_call(_row("effis"), "effis") is None
    # Second call hits the URL cache; the row lookup must not be needed.
    assert await _proxy_call(None, "effis") is None
    assert await _proxy_call(_row("eve_retrieval"), "eve_retrieval") == _TOKEN
    assert await _proxy_call(None, "eve_retrieval") == _TOKEN


# ── logs ───────────────────────────────────────────────────────────────────────


async def test_token_never_logged_and_receivers_announced_once(caplog):
    caplog.set_level(logging.DEBUG)
    await _proxy_call(_row("eve_retrieval"), "eve_retrieval")
    await _proxy_call(_row("effis"), "effis")
    mcp_proxy._server_url_cache.clear()
    await _proxy_call(_row("eve_retrieval"), "eve_retrieval")
    await _direct_connections([_srv("eve_retrieval_dev"), _srv("effis")])

    rendered = [r.getMessage() for r in caplog.records]
    assert not any(_TOKEN in line for line in rendered)
    announced = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.INFO and "mcp.user_token_forwarded" in r.getMessage()
    ]
    assert announced == [
        "mcp.user_token_forwarded server=eve_retrieval",
        "mcp.user_token_forwarded server=eve_retrieval_dev",
    ]
