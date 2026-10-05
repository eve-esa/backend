from src.routers.mcp_proxy import (
    _is_error_message,
    _tool_call_outcome,
    _user_eve_token_var,
    _DynamicBearerAuth,
)
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest


def test_is_error_message_json_rpc_error():
    assert _is_error_message([{"error": {"code": -32603, "message": "boom"}}]) is True


def test_is_error_message_result_is_error():
    assert _is_error_message([{"result": {"isError": True, "content": []}}]) is True


def test_is_error_message_success():
    assert _is_error_message([{"result": {"isError": False, "content": []}}]) is False


def test_is_error_message_unparseable_returns_none():
    assert _is_error_message(None) is None


def test_tool_call_outcome_unknown_when_mcp_unparseable_and_http_ok():
    assert _tool_call_outcome(http_failed=False, is_error=None) == "unknown"


def test_tool_call_outcome_error_when_http_failed_even_if_mcp_unknown():
    assert _tool_call_outcome(http_failed=True, is_error=None) == "error"


def test_tool_call_outcome_success_only_when_mcp_confirmed_ok():
    assert _tool_call_outcome(http_failed=False, is_error=False) == "success"


# ── X-EVE-Token propagation ────────────────────────────────────────────────────
# The MCP proxy authenticates to AgentCore with a shared Cognito M2M token, but
# user-scoped EVE endpoints (notably ``eve_retrieval`` → ``POST /retrieve``) must
# run as the caller: ``apply_private_collections_to_request`` keeps only
# collections owned by the requesting user, so a machine ``EVE_API_KEY`` silently
# drops every private collection. The proxy forwards the caller's EVE credential
# as ``X-EVE-Token`` (the MCP server reads it before its ``EVE_API_KEY`` fallback)
# to preserve that identity. These tests lock the wiring.


@pytest.mark.asyncio
async def test_dynamic_bearer_forwards_user_token_as_x_eve_token():
    """The caller's EVE credential is attached as ``X-EVE-Token`` on egress."""
    token_reset = _user_eve_token_var.set("caller-jwt")
    try:
        provider = MagicMock()
        provider.get_token = AsyncMock(return_value="cognito-m2m")
        auth = _DynamicBearerAuth(provider)

        request = httpx.Request("POST", "https://agentcore.example/mcp")
        gen = auth.async_auth_flow(request)
        sent = await gen.asend(None)

        # Cognito M2M authenticates the proxy to AgentCore; X-EVE-Token
        # authenticates the user to the downstream EVE API.
        assert sent.headers["Authorization"] == "Bearer cognito-m2m"
        assert sent.headers["X-EVE-Token"] == "caller-jwt"

        with pytest.raises(StopAsyncIteration):
            await gen.asend(httpx.Response(200))
    finally:
        _user_eve_token_var.reset(token_reset)


@pytest.mark.asyncio
async def test_dynamic_bearer_omits_x_eve_token_when_no_user_token():
    """No caller token (e.g. a housekeeping hop) must not synthesize a header."""
    token_reset = _user_eve_token_var.set(None)
    try:
        provider = MagicMock()
        provider.get_token = AsyncMock(return_value="cognito-m2m")
        auth = _DynamicBearerAuth(provider)

        request = httpx.Request("POST", "https://agentcore.example/mcp")
        gen = auth.async_auth_flow(request)
        sent = await gen.asend(None)

        headers = {k.lower(): v for k, v in sent.headers.items()}
        assert "x-eve-token" not in headers
        assert sent.headers["Authorization"] == "Bearer cognito-m2m"

        with pytest.raises(StopAsyncIteration):
            await gen.asend(httpx.Response(200))
    finally:
        _user_eve_token_var.reset(token_reset)


# ── No session DELETE upstream ─────────────────────────────────────────────────
# AgentCore hands out an ``Mcp-Session-Id`` but is stateless: the DELETE the MCP
# client sends on close answers 404 and logs a WARNING on every proxied call.


@pytest.mark.asyncio
async def test_upstream_client_does_not_delete_the_session_on_close():
    """A stateful upstream (one that issues a session id) gets no DELETE."""
    from fastmcp import Client, FastMCP
    from fastmcp.client.transports.http import StreamableHttpTransport

    upstream = FastMCP("upstream")

    @upstream.tool
    def ping() -> str:
        return "pong"

    app = upstream.http_app(stateless_http=False)
    methods: list[str] = []

    async def record(request: httpx.Request) -> None:
        methods.append(request.method)

    def client_factory(**kwargs) -> httpx.AsyncClient:
        kwargs.pop("follow_redirects", None)
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            event_hooks={"request": [record]},
            **kwargs,
        )

    transport = StreamableHttpTransport(
        "http://upstream.test/mcp", httpx_client_factory=client_factory
    )
    async with app.lifespan(app):
        async with Client(transport) as client:
            await client.list_tools()
            assert transport.get_session_id(), "upstream must issue a session id"

    assert "POST" in methods
    assert "DELETE" not in methods
