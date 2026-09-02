from typing import Optional
from unittest.mock import AsyncMock, patch

import pytest

from src.database.models.mcp_server import MCPServer, ToolConfig, ToolTransport
from tests.utils.cleaner import cleanup_models
from tests.utils.utils import create_test_user_and_token

_ROUTER = "src.routers.mcp_server"


def _mcp_server(*, user_id: Optional[str], name: str = "test-server") -> MCPServer:
    return MCPServer(
        user_id=user_id,
        name=name,
        config=ToolConfig(
            url="https://example.com/mcp",
            transport=ToolTransport.STREAMABLE_HTTP,
        ),
    )


@pytest.mark.asyncio
async def test_list_mcp_servers_requires_auth(async_client):
    response = await async_client.get("/mcp-servers")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_list_mcp_servers_returns_catalog_for_authenticated_user(async_client):
    owner, owner_token = await create_test_user_and_token()
    other, other_token = await create_test_user_and_token()
    owner_server = _mcp_server(user_id=owner.id, name="owner-server")
    global_server = _mcp_server(user_id=None, name="global-server")
    await owner_server.save()
    await global_server.save()
    try:
        response = await async_client.get(
            "/mcp-servers",
            headers={"Authorization": f"Bearer {other_token}"},
        )
        assert response.status_code == 200
        names = {item["name"] for item in response.json()["data"]}
        assert "owner-server" in names
        assert "global-server" in names
    finally:
        await cleanup_models([owner_server, global_server, owner, other])


@pytest.mark.asyncio
@patch(f"{_ROUTER}._load_mcp_tools_for_servers", new_callable=AsyncMock, return_value=[])
async def test_get_mcp_server_requires_auth(mock_load_tools, async_client):
    server = _mcp_server(user_id="owner-user")
    await server.save()
    try:
        response = await async_client.get(f"/mcp-servers/{server.id}")
        assert response.status_code == 401
        mock_load_tools.assert_not_called()
    finally:
        await cleanup_models([server])


@pytest.mark.asyncio
@patch(f"{_ROUTER}._load_mcp_tools_for_servers", new_callable=AsyncMock, return_value=[])
async def test_get_mcp_server_allows_any_authenticated_user(mock_load_tools, async_client):
    owner, _ = await create_test_user_and_token()
    other, other_token = await create_test_user_and_token()
    server = _mcp_server(user_id=owner.id, name="shared-catalog-server")
    await server.save()
    try:
        response = await async_client.get(
            f"/mcp-servers/{server.id}",
            headers={"Authorization": f"Bearer {other_token}"},
        )
        assert response.status_code == 200
        assert response.json()["name"] == "shared-catalog-server"
        mock_load_tools.assert_awaited_once()
    finally:
        await cleanup_models([server, owner, other])


@pytest.mark.asyncio
@patch(f"{_ROUTER}._load_mcp_tools_for_servers", new_callable=AsyncMock, return_value=[])
async def test_get_mcp_server_reports_no_error_when_discovery_returns_nothing(
    mock_load_tools, async_client
):
    """A server that genuinely exposes no tools must NOT look like a failure."""
    owner, token = await create_test_user_and_token()
    server = _mcp_server(user_id=owner.id, name="empty-but-healthy")
    await server.save()
    try:
        response = await async_client.get(
            f"/mcp-servers/{server.id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["tools"] == []
        assert body["tools_error"] is None
    finally:
        await cleanup_models([server, owner])


@pytest.mark.asyncio
@patch(
    f"{_ROUTER}._load_mcp_tools_for_servers",
    new_callable=AsyncMock,
    side_effect=RuntimeError("421 Misdirected Request"),
)
async def test_get_mcp_server_surfaces_discovery_failure(mock_load_tools, async_client):
    """An unreachable server still answers 200, but says why the list is empty.

    Before tools_error existed this response was byte-identical to the healthy-but-empty
    case above, which is how a wall of unreachable toolkits could read as a wall of
    empty ones.
    """
    owner, token = await create_test_user_and_token()
    server = _mcp_server(user_id=owner.id, name="unreachable")
    await server.save()
    try:
        response = await async_client.get(
            f"/mcp-servers/{server.id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["tools"] == []
        assert body["tools_error"] is not None
        assert "RuntimeError" in body["tools_error"]
        assert "421" in body["tools_error"]
    finally:
        await cleanup_models([server, owner])


# ─── FEATURE_MCP_SERVER_REGISTRATION (config.py, _require_registration_enabled) ──

_CREATE_PAYLOAD = {
    "name": "created-via-api",
    "config": {"url": "https://example.com/mcp", "transport": "streamable_http"},
}


@pytest.mark.asyncio
async def test_create_mcp_server_returns_404_when_registration_disabled(
    async_client, monkeypatch
):
    """404, not 403: with the flag off the write route must look unmounted."""
    monkeypatch.setattr(
        "src.routers.mcp_server.config.FEATURE_MCP_SERVER_REGISTRATION", False
    )
    owner, token = await create_test_user_and_token()
    try:
        response = await async_client.post(
            "/mcp-servers",
            json=_CREATE_PAYLOAD,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 404
    finally:
        await cleanup_models([owner])


@pytest.mark.asyncio
async def test_create_mcp_server_succeeds_when_registration_enabled(
    async_client, monkeypatch
):
    monkeypatch.setattr(
        "src.routers.mcp_server.config.FEATURE_MCP_SERVER_REGISTRATION", True
    )
    owner, token = await create_test_user_and_token()
    created = None
    try:
        response = await async_client.post(
            "/mcp-servers",
            json=_CREATE_PAYLOAD,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["name"] == "created-via-api"
        created = await MCPServer.find_by_id(body["id"])
        assert created is not None
    finally:
        models = [owner]
        if created is not None:
            models.insert(0, created)
        await cleanup_models(models)
