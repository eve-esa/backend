"""Tests for resilient per-server MCP tool loading."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.database.models.mcp_server import MCPServer, ToolConfig, ToolTransport
from src.schemas.generation_request import GenerationRequest
from src.services.agents.core.runner import _build_tools, _load_mcp_tools_for_servers

_RUNNER = "src.services.agents.core.runner"


def _mcp_server(name: str, url: str) -> MCPServer:
    return MCPServer(
        name=name,
        config=ToolConfig(url=url, transport=ToolTransport.STREAMABLE_HTTP),
    )


class TestLoadMcpToolsForServers:
    @pytest.mark.asyncio
    async def test_partial_failure_still_returns_other_server_tools(self):
        good_tool = MagicMock(name="good_tool")
        good_tool.name = "effis_search"

        async def get_tools(*, server_name=None):
            if server_name == "effis":
                return [good_tool]
            raise RuntimeError("MCP server unavailable")

        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(side_effect=get_tools)

        with patch(f"{_RUNNER}._mcp_adapters_available", True), patch(
            f"{_RUNNER}.MultiServerMCPClient",
            return_value=mock_client,
        ), patch(f"{_RUNNER}.get_cognito_token_provider", return_value=None), patch(
            f"{_RUNNER}.LatencyInterceptor", return_value=MagicMock()
        ), patch(f"{_RUNNER}.logger"):
            tools = await _load_mcp_tools_for_servers(
                [
                    _mcp_server("effis", "https://effis.example/mcp"),
                    _mcp_server("dummy_image", "https://dummy.example/mcp"),
                ]
            )

        assert tools == [good_tool]
        assert mock_client.get_tools.await_count == 2

    @pytest.mark.asyncio
    async def test_all_servers_fail_returns_empty_list(self):
        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(side_effect=RuntimeError("down"))

        with patch(f"{_RUNNER}._mcp_adapters_available", True), patch(
            f"{_RUNNER}.MultiServerMCPClient",
            return_value=mock_client,
        ), patch(f"{_RUNNER}.get_cognito_token_provider", return_value=None), patch(
            f"{_RUNNER}.LatencyInterceptor", return_value=MagicMock()
        ), patch(f"{_RUNNER}.logger"):
            tools = await _load_mcp_tools_for_servers(
                [_mcp_server("effis", "https://effis.example/mcp")]
            )

        assert tools == []


class TestBuildTools:
    @pytest.mark.asyncio
    async def test_returns_mcp_tools_from_partial_load(self):
        mcp_tool = MagicMock(name="effis_search")
        request = GenerationRequest(query="test")
        request.mcp_server_configs = [_mcp_server("effis", "https://effis.example/mcp")]

        with patch(f"{_RUNNER}._langgraph_available", True), patch(
            f"{_RUNNER}._load_mcp_tools_for_servers",
            AsyncMock(return_value=[mcp_tool]),
        ), patch(f"{_RUNNER}.logger"):
            tools = await _build_tools(request)

        assert tools == [mcp_tool]

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_mcp_servers_configured(self):
        request = GenerationRequest(query="test")

        with patch(f"{_RUNNER}._langgraph_available", True):
            tools = await _build_tools(request)

        assert tools == []
