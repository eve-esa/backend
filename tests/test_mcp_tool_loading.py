"""Tests for resilient per-server MCP tool loading and discovery cache."""

import gc
import weakref
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.database.models.mcp_server import MCPServer, ToolConfig, ToolTransport
from src.schemas.generation_request import GenerationRequest
from src.services.agents.core.runner import _build_tools, _load_mcp_tools_for_servers
from src.services.mcp.tool_cache import clear_mcp_tool_cache

_RUNNER = "src.services.agents.core.runner"
_LOADER = "src.services.mcp.tool_loader"

pytestmark = pytest.mark.no_db


def _mcp_server(name: str, url: str) -> MCPServer:
    return MCPServer(
        name=name,
        config=ToolConfig(url=url, transport=ToolTransport.STREAMABLE_HTTP),
    )


@pytest.fixture(autouse=True)
def _clear_mcp_tool_cache_fixture():
    clear_mcp_tool_cache()
    yield
    clear_mcp_tool_cache()


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

        with patch(f"{_LOADER}._mcp_adapters_available", True), patch(
            f"{_LOADER}.MultiServerMCPClient",
            return_value=mock_client,
        ), patch(f"{_LOADER}.get_cognito_token_provider", return_value=None), patch(
            f"{_LOADER}.LatencyInterceptor", return_value=MagicMock()
        ), patch(f"{_LOADER}.logger"):
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

        with patch(f"{_LOADER}._mcp_adapters_available", True), patch(
            f"{_LOADER}.MultiServerMCPClient",
            return_value=mock_client,
        ), patch(f"{_LOADER}.get_cognito_token_provider", return_value=None), patch(
            f"{_LOADER}.LatencyInterceptor", return_value=MagicMock()
        ), patch(f"{_LOADER}.logger"):
            tools = await _load_mcp_tools_for_servers(
                [_mcp_server("effis", "https://effis.example/mcp")]
            )

        assert tools == []

    @pytest.mark.asyncio
    async def test_routes_through_proxy_when_bearer_token_and_base_url_set(self):
        captured_connections: dict = {}

        def make_client(connections, **kwargs):
            captured_connections.update(connections)
            mock_client = MagicMock()
            mock_client.get_tools = AsyncMock(return_value=[])
            return mock_client

        with patch(f"{_LOADER}._mcp_adapters_available", True), patch(
            f"{_LOADER}.MultiServerMCPClient", side_effect=make_client
        ), patch(
            f"{_LOADER}.backend_mcp_proxy_url",
            return_value="http://127.0.0.1:8000/mcp/effis",
        ), patch(f"{_LOADER}.get_cognito_token_provider", return_value=None), patch(
            f"{_LOADER}.LatencyInterceptor", return_value=MagicMock()
        ), patch(f"{_LOADER}.ErrorLoggingInterceptor", return_value=MagicMock()), patch(
            f"{_LOADER}.logger"
        ):
            await _load_mcp_tools_for_servers(
                [_mcp_server("effis", "https://agentcore.example/mcp")],
                mcp_proxy_bearer_token="user-jwt",
            )

        assert captured_connections["effis"]["url"] == "http://127.0.0.1:8000/mcp/effis"
        assert captured_connections["effis"]["headers"]["Authorization"] == (
            "Bearer user-jwt"
        )

    @pytest.mark.asyncio
    async def test_client_survives_gc_after_load_returns(self):
        """Live-repro bug: ArtifactInterceptor.bind_client() only keeps a
        weakref to the MultiServerMCPClient (see artifact_ingestion.py). The
        local ``client`` variable inside the tool_loader discovery path used
        to be the only strong reference, so it was garbage-collected the
        moment the loader returned — silently breaking ResourceLink
        resolution for every subsequent tool call in the run. The returned
        tools list must itself keep the client alive, even on the
        no-user-id / uncached path (see ``test_no_user_id_bypasses_cache``).
        """
        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(return_value=[])

        with patch(f"{_LOADER}._mcp_adapters_available", True), patch(
            f"{_LOADER}.MultiServerMCPClient",
            return_value=mock_client,
        ), patch(f"{_LOADER}.get_cognito_token_provider", return_value=None), patch(
            f"{_LOADER}.LatencyInterceptor", return_value=MagicMock()
        ), patch(f"{_LOADER}.ErrorLoggingInterceptor", return_value=MagicMock()), patch(
            f"{_LOADER}.logger"
        ):
            tools = await _load_mcp_tools_for_servers(
                [_mcp_server("effis", "https://effis.example/mcp")]
            )

        ref = weakref.ref(mock_client)
        del mock_client
        gc.collect()
        assert ref() is not None, (
            "the MultiServerMCPClient was garbage-collected even though the "
            "returned tools list should hold a strong reference to it"
        )
        assert getattr(tools, "_mcp_client", None) is ref()

    @pytest.mark.asyncio
    async def test_cache_hit_skips_second_get_tools(self):
        good_tool = MagicMock()
        good_tool.name = "effis_search"
        mock_client = MagicMock()
        mock_client.connections = {
            "effis": {
                "url": "http://127.0.0.1:8000/mcp/effis",
                "headers": {"Authorization": "Bearer old-jwt"},
            }
        }
        mock_client.get_tools = AsyncMock(return_value=[good_tool])
        configs = [_mcp_server("effis", "https://effis.example/mcp")]

        with patch(f"{_LOADER}._mcp_adapters_available", True), patch(
            f"{_LOADER}.MultiServerMCPClient",
            return_value=mock_client,
        ), patch(
            f"{_LOADER}.backend_mcp_proxy_url",
            return_value="http://127.0.0.1:8000/mcp/effis",
        ), patch(f"{_LOADER}.get_cognito_token_provider", return_value=None), patch(
            f"{_LOADER}.LatencyInterceptor", return_value=MagicMock()
        ), patch(f"{_LOADER}.ErrorLoggingInterceptor", return_value=MagicMock()), patch(
            f"{_LOADER}.logger"
        ):
            first = await _load_mcp_tools_for_servers(
                configs,
                mcp_proxy_bearer_token="old-jwt",
                mcp_user_id="user-1",
            )
            second = await _load_mcp_tools_for_servers(
                configs,
                mcp_proxy_bearer_token="new-jwt",
                mcp_user_id="user-1",
            )

        assert first == second == [good_tool]
        assert mock_client.get_tools.await_count == 1
        assert mock_client.connections["effis"]["headers"]["Authorization"] == (
            "Bearer new-jwt"
        )

    @pytest.mark.asyncio
    async def test_cache_miss_when_server_config_changes(self):
        good_tool = MagicMock()
        good_tool.name = "effis_search"
        mock_client = MagicMock()
        mock_client.connections = {}
        mock_client.get_tools = AsyncMock(return_value=[good_tool])

        with patch(f"{_LOADER}._mcp_adapters_available", True), patch(
            f"{_LOADER}.MultiServerMCPClient",
            return_value=mock_client,
        ), patch(f"{_LOADER}.get_cognito_token_provider", return_value=None), patch(
            f"{_LOADER}.LatencyInterceptor", return_value=MagicMock()
        ), patch(f"{_LOADER}.ErrorLoggingInterceptor", return_value=MagicMock()), patch(
            f"{_LOADER}.logger"
        ):
            await _load_mcp_tools_for_servers(
                [_mcp_server("effis", "https://effis.example/mcp")],
                mcp_user_id="user-1",
            )
            await _load_mcp_tools_for_servers(
                [_mcp_server("effis", "https://effis.example/mcp-v2")],
                mcp_user_id="user-1",
            )

        assert mock_client.get_tools.await_count == 2

    @pytest.mark.asyncio
    async def test_no_user_id_bypasses_cache(self):
        good_tool = MagicMock()
        good_tool.name = "effis_search"
        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(return_value=[good_tool])
        configs = [_mcp_server("effis", "https://effis.example/mcp")]

        with patch(f"{_LOADER}._mcp_adapters_available", True), patch(
            f"{_LOADER}.MultiServerMCPClient",
            return_value=mock_client,
        ), patch(f"{_LOADER}.get_cognito_token_provider", return_value=None), patch(
            f"{_LOADER}.LatencyInterceptor", return_value=MagicMock()
        ), patch(f"{_LOADER}.ErrorLoggingInterceptor", return_value=MagicMock()), patch(
            f"{_LOADER}.logger"
        ):
            await _load_mcp_tools_for_servers(configs)
            await _load_mcp_tools_for_servers(configs)

        assert mock_client.get_tools.await_count == 2


class TestBuildTools:
    @pytest.mark.asyncio
    async def test_returns_mcp_tools_from_partial_load(self):
        mcp_tool = MagicMock(name="effis_search")
        request = GenerationRequest(query="test")
        request.mcp_server_configs = [_mcp_server("effis", "https://effis.example/mcp")]

        with patch(f"{_RUNNER}._langgraph_available", True), patch(
            f"{_RUNNER}._load_mcp_tools_for_servers",
            AsyncMock(return_value=[mcp_tool]),
        ) as load_mock, patch(f"{_RUNNER}.logger"):
            request.mcp_proxy_bearer_token = "user-jwt"
            request.mcp_user_id = "user-1"
            tools = await _build_tools(request)

        load_mock.assert_awaited_once_with(
            request.mcp_server_configs,
            mcp_proxy_bearer_token="user-jwt",
            mcp_user_id="user-1",
        )
        assert tools == [mcp_tool]

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_mcp_servers_configured(self):
        request = GenerationRequest(query="test")

        with patch(f"{_RUNNER}._langgraph_available", True):
            tools = await _build_tools(request)

        assert tools == []

    @pytest.mark.asyncio
    async def test_client_keepalive_survives_tools_extend(self):
        """_build_tools copies elements via tools.extend(mcp_tools) — a plain
        list.extend does NOT carry over mcp_tools's own `_mcp_client`
        attribute, so the client reference must be re-attached to the final
        list _build_tools returns, or it's lost right where the caller needs
        it kept alive for the whole graph run.
        """
        from src.services.agents.core.runner import _MCPToolsWithClient

        mcp_tool = MagicMock(name="effis_search")
        sentinel_client = MagicMock()
        wrapped = _MCPToolsWithClient([mcp_tool], sentinel_client)

        request = GenerationRequest(query="test")
        request.mcp_server_configs = [_mcp_server("effis", "https://effis.example/mcp")]

        with patch(f"{_RUNNER}._langgraph_available", True), patch(
            f"{_RUNNER}._load_mcp_tools_for_servers",
            AsyncMock(return_value=wrapped),
        ), patch(f"{_RUNNER}.logger"):
            tools = await _build_tools(request)

        assert tools == [mcp_tool]
        assert getattr(tools, "_mcp_client", None) is sentinel_client


class TestMcpToolCache:
    def test_ttl_expiry_drops_stale_entry(self):
        from src.services.mcp import tool_cache

        good_tool = MagicMock()
        client = MagicMock()
        client.connections = {}
        cache_key = "user-1|effis:https://x/mcp:streamable_http"

        tool_cache._store_entry(cache_key, client, [good_tool], uses_proxy=False)
        tool_cache._mcp_tools_cache[cache_key].expiry = 0.0

        assert tool_cache._get_cached_entry(cache_key) is None
        assert cache_key not in tool_cache._mcp_tools_cache
