"""
Test the hybrid MCP client implementation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.mcp_client_service import MCPClientService
from src.config import Config


class TestMCPClientService:
    """Test the hybrid MCP client service."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = MagicMock(spec=Config)
        config.get_mcp_server_url.return_value = "https://test-mcp-server.com/mcp"
        config.get_mcp_headers.return_value = {"Authorization": "Bearer test-token"}
        return config

    @pytest.mark.asyncio
    async def test_initialization_with_fastapi_mcp_available(self, mock_config):
        """Test that the service initializes correctly with FastAPI MCP Client available."""
        with patch("src.services.mcp_client_service.FASTAPI_MCP_AVAILABLE", True):
            with patch("src.services.mcp_client_service.MCPClient") as mock_mcp_client:
                mock_client_instance = MagicMock()
                mock_mcp_client.return_value = mock_client_instance

                service = MCPClientService(config=mock_config)

                assert service._fastapi_mcp_client is not None
                mock_mcp_client.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialization_without_fastapi_mcp(self, mock_config):
        """Test that the service initializes correctly without FastAPI MCP Client."""
        with patch("src.services.mcp_client_service.FASTAPI_MCP_AVAILABLE", False):
            service = MCPClientService(config=mock_config)

            assert service._fastapi_mcp_client is None

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_config):
        """Test that the service works as an async context manager."""
        with patch("src.services.mcp_client_service.FASTAPI_MCP_AVAILABLE", True):
            with patch("src.services.mcp_client_service.MCPClient") as mock_mcp_client:
                mock_client_instance = AsyncMock()
                mock_mcp_client.return_value = mock_client_instance

                async with MCPClientService(config=mock_config) as service:
                    assert service is not None

                # Verify close was called
                mock_client_instance.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_tool_stream_with_fastapi_mcp(self, mock_config):
        """Test streaming tool call with FastAPI MCP Client."""
        with patch("src.services.mcp_client_service.FASTAPI_MCP_AVAILABLE", True):
            with patch("src.services.mcp_client_service.MCPClient") as mock_mcp_client:
                mock_client_instance = AsyncMock()
                mock_client_instance.call_operation.return_value = self._mock_stream()
                mock_mcp_client.return_value = mock_client_instance

                service = MCPClientService(config=mock_config)

                events = []
                async for event in service.call_tool_stream(
                    "test_tool", {"arg": "value"}
                ):
                    events.append(event)

                assert len(events) == 3
                assert events[0] == {"type": "start", "tool": "test_tool"}
                assert events[1] == {"type": "progress", "data": "processing"}
                assert events[2] == {"type": "complete", "result": "success"}

    @pytest.mark.asyncio
    async def test_call_tool_stream_fallback(self, mock_config):
        """Test streaming tool call falls back to regular call when FastAPI MCP fails."""
        with patch("src.services.mcp_client_service.FASTAPI_MCP_AVAILABLE", True):
            with patch("src.services.mcp_client_service.MCPClient") as mock_mcp_client:
                mock_client_instance = AsyncMock()
                mock_client_instance.call_operation.side_effect = Exception(
                    "Streaming failed"
                )
                mock_mcp_client.return_value = mock_client_instance

                service = MCPClientService(config=mock_config)

                # Mock the regular call_tool method
                service.call_tool = AsyncMock(
                    return_value={"content": ["fallback result"]}
                )

                events = []
                async for event in service.call_tool_stream(
                    "test_tool", {"arg": "value"}
                ):
                    events.append(event)

                assert len(events) == 1
                assert events[0] == {"content": ["fallback result"]}

    def _mock_stream(self):
        """Create a mock async stream."""

        async def stream():
            yield {"type": "start", "tool": "test_tool"}
            yield {"type": "progress", "data": "processing"}
            yield {"type": "complete", "result": "success"}

        return stream()


class TestMCPClientRouter:
    """Test the MCP client router endpoints."""

    @pytest.mark.asyncio
    async def test_get_mcp_status(self):
        """Test the MCP status endpoint."""
        from src.routers.mcp_client import get_mcp_status

        with patch(
            "src.services.mcp_client_service.MCPClientService"
        ) as mock_service_class:
            mock_service = AsyncMock()
            mock_service_class.return_value.__aenter__.return_value = mock_service
            mock_service.list_tools.return_value = [
                {"name": "tool1"},
                {"name": "tool2"},
            ]
            mock_service._fastapi_mcp_client = MagicMock()
            mock_service._server_url = "https://test-server.com/mcp"

            result = await get_mcp_status()

            assert result["status"] == "connected"
            assert result["fastapi_mcp_client_available"] is True
            assert result["streaming_supported"] is True
            assert result["tools_count"] == 2
            assert result["server_url"] == "https://test-server.com/mcp"

    @pytest.mark.asyncio
    async def test_get_mcp_status_error(self):
        """Test the MCP status endpoint with error."""
        from src.routers.mcp_client import get_mcp_status

        with patch(
            "src.services.mcp_client_service.MCPClientService"
        ) as mock_service_class:
            mock_service_class.side_effect = Exception("Connection failed")

            result = await get_mcp_status()

            assert result["status"] == "error"
            assert result["error"] == "Connection failed"
            assert result["fastapi_mcp_client_available"] is False
            assert result["streaming_supported"] is False
            assert result["tools_count"] == 0
