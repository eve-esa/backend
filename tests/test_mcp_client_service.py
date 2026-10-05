"""Connection configs built by MultiServerMCPClientService."""

from unittest.mock import patch

import pytest

from src.services.mcp_client_service import MultiServerMCPClientService

pytestmark = pytest.mark.no_db


def test_streamable_http_server_does_not_terminate_on_close():
    """Stateless MCP servers answer the closing session DELETE with 404."""
    with patch(
        "src.services.mcp_client_service.MultiServerMCPClient"
    ) as client_cls:
        MultiServerMCPClientService(
            server_configs={
                "demo": {
                    "transport": "streamable_http",
                    "url": "https://mcp.example/mcp",
                    "headers": {"X-Key": "k"},
                }
            }
        )

    connections = client_cls.call_args.args[0]
    assert connections["demo"] == {
        "url": "https://mcp.example/mcp",
        "transport": "streamable_http",
        "headers": {"X-Key": "k"},
        "terminate_on_close": False,
    }
