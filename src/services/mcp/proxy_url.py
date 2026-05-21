"""Public MCP proxy URL helpers (must stay in sync with ``MCPProxyDispatcher`` paths)."""

from typing import Optional
from urllib.parse import quote

from src.config import MCP_PROXY_BASE_URL, MCP_PROXY_INTERNAL_BASE_URL


def resolve_public_mcp_url(server_name: str) -> str:
    """URL returned to clients; absolute when ``MCP_PROXY_BASE_URL`` is set, else relative."""
    encoded = quote(server_name, safe="")
    base = (MCP_PROXY_BASE_URL or "").strip().rstrip("/")
    if base:
        return f"{base}/mcp/{encoded}"
    return f"/mcp/{encoded}"


def backend_mcp_proxy_url(server_name: str) -> Optional[str]:
    """Full proxy URL for server-side MCP clients (e.g. LangChain ``MultiServerMCPClient``).

    Prefers ``MCP_PROXY_INTERNAL_BASE_URL`` so the backend can reach its own ASGI stack
    when ``MCP_PROXY_BASE_URL`` points at a host-only URL (typical Docker port publish).
    """
    internal = (MCP_PROXY_INTERNAL_BASE_URL or "").strip().rstrip("/")
    public = (MCP_PROXY_BASE_URL or "").strip().rstrip("/")
    base = internal or public
    if not base:
        return None
    encoded = quote(server_name, safe="")
    return f"{base}/mcp/{encoded}"
