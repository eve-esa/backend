"""MCP tool discovery for the agentic runner."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from src.services.agents.core.interceptors import ErrorLoggingInterceptor
from src.services.agents.graphs_bundle import graphs_base_module
from src.services.mcp.proxy_url import backend_mcp_proxy_url
from src.services.mcp.tool_cache import get_or_load_mcp_tools
from src.services.mcp_auth import get_cognito_token_provider

logger = logging.getLogger(__name__)

LatencyInterceptor = graphs_base_module().LatencyInterceptor

_mcp_adapters_available = False
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    _mcp_adapters_available = True
except Exception:
    MultiServerMCPClient = None  # type: ignore[misc, assignment]


def _build_mcp_connections(
    mcp_server_configs: List[Any],
    *,
    mcp_proxy_bearer_token: Optional[str],
    cognito_auth_header: Optional[str],
) -> tuple[Dict[str, Any], bool]:
    """Build MultiServerMCPClient connection configs from Mongo-backed servers."""
    connections: Dict[str, Any] = {}
    uses_proxy = False

    for srv in mcp_server_configs:
        transport = (
            srv.config.transport.value if srv.config.transport else "streamable_http"
        )
        if transport not in ("streamable_http", "sse"):
            raise ValueError(
                f"MCP server {srv.name!r} uses unsupported transport {transport!r}. "
                "Only 'streamable_http' and 'sse' are supported."
            )

        if not srv.config.url:
            logger.warning("Skipping MCP server %r: missing URL in config", srv.name)
            continue

        headers: Dict[str, str] = dict(srv.config.headers or {})
        proxy_http_url = (
            backend_mcp_proxy_url(srv.name) if mcp_proxy_bearer_token else None
        )
        if proxy_http_url:
            uses_proxy = True
            headers["Authorization"] = f"Bearer {mcp_proxy_bearer_token}"
            url = proxy_http_url
        else:
            if cognito_auth_header and "Authorization" not in headers:
                headers["Authorization"] = cognito_auth_header
            url = srv.config.url

        connections[srv.name] = {
            "transport": "streamable_http" if transport == "streamable_http" else "sse",
            "url": url,
            "headers": headers,
        }

    return connections, uses_proxy


async def _discover_mcp_tools_uncached(
    mcp_server_configs: List[Any],
    *,
    mcp_proxy_bearer_token: Optional[str] = None,
) -> tuple[Any, List[Any], bool]:
    """Discover MCP tools from live servers (cache miss path)."""
    cognito_auth_header: Optional[str] = None
    token_provider = get_cognito_token_provider()
    if token_provider:
        try:
            token = await token_provider.get_token()
            cognito_auth_header = f"Bearer {token}"
        except Exception as exc:
            logger.warning("Failed to obtain Cognito token for MCP auth: %s", exc)

    connections, uses_proxy = _build_mcp_connections(
        mcp_server_configs,
        mcp_proxy_bearer_token=mcp_proxy_bearer_token,
        cognito_auth_header=cognito_auth_header,
    )
    if not connections:
        return MultiServerMCPClient({}), [], uses_proxy

    client = MultiServerMCPClient(
        connections,
        tool_name_prefix=True,
        tool_interceptors=[LatencyInterceptor(), ErrorLoggingInterceptor()],
    )

    async def _load_one(server_name: str) -> tuple[str, Optional[List[Any]]]:
        try:
            server_tools = await client.get_tools(server_name=server_name)
            logger.info(
                "Loaded %d MCP tool(s) from server %r: %s",
                len(server_tools),
                server_name,
                [t.name for t in server_tools],
            )
            return server_name, server_tools
        except Exception as exc:
            logger.error(
                "Failed to load MCP tools from server %r: %s",
                server_name,
                exc,
                exc_info=True,
            )
            return server_name, None

    results = await asyncio.gather(
        *[_load_one(server_name) for server_name in connections]
    )

    tools: List[Any] = []
    failed_servers: List[str] = []
    for server_name, server_tools in results:
        if server_tools is not None:
            tools.extend(server_tools)
        else:
            failed_servers.append(server_name)

    if tools:
        logger.info(
            "Loaded %d MCP tool(s) total from %d/%d MCP server(s)",
            len(tools),
            len(connections) - len(failed_servers),
            len(connections),
        )
    elif failed_servers:
        logger.warning(
            "No MCP tools loaded; all %d configured server(s) failed: %s",
            len(failed_servers),
            failed_servers,
        )
    return client, tools, uses_proxy


async def load_mcp_tools_for_servers(
    mcp_server_configs: List[Any],
    *,
    mcp_proxy_bearer_token: Optional[str] = None,
    mcp_user_id: Optional[str] = None,
) -> List[Any]:
    """Connect to MCP servers and load tools, using the in-process cache when possible."""
    if not _mcp_adapters_available or not mcp_server_configs:
        return []

    return await get_or_load_mcp_tools(
        user_id=mcp_user_id,
        configs=mcp_server_configs,
        mcp_proxy_bearer_token=mcp_proxy_bearer_token,
        loader=lambda: _discover_mcp_tools_uncached(
            mcp_server_configs,
            mcp_proxy_bearer_token=mcp_proxy_bearer_token,
        ),
    )
