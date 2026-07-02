"""In-process TTL cache for agentic MCP tool discovery.

Caches ``(MultiServerMCPClient, tools)`` so repeat agentic requests skip
per-server ``get_tools()`` round-trips. Mirrors the locking/TTL pattern in
``mcp_proxy._server_url_cache``.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List, Optional

from src.config import MCP_TOOLS_CACHE_TTL

_mcp_tools_cache: dict[str, McpToolsCacheEntry] = {}
_mcp_tools_locks: dict[str, asyncio.Lock] = {}
_registry_lock = asyncio.Lock()


@dataclass
class McpToolsCacheEntry:
    client: Any
    tools: list
    expiry: float
    uses_proxy: bool


def build_mcp_tools_cache_key(user_id: str, configs: list) -> str:
    """Stable cache key from user id and resolved server configs."""
    parts: list[str] = []
    for srv in sorted(configs, key=lambda s: s.name):
        transport = (
            srv.config.transport.value if srv.config.transport else "streamable_http"
        )
        parts.append(f"{srv.name}:{srv.config.url}:{transport}")
    return f"{user_id}|" + "|".join(parts)


def refresh_proxy_bearer_token(client: Any, bearer_token: str) -> None:
    """Update proxy ingress Authorization on cached connection configs."""
    auth_header = f"Bearer {bearer_token}"
    for conn in client.connections.values():
        headers = conn.get("headers")
        if headers is None:
            conn["headers"] = {"Authorization": auth_header}
        else:
            headers["Authorization"] = auth_header


def clear_mcp_tool_cache() -> None:
    """Reset the in-process cache (used by tests and hot reload)."""
    _mcp_tools_cache.clear()
    _mcp_tools_locks.clear()


def _get_cached_entry(cache_key: str) -> Optional[McpToolsCacheEntry]:
    entry = _mcp_tools_cache.get(cache_key)
    if entry is None:
        return None
    if time.monotonic() >= entry.expiry:
        _mcp_tools_cache.pop(cache_key, None)
        return None
    return entry


def _return_cached_tools(
    entry: McpToolsCacheEntry,
    mcp_proxy_bearer_token: Optional[str],
) -> List[Any]:
    if entry.uses_proxy and mcp_proxy_bearer_token:
        refresh_proxy_bearer_token(entry.client, mcp_proxy_bearer_token)
    return entry.tools


async def _get_lock(cache_key: str) -> asyncio.Lock:
    if cache_key not in _mcp_tools_locks:
        async with _registry_lock:
            if cache_key not in _mcp_tools_locks:
                _mcp_tools_locks[cache_key] = asyncio.Lock()
    return _mcp_tools_locks[cache_key]


def _store_entry(
    cache_key: str,
    client: Any,
    tools: list,
    *,
    uses_proxy: bool,
) -> None:
    _mcp_tools_cache[cache_key] = McpToolsCacheEntry(
        client=client,
        tools=tools,
        expiry=time.monotonic() + MCP_TOOLS_CACHE_TTL,
        uses_proxy=uses_proxy,
    )


async def get_or_load_mcp_tools(
    *,
    user_id: Optional[str],
    configs: list,
    mcp_proxy_bearer_token: Optional[str],
    loader: Callable[[], Awaitable[tuple[Any, list, bool]]],
) -> List[Any]:
    """Return cached tools or invoke ``loader`` to discover and cache them.

    ``loader`` must return ``(client, tools, uses_proxy)``. Empty tool lists
    are not cached so transient failures can be retried on the next request.
    """
    if not user_id or not configs:
        _client, tools, _uses_proxy = await loader()
        return tools

    cache_key = build_mcp_tools_cache_key(user_id, configs)
    entry = _get_cached_entry(cache_key)
    if entry is not None:
        return _return_cached_tools(entry, mcp_proxy_bearer_token)

    lock = await _get_lock(cache_key)
    async with lock:
        entry = _get_cached_entry(cache_key)
        if entry is not None:
            return _return_cached_tools(entry, mcp_proxy_bearer_token)

        client, tools, uses_proxy = await loader()
        if tools:
            _store_entry(cache_key, client, tools, uses_proxy=uses_proxy)
        return tools
