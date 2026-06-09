"""MCP gateway: user JWT on ingress, Cognito M2M on egress via FastMCP proxy.

Requires ``fastmcp>=3`` so ``create_proxy`` is exported from ``fastmcp.server``; see
https://gofastmcp.com/servers/providers/proxy
"""

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack
from typing import Any

import httpx
from fastmcp import settings as fastmcp_settings
from fastmcp.client.transports.http import StreamableHttpTransport
from fastmcp.server import create_proxy
from fastmcp.server.middleware.caching import ResponseCachingMiddleware
from fastmcp.server.providers.proxy import ProxyClient
from src.config import MCP_TOOLS_CACHE_TTL
from src.database.mongo import get_collection
from src.middlewares.auth import get_user_id_from_bearer_token
from src.services.mcp.auth import CognitoTokenProvider, get_cognito_token_provider
from src.services.mcp.usage import track_usage

logger = logging.getLogger(__name__)

_proxy_apps: dict[str, Any] = {}
_proxy_lifespan_stacks: dict[str, AsyncExitStack] = {}
_proxy_build_lock = asyncio.Lock()

# Cache of (server_name, user_id) → (agentcore_url, expiry_monotonic).
# Avoids a MongoDB round-trip on every proxy request when the result is stable.
# TTL matches MCP_TOOLS_CACHE_TTL so both caches expire together.
# Per-server_name locks ensure at most one concurrent cold-miss fetch.
_server_url_cache: dict[tuple[str, str], tuple[str, float]] = {}
_server_url_locks: dict[str, asyncio.Lock] = {}
_server_url_registry_lock = asyncio.Lock()


async def _get_server_url_lock(server_name: str) -> asyncio.Lock:
    if server_name not in _server_url_locks:
        async with _server_url_registry_lock:
            if server_name not in _server_url_locks:
                _server_url_locks[server_name] = asyncio.Lock()
    return _server_url_locks[server_name]


class _DynamicBearerAuth(httpx.Auth):
    """httpx Auth that fetches a fresh Cognito token on every request.

    This prevents the proxy from getting stuck with a stale cached token after
    the Cognito JWT expires.  ``CognitoTokenProvider.get_token()`` is already
    internally cached and only hits the network when the token approaches
    expiry, so there is no performance penalty on the hot path.

    On a 401 response from AgentCore the auth flow invalidates the token cache
    and retries the request exactly once with a freshly-issued token.  httpx
    supports this two-yield pattern natively via ``async_auth_flow``.
    """

    def __init__(self, provider: CognitoTokenProvider) -> None:
        self._provider = provider

    def auth_flow(self, request: httpx.Request):
        # Sync fallback — should not be reached by httpx.AsyncClient.
        yield request  # pragma: no cover

    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        token = await self._provider.get_token()
        request.headers["Authorization"] = f"Bearer {token}"

        # Yield the request; httpx sends it and feeds the response back via send().
        response = yield request

        if response.status_code != 401:
            # Happy path — nothing more to do.
            return

        # ------------------------------------------------------------------ #
        # AgentCore returned 401.  This is intermittent and happens when the  #
        # cached Cognito token expires between our local expiry check and the  #
        # actual HTTP round-trip (clock skew / narrow expiry window).          #
        # Strategy: invalidate the in-memory cache, fetch a fresh token from  #
        # Cognito, and retry the request exactly once.                         #
        # ------------------------------------------------------------------ #
        url_hint = str(request.url)[:80]
        logger.warning(
            "[MCP proxy] AgentCore returned 401 Unauthorized for %s – "
            "Cognito token may be stale. Invalidating cache and retrying with "
            "a fresh token (single retry).",
            url_hint,
        )
        self._provider.invalidate()
        token = await self._provider.get_token()
        request.headers["Authorization"] = f"Bearer {token}"
        logger.info(
            "[MCP proxy] Retrying request to %s with freshly-issued Cognito token.",
            url_hint,
        )
        yield request


async def build_proxy_app(agentcore_url: str, provider: CognitoTokenProvider):
    """Return a cached ASGI app that proxies MCP to ``agentcore_url`` with Cognito auth.

    The proxy is cached by URL only.  Authentication is handled dynamically
    via ``_DynamicBearerAuth`` so tokens are always fresh regardless of how
    long the proxy has been alive.
    """
    cache_key = agentcore_url
    if cache_key in _proxy_apps:
        return _proxy_apps[cache_key]

    async with _proxy_build_lock:
        if cache_key in _proxy_apps:
            return _proxy_apps[cache_key]
        transport = StreamableHttpTransport(
            agentcore_url,
            auth=_DynamicBearerAuth(provider),
        )
        backend = ProxyClient(transport)
        proxy = create_proxy(backend, name=f"proxy-{agentcore_url[-8:]}")

        # Cache the `tools/list` JSON-RPC response at the proxy server layer so
        # that the MCP SDK's per-call output-schema validation
        # (`ClientSession._validate_tool_result` → `list_tools()` in
        # mcp/client/session.py) does not fan out to AgentCore on every
        # `tools/call`.  FastMCP's built-in `ProxyProvider._tools_cache` only
        # accelerates internal `_get_tool(name)` lookups; it does not
        # short-circuit incoming `tools/list` requests
        # (https://gofastmcp.com/servers/providers/proxy#component-list-caching).
        # Explicitly disable `call_tool` caching — tool execution must always
        # reach upstream (default TTL is 1 h, which would silently serve stale
        # results).
        proxy.add_middleware(
            ResponseCachingMiddleware(
                list_tools_settings={
                    "enabled": True,
                    "ttl": int(MCP_TOOLS_CACHE_TTL),
                },
                list_resources_settings={"enabled": False},
                list_prompts_settings={"enabled": False},
                read_resource_settings={"enabled": False},
                get_prompt_settings={"enabled": False},
                call_tool_settings={"enabled": False},
            )
        )
        http_app = proxy.http_app(stateless_http=True)
        stack = AsyncExitStack()
        await stack.enter_async_context(http_app.lifespan(http_app))
        _proxy_lifespan_stacks[cache_key] = stack
        _proxy_apps[cache_key] = http_app
        logger.info("MCP proxy sub-app started (lifespan): %s", cache_key[:64])
        return http_app


async def shutdown_mcp_proxy_lifespans() -> None:
    """Close all FastMCP ``http_app()`` lifespans (StreamableHTTPSessionManager task groups)."""
    async with _proxy_build_lock:
        for stack in _proxy_lifespan_stacks.values():
            await stack.aclose()
        _proxy_lifespan_stacks.clear()
        _proxy_apps.clear()


class MCPProxyDispatcher:
    """
    Outermost ASGI middleware. Intercepts ``/mcp/{server_name}/*`` requests,
    authenticates the user, resolves the server from MongoDB, rewrites the path,
    and forwards to a FastMCP proxy sub-app (Cognito on upstream).
    All other requests pass through to the FastAPI app unchanged.
    """

    def __init__(self, main_app):
        self.main_app = main_app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path: str = scope.get("path", "")
            if path.startswith("/mcp/"):
                parts = path.split("/")
                if len(parts) >= 3:
                    server_name = parts[2]
                    # ``proxy.http_app()`` registers Streamable HTTP at ``streamable_http_path``
                    # (default ``/mcp``). Rewriting outer ``/mcp/{name}`` to ``/`` produced 404 and
                    # MCP clients surfaced it as ``Session terminated``.
                    mcp_path = (
                        fastmcp_settings.streamable_http_path.rstrip("/") or "/mcp"
                    )
                    tail = "/".join(parts[3:]) if len(parts) > 3 else ""
                    remaining = f"{mcp_path}/{tail}".rstrip("/") if tail else mcp_path
                    try:
                        proxy_app = await self._resolve_proxy(scope, server_name)
                    except PermissionError as exc:
                        await self._send_error(send, 401, str(exc))
                        return
                    except LookupError as exc:
                        await self._send_error(send, 404, str(exc))
                        return
                    except Exception as exc:
                        logger.exception("MCP proxy resolution failed: %s", exc)
                        await self._send_error(send, 503, str(exc))
                        return

                    scope = {**scope, "path": remaining, "raw_path": remaining.encode()}
                    await proxy_app(scope, receive, send)
                    return

        await self.main_app(scope, receive, send)

    async def _resolve_proxy(self, scope, server_name: str):
        headers = dict(scope.get("headers", []))
        auth = headers.get(b"authorization", b"").decode()
        if not auth.startswith("Bearer "):
            raise PermissionError("Missing or malformed Authorization header")

        user_id = await get_user_id_from_bearer_token(auth[7:])

        cache_key = (server_name, user_id)
        now = time.monotonic()
        cached = _server_url_cache.get(cache_key)

        if cached and now < cached[1]:
            agentcore_url = cached[0]
        else:
            lock = await _get_server_url_lock(server_name)
            async with lock:
                # Re-check after acquiring lock — another coroutine may have
                # already refreshed the entry while we were waiting.
                cached = _server_url_cache.get(cache_key)
                if cached and time.monotonic() < cached[1]:
                    agentcore_url = cached[0]
                else:
                    mcp_servers = get_collection("mcp_servers")
                    server = await mcp_servers.find_one(
                        {
                            "name": server_name,
                            "enabled": True,
                            "$or": [{"user_id": user_id}, {"user_id": None}],
                        },
                        {"config.url": 1},
                    )
                    if (
                        not server
                        or not server.get("config")
                        or not server["config"].get("url")
                    ):
                        raise LookupError(
                            f"MCP server '{server_name}' not found or not accessible"
                        )
                    agentcore_url = server["config"]["url"]
                    _server_url_cache[cache_key] = (
                        agentcore_url,
                        time.monotonic() + MCP_TOOLS_CACHE_TTL,
                    )
                    logger.debug(
                        "[MCP proxy] cached URL for server=%r user=%s (TTL=%.0fs)",
                        server_name,
                        user_id,
                        MCP_TOOLS_CACHE_TTL,
                    )

        await track_usage(
            user_id=user_id,
            server_name=server_name,
            request_method=scope.get("method", "UNKNOWN"),
        )

        provider = get_cognito_token_provider()
        if provider is None:
            raise RuntimeError("AgentCore authentication is not configured")

        return await build_proxy_app(agentcore_url, provider)

    @staticmethod
    async def _send_error(send, status: int, detail: str):
        body = json.dumps({"detail": detail}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [[b"content-type", b"application/json"]],
            }
        )
        await send({"type": "http.response.body", "body": body})
