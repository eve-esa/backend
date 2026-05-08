"""MCP gateway: user JWT on ingress, Cognito M2M on egress via FastMCP proxy.

Requires ``fastmcp>=3`` so ``create_proxy`` is exported from ``fastmcp.server``; see
https://gofastmcp.com/servers/providers/proxy
"""

import asyncio
import base64
import json
import logging
from contextlib import AsyncExitStack
from typing import Any

from fastmcp import settings as fastmcp_settings
from fastmcp.client.transports.http import StreamableHttpTransport
from fastmcp.server import create_proxy
from fastmcp.server.providers.proxy import ProxyClient

from src.database.mongo import get_collection
from src.services.mcp.auth import get_cognito_token_provider
from src.services.mcp.usage import track_usage

logger = logging.getLogger(__name__)

_proxy_apps: dict[str, Any] = {}
_proxy_lifespan_stacks: dict[str, AsyncExitStack] = {}
_proxy_build_lock = asyncio.Lock()


async def build_proxy_app(agentcore_url: str, cognito_token: str):
    """Return a cached ASGI app that proxies MCP to ``agentcore_url`` with Cognito auth."""
    cache_key = f"{agentcore_url}:{cognito_token[:16]}"
    if cache_key in _proxy_apps:
        return _proxy_apps[cache_key]

    async with _proxy_build_lock:
        if cache_key in _proxy_apps:
            return _proxy_apps[cache_key]
        transport = StreamableHttpTransport(
            agentcore_url,
            auth=cognito_token,
        )
        backend = ProxyClient(transport)
        proxy = create_proxy(backend, name=f"proxy-{agentcore_url[-8:]}")
        http_app = proxy.http_app()
        stack = AsyncExitStack()
        await stack.enter_async_context(http_app.lifespan(http_app))
        _proxy_lifespan_stacks[cache_key] = stack
        _proxy_apps[cache_key] = http_app
        logger.info("MCP proxy sub-app started (lifespan): %s", cache_key[:48])
        return http_app


async def shutdown_mcp_proxy_lifespans() -> None:
    """Close all FastMCP ``http_app()`` lifespans (StreamableHTTPSessionManager task groups)."""
    async with _proxy_build_lock:
        for stack in _proxy_lifespan_stacks.values():
            await stack.aclose()
        _proxy_lifespan_stacks.clear()
        _proxy_apps.clear()


def _decode_jwt_claims_unverified(token: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) < 2:
        raise PermissionError("Invalid token format")
    payload = parts[1]
    padding = "=" * (-len(payload) % 4)
    try:
        raw = base64.urlsafe_b64decode(payload + padding)
        claims = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise PermissionError("Invalid token payload") from exc
    if not isinstance(claims, dict):
        raise PermissionError("Invalid token payload")
    return claims


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

        claims = _decode_jwt_claims_unverified(auth[7:])
        user_id = claims.get("sub")
        if not user_id:
            raise PermissionError("Invalid token payload")

        mcp_servers = get_collection("mcp_servers")
        server = await mcp_servers.find_one(
            {
                "name": server_name,
                "enabled": True,
                "$or": [{"user_id": user_id}, {"user_id": None}],
            },
            {"config.url": 1},
        )
        if not server or not server.get("config") or not server["config"].get("url"):
            raise LookupError(f"MCP server '{server_name}' not found or not accessible")

        await track_usage(
            user_id=user_id,
            server_name=server_name,
            request_method=scope.get("method", "UNKNOWN"),
        )

        provider = get_cognito_token_provider()
        if provider is None:
            raise RuntimeError("AgentCore authentication is not configured")
        cognito_token = await provider.get_token()

        return await build_proxy_app(server["config"]["url"], cognito_token)

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
