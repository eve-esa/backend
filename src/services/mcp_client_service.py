from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from src.config import Config, WILEY_AUTH_TOKEN, WILEY_MCP_TIMEOUT_S
from src.database.models.mcp_server import MCPServer, ToolTransport

from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.websocket import websocket_client
from mcp.client.session import ClientSession
from urllib.parse import quote
from contextlib import AsyncExitStack


logger = logging.getLogger(__name__)


def _extract_exception_details(exc: BaseException) -> List[str]:
    """Flatten nested ExceptionGroup errors into readable detail lines."""
    details: List[str] = []
    if isinstance(exc, BaseExceptionGroup):
        for sub_exc in exc.exceptions:
            details.extend(_extract_exception_details(sub_exc))
    else:
        details.append(f"{type(exc).__name__}: {exc}")
    return details


# Scholar Gateway v2 authenticates every request with a static API key in this
# header. There is no OAuth exchange and no access token to cache or refresh.
WILEY_API_KEY_HEADER = "X-API-Key"
_WILEY_HOST = "scholargateway.ai"


def is_wiley_url(url: Optional[str]) -> bool:
    """True when ``url`` points at the Wiley Scholar Gateway."""
    if not url:
        return False
    host = (urlparse(url).hostname or "").lower()
    return host == _WILEY_HOST or host.endswith("." + _WILEY_HOST)


def with_wiley_api_key(
    url: Optional[str], headers: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Return a copy of ``headers`` carrying the Wiley API key for a Wiley URL.

    The key is sent as is, never with a Bearer prefix, and only to the Wiley
    host. A header already configured for the server wins. With
    ``WILEY_AUTH_TOKEN`` unset the headers are returned without a key.
    """
    merged = dict(headers or {})
    if not is_wiley_url(url):
        return merged
    if any(k.lower() == WILEY_API_KEY_HEADER.lower() for k in merged):
        return merged
    if WILEY_AUTH_TOKEN:
        merged[WILEY_API_KEY_HEADER] = WILEY_AUTH_TOKEN
    return merged


def _has_api_key_header(headers: Dict[str, Any]) -> bool:
    return any(k.lower() == WILEY_API_KEY_HEADER.lower() for k in headers)


class MultiServerMCPClientService:
    """Service to interact with multiple MCP servers using LangChain's MultiServerMCPClient."""

    # Shared singleton across the process
    _shared_instance: Optional["MultiServerMCPClientService"] = None

    def __init__(
        self,
        config: Optional[Config] = None,
        server_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self._config = config or Config()
        self._client: Optional[MultiServerMCPClient] = None

        # Load server configurations
        self._load_server_configs(server_configs=server_configs)

        if not self._client:
            raise ValueError(
                "No MCP servers configured or failed to initialize MultiServerMCPClient"
            )

    @staticmethod
    def build_server_config_from_model(
        mcp_server: MCPServer,
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """Build MultiServerMCPClient config from a persisted MCPServer model."""
        if not mcp_server.config or not mcp_server.config.transport:
            return None

        server_config: Optional[Dict[str, Any]] = None
        if mcp_server.config.transport == ToolTransport.STDIO:
            if not mcp_server.config.command:
                return None
            server_config = {
                "transport": "stdio",
                "command": mcp_server.config.command,
                "args": mcp_server.config.args or [],
                "env": mcp_server.config.env or {},
                "enabled": mcp_server.enabled,
            }
        elif mcp_server.config.transport == ToolTransport.STREAMABLE_HTTP:
            if not mcp_server.config.url:
                return None
            server_config = {
                "transport": "streamable_http",
                "url": mcp_server.config.url,
                "headers": mcp_server.config.headers or {},
                "enabled": mcp_server.enabled,
            }

        if not server_config:
            return None

        return {mcp_server.name: server_config}

    @classmethod
    def from_mcp_server_model(cls, mcp_server: MCPServer) -> "MultiServerMCPClientService":
        server_configs = cls.build_server_config_from_model(mcp_server)
        if not server_configs:
            raise ValueError(f"MCP server '{mcp_server.name}' has invalid configuration")
        return cls(server_configs=server_configs)

    def _load_server_configs(
        self, server_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> None:
        """Load MCP server configurations and initialize MultiServerMCPClient."""
        if server_configs is None:
            server_configs = self._config.get_mcp_servers()

        if not server_configs:
            logger.warning("No MCP servers configured")
            return

        # Filter enabled servers and prepare configuration for MultiServerMCPClient
        enabled_servers = {}

        for server_name, server_config in server_configs.items():
            if not server_config.get("enabled", True):
                continue

            transport = server_config.get("transport")
            if not transport:
                logger.warning(
                    f"Skipping server '{server_name}' - no transport configured"
                )
                continue

            # Prepare server config based on transport type
            if transport == "stdio":
                # For stdio transport, need command and args
                command = server_config.get("command")
                args = server_config.get("args")
                if not command or not args:
                    logger.warning(
                        f"Skipping server '{server_name}' - stdio transport requires command and args"
                    )
                    continue

                enabled_servers[server_name] = {
                    "command": command,
                    "args": args,
                    "transport": transport,
                }

            elif transport in ["streamable_http", "websocket"]:
                # For HTTP/WebSocket transport, need URL and optionally headers
                url = server_config.get("url")
                if not url:
                    logger.warning(
                        f"Skipping server '{server_name}' - {transport} transport requires URL"
                    )
                    continue

                server_config_for_client = {"url": url, "transport": transport}
                if transport == "streamable_http":
                    # Stateless servers answer the closing DELETE with 404.
                    server_config_for_client["terminate_on_close"] = False

                # Add headers if present; a Wiley server also gets its API key,
                # so tool listing through MultiServerMCPClient is authenticated.
                headers = with_wiley_api_key(url, server_config.get("headers"))
                if headers:
                    server_config_for_client["headers"] = headers

                enabled_servers[server_name] = server_config_for_client

            else:
                logger.warning(
                    f"Skipping server '{server_name}' - unsupported transport: {transport}"
                )
                continue

        if not enabled_servers:
            logger.error("No valid MCP server configurations found")
            return

        try:
            # Initialize MultiServerMCPClient with enabled servers
            self._client = MultiServerMCPClient(enabled_servers)
            logger.info(
                f"MultiServerMCPClient initialized with {len(enabled_servers)} servers: {list(enabled_servers.keys())}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize MultiServerMCPClient: {e}")
            self._client = None

    @classmethod
    def get_shared(
        cls, config: Optional[Config] = None
    ) -> "MultiServerMCPClientService":
        """Return a process-wide shared instance, initializing once on first use."""
        if cls._shared_instance is None:
            cls._shared_instance = cls(config=config)
        return cls._shared_instance

    async def list_tools_from_server(self, server_name: str) -> List[Dict[str, Any]]:
        """List available tools from a specific MCP server."""
        if not self._client:
            raise RuntimeError("MultiServerMCPClient not initialized")

        try:
            # Get tools from the specific server
            tools = await self._client.get_tools(server_name=server_name)

            # Convert to our standard format
            formatted_tools = []
            for tool in tools:
                tool_data = {
                    "name": tool.name,
                    "description": getattr(tool, "description", None),
                    "server": server_name,
                }
                # Parse JSON string in description field if present
                if isinstance(tool_data.get("description"), str):
                    try:
                        import json

                        tool_data["description"] = json.loads(tool_data["description"])
                    except (json.JSONDecodeError, TypeError):
                        # Keep as string if parsing fails
                        pass
                formatted_tools.append(tool_data)

            return formatted_tools
        except Exception as e:
            logger.exception("Failed to list tools from server '%s'", server_name)
            details = _extract_exception_details(e)
            if details:
                logger.error(
                    "Tool discovery error details for '%s': %s",
                    server_name,
                    " | ".join(details),
                )
            return []

    async def list_tools_from_all_servers(self) -> List[Dict[str, Any]]:
        """List available tools from all enabled MCP servers."""
        if not self._client:
            raise RuntimeError("MultiServerMCPClient not initialized")

        all_tools = []

        try:
            # Get tools from all servers
            tools = await self._client.get_tools()

            # Convert to our standard format
            for tool in tools:
                tool_data = {
                    "name": tool.name,
                    "description": getattr(tool, "description", None),
                    "server": getattr(tool, "server", "unknown"),
                }
                # Parse JSON string in description field if present
                if isinstance(tool_data.get("description"), str):
                    try:
                        import json

                        tool_data["description"] = json.loads(tool_data["description"])
                    except (json.JSONDecodeError, TypeError):
                        # Keep as string if parsing fails
                        pass
                all_tools.append(tool_data)

        except Exception as e:
            logger.exception("Failed to list tools from all servers")
            details = _extract_exception_details(e)
            if details:
                logger.error(
                    "Tool discovery error details for all servers: %s",
                    " | ".join(details),
                )
            return []

        return all_tools

    async def call_tool_on_server(
        self, server_name: str, name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Call a specific tool on a specific MCP server."""
        if not self._client:
            raise RuntimeError("MultiServerMCPClient not initialized")

        try:
            return await self._call_tool_with_original_client(
                server_name, name, arguments
            )

        except Exception as e:
            logger.error(f"Failed to call tool '{name}' on server '{server_name}': {e}")
            return {
                "content": [],
                "is_error": True,
                "server": server_name,
                "tool_name": name,
                "arguments": arguments or {},
                "error": str(e),
            }

    async def _call_tool_with_original_client(
        self, server_name: str, name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Call tool over HTTP or WebSocket transport."""
        server_configs = self._config.get_mcp_servers()
        if server_name not in server_configs:
            raise ValueError(f"Server '{server_name}' not found in configuration")

        server_config = server_configs[server_name]
        logger.debug(
            f"Server config for {server_name}: {server_config} (type: {type(server_config)})"
        )
        transport = server_config.get("transport")

        if transport == "stdio":
            return await self._call_tool_over_stdio(server_config, name, arguments)
        elif transport in ["streamable_http", "websocket"]:
            return await self._call_tool_over_network(server_config, name, arguments)
        else:
            return {
                "content": [],
                "is_error": True,
                "server": server_name,
                "tool_name": name,
                "arguments": arguments or {},
                "error": f"Unsupported transport: {transport}",
            }

    async def _call_tool_over_stdio(
        self,
        server_config: Dict[str, Any],
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Call tool over stdio transport."""
        logger.debug(
            f"_call_tool_over_stdio called with server_config: {server_config} (type: {type(server_config)})"
        )
        logger.debug(
            f"server_config.get('command'): {server_config.get('command') if hasattr(server_config, 'get') else 'NO GET METHOD'}"
        )

        command = server_config.get("command")
        args = server_config.get("args", [])

        if not command:
            raise ValueError("Command not configured for stdio transport")

        # For stdio transport, we need to use the mcp.client.stdio module
        try:
            from mcp.client.stdio import stdio_client, StdioServerParameters
        except ImportError:
            raise RuntimeError("stdio transport requires mcp.client.stdio module")

        stack = AsyncExitStack()
        await stack.__aenter__()

        try:
            # Create StdioServerParameters object
            env = server_config.get("env", {})
            stdio_params = StdioServerParameters(command=command, args=args, env=env)

            # Create stdio transport
            transport_obj = await stack.enter_async_context(stdio_client(stdio_params))

            # Handle different transport return types
            if len(transport_obj) == 3:
                read, write, _get_session_id = transport_obj
            else:
                read, write = transport_obj

            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            result = await session.call_tool(name=name, arguments=arguments or {})

            # Convert to our standard format
            def _serialize_content_item(item: Any) -> Any:
                if hasattr(item, "model_dump"):
                    data = item.model_dump()
                    if isinstance(data.get("text"), str):
                        try:
                            import json

                            data["text"] = json.loads(data["text"])
                        except (json.JSONDecodeError, TypeError):
                            pass
                    return data
                try:
                    return dict(item)
                except Exception:
                    return str(item)

            processed = {
                "content": [_serialize_content_item(c) for c in result.content],
                "is_error": getattr(result, "is_error", None)
                or getattr(result, "isError", None)
                or False,
                "server": "unknown",  # We don't have server name in this context
                "tool_name": name,
                "arguments": arguments or {},
            }

            return processed

        finally:
            try:
                await stack.aclose()
            except BaseException as e:
                logger.debug(f"Ignoring transport teardown error: {e}")

    async def _call_tool_over_network(
        self,
        server_config: Dict[str, Any],
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Call tool over HTTP or WebSocket transport."""
        url = server_config.get("url")
        transport = server_config.get("transport")
        # Start with configured headers; a Wiley server gets its API key.
        headers = with_wiley_api_key(url, server_config.get("headers"))
        if is_wiley_url(url) and not _has_api_key_header(headers):
            logger.error("Wiley API key missing: WILEY_AUTH_TOKEN is not set")
            raise RuntimeError("WILEY_AUTH_TOKEN is not set in environment")

        # If Authorization provided but without Bearer prefix, add it
        auth_value = headers.get("Authorization")
        if (
            auth_value
            and not auth_value.startswith("Bearer ")
            and not auth_value.startswith("Basic ")
        ):
            headers["Authorization"] = f"Bearer {auth_value}"

        if not url:
            raise ValueError("URL not configured for network transport")

        stack = AsyncExitStack()
        await stack.__aenter__()

        try:
            if transport == "websocket":
                # WebSocket transport
                ws_url = url
                auth_value = headers.get("Authorization", "")
                if auth_value:
                    sep = "&" if "?" in ws_url else "?"
                    ws_url = f"{ws_url}{sep}auth_header=Authorization&auth_value={quote(auth_value)}"
                transport_obj = await stack.enter_async_context(
                    websocket_client(ws_url)
                )
            else:
                # HTTP transport
                transport_obj = await stack.enter_async_context(
                    streamablehttp_client(
                        url, headers=headers, terminate_on_close=False
                    )
                )

            # Handle different transport return types
            if len(transport_obj) == 3:
                read, write, _get_session_id = transport_obj
            else:
                read, write = transport_obj

            # Deadline per MCP request (initialize, call_tool): a timeout
            # raises McpError, which call_tool_on_server turns into is_error.
            session = await stack.enter_async_context(
                ClientSession(
                    read,
                    write,
                    read_timeout_seconds=timedelta(seconds=WILEY_MCP_TIMEOUT_S),
                )
            )
            await session.initialize()
            result = await session.call_tool(name=name, arguments=arguments or {})

            # Convert to our standard format
            def _serialize_content_item(item: Any) -> Any:
                if hasattr(item, "model_dump"):
                    data = item.model_dump()
                    if isinstance(data.get("text"), str):
                        try:
                            import json

                            data["text"] = json.loads(data["text"])
                        except (json.JSONDecodeError, TypeError):
                            pass
                    return data
                try:
                    return dict(item)
                except Exception:
                    return str(item)

            processed = {
                "content": [_serialize_content_item(c) for c in result.content],
                "is_error": getattr(result, "is_error", None)
                or getattr(result, "isError", None)
                or False,
                "server": "unknown",  # We don't have server name in this context
                "tool_name": name,
                "arguments": arguments or {},
            }

            return processed

        finally:
            try:
                await stack.aclose()
            except BaseException as e:
                logger.debug(f"Ignoring transport teardown error: {e}")

    async def call_tool_on_all_servers(
        self, name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Call a specific tool on all enabled MCP servers and aggregate results."""
        if not self._client:
            raise RuntimeError("MultiServerMCPClient not initialized")

        results = {}

        try:
            # Get server names from the client
            server_names = list(self._client.connections.keys())

            for server_name in server_names:
                try:
                    result = await self.call_tool_on_server(
                        server_name, name, arguments
                    )
                    results[server_name] = result
                except Exception as e:
                    logger.error(
                        f"Failed to call tool '{name}' on server '{server_name}': {e}"
                    )
                    results[server_name] = {
                        "content": [],
                        "is_error": True,
                        "server": server_name,
                        "tool_name": name,
                        "arguments": arguments or {},
                        "error": str(e),
                    }
        except Exception as e:
            logger.error(f"Failed to call tool '{name}' on all servers: {e}")
            return {}

        return results

    def get_server_names(self) -> List[str]:
        """Get list of enabled server names."""
        if not self._client:
            return []
        return list(self._client.connections.keys())

    def get_server_status(self) -> Dict[str, Any]:
        """Get status of all configured servers."""
        if not self._client:
            return {}

        status = {}
        for server_name in self._client.connections.keys():
            status[server_name] = {
                "enabled": True,
                "transport": getattr(
                    self._client.connections[server_name], "transport", "unknown"
                ),
                "connected": True,  # If we can access it, it's connected
            }
        return status

    async def close(self) -> None:
        """Close all open connections."""
        if self._client:
            try:
                # MultiServerMCPClient doesn't have a close method, but we can clean up
                self._client = None
                logger.info("MultiServerMCPClient connections closed")
            except Exception as e:
                logger.debug(f"Error closing MultiServerMCPClient: {e}")

    async def __aenter__(self) -> "MultiServerMCPClientService":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
