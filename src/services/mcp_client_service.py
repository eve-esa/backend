from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import time
from urllib.parse import urlparse

from src.config import Config, WILEY_AUTH_TOKEN

# Import MultiServerMCPClient from langchain_mcp_adapters
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    MULTI_SERVER_MCP_AVAILABLE = True
except ImportError:
    MULTI_SERVER_MCP_AVAILABLE = False
    MultiServerMCPClient = None

# Import original MCP client libraries for actual tool execution
try:
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.client.websocket import websocket_client
    from mcp.client.session import ClientSession
    from urllib.parse import quote
    from contextlib import AsyncExitStack

    ORIGINAL_MCP_AVAILABLE = True
except ImportError:
    ORIGINAL_MCP_AVAILABLE = False
    streamablehttp_client = None
    websocket_client = None
    ClientSession = None
    quote = None
    AsyncExitStack = None


logger = logging.getLogger(__name__)


class MultiServerMCPClientService:
    """Service to interact with multiple MCP servers using LangChain's MultiServerMCPClient."""

    # Shared singleton and token cache across the process
    _shared_instance: Optional["MultiServerMCPClientService"] = None
    _wiley_access_token_shared: Optional[str] = None
    _wiley_access_token_expiry_epoch_shared: float = 0.0

    def __init__(self, config: Optional[Config] = None) -> None:
        self._config = config or Config()
        self._client: Optional[MultiServerMCPClient] = None

        if not MULTI_SERVER_MCP_AVAILABLE:
            raise ImportError(
                "MultiServerMCPClient from langchain_mcp_adapters is not available"
            )

        # Load server configurations
        self._load_server_configs()

        if not self._client:
            raise ValueError(
                "No MCP servers configured or failed to initialize MultiServerMCPClient"
            )

    def _load_server_configs(self) -> None:
        """Load MCP server configurations and initialize MultiServerMCPClient."""
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

                # Add headers if present
                headers = server_config.get("headers")
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
            logger.error(f"Failed to list tools from server '{server_name}': {e}")
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
            logger.error(f"Failed to list tools from all servers: {e}")
            return []

        return all_tools

    async def call_tool_on_server(
        self, server_name: str, name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Call a specific tool on a specific MCP server."""
        if not self._client:
            raise RuntimeError("MultiServerMCPClient not initialized")

        try:
            # Since MultiServerMCPClient doesn't provide actual connection objects,
            # we'll use the fallback approach directly
            if not ORIGINAL_MCP_AVAILABLE:
                raise RuntimeError(
                    "Original MCP client libraries not available for tool execution"
                )

            logger.info(f"Using fallback MCP client for server '{server_name}'")
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
        """Fallback method using original MCP client libraries."""
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
        # Start with configured headers if present
        headers = dict(server_config.get("headers", {}))

        # Ensure Authorization header is a Bearer token for Wiley servers.
        # If missing or expired, fetch a new token using client_credentials.
        if url:
            parsed = urlparse(url)
            # Wiley OAuth token endpoint lives at /oauth2/token on the same host
            token_endpoint = f"{parsed.scheme}://{parsed.netloc}/oauth2/token?grant_type=client_credentials"

            # Reuse token if valid; otherwise refresh it
            now = time.time()
            token_value: Optional[str] = None

            # Use class-level shared cache to persist across requests/instances
            cls = type(self)
            if (
                cls._wiley_access_token_shared
                and now < cls._wiley_access_token_expiry_epoch_shared
            ):
                token_value = cls._wiley_access_token_shared
            else:
                try:
                    import httpx

                    async with httpx.AsyncClient(timeout=10.0) as client:
                        if not WILEY_AUTH_TOKEN:
                            raise RuntimeError(
                                "WILEY_AUTH_TOKEN is not set in environment"
                            )
                        resp = await client.post(
                            token_endpoint,
                            headers={"Authorization": WILEY_AUTH_TOKEN},
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        token_value = data.get("access_token")
                        expires_in = int(data.get("expires_in", 3600))
                        cls._wiley_access_token_expiry_epoch_shared = now + max(
                            0, expires_in - 60
                        )
                        cls._wiley_access_token_shared = token_value
                except Exception as e:
                    logger.error(f"Failed to obtain Wiley token: {e}")
                    raise

            if token_value:
                headers["Authorization"] = f"Bearer {token_value}"

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
