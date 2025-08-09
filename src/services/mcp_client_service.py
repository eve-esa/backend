from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, AsyncIterator

from mcp.client.streamable_http import streamablehttp_client
from mcp.client.websocket import websocket_client
from mcp.client.session import ClientSession
from urllib.parse import quote
from contextlib import AsyncExitStack

from src.config import Config

# Import FastAPI MCP Client for streaming capabilities
try:
    from fastapi_mcp_client import MCPClient, MCPClientConfig
    from fastapi_mcp_client.exceptions import (
        MCPClientError,
        MCPConnectionError,
        MCPStreamError,
    )

    FASTAPI_MCP_AVAILABLE = True
except ImportError:
    FASTAPI_MCP_AVAILABLE = False
    MCPClient = None
    MCPClientConfig = None
    MCPClientError = Exception
    MCPConnectionError = Exception
    MCPStreamError = Exception


logger = logging.getLogger(__name__)


class MCPClientService:
    """Service to interact with an MCP server from the backend."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self._config = config or Config()

        self._server_url: str = self._config.get_mcp_server_url()
        if not self._server_url:
            raise ValueError(
                "MCP server_url not configured (config.yaml -> mcp.server_url)"
            )

        # Headers can contain Authorization etc. Read ONLY from config.yaml
        self._headers: Dict[str, str] = self._config.get_mcp_headers()
        auth_value = (self._headers.get("Authorization") or "").strip()
        if (not auth_value) or ("${" in auth_value and "}" in auth_value):
            raise ValueError(
                "MCP Authorization header not configured (config.yaml -> mcp.headers.Authorization)"
            )

        # Initialize FastAPI MCP Client if available
        self._fastapi_mcp_client: Optional[MCPClient] = None
        if FASTAPI_MCP_AVAILABLE and self._server_url.startswith(
            ("http://", "https://")
        ):
            try:
                # Extract base URL and path from server URL
                if "/mcp" in self._server_url:
                    base_url = self._server_url.split("/mcp")[0]
                else:
                    base_url = self._server_url.rstrip("/")

                # Create FastAPI MCP Client configuration
                fastapi_config = MCPClientConfig(
                    base_url=base_url,
                    default_headers=self._headers,
                    timeout=30.0,
                    log_level="INFO",
                )

                self._fastapi_mcp_client = MCPClient(
                    base_url=base_url, config=fastapi_config
                )
                logger.info("FastAPI MCP Client initialized for streaming operations")
            except Exception as e:
                logger.warning(f"Failed to initialize FastAPI MCP Client: {e}")
                self._fastapi_mcp_client = None

    def _open_session(self):
        """Open MCP session using appropriate transport based on URL scheme."""
        try:
            if self._server_url.startswith(("ws://", "wss://")):
                # WebSocket transport
                ws_url = self._server_url
                auth_value = self._headers.get("Authorization", "")
                if auth_value:
                    sep = "&" if "?" in ws_url else "?"
                    ws_url = f"{ws_url}{sep}auth_header=Authorization&auth_value={quote(auth_value)}"
                return websocket_client(ws_url)
            else:
                # HTTP transport for http:// and https:// URLs
                return streamablehttp_client(
                    self._server_url,
                    headers=self._headers,
                    terminate_on_close=False,
                )
        except Exception as e:
            logger.error(f"Failed to create MCP session: {e}")
            raise

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the MCP server."""
        stack = AsyncExitStack()
        await stack.__aenter__()
        tools: List[Dict[str, Any]] = []
        try:
            transport = await stack.enter_async_context(self._open_session())
            # Handle different transport return types
            if len(transport) == 3:
                read, write, _get_session_id = transport
            else:
                read, write = transport
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            result = await session.list_tools()

            def _serialize_tool(t: Any) -> Dict[str, Any]:
                tool_data = {
                    "name": t.name,
                    "description": getattr(t, "description", None),
                }
                # Parse JSON string in description field if present
                if isinstance(tool_data.get("description"), str):
                    try:
                        import json

                        tool_data["description"] = json.loads(tool_data["description"])
                    except (json.JSONDecodeError, TypeError):
                        # Keep as string if parsing fails
                        pass
                return tool_data

            tools = [_serialize_tool(t) for t in result.tools]
        except Exception as e:
            logger.error(f"Failed to list tools: {e}")
            raise
        finally:
            try:
                await stack.aclose()
            except BaseException as e:
                logger.debug(f"Ignoring transport teardown error: {e}")
        return tools

    async def call_tool(
        self, name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Call a specific tool on the MCP server."""
        stack = AsyncExitStack()
        await stack.__aenter__()
        processed: Dict[str, Any] = {"content": [], "is_error": True}
        try:
            transport = await stack.enter_async_context(self._open_session())
            # Handle different transport return types
            if len(transport) == 3:
                read, write, _get_session_id = transport
            else:
                read, write = transport
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            result = await session.call_tool(name=name, arguments=arguments or {})

            def _serialize_content_item(item: Any) -> Any:
                if hasattr(item, "model_dump"):
                    data = item.model_dump()
                    # Parse JSON string in text field if present
                    if isinstance(data.get("text"), str):
                        try:
                            import json

                            data["text"] = json.loads(data["text"])
                        except (json.JSONDecodeError, TypeError):
                            # Keep as string if parsing fails
                            pass
                    return data
                try:
                    return dict(item)
                except Exception:
                    return str(item)

            processed = {
                "content": [_serialize_content_item(c) for c in result.content],
                "is_error": getattr(result, "is_error", None)
                or getattr(result, "isError", None),
            }
        except Exception as e:
            logger.error(f"Failed to call tool '{name}': {e}")
            raise
        finally:
            try:
                await stack.aclose()
            except BaseException as e:
                logger.debug(f"Ignoring transport teardown error: {e}")
        return processed

    async def call_tool_stream(
        self, name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Call a specific tool on the MCP server with streaming support.

        This method uses the FastAPI MCP Client for streaming operations when available,
        falling back to the regular call_tool method if streaming is not supported.
        """
        if self._fastapi_mcp_client is not None:
            try:
                logger.debug(
                    f"Using FastAPI MCP Client for streaming tool call: {name}"
                )
                async for event in self._fastapi_mcp_client.call_operation(
                    name, arguments, stream=True
                ):
                    yield event
            except (MCPConnectionError, MCPStreamError) as e:
                logger.warning(f"FastAPI MCP Client streaming failed: {e}")
                # Fall back to non-streaming
                result = await self.call_tool(name, arguments)
                yield result
            except Exception as e:
                logger.error(f"Unexpected error in streaming tool call: {e}")
                # Fall back to non-streaming
                result = await self.call_tool(name, arguments)
                yield result
        else:
            # Fall back to non-streaming if FastAPI MCP Client is not available
            logger.debug(
                f"FastAPI MCP Client not available, using regular tool call: {name}"
            )
            result = await self.call_tool(name, arguments)
            yield result

    async def close(self) -> None:
        """Close any open connections."""
        if self._fastapi_mcp_client is not None:
            try:
                await self._fastapi_mcp_client.close()
            except Exception as e:
                logger.debug(f"Error closing FastAPI MCP Client: {e}")

    async def __aenter__(self) -> "MCPClientService":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
