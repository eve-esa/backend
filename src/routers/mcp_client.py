from __future__ import annotations

from typing import Any, Dict
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import json

from src.services.mcp_client_service import MCPClientService


router = APIRouter()


class ToolCallBody(BaseModel):
    arguments: Dict[str, Any] = Field(default_factory=dict)


@router.get("/mcp/tools")
async def get_mcp_tools() -> Dict[str, Any]:
    """List tools exposed by the configured MCP server."""
    try:
        async with MCPClientService() as service:
            tools = await service.list_tools()
            return {"tools": tools}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/mcp/tools/{tool_name}")
async def call_mcp_tool(tool_name: str, body: ToolCallBody) -> Dict[str, Any]:
    """Invoke a tool on the MCP server with optional arguments."""
    try:
        async with MCPClientService() as service:
            result = await service.call_tool(tool_name, body.arguments or {})
            return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/mcp/tools/{tool_name}/stream")
async def call_mcp_tool_stream(tool_name: str, body: ToolCallBody) -> StreamingResponse:
    """Invoke a tool on the MCP server with streaming support."""

    async def generate_stream():
        try:
            async with MCPClientService() as service:
                async for event in service.call_tool_stream(
                    tool_name, body.arguments or {}
                ):
                    # Convert event to JSON string with newline for SSE format
                    yield f"data: {json.dumps(event, default=str)}\n\n"
        except Exception as e:
            # Send error as SSE event
            error_event = {"error": str(e), "tool_name": tool_name}
            yield f"data: {json.dumps(error_event, default=str)}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
        },
    )


@router.get("/mcp/status")
async def get_mcp_status() -> Dict[str, Any]:
    """Get the status of MCP client capabilities."""
    try:
        async with MCPClientService() as service:
            # Check if FastAPI MCP Client is available
            fastapi_mcp_available = (
                hasattr(service, "_fastapi_mcp_client")
                and service._fastapi_mcp_client is not None
            )

            # Test basic connectivity
            tools = await service.list_tools()

            return {
                "status": "connected",
                "fastapi_mcp_client_available": fastapi_mcp_available,
                "streaming_supported": fastapi_mcp_available,
                "tools_count": len(tools),
                "server_url": (
                    service._server_url if hasattr(service, "_server_url") else None
                ),
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "fastapi_mcp_client_available": False,
            "streaming_supported": False,
            "tools_count": 0,
        }
