from __future__ import annotations

from typing import Any, Dict
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.services.mcp_client_service import MultiServerMCPClientService


router = APIRouter()


class ToolCallBody(BaseModel):
    arguments: Dict[str, Any] = Field(default_factory=dict)


@router.get("/mcp/tools")
async def get_mcp_tools() -> Dict[str, Any]:
    """List tools exposed by all configured MCP servers."""
    try:
        async with MultiServerMCPClientService() as service:
            tools = await service.list_tools_from_all_servers()
            return {"tools": tools}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/mcp/servers/{server_name}/tools")
async def get_mcp_tools_from_server(server_name: str) -> Dict[str, Any]:
    """List tools exposed by a specific MCP server."""
    try:
        async with MultiServerMCPClientService() as service:
            tools = await service.list_tools_from_server(server_name)
            return {"tools": tools, "server": server_name}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/mcp/tools/{tool_name}")
async def call_mcp_tool_on_all_servers(
    tool_name: str, body: ToolCallBody
) -> Dict[str, Any]:
    """Invoke a tool on all configured MCP servers with optional arguments."""
    try:
        async with MultiServerMCPClientService() as service:
            result = await service.call_tool_on_all_servers(
                tool_name, body.arguments or {}
            )
            return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/mcp/servers/{server_name}/tools/{tool_name}")
async def call_mcp_tool_on_server(
    tool_name: str, server_name: str, body: ToolCallBody
) -> Dict[str, Any]:
    """Invoke a tool on a specific MCP server with optional arguments."""
    try:
        async with MultiServerMCPClientService() as service:
            result = await service.call_tool_on_server(
                server_name, tool_name, body.arguments or {}
            )
            return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/mcp/servers")
async def get_mcp_servers() -> Dict[str, Any]:
    """Get list of configured MCP servers and their status."""
    try:
        async with MultiServerMCPClientService() as service:
            server_names = service.get_server_names()
            server_status = service.get_server_status()
            return {"servers": server_names, "status": server_status}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/mcp/status")
async def get_mcp_status() -> Dict[str, Any]:
    """Get the status of MCP client capabilities."""
    try:
        async with MultiServerMCPClientService() as service:
            server_names = service.get_server_names()
            server_status = service.get_server_status()

            # Get total tools count from all servers
            all_tools = await service.list_tools_from_all_servers()

            return {
                "status": "connected",
                "servers_count": len(server_names),
                "servers": server_names,
                "server_status": server_status,
                "total_tools_count": len(all_tools),
                "multi_server_support": True,
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "multi_server_support": True,
            "servers_count": 0,
            "total_tools_count": 0,
        }
