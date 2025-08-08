from __future__ import annotations

from typing import Any, Dict
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.services.mcp_client_service import MCPClientService


router = APIRouter()


class ToolCallBody(BaseModel):
    arguments: Dict[str, Any] = Field(default_factory=dict)


@router.get("/mcp/tools")
async def get_mcp_tools() -> Dict[str, Any]:
    """List tools exposed by the configured MCP server."""
    try:
        service = MCPClientService()
        tools = await service.list_tools()
        return {"tools": tools}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/mcp/tools/{tool_name}")
async def call_mcp_tool(tool_name: str, body: ToolCallBody) -> Dict[str, Any]:
    """Invoke a tool on the MCP server with optional arguments."""
    try:
        service = MCPClientService()
        result = await service.call_tool(tool_name, body.arguments or {})
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
