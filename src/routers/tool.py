from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from src.database.models.tool import Tool, ToolConfig, ToolTransport, ToolType
from src.database.mongo_model import PaginatedResponse
from src.middlewares.auth import get_current_user
from src.schemas.common import Pagination
from src.schemas.tool import ToolRequest, ToolUpdate
from src.database.models.user import User


router = APIRouter()


def _build_tool_config_from_request(request: ToolRequest) -> ToolConfig:
    transport: Optional[ToolTransport] = (
        ToolTransport(request.config.transport) if request.config.transport else None
    )

    return ToolConfig(
        url=request.config.url,
        transport=transport,
        headers=request.config.headers,
        command=request.config.command,
        args=request.config.args,
        env=request.config.env,
    )


@router.get("/tools", response_model=PaginatedResponse[Tool])
async def list_tools(
    pagination: Pagination = Depends(),
    requesting_user: User = Depends(get_current_user),
):
    return await Tool.find_all_with_pagination(
        limit=pagination.limit,
        page=pagination.page,
        sort=[("timestamp", -1)],
        filter_dict={"user_id": requesting_user.id},
    )


@router.post("/tools", response_model=Tool)
async def create_tool(
    request: ToolRequest, requesting_user: User = Depends(get_current_user)
):
    try:
        tool = Tool(
            user_id=requesting_user.id,
            name=request.name,
            provider=request.provider,
            description=request.description,
            type=ToolType(request.type) if request.type else ToolType.MCP,
            enabled=request.enabled,
            environment=request.environment,
            config=_build_tool_config_from_request(request),
        )

        await tool.save()
        return tool
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.get("/tools/{tool_id}", response_model=Tool)
async def get_tool(tool_id: str, requesting_user: User = Depends(get_current_user)):
    tool = await Tool.find_by_id(tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    if tool.user_id != requesting_user.id:
        raise HTTPException(status_code=403, detail="Not allowed to access this tool")
    return tool


@router.patch("/tools/{tool_id}", response_model=Tool)
async def update_tool(
    tool_id: str,
    request: ToolUpdate,
    requesting_user: User = Depends(get_current_user),
):
    tool = await Tool.find_by_id(tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    if tool.user_id != requesting_user.id:
        raise HTTPException(status_code=403, detail="Not allowed to update this tool")

    if request.name is not None:
        tool.name = request.name
    if request.provider is not None:
        tool.provider = request.provider
    if request.description is not None:
        tool.description = request.description
    if request.type is not None:
        tool.type = ToolType(request.type)
    if request.enabled is not None:
        tool.enabled = request.enabled
    if request.environment is not None:
        tool.environment = request.environment
    if request.config is not None:
        # Partial update for nested config
        if request.config.url is not None:
            tool.config.url = request.config.url
        if request.config.transport is not None:
            tool.config.transport = ToolTransport(request.config.transport)
        if request.config.headers is not None:
            tool.config.headers = request.config.headers
        if request.config.command is not None:
            tool.config.command = request.config.command
        if request.config.args is not None:
            tool.config.args = request.config.args
        if request.config.env is not None:
            tool.config.env = request.config.env

    tool.updated_at = datetime.now(timezone.utc)
    await tool.save()
    return tool


@router.delete("/tools/{tool_id}")
async def delete_tool(tool_id: str, requesting_user: User = Depends(get_current_user)):
    tool = await Tool.find_by_id(tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    if tool.user_id != requesting_user.id:
        raise HTTPException(status_code=403, detail="Not allowed to delete this tool")

    await tool.delete()
    return {"message": "Tool deleted successfully"}
