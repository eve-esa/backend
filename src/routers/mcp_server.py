from datetime import datetime, timezone
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from src.database.models.mcp_server import MCPServer, ToolConfig, ToolTransport, ToolType
from src.database.mongo_model import PaginatedResponse
from src.middlewares.auth import get_current_user
from src.schemas.common import Pagination
from src.schemas.mcp_server import MCPServerDetail, MCPServerRequest, MCPServerUpdate
from src.database.models.user import User
from src.services.mcp_client_service import MultiServerMCPClientService


router = APIRouter()
logger = logging.getLogger(__name__)


def _build_tool_config_from_request(request: MCPServerRequest) -> ToolConfig:
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


async def _get_owned_mcp_server(
    server_id: str, requesting_user: User, action: str = "access"
) -> MCPServer:
    mcp_server = await MCPServer.find_by_id(server_id)
    if not mcp_server:
        raise HTTPException(status_code=404, detail="MCP server not found")
    if mcp_server.user_id != requesting_user.id:
        raise HTTPException(status_code=403, detail=f"Not allowed to {action} this MCP server")
    return mcp_server


@router.get("/mcp-servers", response_model=PaginatedResponse[MCPServer])
async def list_mcp_servers(
    pagination: Pagination = Depends(),
    requesting_user: User = Depends(get_current_user),
):
    """
    List MCP servers owned by the current user.

    :param pagination: Pagination parameters.\n
    :type pagination: Pagination\n
    :param requesting_user: Authenticated user injected by dependency.\n
    :type requesting_user: User\n
    :return: Paginated MCP servers for the user.\n
    :rtype: PaginatedResponse[MCPServer]\n
    """
    return await MCPServer.find_all_with_pagination(
        limit=pagination.limit,
        page=pagination.page,
        sort=[("timestamp", -1)],
        filter_dict={"user_id": requesting_user.id},
    )


@router.post("/mcp-servers", response_model=MCPServer)
async def create_mcp_server(
    request: MCPServerRequest, requesting_user: User = Depends(get_current_user)
):
    """
    Create a new MCP server configuration for the current user.

    :param request: Tool details and configuration.\n
    :type request: MCPServerRequest\n
    :param requesting_user: Authenticated user injected by dependency.\n
    :type requesting_user: User\n
    :return: Created MCP server.\n
    :rtype: MCPServer\n
    :raises HTTPException:\n
        - 400: Invalid request data.
        - 500: Server error.
    """
    try:
        mcp_server = MCPServer(
            user_id=requesting_user.id,
            name=request.name,
            provider=request.provider,
            description=request.description,
            type=ToolType(request.type) if request.type else ToolType.MCP,
            enabled=request.enabled,
            environment=request.environment,
            config=_build_tool_config_from_request(request),
        )

        await mcp_server.save()
        return mcp_server
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.get("/mcp-servers/{server_id}", response_model=MCPServerDetail)
async def get_mcp_server(
    server_id: str, requesting_user: User = Depends(get_current_user)
):
    """
    Get an MCP server by id owned by the current user.

    :param server_id: MCP server identifier.\n
    :type server_id: str\n
    :param requesting_user: Authenticated user injected by dependency.\n
    :type requesting_user: User\n
    :return: MCP server.\n
    :rtype: MCPServerDetail\n
    :raises HTTPException:\n
        - 404: MCP server not found.
        - 403: Not allowed to access this MCP server.
    """
    mcp_server = await _get_owned_mcp_server(server_id, requesting_user, action="access")
    tools = []
    service: Optional[MultiServerMCPClientService] = None
    try:
        service = MultiServerMCPClientService.from_mcp_server_model(mcp_server)
        tools = await service.list_tools_from_server(mcp_server.name)
    except Exception as exc:
        logger.warning(
            "Failed loading tools for MCP server '%s' (%s): %s",
            mcp_server.name,
            mcp_server.id,
            exc,
        )
    finally:
        if service:
            await service.close()

    return MCPServerDetail(**mcp_server.model_dump(), tools=tools)


@router.patch("/mcp-servers/{server_id}", response_model=MCPServer)
async def update_mcp_server(
    server_id: str,
    request: MCPServerUpdate,
    requesting_user: User = Depends(get_current_user),
):
    """
    Update an existing MCP server owned by the current user.

    Supports partial updates of both top-level and nested config fields.

    :param server_id: MCP server identifier.\n
    :type server_id: str\n
    :param request: Update payload.\n
    :type request: MCPServerUpdate\n
    :param requesting_user: Authenticated user injected by dependency.\n
    :type requesting_user: User\n
    :return: Updated MCP server.\n
    :rtype: MCPServer\n
    :raises HTTPException:\n
        - 404: MCP server not found.
        - 403: Not allowed to update this MCP server.
    """
    mcp_server = await _get_owned_mcp_server(server_id, requesting_user, action="update")

    if request.name is not None:
        mcp_server.name = request.name
    if request.provider is not None:
        mcp_server.provider = request.provider
    if request.description is not None:
        mcp_server.description = request.description
    if request.type is not None:
        mcp_server.type = ToolType(request.type)
    if request.enabled is not None:
        mcp_server.enabled = request.enabled
    if request.environment is not None:
        mcp_server.environment = request.environment
    if request.config is not None:
        # Partial update for nested config
        if request.config.url is not None:
            mcp_server.config.url = request.config.url
        if request.config.transport is not None:
            mcp_server.config.transport = ToolTransport(request.config.transport)
        if request.config.headers is not None:
            mcp_server.config.headers = request.config.headers
        if request.config.command is not None:
            mcp_server.config.command = request.config.command
        if request.config.args is not None:
            mcp_server.config.args = request.config.args
        if request.config.env is not None:
            mcp_server.config.env = request.config.env

    mcp_server.updated_at = datetime.now(timezone.utc)
    await mcp_server.save()
    return mcp_server


@router.delete("/mcp-servers/{server_id}")
async def delete_mcp_server(
    server_id: str, requesting_user: User = Depends(get_current_user)
):
    """
    Delete an MCP server owned by the current user.

    :param server_id: MCP server identifier.\n
    :type server_id: str\n
    :param requesting_user: Authenticated user injected by dependency.\n
    :type requesting_user: User\n
    :return: Confirmation message.\n
    :rtype: dict\n
    :raises HTTPException:\n
        - 404: MCP server not found.
        - 403: Not allowed to delete this MCP server.
    """
    mcp_server = await _get_owned_mcp_server(server_id, requesting_user, action="delete")

    await mcp_server.delete()
    return {"message": "MCP server deleted successfully"}
