import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from botocore.exceptions import BotoCoreError, ClientError
from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import ValidationError
from pymongo.errors import PyMongoError

from src import config
from src.database.models.mcp_server import (
    MCPServer,
    ToolConfig,
    ToolTransport,
    ToolType,
    UserCredentialSpec,
)
from src.database.models.mcp_server_credential import MCPServerCredential
from src.database.models.user import User
from src.database.mongo_model import PaginatedResponse
from src.middlewares.auth import get_bearer_token, get_current_user
from src.schemas.common import Pagination
from src.schemas.mcp_server import (
    MCPServerCredentialFieldPublic,
    MCPServerPublic,
    MCPServerPublicDetail,
    MCPServerRequest,
    MCPServerUpdate,
    MCPServerUserCredentialsPublic,
    ToolkitCredentialsRequest,
)
from src.services.custom_model_cipher import CipherConfigError
from src.services.generate_answer_agentic import _load_mcp_tools_for_servers
from src.services.mcp import user_credentials
from src.services.mcp.attribution import attribution_from_citation
from src.services.mcp.user_environment import public_accounts
from src.services.mcp.proxy_url import resolve_public_mcp_url
from src.services.request_rate_limiter import enforce_request_rate

router = APIRouter()
logger = logging.getLogger(__name__)


def _credentials_public(
    spec: UserCredentialSpec, credential: Optional[MCPServerCredential]
) -> MCPServerUserCredentialsPublic:
    """The spec as the client may see it, plus the caller's stored state. No values."""
    return MCPServerUserCredentialsPublic(
        fields=[
            MCPServerCredentialFieldPublic(
                name=field.name, label=field.label, secret=field.secret, help_text=field.help_text
            )
            for field in spec.fields
        ],
        required=spec.required,
        help_url=spec.help_url,
        configured=credential is not None and user_credentials.is_complete(credential, spec),
        hints=dict(credential.hints) if credential else {},
        verified_at=credential.verified_at if credential else None,
        updated_at=credential.updated_at if credential else None,
    )


def _to_public_mcp_server(
    mcp_server: MCPServer, credential: Optional[MCPServerCredential] = None
) -> MCPServerPublic:
    transport = None
    public_url = None
    if mcp_server.config and mcp_server.config.transport:
        transport = mcp_server.config.transport.value
        if mcp_server.config.transport == ToolTransport.STREAMABLE_HTTP:
            public_url = resolve_public_mcp_url(mcp_server.name)

    spec = user_credentials.effective_spec(mcp_server)
    return MCPServerPublic(
        id=mcp_server.id,
        timestamp=mcp_server.timestamp,
        name=mcp_server.name,
        provider=mcp_server.provider,
        description=mcp_server.description,
        type=mcp_server.type,
        enabled=mcp_server.enabled,
        environment=getattr(mcp_server, "environment", None),
        config={"transport": transport, "url": public_url},
        attribution=attribution_from_citation(
            mcp_server.metadata.citation if mcp_server.metadata else None,
            server_name=mcp_server.name,
        ),
        created_at=mcp_server.created_at,
        updated_at=mcp_server.updated_at,
        deleted_at=mcp_server.deleted_at,
        user_credentials=_credentials_public(spec, credential) if spec else None,
        accounts=public_accounts(mcp_server),
    )


async def _credentials_by_server(
    servers: list[MCPServer], user_id: str
) -> Dict[str, MCPServerCredential]:
    """The caller's stored rows for the servers that ask for credentials, in one query."""
    ids = [str(s.id) for s in servers if s.id and user_credentials.effective_spec(s)]
    if not ids:
        return {}
    return await user_credentials.status_for(user_id=user_id, server_ids=ids)


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


def _require_registration_enabled() -> None:
    """Refuse the write routes when server registration is turned off.

    404 rather than 403: with the flag off the environment behaves as if the
    routes were never mounted, so a probe learns nothing about the feature.
    """
    if not config.FEATURE_MCP_SERVER_REGISTRATION:
        raise HTTPException(status_code=404, detail="Not Found")


def _visible_filter(user_id: str) -> Dict[str, Any]:
    """Rows a user may see: globally managed ones and their own, never soft-deleted.

    The same ownership clause as the send path (routers/message.py) and the MCP
    proxy (routers/mcp_proxy.py). ``enabled`` is not filtered: the client shows
    disabled rows greyed out.
    """
    return {
        "deleted_at": None,
        "$or": [{"user_id": user_id}, {"user_id": None}],
    }


def _object_id_or_404(server_id: str) -> ObjectId:
    try:
        return ObjectId(server_id)
    except (InvalidId, TypeError):
        raise HTTPException(status_code=404, detail="MCP server not found")


async def _get_owned_mcp_server(server_id: str, requesting_user: User) -> MCPServer:
    """A row the caller owns, else 404.

    404 rather than 403 for another user's row or a global one, so a probe
    cannot learn that an id exists.
    """
    mcp_server = await MCPServer.find_one(
        {
            "_id": _object_id_or_404(server_id),
            "user_id": requesting_user.id,
            "deleted_at": None,
        }
    )
    if not mcp_server:
        raise HTTPException(status_code=404, detail="MCP server not found")
    return mcp_server


async def _get_visible_mcp_server(server_id: str, requesting_user: User) -> MCPServer:
    mcp_server = await MCPServer.find_one(
        {"_id": _object_id_or_404(server_id), **_visible_filter(requesting_user.id)}
    )
    if not mcp_server:
        raise HTTPException(status_code=404, detail="MCP server not found")
    return mcp_server


@router.get("/mcp-servers", response_model=PaginatedResponse[MCPServerPublic])
async def list_mcp_servers(
    pagination: Pagination = Depends(),
    requesting_user: User = Depends(get_current_user),
):
    """
    List the MCP servers the current user can use: global ones and their own.

    :param pagination: Pagination parameters.\n
    :type pagination: Pagination\n
    :param requesting_user: Authenticated user injected by dependency.\n
    :type requesting_user: User\n
    :return: Paginated sanitized MCP servers.\n
    :rtype: PaginatedResponse[MCPServerPublic]\n
    """
    result = await MCPServer.find_all_with_pagination(
        limit=pagination.limit,
        page=pagination.page,
        sort=[("timestamp", -1)],
        filter_dict=_visible_filter(requesting_user.id),
    )
    credentials = await _credentials_by_server(result.data, requesting_user.id)
    return PaginatedResponse[MCPServerPublic](
        data=[
            _to_public_mcp_server(server, credentials.get(str(server.id)))
            for server in result.data
        ],
        meta=result.meta,
    )


@router.post("/mcp-servers", response_model=MCPServerPublic)
async def create_mcp_server(
    request: MCPServerRequest, requesting_user: User = Depends(get_current_user)
):
    """
    Create a new MCP server configuration for the current user.

    :param request: Tool details and configuration.\n
    :type request: MCPServerRequest\n
    :param requesting_user: Authenticated user injected by dependency.\n
    :type requesting_user: User\n
    :return: Created MCP server (sanitized).\n
    :rtype: MCPServerPublic\n
    :raises HTTPException:\n
        - 400: Invalid request data.
        - 404: MCP server registration is disabled in this environment.
        - 500: Server error.
    """
    _require_registration_enabled()
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
        return _to_public_mcp_server(mcp_server)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.get("/mcp-servers/{server_id}", response_model=MCPServerPublicDetail)
async def get_mcp_server(
    server_id: str,
    requesting_user: User = Depends(get_current_user),
    bearer_token: str = Depends(get_bearer_token),
):
    """
    Get an MCP server by id.

    Global servers are readable by every user; a user-owned server only by its
    owner. Anything else answers 404.

    :param server_id: MCP server identifier.\n
    :type server_id: str\n
    :param requesting_user: Authenticated user injected by dependency.\n
    :type requesting_user: User\n
    :return: MCP server (sanitized).\n
    :rtype: MCPServerPublicDetail\n
    :raises HTTPException:\n
        - 401: Missing or invalid authentication.
        - 404: MCP server not found.
    """
    mcp_server = await _get_visible_mcp_server(server_id, requesting_user)
    tools = []
    tools_error: Optional[str] = None
    try:
        loaded_tools = await _load_mcp_tools_for_servers(
            [mcp_server], mcp_proxy_bearer_token=bearer_token, mcp_user_id=requesting_user.id
        )
        tools = [
            {
                "name": getattr(tool, "name", "unknown"),
                "description": getattr(tool, "description", None),
                "server": mcp_server.name,
            }
            for tool in loaded_tools
        ]
    except Exception as exc:
        # Still not a 5xx: one unreachable server must not break a page that lists many. But the
        # failure now reaches the caller instead of being flattened into an empty list, which is the
        # whole point. The class name is included because the message alone is often empty
        # (TimeoutError) or a bare status code.
        tools_error = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        logger.warning(
            "Failed loading tools for MCP server '%s' (%s): %s",
            mcp_server.name,
            mcp_server.id,
            tools_error,
        )

    credentials = await _credentials_by_server([mcp_server], requesting_user.id)
    return MCPServerPublicDetail(
        **_to_public_mcp_server(mcp_server, credentials.get(str(mcp_server.id))).model_dump(),
        tools=tools,
        tools_error=tools_error,
    )


@router.patch("/mcp-servers/{server_id}", response_model=MCPServerPublic)
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
    :return: Updated MCP server (sanitized).\n
    :rtype: MCPServerPublic\n
    :raises HTTPException:\n
        - 404: MCP server not found or not owned by the caller, or registration is
          disabled in this environment.
    """
    _require_registration_enabled()
    mcp_server = await _get_owned_mcp_server(server_id, requesting_user)

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
    return _to_public_mcp_server(mcp_server)


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
        - 404: MCP server not found or not owned by the caller.
    """
    mcp_server = await _get_owned_mcp_server(server_id, requesting_user)

    await mcp_server.delete()
    return {"message": "MCP server deleted successfully"}


async def _get_credential_target(
    server_id: str, requesting_user: User
) -> tuple[MCPServer, UserCredentialSpec]:
    """A visible row that asks for user credentials, with its spec, else 404.

    The flag off, a user-owned row and a row without a valid spec all look like
    a missing row, so a probe learns nothing about the feature.
    """
    mcp_server = await _get_visible_mcp_server(server_id, requesting_user)
    spec = user_credentials.effective_spec(mcp_server)
    if spec is None:
        raise HTTPException(status_code=404, detail="MCP server not found")
    return mcp_server, spec


# At most 5 values of a few hundred characters each fit well under this.
CREDENTIALS_BODY_MAX_BYTES = 8 * 1024


def _credentials_error(status_code: int, code: str, message: str, **extra: Any) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code, "message": message, **extra})


async def _read_credentials_body(request: Request) -> ToolkitCredentialsRequest:
    """Parse the body by hand so a malformed one is refused without echoing it.

    FastAPI's default 422 copies the offending ``input`` into the response, and
    here the input is a secret. A body over ``CREDENTIALS_BODY_MAX_BYTES`` is
    refused from its ``Content-Length`` when sent, else as soon as the read
    passes the cap, so it is never held in memory whole.
    """
    malformed = _credentials_error(
        422,
        "credentials_invalid",
        "The body must be a JSON object with a values map of at most 5 strings.",
    )
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > CREDENTIALS_BODY_MAX_BYTES:
                raise malformed
        except ValueError:
            raise malformed from None
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > CREDENTIALS_BODY_MAX_BYTES:
            raise malformed
    try:
        payload = json.loads(bytes(raw))
    except ValueError:
        raise malformed from None
    try:
        return ToolkitCredentialsRequest.model_validate(payload)
    except ValidationError:
        raise malformed from None


@router.put(
    "/mcp-servers/{server_id}/credentials",
    response_model=MCPServerUserCredentialsPublic,
    include_in_schema=config.FEATURE_TOOLKIT_CREDENTIALS,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {"schema": ToolkitCredentialsRequest.model_json_schema()}
            },
        }
    },
)
async def save_mcp_server_credentials(
    server_id: str,
    request: Request,
    requesting_user: User = Depends(get_current_user),
    _rate: None = Depends(enforce_request_rate("credentials")),
):
    """
    Save the current user's own credentials for a toolkit.

    The values are checked with the provider when the toolkit declares a check,
    then stored encrypted. They are never returned: the answer carries only the
    masked hints of the non-secret fields.

    :param server_id: MCP server identifier.\n
    :type server_id: str\n
    :param request: Body ``{"values": {<field>: <value>}}``, at most 5 fields.\n
    :type request: Request\n
    :param requesting_user: Authenticated user injected by dependency.\n
    :type requesting_user: User\n
    :return: The toolkit's credential fields and the caller's stored state.\n
    :rtype: MCPServerUserCredentialsPublic\n
    :raises HTTPException:\n
        - 404: Feature off, MCP server not found, or it asks for no credentials.
        - 409: ``credentials_conflict``: the row changed during the save.
        - 422: ``credentials_invalid`` (with ``field`` when known, also for a body over 8 KiB)
          or ``credentials_rejected``.
        - 429: Rate limited (``credentials`` bucket, enforced in every mode: each save
          may send a token request to the provider).
        - 503: ``credentials_check_unavailable`` or ``credentials_storage_unavailable``.
    """
    mcp_server, spec = await _get_credential_target(server_id, requesting_user)
    body = await _read_credentials_body(request)
    try:
        credential = await user_credentials.save(
            user_id=requesting_user.id,
            server_id=str(mcp_server.id),
            spec=spec,
            values=body.values,
        )
    except user_credentials.CredentialsInvalid as exc:
        message = f"{exc.field} {exc.message}" if exc.field else exc.message
        raise _credentials_error(422, "credentials_invalid", message, field=exc.field) from None
    except user_credentials.CredentialsRejected:
        raise _credentials_error(
            422, "credentials_rejected", "The provider rejected these credentials."
        ) from None
    except user_credentials.CredentialsCheckUnavailable:
        raise _credentials_error(
            503,
            "credentials_check_unavailable",
            "The provider could not be reached to check these credentials.",
        ) from None
    except user_credentials.CredentialsConflict:
        raise _credentials_error(
            409,
            "credentials_conflict",
            "The credentials changed while saving. Try again.",
        ) from None
    except (CipherConfigError, ClientError, BotoCoreError, PyMongoError) as exc:
        logger.error(
            "mcp.user_credentials.storage_unavailable server=%s error=%s",
            mcp_server.id,
            type(exc).__name__,
        )
        raise _credentials_error(
            503,
            "credentials_storage_unavailable",
            "The credentials could not be stored, try again later.",
        ) from None
    logger.info(
        "mcp.user_credentials.saved server=%s user=%s", mcp_server.id, requesting_user.id
    )
    return _credentials_public(spec, credential)


@router.delete(
    "/mcp-servers/{server_id}/credentials",
    status_code=204,
    include_in_schema=config.FEATURE_TOOLKIT_CREDENTIALS,
)
async def delete_mcp_server_credentials(
    server_id: str, requesting_user: User = Depends(get_current_user)
):
    """
    Remove the current user's own credentials for a toolkit.

    A stored row is always removed, even with the feature off, the spec gone or
    the toolkit deleted, so a user can take back their secret. Without a row the
    route answers like the save: 204 for a toolkit that asks for credentials,
    else 404.

    :param server_id: MCP server identifier.\n
    :type server_id: str\n
    :param requesting_user: Authenticated user injected by dependency.\n
    :type requesting_user: User\n
    :return: Empty response.\n
    :rtype: Response\n
    :raises HTTPException:\n
        - 404: No stored row and the feature is off, the MCP server is not found,
          or it asks for no credentials.
    """
    oid = _object_id_or_404(server_id)
    removed = await user_credentials.delete(user_id=requesting_user.id, server_id=str(oid))
    if not removed:
        await _get_credential_target(str(oid), requesting_user)
    logger.info(
        "mcp.user_credentials.deleted server=%s user=%s removed=%s",
        oid,
        requesting_user.id,
        removed,
    )
    return Response(status_code=204)
