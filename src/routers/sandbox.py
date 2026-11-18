"""Conversation files and sandbox, as resources under a conversation.

``files`` is the conversation's saved tree in object storage
(``users/{user_id}/conversations/{conversation_id}/``): ``workspace/``,
``artifacts/``, ``large_tool_results/`` and ``workspace_state``. ``sandbox``
is the live provider session; ``sandbox/files`` is its tree.

Read-only today. Behind ``FEATURE_SANDBOX_API`` (off: 404, and the routes are
left out of the schema). Another user's conversation answers 404, so the
route does not confirm that it exists.
"""

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi.responses import StreamingResponse

from src import config
from src.database.models.conversation import Conversation
from src.database.models.user import User
from src.middlewares.auth import get_current_user
from src.routers.conversation import stream_sandbox_output
from src.schemas.sandbox import FileListing, SandboxRead
from src.services.agents.core import sandbox_inspect
from src.services.agents.core.sandbox_inspect import (
    InvalidPath,
    SandboxGone,
    SandboxNotFound,
    conversation_prefix,
    relative_path,
)


def require_sandbox_api_enabled() -> None:
    """404 while ``FEATURE_SANDBOX_API`` is off, as if the routes did not exist."""
    if not config.FEATURE_SANDBOX_API:
        raise HTTPException(status_code=404, detail="Not Found")


router = APIRouter(
    dependencies=[Depends(require_sandbox_api_enabled)],
    include_in_schema=config.FEATURE_SANDBOX_API,
)

FILES_MAX_LIMIT = 1000
SANDBOX_FILES_MAX_LIMIT = 2000


async def _own_conversation(conversation_id: str, user: User) -> Conversation:
    conversation = await Conversation.find_by_id(conversation_id)
    if not conversation or conversation.user_id != user.id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


def _prefix(raw: str) -> str:
    try:
        return relative_path(raw, allow_prefix=True)
    except InvalidPath:
        raise HTTPException(status_code=404, detail="File not found")


@router.get(
    "/conversations/{conversation_id}/files",
    response_model=FileListing,
)
async def list_conversation_files(
    conversation_id: str = Path(..., description="Conversation ID"),
    prefix: str = Query("", description="Only paths that start with this"),
    limit: int = Query(FILES_MAX_LIMIT, ge=1, le=FILES_MAX_LIMIT),
    requesting_user: User = Depends(get_current_user),
) -> FileListing:
    """List the conversation's saved files, paths relative to the conversation.

    When ``truncated`` is true, the newest ``limit`` objects are returned.

    Raises:
        HTTPException: 502 when object storage could not be listed.
    """
    await _own_conversation(conversation_id, requesting_user)
    relative = _prefix(prefix)
    try:
        return await sandbox_inspect.list_conversation_files(
            requesting_user.id, conversation_id, relative, limit
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Storage not readable: {type(exc).__name__}"
        )


@router.get("/conversations/{conversation_id}/files/{file_path:path}")
async def get_conversation_file(
    conversation_id: str = Path(..., description="Conversation ID"),
    file_path: str = Path(..., description="Path relative to the conversation"),
    requesting_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Download one saved file. Types outside the tool allowlist are attachments."""
    await _own_conversation(conversation_id, requesting_user)
    try:
        relative = relative_path(file_path, allow_prefix=False)
    except InvalidPath:
        raise HTTPException(status_code=404, detail="File not found")
    key = conversation_prefix(requesting_user.id, conversation_id) + relative
    return await stream_sandbox_output(key)


@router.get(
    "/conversations/{conversation_id}/sandbox",
    response_model=SandboxRead,
)
async def get_conversation_sandbox(
    conversation_id: str = Path(..., description="Conversation ID"),
    requesting_user: User = Depends(get_current_user),
) -> SandboxRead:
    """The conversation's sandbox as the pool records it. The provider is not asked."""
    await _own_conversation(conversation_id, requesting_user)
    return await sandbox_inspect.get_sandbox(conversation_id)


@router.get(
    "/conversations/{conversation_id}/sandbox/files",
    response_model=FileListing,
)
async def list_sandbox_files(
    conversation_id: str = Path(..., description="Conversation ID"),
    prefix: str = Query("", description="Only paths that start with this"),
    limit: int = Query(SANDBOX_FILES_MAX_LIMIT, ge=1, le=SANDBOX_FILES_MAX_LIMIT),
    requesting_user: User = Depends(get_current_user),
) -> FileListing:
    """List files in the live sandbox, paths relative to the session root.

    Attaches to the session and detaches again. The idle deadline does not move.

    Raises:
        HTTPException: 404 when the conversation has no sandbox, 410 when the
        provider reports it ended, 502 when the provider or the listing failed.
    """
    await _own_conversation(conversation_id, requesting_user)
    try:
        return await sandbox_inspect.list_sandbox_files(
            conversation_id, _prefix(prefix), limit
        )
    except SandboxNotFound:
        raise HTTPException(status_code=404, detail="Sandbox not found")
    except SandboxGone:
        raise HTTPException(status_code=410, detail="Sandbox gone")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Sandbox not readable: {type(exc).__name__}"
        )
