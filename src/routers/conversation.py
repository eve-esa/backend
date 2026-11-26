import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from bson import ObjectId
from bson.errors import InvalidId

from src.schemas.conversation import (
    ConversationDetail,
    ConversationCreate,
    ConversationNameUpdate,
)
from src.config import AGENTIC_TURN_DEADLINE, CLASSIC_GENERATION_TIMEOUT
from src.database.models.conversation import Conversation
from src.database.models.message import Message

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from src.database.models.user import User
from src.middlewares.auth import get_current_user
from src.database.mongo_model import PaginatedResponse
from src.schemas.common import Pagination
from src.routers.artifact import content_disposition
from src.routers.message import stop_active_generation
from src.services.artifact_cleanup import (
    ArtifactCascadeIncomplete,
    delete_conversation_artifacts,
)
from src.services.cancel_manager import get_cancel_manager
from src.services.checkpoint_retention import delete_thread_checkpoints
from src.services.conversation_documents import delete_conversation_documents
from src.services.agents.core.sandbox_pool import discard_conversation_sandbox
from src.services.agents.core.workspace import workspace_prefix
from src.services.storage import (
    OPAQUE_CONTENT_TYPE,
    TOOL_INLINE_CONTENT_TYPES,
    ObjectNotFoundError,
    storage_service,
)

router = APIRouter()
logger = logging.getLogger(__name__)

# How long a conversation delete waits for a running turn it stopped to
# unwind. The turn persists its partial answer as it ends, so the delete
# waits for that write before it removes the messages and the files.
DELETE_PREEMPT_TIMEOUT_S = 10.0

# How long GET /conversations/{id} waits for the running-turn lookup. Every
# Redis client built from redis_client_kwargs() has socket_timeout=None, the
# cancel manager's shared client included, so a hung store would otherwise
# hold the page load.
ACTIVE_MESSAGE_LOOKUP_TIMEOUT_S = 0.25

# After a lookup times out or fails, this worker answers null without asking
# the store for this many seconds. Each timed-out read drops one connection of
# the shared client that also carries Stop, so without the window a latency
# spike would turn every page load into a reconnect. Same idea as the request
# rate limiter's REQUEST_RATE_LIMIT_SKIP_S, shorter: a null here shows a
# running turn as ended.
ACTIVE_MESSAGE_SKIP_S = 10.0
_lookup_skip_until = 0.0
_clock = time.monotonic

# Added to the longest turn deadline: the turn clock starts after the stream
# subscriber attaches and the mapping is cleared after the final save.
ACTIVE_MESSAGE_AGE_MARGIN_S = 60


def _max_turn_age_s(agentic_deadline_s: int, classic_timeout_s: int) -> int:
    """Oldest age a running turn's message can have, in seconds.

    The longer of the two turn deadlines plus the margin. A classic timeout
    of 0 or less means no classic deadline: the agentic one bounds the check.
    """
    longest = agentic_deadline_s
    if classic_timeout_s > 0:
        longest = max(longest, classic_timeout_s)
    return longest + ACTIVE_MESSAGE_AGE_MARGIN_S


# The mapping has no expiry: a worker that dies mid-turn leaves it behind. A
# mapped message older than the longest turn deadline plus a margin is a dead
# turn, not a running one. The age comes from the ObjectId, so the check costs
# no read.
ACTIVE_MESSAGE_MAX_AGE_S = _max_turn_age_s(
    AGENTIC_TURN_DEADLINE, CLASSIC_GENERATION_TIMEOUT
)


def _is_live_turn(conversation_id: str, message_id: str) -> bool:
    """True when ``message_id`` is an ObjectId younger than ACTIVE_MESSAGE_MAX_AGE_S."""
    try:
        born = ObjectId(message_id).generation_time
    except (InvalidId, TypeError):
        logger.debug(
            "conversation.active_message.invalid_id conversation_id=%s",
            conversation_id,
        )
        return False
    age_s = (datetime.now(timezone.utc) - born).total_seconds()
    if age_s > ACTIVE_MESSAGE_MAX_AGE_S:
        logger.debug(
            "conversation.active_message.stale conversation_id=%s age_s=%d",
            conversation_id,
            int(age_s),
        )
        return False
    return True


async def _active_message_id(conversation_id: str) -> Optional[str]:
    """The message id of the turn still running on the conversation, else None.

    Read only: the cancel manager's conversation to message mapping, written
    when a stream starts and cleared when the turn ends. A store error, a
    slow store, or a mapping past the turn deadline answers None: the
    conversation still loads. After a slow or failed lookup the store is
    not asked again for ACTIVE_MESSAGE_SKIP_S.
    """
    global _lookup_skip_until
    if _clock() < _lookup_skip_until:
        logger.debug(
            "conversation.active_message.skipped conversation_id=%s",
            conversation_id,
        )
        return None
    try:
        message_id = await asyncio.wait_for(
            get_cancel_manager().get_message_for_conversation_async(conversation_id),
            timeout=ACTIVE_MESSAGE_LOOKUP_TIMEOUT_S,
        )
        if message_id and _is_live_turn(conversation_id, message_id):
            return message_id
        return None
    except asyncio.TimeoutError:
        _lookup_skip_until = _clock() + ACTIVE_MESSAGE_SKIP_S
        logger.warning(
            "conversation.active_message.lookup_timeout conversation_id=%s",
            conversation_id,
        )
    except Exception as e:
        _lookup_skip_until = _clock() + ACTIVE_MESSAGE_SKIP_S
        logger.warning(
            "conversation.active_message.lookup_failed conversation_id=%s err=%s",
            conversation_id,
            type(e).__name__,
        )
    return None


@router.get("/conversations", response_model=PaginatedResponse[Conversation])
async def list_conversations(
    request_user: User = Depends(get_current_user), pagination: Pagination = Depends()
) -> PaginatedResponse[Conversation]:
    """
    List conversations owned by the current user.

    Args:
        request_user (User): Authenticated user injected by dependency.
        pagination (Pagination): Pagination parameters.

    Returns:
        Paginated conversations for the user.

    Raises:
        HTTPException: 500 for server errors.
    """
    try:
        result = await Conversation.find_all_with_pagination(
            filter_dict={"user_id": request_user.id},
            page=pagination.page,
            limit=pagination.limit,
            sort=[("timestamp", -1)],
        )

        return result

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: str,
    requesting_user: User = Depends(get_current_user),
) -> ConversationDetail:
    """
    Get a conversation and its messages.

    Args:
        conversation_id (str): Conversation identifier.
        requesting_user (User): Authenticated user injected by dependency.

    Returns:
        Conversation with messages and metadata. ``active_message_id`` is the
        id of the assistant message still being generated (for example after
        a reload closed its stream), or null when no turn runs or the turn is
        past its deadline.

    Raises:
        HTTPException: 404 if not found; 403 if access is forbidden; 500 for server errors.
    """
    try:
        conversation = await Conversation.find_by_id(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        if conversation.user_id != requesting_user.id:
            raise HTTPException(
                status_code=403,
                detail="You are not allowed to get this conversation",
            )

        # Read the running turn before the messages: when it reports none,
        # the turn had already ended and saved its answer on the primary, so
        # the messages are read there too (a secondary can still hold the
        # empty row). While a turn runs the client polls again, so any
        # member will do.
        active_message_id = await _active_message_id(conversation_id)

        messages = await Message.find_all(
            filter_dict={"conversation_id": conversation_id},
            sort=[("timestamp", 1)],
            primary=active_message_id is None,
        )

        return ConversationDetail(
            id=conversation.id,
            user_id=conversation.user_id,
            name=conversation.name,
            timestamp=conversation.timestamp,
            messages=messages,
            active_message_id=active_message_id,
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.post("/conversations", response_model=Conversation)
async def create_conversation(
    request: ConversationCreate,
    requesting_user: User = Depends(get_current_user),
) -> Conversation:
    """
    Create a new conversation for the current user.

    Args:
        request (ConversationCreate): New conversation payload.
        requesting_user (User): Authenticated user injected by dependency.

    Returns:
        Created conversation.

    Raises:
        HTTPException: 500 for server errors.
    """
    try:
        return await Conversation.create(
            user_id=requesting_user.id,
            name=request.name,
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.patch("/conversations/{conversation_id}")
async def update_conversation_name(
    conversation_id: str,
    request: ConversationNameUpdate,
    requesting_user: User = Depends(get_current_user),
) -> Conversation:
    """
    Update a conversation's name.

    Args:
        conversation_id (str): Conversation identifier.
        request (ConversationNameUpdate): New name payload.
        requesting_user (User): Authenticated user injected by dependency.

    Returns:
        Updated conversation.

    Raises:
        HTTPException: 404 if not found; 403 if update is forbidden; 500 for server errors.
    """
    try:
        conversation = await Conversation.find_by_id(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        if conversation.user_id != requesting_user.id:
            raise HTTPException(
                status_code=403,
                detail="You are not allowed to update this conversation",
            )

        conversation.name = request.name
        await conversation.save()

        return conversation

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    requesting_user: User = Depends(get_current_user),
) -> dict:
    """
    Delete a conversation, its messages, its attached documents (collection, rows,
    vectors and slots), its files and its agent memory (LangGraph checkpoints).

    A running turn is stopped first, through the same path as ``/stop``, and
    awaited up to ``DELETE_PREEMPT_TIMEOUT_S``, so it cannot write after the
    delete. A failure deleting the attached documents answers 500
    ``conversation_document_cleanup_failed`` before anything is deleted from the
    conversation, so a retry can finish. The sandbox is destroyed next, without
    a workspace snapshot. Files are the conversation's artifacts (except one
    another of the owner's conversations still shows), its offloaded tool
    results and its saved workspace; they go before the messages and the
    conversation row. When they are not all erased the answer is a retryable 503 and the
    conversation stays, so a retry finishes the job. A failure deleting the
    checkpoints is logged as ``checkpoint.delete_failed`` and does not fail the
    request.

    Args:
        conversation_id (str): Conversation identifier.
        requesting_user (User): Authenticated user injected by dependency.

    Returns:
        Confirmation message.

    Raises:
        HTTPException: 404 if not found; 403 if deletion is forbidden; 503 with
        ``detail.code`` ``conversation_delete_incomplete`` if its files were
        not all erased; 500 for server errors.
    """
    try:
        # Primary: a delete right after the create must find the row.
        conversation = await Conversation.find_by_id_on_primary(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        if conversation.user_id != requesting_user.id:
            raise HTTPException(
                status_code=403,
                detail="You are not allowed to delete this conversation",
            )

        if not await stop_active_generation(
            conversation_id, DELETE_PREEMPT_TIMEOUT_S
        ):
            logger.warning(
                "conversation.delete.preempt_timeout conversation_id=%s",
                conversation_id,
            )
        # First: it raises when Qdrant refuses, and nothing is deleted yet then.
        await delete_conversation_documents(requesting_user.id, conversation_id)
        # Before the files: a live sandbox would save its workspace again
        # after the erase. Best effort; the idle reaper and the provider
        # expiry still stop it.
        await discard_conversation_sandbox(conversation_id)
        # Before the messages and the row: tool results and the workspace have
        # no Mongo row, so once the conversation is gone nothing finds them.
        try:
            await delete_conversation_artifacts(requesting_user.id, conversation_id)
        except ArtifactCascadeIncomplete as exc:
            logger.warning(
                "conversation.delete_incomplete conversation_id=%s failed=%s",
                conversation_id,
                ",".join(exc.failed),
            )
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "conversation_delete_incomplete",
                    "message": "The conversation files could not all be deleted. Try again later.",
                },
            )
        await Message.delete_many({"conversation_id": conversation_id})
        await delete_thread_checkpoints(conversation_id)
        await conversation.delete()
        return {"message": "Conversation deleted successfully"}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


def _workspace_relative(file_path: str) -> str:
    raw = file_path.replace("\\", "/").lstrip("/")
    if (
        not raw
        or raw.endswith("/")
        or any(part in {"", ".", ".."} for part in raw.split("/"))
    ):
        raise HTTPException(status_code=404, detail="File not found")
    return raw


@router.get("/conversations/{conversation_id}/workspace/{file_path:path}")
async def get_workspace_file(
    conversation_id: str,
    file_path: str,
    requesting_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Stream one file from the conversation workspace prefix.

    The bytes are the saved workspace object. This route does not copy them
    into the artifact store. Workspace files are sandbox output, so they get
    the tool allowlist: anything outside it is served as
    ``application/octet-stream`` with ``attachment``, which a browser tab or a
    blob URL never renders.
    """
    conversation = await Conversation.find_by_id(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if conversation.user_id != requesting_user.id:
        raise HTTPException(
            status_code=403,
            detail="You are not allowed to get this conversation",
        )
    relative = _workspace_relative(file_path)
    key = f"{workspace_prefix(requesting_user.id, conversation_id)}{relative}"
    return await stream_sandbox_output(key)


async def stream_sandbox_output(key: str) -> StreamingResponse:
    """Stream one stored object written by a tool or the sandbox.

    Only ``TOOL_INLINE_CONTENT_TYPES`` keep their type and render inline.
    Anything else is ``application/octet-stream`` with ``attachment``.

    Raises:
        HTTPException: 404 when the object is missing.
    """
    try:
        response = await storage_service.get_object(key)
    except ObjectNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    filename = key.rsplit("/", 1)[-1]
    content_type = response.get("ContentType")
    if content_type in TOOL_INLINE_CONTENT_TYPES:
        disposition = content_disposition("inline", filename)
    else:
        content_type = OPAQUE_CONTENT_TYPE
        disposition = content_disposition("attachment", filename)
    headers = {
        "Cache-Control": "private, no-cache",
        "Vary": "Authorization",
        "X-Content-Type-Options": "nosniff",
        "Content-Disposition": disposition,
    }
    length = response.get("ContentLength")
    if length is not None:
        headers["Content-Length"] = str(length)
    return StreamingResponse(
        storage_service.stream_body(response["Body"]),
        media_type=content_type,
        headers=headers,
    )
