"""AG-UI 1.0 stream of one agentic turn: ``POST /conversations/{id}/agui``.

Behind ``FEATURE_AGUI_STREAM`` (off: 404). The route takes the same guards as
``stream-generate-agentic`` in the same order, reads one ``RunAgentInput``,
starts the same runner on the same stream bus, and pipes the legacy frames
through ``AguiAdapter`` into SDK-encoded SSE events with the usual
keepalive comment. Stop stays ``POST /conversations/{id}/stop``.

History is kept by the server: only the trailing user message of the input is
read. ``forwardedProps.eve`` passes through an allowlist onto a
``GenerationRequest``, so every field keeps its own validation.

Errors before the stream are HTTP errors with ``{"detail": {"code",
"message"}}`` (contract 1.6); after the stream opened they travel as
``RUN_ERROR`` (contract 1.7, inside the adapter).

Logging: codes, counts and ids only, never user text.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from ag_ui.core import RunAgentInput, UserMessage
from bson import ObjectId
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from src import config
from src.config import AGENTIC_TURN_DEADLINE
from src.database.models.collection import Collection as CollectionModel
from src.database.models.conversation import Conversation
from src.database.models.message import Message
from src.database.models.user import User
from src.middlewares.auth import get_bearer_token, get_current_user
from src.observability.context import current_trace_id
from src.routers.message import (
    CHAT_RATE_LIMIT,
    _prepare_agentic_request,
    resolve_artifact_attachments,
)
from src.schemas.generation_request import GenerationRequest
from src.services.agui.adapter import (
    AguiAdapter,
    RunContext,
    encode_sse,
    pre_stream_error,
)
from src.services.cancel_manager import get_cancel_manager
from src.services.generate_answer import (
    build_error_payload,
    finish_turn_if_cancelled_unstarted,
)
from src.services.generate_answer_agentic import run_agentic_generation_to_bus
from src.services.load_shedding import acquire_generation_slot_or_raise
from src.services.stream_bus import get_stream_bus
from src.services.token_rate_limiter import enforce_token_budget_or_raise
from src.utils.error_logger import (
    get_error_logger,
    set_conversation_context,
    set_message_context,
    set_user_context,
)
from src.utils.sse_keepalive import DEFAULT_KEEPALIVE_INTERVAL_S, with_sse_keepalive

logger = logging.getLogger(__name__)

router = APIRouter()

ROUTE_NAME = "agui"
# Module level so a test can shorten it.
KEEPALIVE_INTERVAL_S = DEFAULT_KEEPALIVE_INTERVAL_S
# The current artifact_ids cap (contract 1.4).
MAX_ATTACHMENTS = 20

# forwardedProps.eve key -> GenerationRequest field (contract 1.3). `agent` is
# not here: the graph is a server setting (contract 8.3).
EVE_PROPS: Dict[str, str] = {
    "model": "llm_type",
    "customModelId": "custom_model_id",
    "tools": "public_mcp_servers",
    "skills": "skills",
    "publicCollections": "public_collections",
    "privateCollections": "private_collections",
    "filters": "filters",
    "year": "year",
    "k": "k",
    "temperature": "temperature",
    "scoreThreshold": "score_threshold",
    "maxNewTokens": "max_new_tokens",
}

# (conversation_id, run_id) pairs between the conflict check and the message
# insert on this worker: two identical POSTs on one worker cannot both pass.
_runs_in_admission: set[Tuple[str, str]] = set()


def agui_http_error(code: str, retry_after: Optional[int] = None) -> HTTPException:
    """The contract 1.6 HTTP error for ``code``."""
    err = pre_stream_error(code, retry_after=retry_after)
    return HTTPException(
        status_code=err.status_code, detail=err.body["detail"], headers=err.headers or None
    )


def require_agui_stream() -> None:
    """404 ``not_found`` while the flag is off, before auth and before the body."""
    if not config.FEATURE_AGUI_STREAM:
        raise agui_http_error("not_found")


def _retry_after(exc: HTTPException) -> Optional[int]:
    value = (exc.headers or {}).get("Retry-After")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _check_protocol_version(version: Optional[str]) -> None:
    if version is None:
        return
    major, _, _ = str(version).partition(".")
    if major != "1":
        raise agui_http_error("protocol_version_unsupported")
    if str(version) != "1.0":
        logger.warning("agui.protocol_minor version=%s", version)


def _read_user_message(body: RunAgentInput) -> Tuple[str, List[str]]:
    """Query text and artifact ids from the trailing ``UserMessage``.

    Only the last message is read (deviation D1). Text parts are joined with a
    newline; an ``image`` or ``document`` part counts only with a ``url`` source
    and ``metadata["eve.artifactId"]``, and its URL is never fetched.
    """
    if not body.messages or not isinstance(body.messages[-1], UserMessage):
        raise agui_http_error("no_user_message")
    content = body.messages[-1].content
    if isinstance(content, str):
        return content, []
    texts: List[str] = []
    artifact_ids: List[str] = []
    skipped = 0
    for part in content:
        kind = getattr(part, "type", None)
        if kind == "text":
            texts.append(part.text)
            continue
        source = getattr(part, "source", None)
        metadata = getattr(part, "metadata", None)
        artifact_id = metadata.get("eve.artifactId") if isinstance(metadata, dict) else None
        if (
            kind in ("image", "document")
            and getattr(source, "type", None) == "url"
            and isinstance(artifact_id, str)
            and artifact_id
            and len(artifact_ids) < MAX_ATTACHMENTS
        ):
            artifact_ids.append(artifact_id)
        else:
            skipped += 1
    if skipped:
        logger.warning("agui.input_part_skipped count=%d", skipped)
    return "\n".join(texts), artifact_ids


def _build_generation_request(
    query: str, artifact_ids: List[str], forwarded_props: Any
) -> GenerationRequest:
    """A ``GenerationRequest`` from the query and the ``forwardedProps.eve`` allowlist."""
    eve: Any = None
    if isinstance(forwarded_props, dict):
        eve = forwarded_props.get("eve")
    if eve is None:
        eve = {}
    if not isinstance(eve, dict):
        raise RequestValidationError(
            [
                {
                    "type": "dict_type",
                    "loc": ("body", "forwardedProps", "eve"),
                    "msg": "Input should be a valid dictionary",
                    "input": None,
                }
            ]
        )
    if "agent" in eve:
        raise agui_http_error("agent_not_allowed")
    fields: Dict[str, Any] = {}
    dropped = 0
    for key, value in eve.items():
        target = EVE_PROPS.get(key)
        if target is None:
            dropped += 1
            continue
        fields[target] = value
    if dropped:
        logger.warning("agui.forwarded_props_dropped count=%d", dropped)
    try:
        return GenerationRequest(
            query=query, artifact_ids=artifact_ids or None, **fields
        )
    except ValidationError as exc:
        errors = []
        for error in exc.errors(include_url=False, include_context=False):
            error = dict(error)
            error.pop("input", None)
            error["loc"] = ("body", "forwardedProps", "eve", *error.get("loc", ()))
            errors.append(error)
        raise RequestValidationError(errors) from None


def _log_ignored_input(body: RunAgentInput) -> None:
    if body.model_extra:
        logger.warning("agui.input_unknown_members count=%d", len(body.model_extra))
    if body.tools:
        logger.warning("agui.input_tools_ignored count=%d", len(body.tools))
    if body.resume:
        # No interrupts exist before H4: every entry names no open interrupt.
        logger.warning("agui.resume_ignored count=%d", len(body.resume))


async def _enforce_budget(user: User) -> None:
    try:
        await enforce_token_budget_or_raise(user)
    except HTTPException as exc:
        if exc.status_code != 429:
            raise
        raise agui_http_error("budget_exhausted", retry_after=_retry_after(exc)) from None


async def _assert_collections_owned(request: GenerationRequest, user_id: str) -> None:
    """403 ``collection_forbidden`` for a selected collection another user owns."""
    oids = []
    for collection_id in [*request.public_collections, *request.private_collections]:
        try:
            oids.append(ObjectId(collection_id))
        except Exception:
            continue
    if not oids:
        return
    foreign = await CollectionModel.find_all(
        limit=1,
        filter_dict={"_id": {"$in": oids}, "user_id": {"$nin": [user_id, None]}},
    )
    if foreign:
        raise agui_http_error("collection_forbidden")


async def _run_id_used(conversation_id: str, run_id: str) -> bool:
    # On the primary: a retried POST lands a moment after the first insert.
    doc = await Message.get_primary_collection().find_one(
        {"conversation_id": conversation_id, "metadata.run_id": run_id},
        projection={"_id": 1},
    )
    return doc is not None


def _run_trace_id() -> str:
    """The request's OpenTelemetry trace id for ``eve.traceId``.

    The contract requires the key on ``RUN_STARTED``. With telemetry off (a
    local stack without a collector) there is no trace, so a random id of the
    same shape stands in; it links to nothing.
    """
    return current_trace_id() or uuid.uuid4().hex


def _parse_frame(data: str) -> Optional[Dict[str, Any]]:
    """One legacy bus chunk (``data: {json}``) as a dict, or None."""
    if not isinstance(data, str):
        return None
    payload = data.strip()
    if not payload.startswith("data:"):
        return None
    payload = payload[len("data:") :].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        frame = json.loads(payload)
    except ValueError:
        return None
    return frame if isinstance(frame, dict) else None


@router.post(
    "/conversations/{conversation_id}/agui",
    response_class=StreamingResponse,
    dependencies=[Depends(require_agui_stream), CHAT_RATE_LIMIT],
)
async def create_agui_run(
    body: RunAgentInput,
    conversation_id: str,
    background_tasks: BackgroundTasks,
    requesting_user: User = Depends(get_current_user),
    bearer_token: str = Depends(get_bearer_token),
) -> StreamingResponse:
    """Run one agentic turn and stream it as AG-UI 1.0 events over SSE.

    Raises:
        HTTPException: contract 1.6 errors (400, 403, 404, 409, 429, 500).
    """
    set_user_context(requesting_user.id)
    set_conversation_context(conversation_id)

    # AG-UI input checks: pure, no I/O, before any slot is taken.
    if body.thread_id != conversation_id:
        raise agui_http_error("thread_mismatch")
    _check_protocol_version(body.protocol_version)
    _log_ignored_input(body)
    query, artifact_ids = _read_user_message(body)
    if not query.strip() and not artifact_ids:
        raise agui_http_error("no_user_message")
    request = _build_generation_request(query, artifact_ids, body.forwarded_props)
    run_id = body.run_id

    # Same guards as stream-generate-agentic, in the same order.
    slot = await acquire_generation_slot_or_raise(
        ROUTE_NAME, slow_after=AGENTIC_TURN_DEADLINE
    )
    gen_task = None
    message = None
    admission = (conversation_id, run_id)
    admitted = False
    try:
        conversation = await Conversation.find_by_id(conversation_id)
        # A conversation of someone else answers 404 too (contract 1.2).
        if not conversation or conversation.user_id != requesting_user.id:
            raise agui_http_error("not_found")
        await _enforce_budget(requesting_user)
        await _assert_collections_owned(request, requesting_user.id)

        if admission in _runs_in_admission or await _run_id_used(conversation_id, run_id):
            raise agui_http_error("run_id_conflict")
        _runs_in_admission.add(admission)
        admitted = True

        request = await _prepare_agentic_request(request, requesting_user, conversation_id)
        request.mcp_proxy_bearer_token = bearer_token
        request.mcp_user_id = str(requesting_user.id)

        attachments = await resolve_artifact_attachments(
            request.artifact_ids, conversation_id, requesting_user
        )

        message = await Message.create(
            conversation_id=conversation_id,
            input=request.query,
            output="",
            documents=[],
            use_rag=False,
            request_input=request,
            metadata={"pipeline": "agentic", "run_id": run_id},
            attachments=attachments,
        )
        _runs_in_admission.discard(admission)
        admitted = False
        set_message_context(message.id)

        cancel_mgr = get_cancel_manager()
        cancel_event = cancel_mgr.create(message.id)
        cancel_mgr.link_conversation(conversation_id, message.id)
        subscriber_ready = asyncio.Event()
        gen_task = asyncio.create_task(
            run_agentic_generation_to_bus(
                request=request,
                conversation_id=conversation_id,
                message_id=message.id,
                user_id=requesting_user.id,
                background_tasks=background_tasks,
                cancel_event=cancel_event,
                subscriber_ready=subscriber_ready,
                deadline_seconds=AGENTIC_TURN_DEADLINE,
            )
        )
        slot.release_when_done(gen_task)
        cancel_mgr.set_task(message.id, gen_task)
        finish_turn_if_cancelled_unstarted(
            gen_task, conversation_id, message.id, cancel_event
        )

        adapter = AguiAdapter(
            RunContext(
                thread_id=conversation_id,
                run_id=run_id,
                message_id=message.id,
                trace_id=_run_trace_id(),
                graph=config.AGENT_GRAPH_TYPE
                if config.AGENT_GRAPH_TYPE in ("react", "deep")
                else None,
            )
        )
        bus = get_stream_bus()
        message_id = message.id

        async def _events():
            yield encode_sse(adapter.start())
            # As on the legacy route, the response stays open until the runner
            # closes the bus after the token charge; nothing is sent after the
            # terminal event.
            async with contextlib.aclosing(
                bus.subscribe(message_id, ready=subscriber_ready)
            ) as frames:
                async for data in frames:
                    if adapter.terminated:
                        continue
                    frame = _parse_frame(data)
                    if frame is None:
                        continue
                    events = adapter.adapt(frame)
                    if events:
                        yield encode_sse(events)
            tail = adapter.finish()
            if tail:
                yield encode_sse(tail)

        response = StreamingResponse(
            with_sse_keepalive(_events(), interval=KEEPALIVE_INTERVAL_S),
            media_type="text/event-stream",
        )
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Connection"] = "keep-alive"
        response.headers["X-Accel-Buffering"] = "no"
        return response

    except HTTPException as http_exc:
        if message:
            await _record_failure(message, http_exc)
        raise
    except Exception as exc:
        logger.error("agui.pre_stream_failed error_type=%s", type(exc).__name__)
        if message:
            await _record_failure(message, exc)
        raise agui_http_error("internal_error") from None
    finally:
        if admitted:
            _runs_in_admission.discard(admission)
        if gen_task is None:
            slot.release()


async def _record_failure(message: Message, exc: BaseException) -> None:
    """Persist the failure on the message and in the error log, as the legacy route does."""
    try:
        existing = dict(getattr(message, "metadata", {}) or {})
        existing["error"] = build_error_payload(exc)
        message.metadata = existing
        await message.save()
    except Exception:
        pass
    with contextlib.suppress(Exception):
        await get_error_logger().log_error_sync(
            error=exc,
            source="router",
            kind="error_handler",
            description="Exception in create_agui_run",
            error_type=type(exc).__name__,
        )
