"""AG-UI 1.0 stream of one agentic turn: ``POST /conversations/{id}/agui``.

Behind ``FEATURE_AGUI_STREAM`` (off: 404). The route takes the same guards as
``stream-generate-agentic`` in the same order, reads one ``RunAgentInput``,
starts the same runner on the same stream bus, and pipes the legacy frames
through ``AguiAdapter`` into SDK-encoded SSE events with the usual
keepalive comment. Stop is ``POST /conversations/{id}/stop`` or, as AG-UI
clients do it, closing the request: a response that ends before its terminal
event cancels the run on the same path as Stop (decision 0008). The legacy
stream route keeps running after a disconnect.

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
import re
import uuid
from typing import Any, Awaitable, Dict, List, Optional, Tuple, TypeVar

from ag_ui.core import RunAgentInput, UserMessage
from bson import ObjectId
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from pydantic import ValidationError
from starlette.types import Receive, Scope, Send

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

try:
    from redis import asyncio as aioredis  # type: ignore
except Exception:  # pragma: no cover - redis is in requirements.txt
    aioredis = None  # type: ignore

logger = logging.getLogger(__name__)

T = TypeVar("T")

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

# A runId is minted by the client (a UUID v4, 36 chars). Bounded so a stored,
# queried and echoed value cannot grow without limit.
RUN_ID_MAX_CHARS = 64
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,%d}$" % RUN_ID_MAX_CHARS)
# The run id claim in Valkey (database of REDIS_URL): set before the conflict
# read, so two POSTs with one runId on two workers cannot both pass. The key
# outlives any retry of one send; after the insert the Mongo row also blocks.
RUN_ID_CLAIM_TTL_S = 3600
RUN_ID_STORE_TIMEOUT_S = 0.5
# The response ends at the turn deadline plus this margin even if the bus close
# is lost: the stream then ends with RUN_ERROR stream_incomplete.
STREAM_DEADLINE_MARGIN_S = 60.0
# The stopped frame of a disconnect goes out like the Stop route's, bounded:
# the response is already gone and nobody waits for it.
DISCONNECT_STOP_TIMEOUT_S = 2.0
_STOPPED_FRAME = f"data: {json.dumps({'type': 'stopped'})}\n\n"
# Strong references to the stop publishes until they end.
_disconnect_stops: set = set()

_CLAIM_STORE = "store"
_CLAIM_LOCAL = "local"
# Fallback when no Valkey is configured (local tests) or it does not answer:
# (conversation_id, run_id) pairs between the claim and the message insert on
# this worker.
_runs_in_admission: set[Tuple[str, str]] = set()
_run_id_store: Any = None

# Pre-stream HTTP errors of the helpers the legacy route shares (string detail)
# mapped to contract codes. A 403 on someone else's item answers as missing.
_ARTIFACT_ERRORS = {404: "artifact_not_found", 403: "artifact_not_found"}
_CUSTOM_MODEL_ERRORS = {
    404: "custom_model_not_found",
    403: "custom_model_not_found",
    422: "custom_model_unusable",
}


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


_MINOR_VERSION = re.compile(r"^1\.\d{1,4}$")


def _check_protocol_version(version: Optional[str]) -> None:
    if version is None:
        return
    major, _, _ = str(version).partition(".")
    if major != "1":
        raise agui_http_error("protocol_version_unsupported")
    if str(version) != "1.0":
        # Client text: logged only when it has the expected shape.
        if _MINOR_VERSION.match(str(version)):
            logger.warning("agui.protocol_minor version=%s", version)
        else:
            logger.warning("agui.protocol_minor version=unrecognised")


def _check_run_id(run_id: Any) -> str:
    """422 for a runId that is not a short id string (contract 1.3)."""
    if isinstance(run_id, str) and _RUN_ID_PATTERN.match(run_id):
        return run_id
    raise RequestValidationError(
        [
            {
                "type": "string_pattern_mismatch",
                "loc": ("body", "runId"),
                "msg": f"runId must be 1 to {RUN_ID_MAX_CHARS} letters, digits, '-' or '_'",
                "input": None,
            }
        ]
    )


def _read_user_message(body: RunAgentInput) -> Tuple[str, List[str], List[Optional[str]]]:
    """Query text, artifact ids and the skipped parts (their ``id``, None when
    the part has none) from the trailing ``UserMessage``.

    Only the last message is read (deviation D1). Text parts are joined with a
    newline; an ``image`` or ``document`` part counts only with a ``url`` source
    and ``metadata["eve.artifactId"]``, and its URL is never fetched.
    """
    if not body.messages or not isinstance(body.messages[-1], UserMessage):
        raise agui_http_error("no_user_message")
    content = body.messages[-1].content
    if isinstance(content, str):
        return content, [], []
    texts: List[str] = []
    artifact_ids: List[str] = []
    skipped: List[Optional[str]] = []
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
            part_id = getattr(part, "id", None)
            skipped.append(part_id if isinstance(part_id, str) and part_id else None)
    if skipped:
        logger.warning("agui.input_part_skipped count=%d", len(skipped))
    return "\n".join(texts), artifact_ids, skipped


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
    if body.context:
        # Deviation D5: context is not read in v1.
        logger.warning("agui.input_context_ignored count=%d", len(body.context))
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


async def _mapped_errors(awaitable: Awaitable[T], codes: Dict[int, str]) -> T:
    """Await a shared helper; its string-detail HTTP errors become contract errors."""
    try:
        return await awaitable
    except HTTPException as exc:
        if isinstance(exc.detail, dict) and "code" in exc.detail:
            raise
        raise agui_http_error(codes.get(exc.status_code, "internal_error")) from None


def _store_client() -> Any:
    """The Valkey client for run id claims, or None without REDIS_URL."""
    global _run_id_store
    if not config.REDIS_URL or aioredis is None:
        return None
    # One client per event loop: its connections belong to the loop that
    # opened them (one per worker in production, one per test in the suite).
    loop = asyncio.get_running_loop()
    if _run_id_store is None or _run_id_store[0] is not loop:
        # Bounded socket waits: the claim sits on the request path.
        _run_id_store = (
            loop,
            aioredis.Redis.from_url(
                config.REDIS_URL,
                socket_timeout=RUN_ID_STORE_TIMEOUT_S,
                socket_connect_timeout=RUN_ID_STORE_TIMEOUT_S,
            ),
        )
    return _run_id_store[1]


def _claim_key(conversation_id: str, run_id: str) -> str:
    return f"agui-run:{conversation_id}:{run_id}"


async def _claim_run_id(conversation_id: str, run_id: str) -> Optional[str]:
    """Reserve ``run_id`` on the thread for every worker; None when taken.

    ``SET NX`` with a TTL in Valkey. Without REDIS_URL, or when the store does
    not answer in time, the claim falls back to this worker's set (fail open,
    like the request limiter): the Mongo read still blocks a used runId.
    """
    client = _store_client()
    if client is not None:
        try:
            taken = await asyncio.wait_for(
                client.set(
                    _claim_key(conversation_id, run_id),
                    "1",
                    nx=True,
                    ex=RUN_ID_CLAIM_TTL_S,
                ),
                RUN_ID_STORE_TIMEOUT_S,
            )
            return _CLAIM_STORE if taken else None
        except Exception as exc:
            logger.warning("agui.run_id_store_failed error_type=%s", type(exc).__name__)
    admission = (conversation_id, run_id)
    if admission in _runs_in_admission:
        return None
    _runs_in_admission.add(admission)
    return _CLAIM_LOCAL


async def _release_run_id(
    conversation_id: str, run_id: str, claim: Optional[str], *, created: bool
) -> None:
    """Drop the claim of a run that created no message; after the insert the
    store key stays until its TTL (the Mongo row blocks from then on)."""
    if claim == _CLAIM_LOCAL:
        _runs_in_admission.discard((conversation_id, run_id))
        return
    if claim == _CLAIM_STORE and not created:
        client = _store_client()
        if client is None:
            return
        with contextlib.suppress(Exception):
            await asyncio.wait_for(
                client.delete(_claim_key(conversation_id, run_id)),
                RUN_ID_STORE_TIMEOUT_S,
            )


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
    include_in_schema=config.FEATURE_AGUI_STREAM,
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
    run_id = _check_run_id(body.run_id)
    _check_protocol_version(body.protocol_version)
    _log_ignored_input(body)
    query, artifact_ids, skipped_parts = _read_user_message(body)
    if not query.strip() and not artifact_ids:
        raise agui_http_error("no_user_message")
    request = _build_generation_request(query, artifact_ids, body.forwarded_props)

    # Same guards as stream-generate-agentic, in the same order.
    slot = await acquire_generation_slot_or_raise(
        ROUTE_NAME, slow_after=AGENTIC_TURN_DEADLINE
    )
    gen_task = None
    message = None
    claim: Optional[str] = None
    try:
        conversation = await Conversation.find_by_id(conversation_id)
        # A conversation of someone else answers 404 too (contract 1.2).
        if not conversation or conversation.user_id != requesting_user.id:
            raise agui_http_error("not_found")
        await _enforce_budget(requesting_user)
        await _assert_collections_owned(request, requesting_user.id)

        claim = await _claim_run_id(conversation_id, run_id)
        if claim is None or await _run_id_used(conversation_id, run_id):
            raise agui_http_error("run_id_conflict")

        request = await _mapped_errors(
            _prepare_agentic_request(request, requesting_user, conversation_id),
            _CUSTOM_MODEL_ERRORS,
        )
        request.mcp_proxy_bearer_token = bearer_token
        request.mcp_user_id = str(requesting_user.id)

        attachments = await _mapped_errors(
            resolve_artifact_attachments(
                request.artifact_ids, conversation_id, requesting_user
            ),
            _ARTIFACT_ERRORS,
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
                skipped_parts=tuple(skipped_parts),
            )
        )
        bus = get_stream_bus()
        message_id = message.id
        stream_budget_s = AGENTIC_TURN_DEADLINE + STREAM_DEADLINE_MARGIN_S

        async def _events():
            yield encode_sse(adapter.start())
            # As on the legacy route, the response stays open until the runner
            # closes the bus after the token charge; nothing is sent after the
            # terminal event. A bus that fails, or a close that never comes,
            # ends the run with the closing sequence and RUN_ERROR
            # stream_incomplete (contract 2.4): exactly one terminal event.
            # Each item is read in its own task (with_sse_keepalive), so the
            # deadline is a per-read wait, not an asyncio.timeout block.
            loop = asyncio.get_running_loop()
            deadline = loop.time() + stream_budget_s
            try:
                async with contextlib.aclosing(
                    bus.subscribe(message_id, ready=subscriber_ready)
                ) as frames:
                    iterator = frames.__aiter__()
                    while True:
                        remaining = deadline - loop.time()
                        if remaining <= 0:
                            raise TimeoutError("agui stream deadline")
                        try:
                            data = await asyncio.wait_for(
                                iterator.__anext__(), remaining
                            )
                        except StopAsyncIteration:
                            break
                        if adapter.terminated:
                            continue
                        frame = _parse_frame(data)
                        if frame is None:
                            continue
                        events = adapter.adapt(frame)
                        if events:
                            yield encode_sse(events)
            except Exception as exc:
                logger.warning(
                    "agui.stream_failed error_type=%s terminated=%s",
                    type(exc).__name__,
                    adapter.terminated,
                )
            tail = adapter.finish()
            if tail:
                yield encode_sse(tail)

        response = AguiStreamingResponse(
            with_sse_keepalive(_events(), interval=KEEPALIVE_INTERVAL_S),
            media_type="text/event-stream",
            on_close=lambda: _stop_on_disconnect(adapter, cancel_event, message_id),
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
        await _release_run_id(
            conversation_id, run_id, claim, created=message is not None
        )
        if gen_task is None:
            slot.release()


class AguiStreamingResponse(StreamingResponse):
    """A ``StreamingResponse`` that calls ``on_close`` once the response is over.

    Over in every way but a cancel of the request task: the body ended, the
    client disconnected (ASGI 2.3, a task listening for ``http.disconnect``;
    2.4, a write that fails), or a send raised. ``on_close`` decides from the run state whether that was a
    disconnect; it is synchronous so a cancelled request still runs it whole.
    The body is closed afterwards, so nothing more is produced for it.
    """

    def __init__(self, content: Any, *, on_close: Any, **kwargs: Any) -> None:
        super().__init__(content, **kwargs)
        self._on_close = on_close

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        cancelled = False
        try:
            await super().__call__(scope, receive, send)
        except asyncio.CancelledError:
            # The request task itself was cancelled (the worker shutting
            # down), not a client that left: the run is handled as before.
            cancelled = True
            raise
        finally:
            if not cancelled:
                try:
                    self._on_close()
                except Exception as exc:
                    logger.warning("agui.on_close_failed error_type=%s", type(exc).__name__)
            aclose = getattr(self.body_iterator, "aclose", None)
            if aclose is not None:
                with contextlib.suppress(Exception):
                    await aclose()


async def _publish_disconnect_stop(message_id: str) -> None:
    """The Stop route's ``bus.stop``: the stopped frame, then the bus close."""
    try:
        await asyncio.wait_for(
            get_stream_bus().stop(message_id, _STOPPED_FRAME), DISCONNECT_STOP_TIMEOUT_S
        )
    except Exception as exc:
        logger.warning(
            "agui.disconnect_stop_failed message_id=%s error_type=%s",
            message_id,
            type(exc).__name__,
        )


def _stop_on_disconnect(
    adapter: AguiAdapter, cancel_event: asyncio.Event, message_id: str
) -> None:
    """Cancel the run like ``POST /stop`` when the response ended before its terminal event.

    Nothing to do after a terminal event (the run is over; the bus may still be
    open for the token charge) or after a Stop already set the cancel event.
    """
    if adapter.terminated or cancel_event.is_set():
        return
    logger.info("agui.client_disconnected message_id=%s", message_id)
    get_cancel_manager().cancel(message_id)
    task = asyncio.get_running_loop().create_task(_publish_disconnect_stop(message_id))
    _disconnect_stops.add(task)
    task.add_done_callback(_disconnect_stops.discard)


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
