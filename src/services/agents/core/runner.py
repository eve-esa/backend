"""Agent runner — all backend integration logic for agent graph execution.

Handles: LLM instantiation, MCP tool loading, system prompt resolution,
conversation history, OpenTelemetry root span, SSE streaming, persistence,
cancellation, error logging, token consumption, and stream bus publishing.
"""

import asyncio
import contextlib
import copy
import functools
import inspect
import json
import logging
import re
import time
import urllib.parse
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from fastapi import BackgroundTasks

from src import config as src_config
from src.config import (
    AGENT_CONTEXT_WINDOW_TOKENS,
    AGENTIC_LLM_TYPE,
    AGENTIC_TIMEOUT,
    MODEL_TIMEOUT,
    PINNED_CONTEXT,
    SUMMARY_KEEP_FRACTION,
    SUMMARY_MAX_TOKENS,
    SUMMARY_TRIGGER_FRACTION,
    TOOL_RESULT_MAX_TOKENS,
    TRUNCATE_ARGS_MAX_CHARS,
)
from src.core.llm_health import is_endpoint_failure
from src.core.llm_manager import LLMType, endpoint_timeout
from bson import ObjectId

from src.database.models.artifact import Artifact
from src.database.models.conversation import Conversation
from src.database.models.message import Message
from src.database.models.user import User
from src.services.custom_model_service import (
    build_custom_model_llm,
    custom_model_metadata,
    ensure_custom_model_has_credentials,
    get_owned_custom_model,
)
from src.schemas.generation_request import GenerationRequest
from src.services.generate_answer import (
    EmptyAnswer,
    _get_conversation_history_from_db,
    build_empty_answer_payload,
    build_endpoint_metadata,
    build_error_payload,
    deadline_error,
    deadline_overrides,
    endpoint_answered_by,
    get_shared_llm_manager,
    is_empty_answer,
    maybe_rollup_and_trim_history,
    persist_message_state,
    is_terminal_event,
    persist_runner_cancel,
    release_stop_handle,
    raise_if_deadline_expired,
    resolve_generated_model_name,
)
from src.services.agents.core.history_search import (
    SEARCH_NOTE,
    build_history_tools,
    note_searchable_history,
)
from src.services.agents.core.registry import get_agent_graph
from src.services.agents.core.sandbox import (
    backend_for_compile,
    close_sandbox,
    large_tool_result_permissions,
    open_sandbox,
)
from src.services.agents.core.sandbox_pool import acquire_sandbox, release_sandbox
from src.services.agents.core.skills import (
    cached_skills_root_key,
    resolve_skills_root,
    skills_root_key,
)
from src.services.checkpoint_retention import build_mongo_checkpointer
from src.services.conversation_documents import attached_document_names
from src.services.mcp.artifact_context import (
    reset_artifact_context,
    set_artifact_context,
)
from src.services.mcp.retrieval_context import (
    get_retrieval_context,
    reset_retrieval_context,
    set_retrieval_context,
)
from src.services.sources.citations import (
    CitationRewriter,
    Edit,
    SourceRegistry,
    citation_payload,
    prior_sources_from_rows,
    shift_offset,
    strip_handles,
)
from src.services.mcp.tool_loader import (
    MCP_SERVER_METADATA_KEY,
    _MCPToolsWithClient,
    load_mcp_tools_for_servers as _load_mcp_tools_for_servers,
)
from src.services.stream_bus import get_stream_bus
from src.services.token_rate_limiter import (
    consume_tokens_for_user,
    count_tokens_for_texts,
)
from src.utils.error_logger import persist_policy_event
from src.utils.helpers import (
    extract_documents_from_retrieval_payload,
    get_mongodb_uri,
    is_retrieval_error_payload,
    stringify_message_content,
    stringify_tool_content,
)
from src.observability.context import (
    add_span_event,
    agent_span,
    cancel_is_stop,
    current_trace_id,
    mark_error,
    set_llm_attributes,
    span_trace_id,
)

logger = logging.getLogger(__name__)

# ─── Optional imports ──────────────────────────────────────────────────────────

_langgraph_available = False
try:
    from langchain_core.messages import (
        AIMessage,
        AIMessageChunk,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )
    from langgraph.checkpoint.mongodb import MongoDBSaver

    _langgraph_available = True
except Exception:
    AIMessage = AIMessageChunk = HumanMessage = SystemMessage = ToolMessage = None  # type: ignore
    MongoDBSaver = None  # type: ignore

try:
    from langgraph.checkpoint.memory import InMemorySaver
except Exception:
    InMemorySaver = None  # type: ignore

# Text-format tool-call helpers come from this backend's OWN
# `src.services.agentic_utils` rather than the external `graphs_utils_module()`
# package: the installed `eve-esa-agents` package's `utils.py` doesn't define
# `might_be_incomplete_text_tool_call` at all, so importing that single
# attribute used to raise inside the `try` block below and silently set
# *every* name in this group (including `has_text_tool_call` and
# `parse_text_tool_calls`, which DO exist there) to None — permanently
# disabling text-format tool-call detection in the streaming path with no
# error surfaced anywhere. `agentic_utils.py` is vendored in this repo and
# kept in sync with what the streaming/turn-buffer logic below needs.
try:
    from src.services.agentic_utils import (
        has_text_tool_call,
        might_be_incomplete_text_tool_call,
        parse_text_tool_calls,
        split_tool_calls_and_answer_text,
        tool_call_label,
    )
except Exception:
    tool_call_label = None  # type: ignore
    has_text_tool_call = None  # type: ignore
    might_be_incomplete_text_tool_call = None  # type: ignore
    parse_text_tool_calls = None  # type: ignore
    split_tool_calls_and_answer_text = None  # type: ignore


def _strip_text_tool_calls(text: str) -> str:
    """``text`` without a text-format tool call, the rule of the stream's turn buffer.

    Recognised ``[TOOL_CALLS]`` calls go and the text after them stays; a marker
    nothing parses leaves no answer. A call written as text is never run by the
    sync path, so it is not an answer either.
    """
    if not (has_text_tool_call and has_text_tool_call(text)):
        return text
    calls, rest = (
        split_tool_calls_and_answer_text(text)
        if split_tool_calls_and_answer_text
        else ([], "")
    )
    return rest if calls else ""


def _content_text(content: Any) -> str:
    """Message content as text: a list of parts is joined by its text parts,
    the stream's rule, never turned into a Python repr."""
    if isinstance(content, list):
        return "".join(
            c.get("text", "") if isinstance(c, dict) else str(c) for c in content
        )
    return content if isinstance(content, str) else str(content or "")


def _final_answer_text(
    messages: List[Any], handles: Optional[SourceRegistry] = None
) -> str:
    """What the non-streaming routes return: the last model message without tool calls.

    The model may copy an [S<k>] handle an earlier streamed turn left in the
    thread: this route hands out none, so it never reaches the user.
    ``handles`` is that conversation's stored Sources (``strip_handles``);
    None strips every handle.
    """
    for msg in reversed(messages):
        if (
            isinstance(msg, AIMessage)
            and msg.content
            and not getattr(msg, "tool_calls", None)
        ):
            return strip_handles(
                _strip_text_tool_calls(_content_text(msg.content)), handles
            )
    return ""


def _mark_empty_answer(span: Any) -> None:
    """Record an empty answer on the turn's root span, as a raised failure would be."""
    if span is None:
        return
    mark_error(span, EmptyAnswer(build_empty_answer_payload()["message"]))
    with contextlib.suppress(Exception):
        span.set_attribute("eve.error.code", "empty_answer")


def _log_empty_answer(conversation_id: Optional[str], message_id: Optional[str]) -> None:
    # Ids only: never the question or the model output.
    logger.warning(
        "generation.empty_answer conversation_id=%s message_id=%s",
        conversation_id,
        message_id,
    )


# ─── Trace serialisation ──────────────────────────────────────────────────────


def _coerce_trace_metadata(value: Any) -> Optional[Dict[str, Any]]:
    """Normalise LangChain metadata objects into a plain dict for persistence."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value or None
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(exclude_none=True)
        return dumped or None
    return None


def _enrich_trace_entry_with_message_metadata(
    entry: Dict[str, Any], msg: Any
) -> None:
    """Attach provider/model metadata and correlation ids to a trace entry."""
    msg_id = getattr(msg, "id", None)
    if msg_id:
        entry["id"] = msg_id

    response_metadata = _coerce_trace_metadata(
        getattr(msg, "response_metadata", None)
    )
    if response_metadata:
        entry["response_metadata"] = response_metadata

    usage_metadata = _coerce_trace_metadata(getattr(msg, "usage_metadata", None))
    if usage_metadata:
        entry["usage_metadata"] = usage_metadata

    if ToolMessage and isinstance(msg, ToolMessage):
        tool_call_id = getattr(msg, "tool_call_id", None)
        if tool_call_id:
            entry["tool_call_id"] = tool_call_id
        status = getattr(msg, "status", None)
        if status:
            entry["status"] = status

    if AIMessage and isinstance(msg, AIMessage):
        invalid_tool_calls = getattr(msg, "invalid_tool_calls", None)
        if invalid_tool_calls:
            entry["invalid_tool_calls"] = invalid_tool_calls


# Per string value in a streamed tool_call event. A write_file call carries a
# whole file; the full arguments stay in Message.trace.
STREAM_TOOL_ARG_MAX_CHARS = 500

# Between the answer text of two model calls in one turn (text, a tool call,
# more text), so the stream and the stored answer do not glue them.
PARAGRAPH_BREAK = "\n\n"


def _stream_tool_args(value: Any) -> Any:
    """Tool call arguments for a stream event, long strings cut."""
    if isinstance(value, str):
        if len(value) <= STREAM_TOOL_ARG_MAX_CHARS:
            return value
        return value[:STREAM_TOOL_ARG_MAX_CHARS] + "…"
    if isinstance(value, dict):
        return {str(k): _stream_tool_args(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stream_tool_args(v) for v in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _stream_tool_args(str(value))


def _stream_args_truncated(value: Any) -> bool:
    """True when ``_stream_tool_args`` would cut a string inside *value*."""
    if isinstance(value, str):
        return len(value) > STREAM_TOOL_ARG_MAX_CHARS
    if isinstance(value, dict):
        return any(_stream_args_truncated(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_stream_args_truncated(v) for v in value)
    if value is None or isinstance(value, (bool, int, float)):
        return False
    return _stream_args_truncated(str(value))


# ─── Step frames ──────────────────────────────────────────────────────────────
# Additive fields on the tool_call and tool_result frames, and the same fields
# under ``step`` in the persisted trace. The legacy client ignores them.

_SKILL_FILE = re.compile(r"^/skills/([^/]+)/SKILL\.md$")
_SKILL_READ_TOOLS = frozenset({"read_file"})
_SUBAGENT_TOOLS = frozenset({"task"})


def _tool_server(tool: Any) -> Optional[str]:
    meta = getattr(tool, "metadata", None)
    if isinstance(meta, dict):
        server = meta.get(MCP_SERVER_METADATA_KEY)
        if isinstance(server, str) and server:
            return server
    return None


# Content prefixes the agents react tools node writes for a failed call. Up to
# the agents fix it leaves ``ToolMessage.status`` at its default, so the prefix
# is the only failure signal there. ``Error:`` (LangChain ToolNode, deep) is
# only stripped before the timeout check: a result may start with it.
_TOOL_ERROR_PREFIXES = ("Tool error:", "Unknown tool:")
_TOOL_ERROR_HEADS = _TOOL_ERROR_PREFIXES + ("Error:",)
# Timeout words, matched only at the head of the error text and before any
# quote, where exception names and messages sit: an echoed query or argument
# (quoted, or later in the text) cannot turn a plain error into a timeout.
_TIMEOUT_HEAD = re.compile(r"timeout|timed out", re.IGNORECASE)
_TIMEOUT_HEAD_CHARS = 60


def _tool_status(lc_status: Any, content: Any) -> str:
    """``ok``, ``error`` or ``timeout`` for one ToolMessage."""
    text = stringify_tool_content(content).lstrip()
    if lc_status != "error" and not text.startswith(_TOOL_ERROR_PREFIXES):
        return "ok"
    if text.startswith("Unknown tool:"):
        return "error"
    prefix = next((p for p in _TOOL_ERROR_HEADS if text.startswith(p)), "")
    head = text[len(prefix) :]
    head = re.split(r"['\"]", head[:_TIMEOUT_HEAD_CHARS], maxsplit=1)[0]
    return "timeout" if _TIMEOUT_HEAD.search(head) else "error"


def _tool_duration_ms(message: Any) -> Optional[int]:
    """The call's own time when the tools node stamped it (agents fix)."""
    meta = getattr(message, "response_metadata", None)
    value = meta.get("duration_ms") if isinstance(meta, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return int(value)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class _StepFrames:
    """Ids, kinds, servers, status and durations of the tool steps of one run.

    ``tool_call_id`` is unique within the run: a missing raw id is generated,
    a raw id already used by another call gets ``~2``, ``~3`` (text-format
    models repeat ``call_1_x`` across steps). A call with the same tool and
    arguments as a call whose result failed is a retry: it gets its own id,
    counts ``attempt`` and names the failed attempt in ``retry_of``.

    ``duration_ms`` is the call's own time when the tools node stamps it on
    the ToolMessage; otherwise it runs from the announcement to the result,
    which on a multi-call step includes the calls that ran before it. Clocks
    use ``time.monotonic`` so the ``perf_counter`` latencies are left alone.

    The trace steps written by ``annotate_trace_calls`` stay referenced, so a
    Stop or a ghost call updates the persisted trace in place.
    """

    def __init__(self, tools: Any = ()) -> None:
        self._server_by_tool: Dict[str, str] = {}
        for tool in tools or []:
            name = getattr(tool, "name", None)
            server = _tool_server(tool)
            if isinstance(name, str) and name and server:
                self._server_by_tool[name] = server
        self._step_by_raw: Dict[str, str] = {}
        self._used: set = set()
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._failed: Dict[str, tuple] = {}
        self._fields_by_announce_key: Dict[str, Dict[str, Any]] = {}
        self._trace_steps: Dict[str, Dict[str, Any]] = {}
        self._seq = 0
        self._last_result_seq = 0

    @staticmethod
    def call_key(name: str, args: Any) -> str:
        return f"{name}:{json.dumps(args, sort_keys=True, default=str)}"

    def _new_id(self, raw_id: Any) -> str:
        base = str(raw_id) if raw_id else f"call_eve_{uuid.uuid4().hex[:16]}"
        candidate, n = base, 1
        while candidate in self._used:
            n += 1
            candidate = f"{base}~{n}"
        return candidate

    def _kind(self, name: str, args: Dict[str, Any], step_id: str) -> tuple:
        if name in _SUBAGENT_TOOLS:
            extra: Dict[str, Any] = {"subagent_run_id": f"sa-{step_id}"}
            subagent = args.get("subagent_type")
            if isinstance(subagent, str) and subagent:
                extra["subagent_name"] = subagent
            return "subagent", extra
        if name in _SKILL_READ_TOOLS:
            match = _SKILL_FILE.match(str(args.get("file_path") or ""))
            if match:
                return "skill", {"skill_name": match.group(1)}
        if name in self._server_by_tool:
            return "mcp", {}
        return "builtin", {}

    def call(
        self,
        name: str,
        args: Any,
        raw_id: Any,
        *,
        announce_key: str,
        model: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fields for a tool_call frame; records the call as pending.

        ``model`` is always our endpoint key (``main``, ``fallback``, ...);
        ``model_name`` the provider's own model id, when the message has one.
        """
        safe_args = args if isinstance(args, dict) else {}
        key = self.call_key(name, args)
        retried = self._failed.pop(key, None)
        step_id = self._new_id(raw_id)
        attempt = retried[1] + 1 if retried is not None else 1
        self._used.add(step_id)
        if raw_id:
            self._step_by_raw[str(raw_id)] = step_id
        kind, extra = self._kind(name, safe_args, step_id)
        server = self._server_by_tool.get(name)
        fields: Dict[str, Any] = {
            "tool_call_id": step_id,
            "kind": kind,
            "server": server,
            "started_at": _utc_now_iso(),
            "attempt": attempt,
            **extra,
        }
        if retried is not None:
            fields["retry_of"] = retried[0]
        if model:
            fields["model"] = model
        if model_name:
            fields["model_name"] = model_name
        if _stream_args_truncated(args):
            fields["args_truncated"] = True
        self._seq += 1
        self._pending[step_id] = {
            "name": name,
            "key": key,
            "started": time.monotonic(),
            "attempt": attempt,
            "retry_of": fields.get("retry_of"),
            "server": server,
            "kind": kind,
            "seq": self._seq,
        }
        self._fields_by_announce_key[announce_key] = fields
        return fields

    def result(
        self,
        name: Any,
        raw_id: Any,
        lc_status: Any,
        content: Any,
        duration_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Fields for a tool_result frame; pairs it with its pending call."""
        step_id: Optional[str] = None
        if raw_id:
            step_id = self._step_by_raw.get(str(raw_id))
        elif name:
            step_id = next(
                (sid for sid, s in self._pending.items() if s["name"] == name), None
            )
        if step_id is None and not raw_id and self._pending:
            step_id = next(iter(self._pending))
        state = self._pending.pop(step_id, None) if step_id else None
        if step_id is None:
            step_id = str(raw_id) if raw_id else self._new_id(None)
            self._used.add(step_id)
        self._seq += 1
        self._last_result_seq = self._seq
        status = _tool_status(lc_status, content)
        tool = str(name or "")
        fields: Dict[str, Any] = {
            "tool_call_id": step_id,
            "status": status,
            "kind": state["kind"] if state else self._kind(tool, {}, step_id)[0],
            "server": state["server"] if state else self._server_by_tool.get(tool),
            "attempt": state["attempt"] if state else 1,
        }
        if duration_ms is not None:
            fields["duration_ms"] = duration_ms
        elif state is not None:
            fields["duration_ms"] = int(
                round((time.monotonic() - state["started"]) * 1000)
            )
        if state is not None and state.get("retry_of"):
            fields["retry_of"] = state["retry_of"]
        if status != "ok":
            fields["error_code"] = "timeout" if status == "timeout" else "tool_error"
            if state is not None:
                self._failed[state["key"]] = (step_id, state["attempt"])
        return fields

    def drop_unanswered(self) -> int:
        """Forget calls a tools step already passed by without a result.

        Called at each agent update: a call announced before the last result
        and still pending is a ghost (a text-format call no tool ran). It is
        neither finished nor cancelled; its trace step reads ``not_run``.
        """
        stale = [
            sid
            for sid, state in self._pending.items()
            if state["seq"] < self._last_result_seq
        ]
        for sid in stale:
            self._pending.pop(sid, None)
            step = self._trace_steps.get(sid)
            if step is not None:
                step["status"] = "not_run"
        return len(stale)

    def cancel_pending(self) -> List[Dict[str, Any]]:
        """Pending calls as ``cancelled`` entries; clears them.

        Their trace steps get the same status and duration, so the stored
        trace says which calls a Stop cut.
        """
        now = time.monotonic()
        cancelled = []
        for step_id, state in self._pending.items():
            entry = {
                "tool_call_id": step_id,
                "status": "cancelled",
                "attempt": state["attempt"],
                "duration_ms": int(round((now - state["started"]) * 1000)),
            }
            cancelled.append(entry)
            step = self._trace_steps.get(step_id)
            if step is not None:
                step["status"] = "cancelled"
                step["duration_ms"] = entry["duration_ms"]
        self._pending.clear()
        return cancelled

    def annotate_trace_calls(self, entry: Dict[str, Any]) -> None:
        """Copy the announced fields onto the trace entry's tool calls."""
        for call in entry.get("tool_calls") or []:
            key = call.get("id") or self.call_key(
                call.get("name", "tool"), call.get("args") or {}
            )
            fields = self._fields_by_announce_key.get(key)
            if fields is not None:
                step = dict(fields)
                call["step"] = step
                self._trace_steps[fields["tool_call_id"]] = step


_USAGE_COUNT_KEYS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "input_token_details",
    "output_token_details",
)


def _usage_row(
    message: Any, endpoint_key: Optional[str], model_name: Optional[str]
) -> Optional[Dict[str, Any]]:
    """One ``final.usage`` row: the counts of one model call, no content.

    ``provider`` is our endpoint key, ``model`` the provider's model id when
    the message names it. None when the provider reported no count.
    """
    usage = getattr(message, "usage_metadata", None)
    if not isinstance(usage, dict):
        return None
    row = {k: usage[k] for k in _USAGE_COUNT_KEYS if usage.get(k) is not None}
    if not row:
        return None
    if endpoint_key:
        row["provider"] = endpoint_key
    if model_name or endpoint_key:
        row["model"] = model_name or endpoint_key
    return row


def _endpoint_switch_frame(endpoint: Optional[Dict[str, Any]]) -> Optional[str]:
    """``model_fallback`` frame when the chain skipped the requested endpoint
    because its circuit is open; None otherwise."""
    if not endpoint:
        return None
    requested = endpoint.get("requested")
    answered = endpoint.get("answered")
    if not requested or not answered or requested == answered:
        return None
    if requested not in (endpoint.get("circuit_open") or []):
        return None
    return f"data: {json.dumps({'type': 'model_fallback', 'from': requested, 'to': answered, 'reason': 'circuit_open'})}\n\n"


def _serialise_trace_entry(
    msg: Any,
    *,
    node: str = "",
    latency_s: Optional[float] = None,
    started_at_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Convert a LangChain message into a JSON-serialisable trace dict."""
    entry: Dict[str, Any] = {"node": node}
    if started_at_s is not None:
        entry["started_at_s"] = started_at_s
    if latency_s is not None:
        entry["latency_s"] = latency_s

    if AIMessage and isinstance(msg, AIMessage):
        entry["role"] = "assistant"
        entry["content"] = stringify_message_content(msg.content)
        tc = getattr(msg, "tool_calls", None)
        if tc:
            tool_calls: List[Dict[str, Any]] = []
            for c in tc:
                tc_entry: Dict[str, Any] = {
                    "name": c.get("name", ""),
                    "args": c.get("args", {}),
                }
                if c.get("id"):
                    tc_entry["id"] = c["id"]
                tool_calls.append(tc_entry)
            entry["tool_calls"] = tool_calls
    elif ToolMessage and isinstance(msg, ToolMessage):
        entry["role"] = "tool"
        entry["name"] = getattr(msg, "name", "tool")
        entry["content"] = stringify_tool_content(msg.content)
    elif HumanMessage and isinstance(msg, HumanMessage):
        entry["role"] = "user"
        entry["content"] = (
            msg.content if isinstance(msg.content, str) else str(msg.content)
        )
    elif SystemMessage and isinstance(msg, SystemMessage):
        entry["role"] = "system"
        entry["content"] = (
            msg.content if isinstance(msg.content, str) else str(msg.content)
        )
    else:
        entry["role"] = "unknown"
        entry["content"] = str(msg)

    _enrich_trace_entry_with_message_metadata(entry, msg)
    return entry


class _TraceTimeline:
    """Tracks per-step ``started_at_s``/``latency_s`` against a run's start.

    Timing rules (shared by the streaming and non-streaming runners):
    - An agent step (an AIMessage) starts when the previous recorded step
      ended, or at the run start for the first step, and ends when the
      message arrives.
    - A tool step (a ToolMessage) starts when the agent step that requested
      it ended, matched by ``tool_call_id`` against the requesting
      AIMessage's ``tool_calls``. Tools requested in parallel by one agent
      step therefore share the same start. A tool call with no match falls
      back to the most recently recorded agent step's end.
    """

    def __init__(self, run_start: float) -> None:
        self._run_start = run_start
        self._last_event_end_s: float = 0.0
        self._last_agent_end_s: float = 0.0
        self._agent_end_by_call_id: Dict[str, float] = {}

    def _elapsed(self, at: Optional[float]) -> float:
        return (at if at is not None else time.perf_counter()) - self._run_start

    def agent_step(
        self, tool_calls: Optional[List[Any]] = None, *, at: Optional[float] = None
    ) -> tuple[float, float]:
        """Record an agent step's end; return ``(started_at_s, latency_s)``.

        Also covers the manual final-answer entry: it follows the exact same
        rule (start at the previous event's end, end now) and carries no
        tool calls.
        """
        end = self._elapsed(at)
        start = self._last_event_end_s
        self._last_event_end_s = end
        self._last_agent_end_s = end
        for call in tool_calls or []:
            call_id = (
                call.get("id")
                if isinstance(call, dict)
                else getattr(call, "id", None)
            )
            if call_id:
                self._agent_end_by_call_id[str(call_id)] = end
        return round(start, 3), round(end - start, 3)

    def tool_step(
        self, tool_call_id: Optional[str], *, at: Optional[float] = None
    ) -> tuple[float, float]:
        """Record a tool step's end; return ``(started_at_s, latency_s)``."""
        end = self._elapsed(at)
        start = self._agent_end_by_call_id.get(
            str(tool_call_id or ""), self._last_agent_end_s
        )
        self._last_event_end_s = end
        return round(start, 3), round(end - start, 3)


def _recoverable_after_agent_fallback(*, fallback_used: bool, has_answer: bool) -> bool:
    """Return True when agent_fallback produced an answer but the graph still raised."""
    return fallback_used and has_answer


# String literals on purpose: agents v0.5.5 does not export these names yet.
_FALLBACK_MARKER = "eve_fallback_used"
_FALLBACK_LATENCY_KEY = "eve_fallback_latency_s"


def _update_messages(update: Dict[str, Any]) -> List[Any]:
    msgs: List[Any] = []
    for out in update.values():
        got = out.get("messages", []) if isinstance(out, dict) else []
        msgs.extend(got if isinstance(got, list) else [got])
    return msgs


@dataclass
class _FallbackUse:
    """In-graph fallback seen during one agent run.

    ``used`` is true when an update names the ``agent_fallback`` node, or a
    message carries ``eve_fallback_used``. ``seconds`` sums
    ``eve_fallback_latency_s`` from those messages: the fallback call only,
    not the failed primary. ``merge_latencies`` stores that sum as
    ``fallback_latency_s``. It is not added to ``node_latencies``.

    An ``agent_fallback`` update with no marker (agents before the stamp)
    leaves ``seconds`` at 0, so only ``node_agent_fallback_s`` is stored.
    """

    used: bool = False
    seconds: float = 0.0

    def observe(self, update: Dict[str, Any]) -> None:
        if "agent_fallback" in update:
            self.used = True
        for msg in _update_messages(update):
            meta = getattr(msg, "response_metadata", None) or {}
            if meta.get(_FALLBACK_MARKER):
                self.used = True
                self.seconds += float(meta.get(_FALLBACK_LATENCY_KEY) or 0.0)

    def merge_latencies(self, latencies: Dict[str, Any]) -> Dict[str, Any]:
        if self.seconds > 0:
            latencies["fallback_latency_s"] = round(self.seconds, 3)
        return latencies


def _latency_node_name(node_name: str) -> Optional[str]:
    """Map a graph node onto a ``metadata.latencies`` key.

    Deep calls the model node ``model``; the back office reads ``node_agent_s``.
    Middleware hook names contain a dot and must not become a Mongo field.
    ``None`` drops the node so its time stays in ``generation_unattributed_s``.
    Trace entries keep the real node name.
    """
    if "." in node_name:
        return None
    if node_name == "model":
        return "agent"
    return node_name


# ─── MCP tool loader ──────────────────────────────────────────────────────────
# Implemented in src.services.mcp.tool_loader; re-exported for backward compat.


# ─── Tool factory ─────────────────────────────────────────────────────────────


def _tool_name(tool: Any) -> str:
    """Lowercased tool name, empty when the attribute is missing or raises."""
    try:
        return str(getattr(tool, "name", "") or "").lower()
    except Exception:
        return ""


def _tool_arg_names(tool: Any) -> set[str]:
    """extraction of accepted argument names from langchain tools.

    ``tool.args`` is a property that builds the JSON schema on the fly and can
    raise for a malformed MCP tool, so introspection failures degrade to "no
    known argument": one bad tool must never break tool loading for the run.
    """
    names: set[str] = set()

    try:
        args = getattr(tool, "args", None)
        if isinstance(args, dict):
            names.update(str(k) for k in args.keys())

        schema = getattr(tool, "args_schema", None) or getattr(
            tool, "input_schema", None
        )
        model_fields = getattr(schema, "model_fields", None) or getattr(
            schema, "__fields__", None
        )
        if isinstance(model_fields, dict):
            names.update(str(k) for k in model_fields.keys())

        if isinstance(schema, dict):
            properties = schema.get("properties")
            if isinstance(properties, dict):
                names.update(str(k) for k in properties.keys())
    except Exception:
        logger.warning(
            "Failed to introspect arguments of tool %r",
            _tool_name(tool) or type(tool).__name__,
            exc_info=True,
        )
        return set()

    return names


# UI-owned RAG keys are authoritative: whatever the model puts in the tool call
# for these is replaced by the value the user selected in the UI. Only the keys
# listed in _RAG_FILL_ONLY_KEYS are filled in when the model omits them.
_RAG_FILL_ONLY_KEYS = frozenset({"query"})

_MISSING = object()


def _rag_mcp_tool_defaults(
    tool: Any,
    request: GenerationRequest,
) -> Dict[str, Any]:
    """Map UI-selected RAG params onto the argument names a RAG MCP tool accepts.

    Only argument names the tool schema actually declares are returned: when the
    schema cannot be introspected nothing is injected, because guessing would
    push every alias (k, top_k, top_n, ...) into a single call.

    ``filters``, ``llm_type`` and ``year`` are mapped even when the request
    leaves them unset: "the user selected no filter" is itself a UI decision and
    must overwrite whatever the model invented. The retrieve tool declares them
    as optional (``filters: dict | None``, ``llm_type: str | None``,
    ``year: list[int] | None``), so sending ``None`` is valid.
    ``start_year``/``end_year`` stay absent: the backend ``POST /retrieve``
    recomputes the year range from ``filters``/``year``.

    ``query`` stays fill-only: the model may pass the user's query explicitly,
    and the retrieve endpoint owns any requery/rewrite before it retrieves.
    """
    accepted = _tool_arg_names(tool)

    def accepts(name: str) -> bool:
        return name in accepted

    defaults: Dict[str, Any] = {}
    if request.query and accepts("query"):
        defaults["query"] = request.query
    if accepts("llm_type"):
        defaults["llm_type"] = request.llm_type
    if request.embeddings_model and accepts("embeddings_model"):
        defaults["embeddings_model"] = request.embeddings_model
    for name in ("k", "top_k", "top_n", "limit", "n_results"):
        if accepts(name):
            defaults[name] = request.k
    if accepts("temperature"):
        defaults["temperature"] = request.temperature
    for name in ("score_threshold", "threshold", "min_score"):
        if accepts(name):
            defaults[name] = request.score_threshold
    if accepts("max_new_tokens"):
        defaults["max_new_tokens"] = request.max_new_tokens
    for name in ("collection_ids", "collections", "collection_names"):
        if request.collection_ids and accepts(name):
            defaults[name] = request.collection_ids
    if accepts("public_collections"):
        defaults["public_collections"] = list(request.public_collections or [])
    if accepts("private_collections"):
        defaults["private_collections"] = list(request.private_collections or [])
    if accepts("filters"):
        defaults["filters"] = request.filters
    if accepts("year"):
        defaults["year"] = request.year
    return defaults


def _is_rag_mcp_tool(tool: Any) -> bool:
    """Return True for MCP tools that should inherit request-scoped RAG params.

    Tools are exposed as ``f"{server_name}_{tool_name}"``, and the server name
    itself may contain underscores, so the unprefixed name is matched as a
    suffix. The ``public_collections`` argument is required for the suffix
    match so that unrelated search tools (esa_moocs ``search_moocs(query,
    top_k)``) are never captured.
    """
    name = _tool_name(tool)
    if not name:
        return False
    if name == "eve_retrieval_retrieve":
        return True
    if name == "retrieve" or name.endswith("_retrieve"):
        return "public_collections" in _tool_arg_names(tool)
    return False


def _rag_tool_names(tools: Any) -> set[str]:
    """Lowercased names of the tools whose output carries retrieval documents."""
    names: set[str] = set()
    for tool in tools or []:
        try:
            if not _is_rag_mcp_tool(tool):
                continue
        except Exception:
            continue
        name = _tool_name(tool)
        if name:
            names.add(name)
    return names


def _retrieval_tool_content(
    content: Any, user_id: Optional[str], conversation_id: Optional[str]
) -> Any:
    """The stored tool result when ``content`` is an offload stub.

    The retrieval interceptor already kept the documents, so this only
    matters for a RAG tool that did not go through it. A stub that is not
    read back stays a stub, and the parse below finds no documents.
    """
    if not user_id or not conversation_id or not isinstance(content, str):
        return content
    if not content.startswith("Tool result too large"):
        return content
    from src.services.agents.core.tool_results import expand_offloaded_content

    return expand_offloaded_content(content, user_id, conversation_id)


def _collect_retrieval_documents(
    all_messages: Any,
    rag_tool_names: set[str],
    *,
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> tuple[List[Dict[str, Any]], int, int]:
    """Documents produced by the retrieval tool during an agentic run.

    Returns ``(documents, retrieval_calls, retrieval_errors)``. Only ToolMessages
    coming from a RAG tool are read: a geocode or image tool must not mark the
    answer as source-backed. A retrieval call that returned nothing still counts
    as a call (the UI then says "no sources found", not "answered without
    sources"); a call that returned an error payload counts as an error instead.

    The documents themselves come from the per-request retrieval context when
    ``RetrievalContextInterceptor`` filled it: the ToolMessage then holds only
    the reduced copy the model read, without ids. The ToolMessage is parsed
    only when no interceptor ran (tests, a RAG tool loaded another way).
    """
    if not rag_tool_names or ToolMessage is None:
        return [], 0, 0

    # ToolMessage.name is set by the tool node, but a provider that drops it
    # leaves the tool_call id as the only link back to the tool that ran.
    names_by_call_id: Dict[str, str] = {}
    for msg in all_messages:
        if AIMessage is None or not isinstance(msg, AIMessage):
            continue
        for call in getattr(msg, "tool_calls", None) or []:
            if isinstance(call, dict):
                call_id, call_name = call.get("id"), call.get("name")
            else:
                call_id, call_name = (
                    getattr(call, "id", None),
                    getattr(call, "name", None),
                )
            if call_id and call_name:
                names_by_call_id[str(call_id)] = str(call_name).lower()

    documents: List[Dict[str, Any]] = []
    retrieval_calls = 0
    retrieval_errors = 0
    # The context is filled inside the tools node, so it is complete even when
    # the run was cancelled before the ToolMessage reached the stream. When it
    # already has these RAG calls, skip the S3 GET: the documents below would
    # be discarded. The collector is synchronous, so a GET that is still
    # required stays sync and runs only on this fallback (no context calls).
    ctx = get_retrieval_context()
    context_calls = ctx.for_tools(rag_tool_names) if ctx is not None else []
    for msg in all_messages:
        if not isinstance(msg, ToolMessage):
            continue
        name = str(getattr(msg, "name", "") or "").lower()
        if not name:
            name = names_by_call_id.get(
                str(getattr(msg, "tool_call_id", "") or ""), ""
            )
        if name not in rag_tool_names:
            continue
        raw = getattr(msg, "content", None)
        content = (
            raw
            if context_calls
            else _retrieval_tool_content(raw, user_id, conversation_id)
        )
        if is_retrieval_error_payload(content):
            retrieval_errors += 1
            continue
        retrieval_calls += 1
        documents.extend(extract_documents_from_retrieval_payload(content))

    if context_calls:
        documents = [doc for call in context_calls for doc in call.documents]
        retrieval_calls = max(retrieval_calls, len(context_calls))
    return documents, retrieval_calls, retrieval_errors


def _apply_rag_defaults(
    merged: Dict[str, Any],
    defaults: Dict[str, Any],
    *,
    tool_name: str,
) -> None:
    """Overwrite UI-owned keys in ``merged`` in place, filling only ``query``.

    A ``None`` in ``defaults`` is a value like any other for UI-owned keys: the
    user cleared that filter, so the model's own guess has to go. Only the
    fill-only keys treat ``None`` as "nothing to contribute".
    """
    forced: Dict[str, Any] = {}
    replaced: List[str] = []
    changed = False

    for key, value in defaults.items():
        if key in _RAG_FILL_ONLY_KEYS:
            if value is None:
                continue
            current = merged.get(key, _MISSING)
            if current is _MISSING or current in (None, ""):
                merged[key] = value
                changed = True
            continue
        previous = merged.get(key, _MISSING)
        merged[key] = value
        forced[key] = value
        if previous is _MISSING:
            changed = True
        elif previous != value:
            changed = True
            replaced.append(f"{key}: {previous!r} -> {value!r}")

    # Checked before the "nothing changed" shortcut below: an empty selection is
    # worth flagging on every call, including the one where the model already
    # sent exactly the (empty) selection the request carries.
    if "public_collections" in defaults and not merged.get("public_collections"):
        logger.warning(
            "RAG tool %s called with no public collection selected", tool_name
        )

    if not changed:
        # The second patched layer (coroutine after ainvoke) re-merges the very
        # same values; logging it again would only duplicate the first line.
        logger.debug("RAG tool %s already carries the UI params", tool_name)
        return

    logger.info("RAG tool %s forced UI params: %s", tool_name, forced)
    if replaced:
        logger.info(
            "RAG tool %s replaced model-supplied values: %s",
            tool_name,
            "; ".join(replaced),
        )


def _merge_tool_defaults(
    tool_input: Any,
    defaults: Dict[str, Any],
    *,
    tool_name: str = "tool",
) -> Any:
    if not isinstance(tool_input, dict) or not defaults:
        return tool_input
    merged = dict(tool_input)
    _apply_rag_defaults(merged, defaults, tool_name=tool_name)
    return merged


def _merge_kwarg_defaults(
    kwargs: Dict[str, Any],
    defaults: Dict[str, Any],
    *,
    tool_name: str = "tool",
) -> Dict[str, Any]:
    merged = dict(kwargs)
    _apply_rag_defaults(merged, defaults, tool_name=tool_name)
    return merged


def _with_request_rag_defaults(
    tool: Any,
    request: GenerationRequest,
) -> Any:
    """Return a per-request tool copy that injects UI RAG params into calls."""
    try:
        if not _is_rag_mcp_tool(tool):
            return tool
        defaults = _rag_mcp_tool_defaults(tool, request)
    except Exception:
        logger.warning(
            "Failed to inspect tool %r for RAG params; using it untouched",
            _tool_name(tool) or type(tool).__name__,
            exc_info=True,
        )
        return tool

    if not defaults:
        return tool

    tool_name = _tool_name(tool) or "tool"

    try:
        bound_tool = copy.copy(tool)
    except Exception:
        logger.warning(
            "Failed to copy MCP RAG tool %r; using original tool object",
            tool_name,
            exc_info=True,
        )
        bound_tool = tool

    original_ainvoke = getattr(bound_tool, "ainvoke", None)
    if callable(original_ainvoke):
        async def ainvoke_with_defaults(
            tool_input: Any, *args: Any, **kwargs: Any
        ) -> Any:
            return await original_ainvoke(
                _merge_tool_defaults(tool_input, defaults, tool_name=tool_name),
                *args,
                **kwargs,
            )

        object.__setattr__(bound_tool, "ainvoke", ainvoke_with_defaults)

    original_invoke = getattr(bound_tool, "invoke", None)
    if callable(original_invoke):
        def invoke_with_defaults(tool_input: Any, *args: Any, **kwargs: Any) -> Any:
            return original_invoke(
                _merge_tool_defaults(tool_input, defaults, tool_name=tool_name),
                *args,
                **kwargs,
            )

        object.__setattr__(bound_tool, "invoke", invoke_with_defaults)

    # functools.wraps keeps the wrapped signature: the graph's tool node reads
    # it to find the ``runtime`` parameter, and without it the MCP request
    # carries no runtime, so the retrieval Sources lose their tool call id.
    original_coroutine = getattr(bound_tool, "coroutine", None)
    if callable(original_coroutine):
        @functools.wraps(original_coroutine)
        async def coroutine_with_defaults(*args: Any, **kwargs: Any) -> Any:
            return await original_coroutine(
                *args,
                **_merge_kwarg_defaults(kwargs, defaults, tool_name=tool_name),
            )

        object.__setattr__(bound_tool, "coroutine", coroutine_with_defaults)

    original_func = getattr(bound_tool, "func", None)
    if callable(original_func):
        @functools.wraps(original_func)
        def func_with_defaults(*args: Any, **kwargs: Any) -> Any:
            return original_func(
                *args,
                **_merge_kwarg_defaults(kwargs, defaults, tool_name=tool_name),
            )

        object.__setattr__(bound_tool, "func", func_with_defaults)

    return bound_tool


async def _build_tools(
    request: GenerationRequest,
    cancel_event: Optional[asyncio.Event] = None,
    *,
    conversation_id: Optional[str] = None,
) -> List[Any]:
    """Return LangChain StructuredTools bound to the current request context.

    ``conversation_id`` adds ``search_history`` and ``read_history``, bound to
    that conversation. The ``simple`` graph does not call tools, so its caller
    leaves this unset.
    """
    if not _langgraph_available:
        return []

    tools: List[Any] = []

    if getattr(request, "mcp_server_configs", None):
        try:
            mcp_tools = await _load_mcp_tools_for_servers(
                request.mcp_server_configs,
                mcp_proxy_bearer_token=request.mcp_proxy_bearer_token,
                mcp_user_id=request.mcp_user_id,
            )
            mcp_client = getattr(mcp_tools, "_mcp_client", None)
            mcp_tools = [
                _with_request_rag_defaults(tool, request)
                for tool in mcp_tools
            ]
            tools.extend(mcp_tools)
            # `tools.extend(...)` above only copies the individual tool
            # references, not `mcp_tools`'s own `_mcp_client` attribute — carry
            # it over explicitly so the client stays alive for as long as the
            # caller holds onto this function's return value (see
            # _MCPToolsWithClient docstring in _load_mcp_tools_for_servers).
            if mcp_client is not None:
                tools = _MCPToolsWithClient(tools, mcp_client)
        except Exception:
            logger.error("MCP tool loading failed; proceeding without MCP tools", exc_info=True)

    if conversation_id:
        tools.extend(build_history_tools(conversation_id))

    return tools


# ─── Shared checkpointer ──────────────────────────────────────────────────────

_agentic_checkpointer: Optional[Any] = None
_agentic_checkpointer_lock = asyncio.Lock()


async def _get_agentic_checkpointer() -> Optional[Any]:
    global _agentic_checkpointer
    if _agentic_checkpointer is not None:
        return _agentic_checkpointer
    async with _agentic_checkpointer_lock:
        if _agentic_checkpointer is not None:
            return _agentic_checkpointer
        try:
            from pymongo import MongoClient

            # Off the event loop: a TTL index build can take a while.
            _agentic_checkpointer = await asyncio.to_thread(
                build_mongo_checkpointer, MongoClient(get_mongodb_uri())
            )
            logger.info("Agentic agent using MongoDB checkpointer")
            return _agentic_checkpointer
        except Exception as exc:
            logger.warning(
                "MongoDB checkpointer unavailable, using in-memory: %s", exc
            )
            try:
                _agentic_checkpointer = InMemorySaver()
                return _agentic_checkpointer
            except Exception:
                return None


# ─── Resolve LLM ──────────────────────────────────────────────────────────────


def _resolve_agentic_llm_type(
    llm_type: Optional[str], *, override: Optional[str] = None
) -> Optional[str]:
    """Resolve the effective LLM type for agentic generation.

    Precedence: explicit ``override`` (e.g. forcing the Fallback model) wins,
    then the request's ``llm_type``, then the ``AGENTIC_LLM_TYPE`` env default.
    Centralising this here keeps resolution consistent for every caller of
    :func:`_build_react_graph`.
    """
    if override is not None:
        return override
    if llm_type and llm_type != AGENTIC_LLM_TYPE:
        logger.info(
            "Agentic graph: overriding AGENTIC_LLM_TYPE %r -> %r llm_type",
            AGENTIC_LLM_TYPE,
            llm_type,
        )
    return llm_type or AGENTIC_LLM_TYPE


async def _resolve_agentic_llm_client(
    request: GenerationRequest,
    *,
    user_id: str,
) -> tuple[Any, Dict[str, Any]]:
    """Resolve the LLM client and metadata for agentic generation."""
    if request.custom_model_id:
        model = request.resolved_custom_model
        if model is None:
            model = await get_owned_custom_model(
                request.custom_model_id, user_id, action="use"
            )
        else:
            ensure_custom_model_has_credentials(model)
        llm = await build_custom_model_llm(model)
        return llm, custom_model_metadata(model)

    manager = get_shared_llm_manager()
    effective_type = _resolve_agentic_llm_type(request.llm_type)
    chain = manager.resolve_chain(effective_type)
    circuit_open = manager.open_circuits(effective_type, chain)
    # resolve_chain has already sunk open circuits to the back, kept the
    # configured order when every one of them is open, and dropped an endpoint
    # a 429 keeps out, so the head is the pick.
    candidate = chain[0]
    # client_for_candidate raises instead of quietly substituting Mistral, so
    # agentic_llm_resolved reports the endpoint that will actually be called.
    llm = manager.client_for_candidate(candidate)
    return llm, {
        "agentic_llm_resolved": candidate,
        "endpoint": build_endpoint_metadata(
            requested=request.llm_type,
            chain=chain,
            answered=candidate,
            circuit_open=circuit_open,
        ),
    }


class _AgenticBudgetTimeout(TimeoutError):
    """No final-answer token within AGENTIC_TIMEOUT (default 120s).

    The clock starts at generation and restarts on each announced tool call
    and each tool result. At 480s AGENTIC_TURN_DEADLINE cuts the turn. Must
    not open the LLM circuit: non-answer chunks mean the endpoint is up.
    """


def _record_endpoint_failure(
    endpoint: Optional[Dict[str, Any]], exc: BaseException
) -> None:
    """Open the answering endpoint's circuit when *exc* means it is down.

    Does not write ``error_logs``: the leftover timeout/error_handler row already
    records the failure. Circuit state lives on ``EndpointHealth`` and on
    ``metadata.endpoint.circuit_open``.
    """
    if isinstance(exc, _AgenticBudgetTimeout):
        return
    answered = (endpoint or {}).get("answered")
    if answered and is_endpoint_failure(exc):
        get_shared_llm_manager().health.record_failure(answered, exc)


def _is_node_timeout_error(exc: BaseException) -> bool:
    """LangGraph ``NodeTimeoutError`` is not a stdlib ``TimeoutError``."""
    if type(exc).__name__ == "NodeTimeoutError":
        return True
    try:
        from langgraph.errors import NodeTimeoutError

        return isinstance(exc, NodeTimeoutError)
    except Exception:
        return False


def _node_from_exception(exc: BaseException) -> Optional[str]:
    node = getattr(exc, "node", None)
    return node if isinstance(node, str) and node else None


def _policy_for_uncaught(exc: BaseException) -> str:
    """Map a bubbled graph/run exception to the agentic policy name."""
    if isinstance(exc, _AgenticBudgetTimeout):
        return "run_timeout"
    if _is_node_timeout_error(exc):
        return "timeout"
    if isinstance(exc, TimeoutError):
        return "run_timeout"
    return "error_handler"


def _record_in_graph_fallback(
    endpoint: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Re-attribute the answer to the fallback model.

    Deliberately does NOT open the primary's circuit: the graph's error_handler
    swallows the node's exception, so the failure cannot be classified here,
    and a prompt-level 400 (the strict-template class) reproducing on every
    send would otherwise park a healthy endpoint indefinitely. Transport-level
    failures still open the circuit through the classified paths.
    """
    return endpoint_answered_by(endpoint, LLMType.Fallback.value)


def _resolve_agent_graph_type(request: GenerationRequest) -> str:
    """Return request-selected graph type, else AGENT_GRAPH_TYPE from env."""
    from src.config import AGENT_GRAPH_TYPE

    raw = getattr(request, "agent", None)
    if raw is None:
        return AGENT_GRAPH_TYPE

    s = str(raw).strip().lstrip("\ufeff").strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    return s or AGENT_GRAPH_TYPE


# ─── Artifact markdown instruction ────────────────────────────────────────────

# The ReAct graph's own lead-in instruction lives in `prompts.yaml` bundled
# with the `eve-esa-agents` package resolved by graphs_bundle.py (see its
# `_prompts_yaml_path`) — that package is an external git dependency, not part
# of this repo, so it can't be edited here. This backend-owned instruction is
# injected as an extra SystemMessage ahead of the user's query instead: it
# ends up positioned *after* the graph's own system prompt in every `agent`
# node invocation (ReactAgent._invoke prepends its instruction fresh each
# call), so it stays visible for the whole tool-calling loop.
_ARTIFACT_MARKDOWN_INSTRUCTION = (
    "The sandbox root has four sibling directories. "
    "workspace/ is your working directory. The shell starts there. "
    "Run a file you wrote as `x`, `workspace/x` or `/workspace/x` as `x`, "
    "for example `python3 x`: in the shell, never run a file through "
    "`workspace/x` or `/workspace/x`. Those paths do not exist there. "
    "A file you create or download belongs in workspace/ and stays for this "
    "conversation. "
    "/skills/ holds installed skills. Read them. Do not write there. "
    "/artifacts/ holds a file a tool returned. Its path is "
    "/artifacts/{id}/{filename}. "
    "/large_tool_results/ holds a tool result that was too large for the "
    "thread. Read it. Do not write there. "
    "To show a file from any of these directories, write normal markdown. "
    "An image is `![filename](path)`. Any other file is `[filename](path)`. "
    "In markdown a file in the working directory is `ndvi.png` or "
    "`/workspace/ndvi.png`. "
    "Never put a scheme or a host in front of a path. "
    "There is no display tool. Do not call display."
)


def _build_initial_messages(
    request: GenerationRequest,
    tools: List[Any],
    history: Optional[List[Any]] = None,
    prior: Optional[List[Any]] = None,
    message_id: Optional[str] = None,
    new_checkpoint: Optional[bool] = None,
    agent_graph_type: Optional[str] = None,
    attached_files: Optional[List[str]] = None,
) -> List[Any]:
    """Return the graph's initial ``messages`` state for this turn.

    Prepends :data:`_ARTIFACT_MARKDOWN_INSTRUCTION` as a ``SystemMessage``
    when MCP tools are in play, since only then can a tool result carry an
    artifact stub worth guarding against being rewritten.  A ``deep`` turn
    skips it: the deep agent carries the rule in its system prompt.

    First turn only: the checkpointed thread state keeps the turn-one copy
    visible on later turns, and injecting another one mid-conversation puts a
    ``system`` message after an ``assistant`` one, which strict chat templates
    refuse (Blablador: 400 "Unexpected role 'system' after role 'assistant'").
    ``new_checkpoint`` overrides that guess. A deep turn that already has a
    thread passes False even though ``history`` is empty.

    ``prior`` is the stored turns seeded into a deep thread that has no
    checkpoint. ``message_id`` is written on the new human message so a later
    summary can name the range it covers. ``attached_files`` (sandbox paths)
    follow the query in that message, so the thread keeps them with the turn
    they came with.
    """
    if new_checkpoint is None:
        new_checkpoint = not history
    messages: List[Any] = []
    if tools and SystemMessage and new_checkpoint and agent_graph_type != "deep":
        messages.append(SystemMessage(content=_ARTIFACT_MARKDOWN_INSTRUCTION))
    if prior:
        messages.extend(prior)
    human_kwargs: Dict[str, Any] = {}
    if message_id:
        human_kwargs["additional_kwargs"] = {"eve_message_id": str(message_id)}
    content = request.query
    block = attached_files_block(attached_files or [])
    if block:
        content = f"{content}\n\n{block}" if content else block
    messages.append(HumanMessage(content=content, **human_kwargs))
    return messages


# A markdown link that names an artifact. The chat loads ``/artifacts/{id}``.
# The sandbox path is ``/artifacts/{id}/file``. An older thread may still say
# ``/tool_artifacts/{id}/file``. A model may keep the id and put a host in
# front, such as ``storage.googleapis.com/.../artifacts/{id}/file``.
_ARTIFACT_MARKDOWN = re.compile(
    r"(!?\[[^\]]*])\([^)\s]*?/(?:tool_)?artifacts/([0-9a-fA-F]{24})(?:/[^)\s]*)?\)"
)


def _markdown_paths_to_artifact_urls(
    answer: str, edits: Optional[List[Edit]] = None
) -> str:
    """Rewrite a markdown artifact link to the URL the chat can load.

    A link that is already ``/artifacts/{id}`` is left as it is. Any other
    form is repaired, and an INFO line names the URL that was wrong. Each
    repair is appended to ``edits``.
    """

    def replace(match: re.Match) -> str:
        artifact_id = match.group(2)
        repaired = f"{match.group(1)}(/artifacts/{artifact_id})"
        original = match.group(0)
        if original != repaired:
            logger.info(
                "artifact.markdown_repaired url=%s to=/artifacts/%s",
                original,
                artifact_id,
            )
            if edits is not None:
                edits.append((match.start(), match.end(), len(repaired)))
        return repaired

    return _ARTIFACT_MARKDOWN.sub(replace, answer)


# A markdown link with any target.
_MARKDOWN_LINK = re.compile(r"(!?\[[^\]]*])\(([^)\s]+)\)")
# A 24 hex run that is not part of a longer one: a candidate artifact id.
_HEX_ID = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{24}(?![0-9a-fA-F])")


def _turn_artifact_filename(target: str) -> Optional[str]:
    """The filename a link names when it can only mean a tool file.

    A bare ``name.png``, ``./name.png`` or ``/artifacts/name.png`` (the id
    left out). A path into another directory, ``/workspace/name.png``
    included, is not a tool file. A target with a scheme is not either.
    """
    if ":" in target:
        return None
    raw = urllib.parse.unquote(target.split("?", 1)[0].split("#", 1)[0]).strip()
    for prefix in ("./", "/artifacts/", "artifacts/"):
        if raw.startswith(prefix):
            raw = raw[len(prefix) :]
            break
    if not raw or "/" in raw or raw in {".", ".."}:
        return None
    return raw


def _links_to_turn_artifact_urls(
    answer: str, edits: Optional[List[Edit]] = None
) -> str:
    """Point a link to a file a tool returned in this turn at its artifact.

    The stub tells the model ``/artifacts/{id}/{filename}``, yet a model may
    write only the filename, or wrap the id in a host it made up, such as
    ``https://storage.googleapis.com/<bucket>/{id}/file.png``. A target that
    holds the id of an artifact the MCP interceptor stored in this turn
    becomes ``/artifacts/{id}``, whatever the scheme and host; a bare name
    is matched against the filenames of those artifacts. Any other link is
    left to the workspace pass. Each rewrite is appended to ``edits``.
    """
    from src.services.mcp.artifact_context import get_artifact_context

    ctx = get_artifact_context()
    if ctx is None:
        return answer
    files = getattr(ctx, "collected_artifact_files", None) or {}
    ids = {i.lower() for i in getattr(ctx, "collected_artifact_ids", None) or []}
    if not files and not ids:
        return answer

    def replace(match: re.Match) -> str:
        target = match.group(2)
        reason = "turn_id"
        found = (m.group(0).lower() for m in _HEX_ID.finditer(target))
        artifact_id = next((i for i in found if i in ids), None)
        if artifact_id is None:
            reason = "turn_filename"
            name = _turn_artifact_filename(target)
            artifact_id = files.get(name) if name else None
        if artifact_id is None:
            return match.group(0)
        repaired = f"{match.group(1)}(/artifacts/{artifact_id})"
        if repaired == match.group(0):
            return repaired
        # The id only: the original link is model text.
        logger.info(
            "artifact.markdown_repaired reason=%s to=/artifacts/%s",
            reason,
            artifact_id,
        )
        if edits is not None:
            edits.append((match.start(), match.end(), len(repaired)))
        return repaired

    return _MARKDOWN_LINK.sub(replace, answer)


# Code the model quotes: a fenced block (to its closing fence or the end of
# the answer), then an inline span closed by a run of the same length.
_FENCED_CODE = re.compile(
    r"^ {0,3}(`{3,}|~{3,})[^\n]*(?:\n.*?(?:^ {0,3}\1[`~]*[ \t]*$|\Z)|\Z)",
    re.MULTILINE | re.DOTALL,
)
_INLINE_CODE = re.compile(r"(?<!`)(`+)(?!`).+?(?<!`)\1(?!`)", re.DOTALL)


def _blank(text: str, regions: List[Tuple[int, int]]) -> str:
    """``text`` with each region turned into spaces, same length."""
    out = list(text)
    for start, end in regions:
        out[start:end] = " " * (end - start)
    return "".join(out)


def _code_regions(text: str) -> List[Tuple[int, int]]:
    fenced = [m.span() for m in _FENCED_CODE.finditer(text)]
    inline = [m.span() for m in _INLINE_CODE.finditer(_blank(text, fenced))]
    return sorted(fenced + inline)


async def rewrite_answer_file_links(
    answer: str, passes: Optional[List[List[Edit]]] = None
) -> str:
    """Point embedded file markdown at the URL the chat can load.

    The instruction asks the model to embed ``/artifacts/{id}/{filename}``
    as normal markdown. The chat loads ``/artifacts/{id}``, so that path is
    rewritten before the answer is stored. A host wrapped around the same id,
    such as ``storage.googleapis.com/.../artifacts/{id}/file``, is rewritten
    in place, and so is any host around the id of an artifact this turn
    stored. A workspace path is published the same way. An artifact the
    answer never mentions stays out of the answer.

    A link inside inline code or a fenced block is an example the model
    quotes and stays as written: the passes run on a copy where code is
    blanked to spaces of the same length, then the code goes back in place.

    ``passes`` receives the edits of each rewrite pass, in order, for
    ``CitationRewriter.remap``.
    """
    if not answer:
        return answer
    from src.services.agents.core.workspace import publish_markdown_files

    regions = _code_regions(answer)
    text = _blank(answer, regions)
    artifact_edits: List[Edit] = []
    filename_edits: List[Edit] = []
    workspace_edits: List[Edit] = []
    text = _markdown_paths_to_artifact_urls(text, artifact_edits)
    text = _links_to_turn_artifact_urls(text, filename_edits)
    text = await publish_markdown_files(text, workspace_edits)
    rewrite_passes = [artifact_edits, filename_edits, workspace_edits]
    for start, end in regions:
        at = start
        for edits in rewrite_passes:
            at = shift_offset(at, edits)
        text = text[:at] + answer[start:end] + text[at + end - start :]
    if passes is not None:
        passes.extend(rewrite_passes)
    return text


# ─── Graph compile helpers ────────────────────────────────────────────────────


def _build_react_graph(
    llm_type: Optional[str],
    tools: List[Any],
    checkpointer: Any,
    *,
    agent: Any,
    history: Optional[List[Any]] = None,
    summary: Optional[str] = None,
    llm_type_override: Optional[str] = None,
    fallback_llm: Any = None,
    llm: Any = None,
    llm_run_timeout: Optional[int] = None,
    llm_idle_timeout: Optional[int] = None,
    on_policy: Any = None,
    sandbox_backend: Any = None,
    skills_root: Any = None,
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> Any:
    """Compile the agent graph using the resolved LLM type.

    ``llm_type_override`` forces a specific model (e.g. ``LLMType.Fallback.value``)
    regardless of the request's ``llm_type``.  ``fallback_llm`` is forwarded to
    ``agent.compile()`` for in-graph node-level fallback.
    ``llm`` is the primary graph model (platform or user custom).  When
    ``fallback_llm`` is not passed, the platform fallback model is wired
    automatically.

    The two node budgets measure different things and must not be confused.
    ``llm_idle_timeout`` is the no-progress cap: its clock resets on every
    streamed chunk, so it is where an endpoint's first-token budget belongs.
    ``llm_run_timeout`` is a hard wall-clock cap on one node attempt that never
    resets, so it has to cover a whole long answer — a first-token budget used
    here caps the *length* of every answer instead of catching a stalled one.
    """
    if llm is None:
        effective_type = _resolve_agentic_llm_type(llm_type, override=llm_type_override)
        llm = get_shared_llm_manager().get_client_for_model(effective_type)
    if fallback_llm is None:
        fallback_llm = get_shared_llm_manager().get_client_for_model(
            LLMType.Fallback.value
        )
    graph_name = getattr(agent, "name", None)
    timed_node = getattr(type(agent), "timed_node", None)
    if inspect.isfunction(timed_node) or inspect.ismethod(timed_node):
        try:
            if "on_policy" not in inspect.signature(timed_node).parameters:
                logger.warning(
                    "Agent graph %s timed_node has no on_policy; "
                    "eve-esa-agents pin is older than feat/agentic-error-logging. "
                    "In-graph retry/timeout/fallback logs will not be written.",
                    graph_name,
                )
        except (TypeError, ValueError):
            pass
    policy_cb = on_policy if on_policy is not None else _graph_on_policy(graph_name)
    from src.services.agents.core.tool_args import ensure_tool_args_are_json

    llm = ensure_tool_args_are_json(llm)
    fallback_llm = ensure_tool_args_are_json(fallback_llm)
    compile_kwargs: Dict[str, Any] = {
        "llm": llm,
        "tools": tools,
        "checkpointer": checkpointer,
        "history": history,
        "summary": summary,
        "fallback_llm": fallback_llm,
        "llm_run_timeout": (
            llm_run_timeout if llm_run_timeout is not None else AGENTIC_TIMEOUT
        ),
        "llm_idle_timeout": (
            llm_idle_timeout if llm_idle_timeout is not None else MODEL_TIMEOUT
        ),
        "on_policy": policy_cb,
    }
    # React's compile has no backend parameter. A MagicMock compile exposes
    # (*args, **kwargs) only, so it must not gain a backend kwarg either.
    if _compile_declares_backend(agent):
        offload = _offload_backend(user_id, conversation_id, sandbox_backend)
        if sandbox_backend is not None:
            compile_kwargs["backend"] = backend_for_compile(
                sandbox_backend, tool_results=offload
            )
            # Route-scoped deny. Offload calls awrite and does not consult it.
            if offload is not None:
                compile_kwargs["permissions"] = large_tool_result_permissions()
        else:
            routes: Dict[str, Any] = {}
            permissions: List[Any] = []
            if skills_root is not None:
                skills_route, skill_permissions = _skills_route(skills_root)
                if skills_route is not None:
                    routes[_SKILLS_SOURCE] = skills_route
                if skill_permissions:
                    permissions.extend(skill_permissions)
            # No conversation: leave /large_tool_results/ on the default
            # StateBackend. A second StateBackend shares the files channel
            # and the composite strips the prefix, so the keys collide.
            if offload is not None:
                routes[_LARGE_TOOL_RESULTS] = offload
                permissions.extend(large_tool_result_permissions())
            from deepagents.backends import CompositeBackend, StateBackend

            compile_kwargs["backend"] = CompositeBackend(
                default=StateBackend(), routes=routes
            )
            if permissions:
                compile_kwargs["permissions"] = permissions
        # create_deep_agent installs SkillsMiddleware only when `skills` is a
        # list of source paths. `/skills/` is the tree uploaded into the
        # sandbox or mounted when there is no sandbox.
        if (
            skills_root is not None
            and "backend" in compile_kwargs
            and _explicit_compile_param(agent, "skills")
        ):
            compile_kwargs["skills"] = [_SKILLS_SOURCE]
    if _explicit_compile_param(agent, "tool_result_max_tokens"):
        compile_kwargs["tool_result_max_tokens"] = TOOL_RESULT_MAX_TOKENS
    summary_kwargs = {
        "context_window": AGENT_CONTEXT_WINDOW_TOKENS,
        "summary_trigger_fraction": SUMMARY_TRIGGER_FRACTION,
        "summary_keep_fraction": SUMMARY_KEEP_FRACTION,
        "summary_max_tokens": SUMMARY_MAX_TOKENS,
        "truncate_args_max_chars": TRUNCATE_ARGS_MAX_CHARS,
        "recall_note": SEARCH_NOTE,
    }
    for name, value in summary_kwargs.items():
        if _explicit_compile_param(agent, name):
            compile_kwargs[name] = value
    return agent.compile(**compile_kwargs)


def _filesystem_skills_backend(skills_root: Any):
    """Read-only ``/skills/`` mount for a turn that has no sandbox.

    Uses the deep graph helpers when they accept the folder. Otherwise builds
    the same mount here. The sandbox path does not use this: it uploads the
    tree and the shell reads it there.
    """
    try:
        from agents.graphs.deep.graph import default_backend, default_permissions
    except ImportError:
        return _local_skills_mount(skills_root)
    try:
        backend = default_backend(skills_root)
    except TypeError:
        return _local_skills_mount(skills_root)
    return backend, default_permissions()


def _local_skills_mount(skills_root: Any):
    """``/skills/`` routed at *skills_root*, read-only, with state as default."""
    from deepagents.backends import CompositeBackend, StateBackend
    from deepagents.backends.filesystem import FilesystemBackend
    from deepagents.middleware.filesystem import FilesystemPermission

    backend = CompositeBackend(
        default=StateBackend(),
        routes={
            "/skills/": FilesystemBackend(root_dir=skills_root, virtual_mode=True),
        },
    )
    permissions = [
        FilesystemPermission(operations=["read"], paths=["/skills/**"], mode="allow"),
        FilesystemPermission(operations=["write"], paths=["/skills/**"], mode="deny"),
    ]
    return backend, permissions


# Skill folders live under this prefix in the sandbox and on the no-sandbox mount.
_SKILLS_SOURCE = "/skills/"
_LARGE_TOOL_RESULTS = "/large_tool_results/"


def _offload_backend(
    user_id: Optional[str], conversation_id: Optional[str], sandbox_backend: Any
) -> Any:
    """The S3 route for ``/large_tool_results/``, or ``None``.

    A conversation stores the files in S3 and, when a sandbox is attached,
    mirrors them there. ``None`` means the caller must not add a route: the
    path stays on the default backend, so the state key remains
    ``/large_tool_results/{id}``.
    """
    if not user_id or not conversation_id:
        return None
    from src.services.agents.core.tool_results import S3ToolResultBackend

    return S3ToolResultBackend(
        user_id=user_id,
        conversation_id=conversation_id,
        sandbox_backend=sandbox_backend,
    )


def _skills_route(skills_root: Any):
    """The ``/skills/`` backend and its permissions, or ``(None, None)``."""
    filesystem = _filesystem_skills_backend(skills_root)
    if filesystem is None:
        return None, None
    backend, permissions = filesystem
    from deepagents.backends import CompositeBackend

    if isinstance(backend, CompositeBackend):
        route = backend.routes.get(_SKILLS_SOURCE)
        if route is not None:
            return route, permissions
    local_backend, local_permissions = _local_skills_mount(skills_root)
    return local_backend.routes.get(_SKILLS_SOURCE), local_permissions


def _on_fresh_tool_results(
    user_id: Optional[str], conversation_id: Optional[str]
):
    """Restore callback for ``acquire_sandbox``. The pool calls it under the lock.

    A reused sandbox does not get this callback. ``restore_tool_results`` logs
    and swallows its own failure.
    """

    async def _restore(lease: Any) -> None:
        if lease is None or not user_id or not conversation_id:
            return
        from src.services.agents.core.tool_results import restore_tool_results
        from src.services.mcp.conversation_artifacts import restore_artifacts

        from src.services.agents.core.workspace import restore_workspace

        await restore_tool_results(lease.backend, user_id, conversation_id)
        await restore_artifacts(lease.backend, user_id, conversation_id)
        await restore_workspace(lease.backend, user_id, conversation_id)

    return _restore


def _explicit_compile_param(agent: Any, name: str) -> bool:
    """True when ``compile`` has an explicit ``name`` parameter.

    A catch-all ``**kwargs`` does not count. Passing a keyword there would
    change the kwargs of a graph that does not know it.
    """
    compile_fn = getattr(agent, "compile", None)
    if compile_fn is None:
        return False
    try:
        parameters = inspect.signature(compile_fn).parameters
    except (TypeError, ValueError):
        return False
    param = parameters.get(name)
    if param is None:
        return False
    return param.kind not in (
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.VAR_KEYWORD,
    )


def _compile_declares_backend(agent: Any) -> bool:
    """True when ``compile`` has an explicit ``backend`` parameter."""
    return _explicit_compile_param(agent, "backend")


def _graph_on_policy(graph_name: Optional[str]):
    """Mongo writer for compile-time policy events (retry / timeout / fallback)."""

    async def on_policy(
        *,
        node: Optional[str] = None,
        policy: str,
        attempt: Optional[int] = None,
        error: Optional[BaseException] = None,
        extra: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        **_kwargs: Any,
    ) -> None:
        desc = description or (
            f"Policy '{policy}' on node '{node}'" if node else f"Policy '{policy}'"
        )
        if attempt is not None:
            desc = f"{desc} (attempt {attempt})"
        await persist_policy_event(
            policy=policy,
            description=desc,
            node=node,
            graph=graph_name,
            error=error,
            attempt=attempt,
            extra=extra,
        )

    return on_policy


# ─── Conversation context ─────────────────────────────────────────────────────

# Thread-state key of the deepagents summarization middleware, and the
# ``additional_kwargs`` the EVE subclass puts on its summary message.
SUMMARIZATION_EVENT_KEY = "_summarization_event"
SUMMARY_TEXT_KEY = "eve_summary"
SUMMARY_FIRST_ID_KEY = "eve_summary_first_message_id"
SUMMARY_LAST_ID_KEY = "eve_summary_last_message_id"
# Metadata LangChain middleware puts on its own model calls. Only the newer
# streaming API filters those calls; stream_mode="messages" still yields them
# from the model node, like answer tokens.
_INTERNAL_CALL_KEY = "lc_internal_call"


def _is_internal_model_call(metadata: Any) -> bool:
    """True for a chunk from a middleware model call, such as the summary."""
    if not isinstance(metadata, dict):
        return False
    return _INTERNAL_CALL_KEY in metadata or metadata.get("lc_source") == "summarization"


# ``event`` of the custom stream payload the deep summarization middleware
# writes around its summary call (agents ``SUMMARY_STREAM_EVENT``).
SUMMARY_STREAM_EVENT = "eve_summarization"
_SUMMARY_PHASES = ("started", "finished")


def _summarizing_event(payload: Any) -> Optional[str]:
    """SSE ``summarizing`` event for a custom stream payload, else None."""
    if not isinstance(payload, dict) or payload.get("event") != SUMMARY_STREAM_EVENT:
        return None
    phase = payload.get("phase")
    if phase not in _SUMMARY_PHASES:
        return None
    event = {
        "type": "summarizing",
        "phase": phase,
        "first_message_id": payload.get("first_message_id"),
        "last_message_id": payload.get("last_message_id"),
    }
    return f"data: {json.dumps(event)}\n\n"


async def _fetch_conversation_context(
    conversation_id: Optional[str],
) -> tuple[List[Any], Optional[str]]:
    """Return ``(history, summary)`` for the given conversation.

    *history* is a list of LangChain messages (HumanMessage / AIMessage) from the
    most-recent turn stored in the DB; *summary* is the rolling summary string if
    one exists.  Both are ``[]`` / ``None`` when there is no conversation yet.
    The agent is responsible for deciding how to integrate them via
    :meth:`~AgentGraph.format_history` and :meth:`~AgentGraph.instruction_text`.
    """
    if not conversation_id:
        return [], None
    history, summary = await _get_conversation_history_from_db(conversation_id)
    return history, summary


def _agent_with_pinned_context(agent: Any) -> Any:
    """Copy *agent* with the standing rules in its system prompt.

    ``PINNED_CONTEXT`` goes in front and :data:`_ARTIFACT_MARKDOWN_INSTRUCTION`
    at the end. The copy keeps the registry instance unchanged. Compaction
    does not edit the system message, so both survive a summary; a
    ``SystemMessage`` in the thread would leave the model's view at the cutoff.
    """
    system = str((getattr(agent, "prompts", None) or {}).get("system") or "").strip()
    if _ARTIFACT_MARKDOWN_INSTRUCTION in system:
        return agent
    pinned = (PINNED_CONTEXT or "").strip()
    parts = [p for p in (pinned, system, _ARTIFACT_MARKDOWN_INSTRUCTION) if p]
    cloned = copy.copy(agent)
    cloned.prompts = dict(getattr(agent, "prompts", None) or {})
    cloned.prompts["system"] = "\n\n".join(parts)
    return cloned


_DOCUMENT_NAME_MAX_CHARS = 120
# Longest the attached-documents lookup may hold up the first token.
_ATTACHED_DOCUMENTS_LOOKUP_S = 2.0


def attached_documents_line(names: List[str]) -> Optional[str]:
    """The system line naming a conversation's documents, or None without any.

    It says the documents exist and that the retrieve tool searches them; it
    does not tell the model to search, which stays the model's choice (forcing
    retrieval was reverted in #306). File names are user input, so each is put
    on one line, stripped of control characters, capped and quoted.
    """
    quoted: List[str] = []
    for name in names:
        text = "".join(ch for ch in " ".join(str(name).split()) if ch.isprintable())
        if len(text) > _DOCUMENT_NAME_MAX_CHARS:
            text = text[: _DOCUMENT_NAME_MAX_CHARS - 3] + "..."
        if text:
            quoted.append(json.dumps(text, ensure_ascii=False))
    if not quoted:
        return None
    return (
        "The user attached these documents to this conversation: "
        + ", ".join(quoted)
        + ". The retrieve tool searches them along with the selected collections."
    )


def _agent_with_system_line(agent: Any, line: Optional[str]) -> Any:
    """Copy *agent* with ``line`` at the end of its system prompt.

    Every graph rebuilds its system message from ``prompts["system"]`` on each
    turn and keeps it out of the thread, so the line follows uploads made
    mid-conversation without a ``system`` message after an ``assistant`` one
    (see :func:`_build_initial_messages`). The registry instance is unchanged.
    """
    if not line:
        return agent
    system = str((getattr(agent, "prompts", None) or {}).get("system") or "").strip()
    cloned = copy.copy(agent)
    cloned.prompts = dict(getattr(agent, "prompts", None) or {})
    cloned.prompts["system"] = f"{system}\n\n{line}" if system else line
    return cloned


async def _with_attached_documents(agent: Any, tools: Any, request: Any) -> Any:
    """*agent* naming the conversation's documents when a retrieve tool can search them.

    The collection comes from the request (``apply_private_collections_to_request``
    already looked it up). A lookup that fails or takes longer than
    ``_ATTACHED_DOCUMENTS_LOOKUP_S`` leaves the turn without the line rather
    than failing or delaying it: the collection is still in
    ``private_collections``.
    """
    collection_id = getattr(request, "attached_collection_id", None)
    if not collection_id or not _rag_tool_names(tools):
        return agent
    try:
        async with asyncio.timeout(_ATTACHED_DOCUMENTS_LOOKUP_S):
            names = await attached_document_names(collection_id)
    except Exception as e:
        logger.warning(
            "Could not list attached documents collection=%s timeout=%s",
            collection_id,
            isinstance(e, TimeoutError),
            exc_info=not isinstance(e, TimeoutError),
        )
        return agent
    return _agent_with_system_line(agent, attached_documents_line(names))


def attached_files_block(paths: List[str]) -> Optional[str]:
    """The block naming this message's attached files by sandbox path, or None."""
    if not paths:
        return None
    return "## Attached files\nThe user attached the following files:\n" + "\n".join(
        paths
    )


async def _attached_files(
    request: Any, user_id: Optional[str], conversation_id: Optional[str], lease: Any
) -> List[str]:
    """Sandbox paths of the files attached to this message, present in ``lease``.

    The router has already moved each attachment into the conversation's
    ``artifacts/`` folder. A new sandbox restored that folder before the
    turn; a reused one gets this message's files copied in here. Without a
    sandbox the paths would name nothing, so there are none. A lookup that
    fails leaves the turn without the block.
    """
    ids = list(dict.fromkeys(getattr(request, "artifact_ids", None) or []))
    if not ids or not user_id or not conversation_id or lease is None:
        return []
    from src.services.mcp.conversation_artifacts import (
        attached_file_paths,
        copy_into_sandbox,
    )

    try:
        oids = [ObjectId(i) for i in ids if ObjectId.is_valid(i)]
        docs = (
            await Artifact.get_primary_collection()
            .find(
                {
                    "_id": {"$in": oids},
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                },
                {"key": 1},
            )
            .to_list(length=len(oids))
        )
        keys = {str(doc["_id"]): doc.get("key") or "" for doc in docs}
        files = attached_file_paths(
            user_id, conversation_id, [keys[i] for i in ids if i in keys]
        )
        if not getattr(lease, "fresh", False):
            await copy_into_sandbox(lease.backend, files)
    except Exception:
        logger.warning(
            "Could not attach files conversation=%s", conversation_id, exc_info=True
        )
        return []
    return [path for path, _key in files]


def _retrieval_call_collections(rag_tool_names: set[str]) -> List[Dict[str, Any]]:
    """The collections each retrieval call of this run searched, in call order.

    Stored as ``metadata.prompts.retrieval_call_collections``: the tool calls in
    the trace carry the model's arguments, before the runner replaced the
    UI-owned ones. Only calls that returned a result are listed; a call that
    failed is not recorded by the interceptor.
    """
    ctx = get_retrieval_context()
    if ctx is None:
        return []
    return [
        {"tool": call.tool_name, **call.collections, "documents": len(call.documents)}
        for call in ctx.for_tools(rag_tool_names)
    ]


# Stands for a checkpoint that exists but could not be read. The thread may
# hold a stale skill list, so the skills check lists again.
_CHECKPOINT_UNREADABLE = object()


async def _read_checkpoint(checkpointer: Any, conversation_id: str) -> tuple[bool, Any]:
    """Whether this conversation already has a checkpoint, and the tuple read.

    A lookup error is treated as "the thread is there": seeding on top of a
    thread that could not be read would duplicate it. The tuple is then
    ``_CHECKPOINT_UNREADABLE``; it is ``None`` when no checkpointer or no
    checkpoint exists.
    """
    if checkpointer is None:
        return False, None
    try:
        found = await checkpointer.aget_tuple(
            {"configurable": {"thread_id": conversation_id}}
        )
    except Exception:
        logger.warning(
            "checkpoint lookup failed conversation=%s", conversation_id, exc_info=True
        )
        return True, _CHECKPOINT_UNREADABLE
    return found is not None, found


# Checkpoint metadata key holding the skills key a turn ran with. LangGraph
# copies string values of ``config["metadata"]`` onto every checkpoint the
# turn writes, so the next turn reads it from the tuple it already has.
_SKILLS_KEY_METADATA = "eve_skills_key"
# Bound on the wait for the local tree scan, not on the scan: a thread cannot
# be cancelled, so a late scan finishes in the background and fills the cache.
_SKILLS_CHECK_TIMEOUT_S = 2.0
# Checks that ran out of time in this process. Under load the wait mostly
# measures the default executor queue, so each one is an INFO line with the
# running count, not a WARNING per turn.
_skills_check_timeouts = 0


def _stored_skills_key(checkpoint: Any) -> Optional[str]:
    metadata = getattr(checkpoint, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    value = metadata.get(_SKILLS_KEY_METADATA)
    return value if isinstance(value, str) and value else None


def _has_skills_load_errors(checkpoint: Any) -> bool:
    """True when the last listing on this thread reported errors."""
    state = getattr(checkpoint, "checkpoint", None)
    if not isinstance(state, dict):
        return False
    values = state.get("channel_values")
    if not isinstance(values, dict):
        return False
    errors = values.get("skills_load_errors")
    return isinstance(errors, list) and bool(errors)


async def _skills_reload_input(
    checkpoint: Any, conversation_id: Optional[str], skills_root: Any
) -> tuple[Dict[str, Any], Dict[str, str]]:
    """Graph input and config metadata that keep a thread's skill list current.

    SkillsMiddleware lists ``/skills/`` once per thread and keeps the list in
    the checkpoint; ``{"skills_metadata": None}`` asks it to list again. The
    decision compares the key of this turn's local tree with the key the last
    turn recorded in its checkpoint metadata, never with the checkpointed list:
    on the sandbox path that list comes from the sandbox listing, and duplicate
    names resolve in listing order, so a list that differs from the local tree
    would force a reload on every turn.

    A stored key that is missing or differs from this turn's key reloads; only
    equal keys keep the list. A key goes missing when the last turn ran with
    no skills tree (no middleware, nothing recorded, the old list still in
    state) or on a thread from before this check: one extra listing then.
    Also reloaded: a checkpoint that could not be read, and a thread whose
    last listing reported errors (a failed listing would otherwise stick to
    the thread until the key changes). With no checkpoint there is nothing to
    reload: the middleware lists anyway.

    A scan failure or timeout keeps the list and re-records the previous key,
    so a change is still caught on a later turn; with no previous key nothing
    is recorded and the next turn reloads. Known gaps: a read from a lagging
    secondary can return an older checkpoint whose key equals this turn's
    key (a change made and reverted in between), and two turns on one thread
    at once can write the old list back; both stay until the key changes
    again.
    """
    if skills_root is None or not conversation_id:
        return {}, {}
    global _skills_check_timeouts
    unreadable = checkpoint is _CHECKPOINT_UNREADABLE
    stored = None if unreadable else _stored_skills_key(checkpoint)
    wanted = cached_skills_root_key(skills_root)
    if wanted is None:
        try:
            wanted = await asyncio.wait_for(
                asyncio.to_thread(skills_root_key, skills_root),
                timeout=_SKILLS_CHECK_TIMEOUT_S,
            )
        except TimeoutError:
            _skills_check_timeouts += 1
            logger.info(
                "skills reload check: no key within %.1f s, kept the list "
                "conversation=%s timeouts=%d",
                _SKILLS_CHECK_TIMEOUT_S,
                conversation_id,
                _skills_check_timeouts,
            )
        except Exception as exc:
            logger.warning(
                "skills reload check failed, kept the list conversation=%s error=%s",
                conversation_id,
                type(exc).__name__,
            )
    recorded = wanted or stored
    metadata_out = {_SKILLS_KEY_METADATA: recorded} if recorded else {}
    if checkpoint is None or wanted is None:
        return {}, metadata_out
    if unreadable:
        reason = "unreadable_checkpoint"
    elif _has_skills_load_errors(checkpoint):
        reason = "load_errors"
    elif stored is None:
        reason = "not_recorded"
    elif wanted != stored:
        reason = "changed"
    else:
        return {}, metadata_out
    logger.info(
        "skills reload conversation=%s reason=%s", conversation_id, reason
    )
    return {"skills_metadata": None}, metadata_out


async def save_agent_summary(graph: Any, conversation_id: str) -> None:
    """Copy the deep agent's latest summary onto ``Conversation.summary``.

    The summarization middleware keeps its last compaction in the thread
    state. Nothing is written until the agent has summarized, or when the
    stored copy already matches.
    """
    try:
        snapshot = await graph.aget_state(
            {"configurable": {"thread_id": conversation_id}}
        )
        event = (getattr(snapshot, "values", None) or {}).get(SUMMARIZATION_EVENT_KEY)
        message = (event or {}).get("summary_message")
        extra = getattr(message, "additional_kwargs", None) or {}
        text = extra.get(SUMMARY_TEXT_KEY)
        if not text:
            return
        first_id = extra.get(SUMMARY_FIRST_ID_KEY)
        last_id = extra.get(SUMMARY_LAST_ID_KEY)
        convo = await Conversation.find_by_id(conversation_id)
        if convo is None:
            return
        if (convo.summary, convo.summary_first_message_id, convo.summary_last_message_id) == (
            text,
            first_id,
            last_id,
        ):
            return
        convo.summary = text
        convo.summary_first_message_id = first_id
        convo.summary_last_message_id = last_id
        await convo.save()
    except Exception:
        logger.warning(
            "saving agent summary failed conversation=%s", conversation_id, exc_info=True
        )


def _schedule_after_turn(
    agent_graph_type: str,
    graph: Any,
    conversation_id: Optional[str],
    background_tasks: Optional[BackgroundTasks],
) -> None:
    """Keep ``Conversation.summary`` current once the turn is persisted.

    The deep agent writes its own summary; the backend stores it. Graphs
    without a summarizer keep the rolling summary.
    """
    if not conversation_id:
        return
    if agent_graph_type == "deep":
        job, args = save_agent_summary, (graph, conversation_id)
    else:
        job, args = maybe_rollup_and_trim_history, (conversation_id,)
    if background_tasks:
        background_tasks.add_task(job, *args)
    else:
        asyncio.create_task(job(*args))


async def _seed_turns(
    conversation_id: str, exclude_message_id: Optional[str]
) -> List[Any]:
    """Stored turns as human/assistant messages, excluding the turn in progress.

    Each human message carries ``eve_message_id`` so a summary can name the
    range. The current message is already in the database with an empty
    output; it is not part of the seed. The new human message added for this
    turn carries that id instead.
    """
    cursor = (
        Message.get_collection()
        .find({"conversation_id": conversation_id}, {"input": 1, "output": 1})
        .sort([("timestamp", 1), ("_id", 1)])
    )
    documents = await cursor.to_list(length=None)
    if exclude_message_id:
        documents = [
            doc
            for doc in documents
            if str(doc.get("_id")) != str(exclude_message_id)
        ]
    elif documents:
        documents = documents[:-1]
    seeded: List[Any] = []
    for doc in documents:
        message_id = str(doc.get("_id"))
        user = doc.get("input")
        if user:
            seeded.append(
                HumanMessage(
                    content=str(user),
                    additional_kwargs={"eve_message_id": message_id},
                )
            )
        output = doc.get("output")
        if output:
            seeded.append(AIMessage(content=str(output)))
    return seeded


async def _prepare_agent_context(
    agent_graph_type: str,
    conversation_id: Optional[str],
    checkpointer: Any,
    message_id: Optional[str],
) -> tuple[List[Any], Optional[str], List[Any], bool, Any, Any]:
    """Return history, summary, seeded turns, new-checkpoint, agent, checkpoint.

    A deep turn with a checkpoint uses that thread and no rolling summary.
    Without one, the stored turns are seeded into the new thread. ``simple``
    and ``react`` keep the rolling summary and the one-turn window. The
    checkpoint tuple is the deep turn's one read of the thread, ``None``, or
    ``_CHECKPOINT_UNREADABLE`` when the read failed.
    """
    history, summary = await _fetch_conversation_context(conversation_id)
    prior: List[Any] = []
    checkpoint: Any = None
    new_checkpoint = not history
    agent = get_agent_graph(agent_graph_type)
    if agent_graph_type == "deep" and conversation_id:
        exists, checkpoint = await _read_checkpoint(checkpointer, conversation_id)
        if exists:
            history, summary = [], None
            new_checkpoint = False
        else:
            prior = await _seed_turns(conversation_id, message_id)
            history, summary = [], None
            new_checkpoint = True
        agent = _agent_with_pinned_context(agent)
    elif agent_graph_type != "simple":
        summary = note_searchable_history(summary)
    return history, summary, prior, new_checkpoint, agent, checkpoint


def _generation_unattributed_s(
    generation_latency: float, node_latencies: Dict[str, float]
) -> float:
    """Seconds inside ``generation_latency`` that never landed on a node.

    ``generation_latency`` minus the sum of the ``node_<name>_s`` values,
    floored at zero.
    """
    return max(0.0, generation_latency - sum(node_latencies.values()))


# The keys the classic RAG path writes and the back office averages
# (routers/message.py stats): summed across the retrieval calls of a run.
_RETRIEVAL_LATENCY_KEYS = (
    "query_embedding_latency",
    "qdrant_retrieval_latency",
    "reranking_latency",
)


def _merge_retrieval_latencies(
    latencies: Dict[str, Any], rag_tool_names: set[str]
) -> Dict[str, Any]:
    """Add the /retrieve endpoint timings of this run to ``metadata.latencies``.

    The retrieval interceptor stashes the ``latencies`` object of every
    retrieval call in the request context. Summed per key so a run with two
    retrieval calls reports the time spent in Qdrant across both.
    """
    ctx = get_retrieval_context()
    if ctx is None:
        return latencies
    calls = ctx.for_tools(rag_tool_names)
    for key in _RETRIEVAL_LATENCY_KEYS:
        values = [
            call.latencies[key]
            for call in calls
            if isinstance(call.latencies.get(key), (int, float))
        ]
        if values:
            latencies[key] = round(sum(values), 6)
    return latencies


def _build_agentic_latencies(
    *,
    total_latency: float,
    generation_latency: float,
    node_latencies: Dict[str, float],
    setup_latencies: Dict[str, float],
    first_token_latency: Any = None,
    include_first_token: bool = False,
) -> Dict[str, Optional[float]]:
    """Latencies persisted on the message as ``metadata.latencies``."""
    # Backoffice reads these from metadata.latencies.
    # These add up to total_latency:
    # setup_mcp_tool_loading_s, setup_checkpointer_s, setup_loading_history_s,
    # setup_llm_resolve_s, setup_graph_compile_s, setup_sandbox_s (each omitted
    # when that step did not run), generation_latency, and other_latency_s
    # (whatever is left:
    # agent selection, callbacks, artifact stubs).
    # These add up to generation_latency, so they are not added on top of it:
    # node_<name>_s and generation_unattributed_s (that difference, floored
    # at zero). first_token_latency is the time until the first token, not a
    # slice of the total. fallback_latency_s is added afterwards by
    # _FallbackUse.merge_latencies.
    latencies: Dict[str, Optional[float]] = {}
    if include_first_token:
        latencies["first_token_latency"] = first_token_latency
    latencies["generation_latency"] = generation_latency
    latencies["total_latency"] = total_latency
    latencies.update({f"node_{k}_s": v for k, v in node_latencies.items()})
    # Only steps that actually ran. Callers omit a key by not recording it.
    latencies.update(setup_latencies)
    latencies["generation_unattributed_s"] = _generation_unattributed_s(
        generation_latency, node_latencies
    )
    latencies["other_latency_s"] = max(
        0.0,
        total_latency - generation_latency - sum(setup_latencies.values()),
    )
    return latencies


# ─── Non-streaming generation ─────────────────────────────────────────────────


# FEATURE_INLINE_CITATIONS: handles are conversation scoped. A streamed turn
# reads the Sources earlier turns stored before it hands out new ones; the
# newest rows carry the highest handles, so the cap never lets a new handle
# collide with one already in the thread.
_PRIOR_SOURCES_TIMEOUT_S = 2.0
_PRIOR_SOURCES_MAX_MESSAGES = 200


async def _load_prior_sources(
    conversation_id: str, message_id: Optional[str]
) -> List[Dict[str, Any]]:
    """The Sources earlier messages of the conversation stored, with their handles."""
    cursor = (
        Message.get_collection()
        .find(
            {
                "conversation_id": conversation_id,
                "$or": [
                    {"metadata.citations": {"$exists": True}},
                    {"metadata.sources_retrieved": {"$exists": True}},
                ],
            },
            {"metadata.citations": 1, "metadata.sources_retrieved": 1},
        )
        .sort("timestamp", -1)
        .limit(_PRIOR_SOURCES_MAX_MESSAGES)
    )
    rows = [
        row
        async for row in cursor
        if message_id is None or str(row.get("_id")) != str(message_id)
    ]
    return prior_sources_from_rows(rows)


async def _seed_conversation_sources(
    registry: SourceRegistry, conversation_id: str, message_id: Optional[str]
) -> bool:
    """Seed ``registry`` from the conversation; False when the lookup failed.

    Without the earlier handles a new one could take a number the thread
    already uses and turn a cited passage into the wrong Source, so the
    caller then runs the turn without handles.
    """
    try:
        prior = await asyncio.wait_for(
            _load_prior_sources(conversation_id, message_id),
            _PRIOR_SOURCES_TIMEOUT_S,
        )
    except Exception as exc:
        logger.warning(
            "citations.prior_sources_unavailable error=%s", type(exc).__name__
        )
        return False
    registry.seed(prior)
    return True


async def generate_answer_agentic(
    request: GenerationRequest,
    *,
    user_id: str,
    conversation_id: Optional[str] = None,
    message_id: Optional[str] = None,
) -> tuple[
    str,
    List[Dict[str, Any]],
    bool,
    Dict[str, Optional[float]],
    Dict[str, Any],
    List[Dict[str, Any]],
    List[str],
]:
    """Run the full agentic generation pipeline without streaming.

    Returns (answer, documents, use_rag, latencies, prompts, trace, artifact_ids).

    ``documents`` holds the retrieval-tool output normalised to the Document
    shape the API returns, so the UI can render sources for an agentic answer.
    ``artifact_ids`` lists any Artifacts the MCP artifact interceptor persisted
    from tool output during this run (see ``src.services.mcp.artifact_ingestion``).
    """
    if not _langgraph_available:
        raise RuntimeError("LangGraph is not available — cannot run agentic generation")

    total_start = time.perf_counter()
    # message_id tags the human message on the thread. Artifacts stay linked
    # via conversation_id; the router attaches artifact_ids to the Message.
    artifact_ctx, artifact_token = set_artifact_context(
        user_id=user_id, conversation_id=conversation_id
    )
    _retrieval_ctx, retrieval_token = set_retrieval_context()
    endpoint_metadata: Optional[Dict[str, Any]] = None
    agent_graph_type: Optional[str] = None
    sandbox_session: Any = None
    sandbox_lease: Any = None

    try:
        setup_latencies: Dict[str, float] = {}

        agent_graph_type = _resolve_agent_graph_type(request)
        step_start = time.perf_counter()
        tools = await _build_tools(
            request,
            conversation_id=None if agent_graph_type == "simple" else conversation_id,
        )
        setup_latencies["setup_mcp_tool_loading_s"] = time.perf_counter() - step_start

        step_start = time.perf_counter()
        # No conversation, no thread: run without a checkpointer, store nothing.
        checkpointer = await _get_agentic_checkpointer() if conversation_id else None
        setup_latencies["setup_checkpointer_s"] = time.perf_counter() - step_start

        step_start = time.perf_counter()
        (
            history,
            summary,
            prior,
            new_checkpoint,
            agent,
            checkpoint,
        ) = await _prepare_agent_context(
            agent_graph_type, conversation_id, checkpointer, message_id
        )
        agent = await _with_attached_documents(agent, tools, request)
        setup_latencies["setup_loading_history_s"] = time.perf_counter() - step_start
        resolved_instruction = agent.instruction_text(history=history, summary=summary)
        step_start = time.perf_counter()
        llm, llm_metadata = await _resolve_agentic_llm_client(
            request, user_id=user_id
        )
        setup_latencies["setup_llm_resolve_s"] = time.perf_counter() - step_start
        endpoint_metadata = llm_metadata.pop("endpoint", None)
        # Time the sandbox only when this graph can take one. An extra
        # perf_counter call would shift the setup clocks of every other graph.
        # Skills resolve before that clock: building the tree is not sandbox time.
        skills_root = None
        if _compile_declares_backend(agent):
            skills_root = resolve_skills_root(request.skill_configs)
            step_start = time.perf_counter()
            # No conversation: the sandbox dies with the turn. A conversation
            # keeps one sandbox and restarts its idle timer on release.
            if conversation_id:
                sandbox_lease = await acquire_sandbox(
                    conversation_id,
                    skills_root,
                    on_fresh=_on_fresh_tool_results(user_id, conversation_id),
                    user_id=user_id,
                )
            else:
                sandbox_session = await open_sandbox(skills_root)
            if sandbox_lease is not None or sandbox_session is not None:
                setup_latencies["setup_sandbox_s"] = time.perf_counter() - step_start
        step_start = time.perf_counter()
        held = sandbox_lease if sandbox_lease is not None else sandbox_session
        # Append, do not rebuild the list. It may be a _MCPToolsWithClient,
        # and a new list would drop the client strong reference.
        if held is not None:
            artifact_ctx.sandbox_backend = held.backend
        attached_files = await _attached_files(
            request, user_id, conversation_id, sandbox_lease
        )
        graph = _build_react_graph(
            request.llm_type,
            tools,
            checkpointer,
            agent=agent,
            history=history,
            summary=summary,
            llm=llm,
            llm_idle_timeout=endpoint_timeout(
                (endpoint_metadata or {}).get("answered")
            ),
            llm_run_timeout=AGENTIC_TIMEOUT,
            sandbox_backend=None if held is None else held.backend,
            skills_root=skills_root,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        setup_latencies["setup_graph_compile_s"] = time.perf_counter() - step_start

        config = {"configurable": {"thread_id": conversation_id}} if conversation_id else {}
        skills_reload: Dict[str, Any] = {}
        if skills_root is not None:
            step_start = time.perf_counter()
            skills_reload, skills_metadata = await _skills_reload_input(
                checkpoint, conversation_id, skills_root
            )
            if skills_metadata:
                config["metadata"] = skills_metadata
            setup_latencies["setup_skills_check_s"] = time.perf_counter() - step_start

        # This route hands out no handles, but the model may copy one an
        # earlier streamed turn left in the thread: the handles that turn
        # stored go, and with the flag on and no lookup, every handle goes.
        handle_registry: Optional[SourceRegistry] = SourceRegistry()
        if not (
            conversation_id
            and await _seed_conversation_sources(
                handle_registry, conversation_id, message_id
            )
        ):
            handle_registry = (
                None if src_config.FEATURE_INLINE_CITATIONS else SourceRegistry()
            )

        async def _run_graph(
            g: Any,
        ) -> tuple[List[Any], List[Dict[str, Any]], Dict[str, float], float, _FallbackUse]:
            """Stream the graph and collect results.

            Returns ``(raw_messages, trace_entries, node_latencies, duration,
            fallback)``. ``fallback.used`` is true when the graph streamed an
            ``agent_fallback`` node update or a message carrying
            ``eve_fallback_used``.
            """
            raw_msgs: List[Any] = []
            trace: List[Dict[str, Any]] = []
            latency_map: Dict[str, float] = {}
            fallback = _FallbackUse()
            start = time.perf_counter()
            # Separate from latency_map/node_latencies below: that dict keeps its
            # existing "elapsed since run start" semantics for metadata.latencies,
            # while the timeline computes the real per-step start/duration that
            # each trace entry now carries.
            trace_timeline = _TraceTimeline(start)
            graph_exc: Optional[Exception] = None
            with agent_span(
                "agentic_generation",
                conversation_id=conversation_id,
                user_id=user_id,
                attributes={
                    "eve.pipeline": "agentic",
                    "eve.stream": False,
                    "eve.agent_graph": agent_graph_type,
                },
            ) as root_span:
                set_llm_attributes(root_span, endpoint_metadata, include_answered=False)
                try:
                    async for update in g.astream(
                        {
                            "messages": _build_initial_messages(
                                request,
                                tools,
                                history,
                                prior=prior,
                                message_id=message_id,
                                new_checkpoint=new_checkpoint,
                                agent_graph_type=agent_graph_type,
                                attached_files=attached_files,
                            ),
                            **skills_reload,
                        },
                        config=config,
                        stream_mode="updates",
                    ):
                        fallback.observe(update)
                        step_time = time.perf_counter()
                        for node_name, node_output in update.items():
                            step_latency_s = step_time - start
                            latency_key = _latency_node_name(node_name)
                            if latency_key is not None:
                                latency_map.setdefault(latency_key, 0.0)
                            msgs = (
                                node_output.get("messages", [])
                                if isinstance(node_output, dict)
                                else []
                            )
                            for msg in msgs:
                                if ToolMessage and isinstance(msg, ToolMessage):
                                    started_at_s, entry_latency_s = (
                                        trace_timeline.tool_step(
                                            getattr(msg, "tool_call_id", None),
                                            at=step_time,
                                        )
                                    )
                                else:
                                    started_at_s, entry_latency_s = (
                                        trace_timeline.agent_step(
                                            getattr(msg, "tool_calls", None),
                                            at=step_time,
                                        )
                                    )
                                trace.append(
                                    _serialise_trace_entry(
                                        msg,
                                        node=node_name,
                                        latency_s=entry_latency_s,
                                        started_at_s=started_at_s,
                                    )
                                )
                                raw_msgs.append(msg)
                            if latency_key is not None:
                                latency_map[latency_key] = step_latency_s
                except Exception as exc:
                    graph_exc = exc
                # While the root is open: a turn that ends with nothing to show
                # is a failed turn in the trace too.
                if graph_exc is None and is_empty_answer(
                    _final_answer_text(raw_msgs, handle_registry),
                    artifact_ctx.collected_artifact_ids,
                ):
                    _mark_empty_answer(root_span)

            if graph_exc is not None:
                # Whitespace or an unrun text call from the fallback is not an
                # answer: the primary's failure stands.
                has_answer = not is_empty_answer(
                    _final_answer_text(raw_msgs, handle_registry),
                    artifact_ctx.collected_artifact_ids,
                )
                if not _recoverable_after_agent_fallback(
                    fallback_used=fallback.used, has_answer=has_answer
                ):
                    raise graph_exc
                logger.warning(
                    "Agent graph raised after successful agent_fallback; "
                    "returning fallback answer: %s",
                    graph_exc,
                )
                await persist_policy_event(
                    policy=_policy_for_uncaught(graph_exc),
                    description=(
                        "Primary agent node failed after agent_fallback produced an answer"
                    ),
                    node=_node_from_exception(graph_exc),
                    source="runner",
                    error=graph_exc,
                    graph=agent_graph_type,
                )
            return (
                raw_msgs,
                trace,
                latency_map,
                time.perf_counter() - start,
                fallback,
            )

        gen_start = time.perf_counter()
        (
            all_messages,
            trace_entries,
            node_latencies,
            gen_latency,
            fallback,
        ) = await _run_graph(graph)

        final_answer = await rewrite_answer_file_links(
            _final_answer_text(all_messages, handle_registry)
        )
        if is_empty_answer(final_answer, artifact_ctx.collected_artifact_ids):
            # The router persists the empty_answer marker from the same check.
            final_answer = ""
            _log_empty_answer(conversation_id, message_id)

        # Only retrieval output reaches the UI as documents. Every ToolMessage is
        # still kept verbatim in the trace by _serialise_trace_entry above.
        rag_tool_names = _rag_tool_names(tools)
        documents, retrieval_calls, _retrieval_errors = _collect_retrieval_documents(
            all_messages,
            rag_tool_names,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        use_rag = retrieval_calls > 0

        total_latency = time.perf_counter() - total_start
        latencies = fallback.merge_latencies(
            _merge_retrieval_latencies(
                _build_agentic_latencies(
                    total_latency=total_latency,
                    generation_latency=gen_latency,
                    node_latencies=node_latencies,
                    setup_latencies=setup_latencies,
                ),
                rag_tool_names,
            )
        )
        if fallback.used:
            endpoint_metadata = _record_in_graph_fallback(endpoint_metadata)
            # Same re-point as the streaming path: the footer reads this key.
            llm_metadata["agentic_llm_resolved"] = LLMType.Fallback.value
        prompts: Dict[str, Any] = {
            "query": request.query,
            "instruction": resolved_instruction,
            "agent_prompts": dict(agent.prompts),
            "agent_graph_type": agent_graph_type,
            "used_fallback_llm": fallback.used,
            "retrieval_call_collections": _retrieval_call_collections(rag_tool_names),
            **llm_metadata,
            # This function persists nothing itself: the router lifts the
            # payload out into metadata.endpoint.
            **({"endpoint": endpoint_metadata} if endpoint_metadata else {}),
        }
        # The router schedules the rolling summary for the other graphs: that
        # job reads the message it saves.
        if agent_graph_type == "deep" and conversation_id:
            asyncio.create_task(save_agent_summary(graph, conversation_id))

        return (
            final_answer,
            documents,
            use_rag,
            latencies,
            prompts,
            trace_entries,
            list(artifact_ctx.collected_artifact_ids),
        )

    except Exception as exc:
        _record_endpoint_failure(endpoint_metadata, exc)
        await persist_policy_event(
            policy=_policy_for_uncaught(exc),
            description="Agentic generation failed",
            source="runner",
            node=_node_from_exception(exc),
            error=exc,
            graph=agent_graph_type,
        )
        raise
    finally:
        if sandbox_lease is not None:
            sandbox_lease.session.start_workspace_sync(user_id, conversation_id or "")
        await release_sandbox(sandbox_lease)
        await close_sandbox(sandbox_session)
        reset_retrieval_context(retrieval_token)
        reset_artifact_context(artifact_token)


# ─── Streaming generation ─────────────────────────────────────────────────────


async def generate_answer_agentic_stream_helper(
    request: GenerationRequest,
    conversation_id: str,
    message_id: str,
    user_id: str,
    output_format: str = "json",
    background_tasks: Optional[BackgroundTasks] = None,
    cancel_event: Optional[asyncio.Event] = None,
) -> AsyncGenerator[str, None]:
    """Stream agentic generation as SSE events.

    Event types emitted:
      status      — pre-answer progress notice (emitted immediately so the
                    stream starts promptly)
      tool_call   — agent invoked a tool (query shown), plus `tool`, `label`
                    and `query` for clients rendering tool activity themselves
      tool_result — tool returned (preview), plus `tool` and `status`
      token       — LLM final-answer token
      final       — complete answer + latencies
      stopped     — cancelled by client
      error       — unhandled exception

    `content` on the tool events stays the ready-made display string the
    structured fields are derived from, so older clients render unchanged.

    Terminal-event rule: a turn that has already put answer text on the wire
    ends with `final`, never `error`. A late failure (a node timing out after
    the answer streamed, say) is still recorded in the persisted
    `metadata.error`, but the client is not handed an error banner on top of an
    answer it has already rendered. Only a turn with nothing streamed ends with
    `error`. A turn that ends without a failure but with no visible text and no
    artifact ends with `error` too, code ``empty_answer``.

    Raises ``ValueError`` when ``conversation_id`` is empty: a streamed turn
    always persists its message in a conversation and runs on its thread.
    """
    if not conversation_id:
        raise ValueError("a streamed agentic turn needs a conversation id")
    if not _langgraph_available:
        error_info = build_error_payload(RuntimeError("LangGraph not available"))
        with contextlib.suppress(Exception):
            await persist_message_state(message_id, error=error_info)
        yield f"data: {json.dumps({'type': 'error', 'code': error_info['code'], 'message': error_info['message']})}\n\n"
        return

    total_start = time.perf_counter()
    accumulated: List[str] = []
    endpoint_metadata: Optional[Dict[str, Any]] = None
    # Bound before the try so the error handlers can still attribute a turn that
    # died before, or during, LLM resolution.
    llm_prompts: Dict[str, Any] = {}
    final_emitted = False
    # Filled in as the graph streams, read back by every persistence path below.
    rag_tool_names: set[str] = set()
    graph_messages: List[Any] = []
    agent_graph_type: Optional[str] = None
    artifact_ctx, artifact_token = set_artifact_context(
        user_id=user_id, conversation_id=conversation_id, message_id=message_id
    )
    _retrieval_ctx, retrieval_token = set_retrieval_context()
    # The invoke_agent root span opens around the graph run below. Until then
    # the request's own trace (same trace id) answers; None with telemetry off.
    root_span: Any = None
    trace_id: Optional[str] = current_trace_id()
    sandbox_lease: Any = None
    # Bound before the try: a Stop that lands mid-run persists the trace with
    # the calls it cut, from the in-loop check or from the cancel handler.
    trace_entries: List[Dict[str, Any]] = []
    step_frames: Optional[_StepFrames] = None
    # One row per model call that reported token counts (``final.usage``).
    usage_rows: List[Dict[str, Any]] = []
    # The answer filter for [S<k>] handles. With FEATURE_INLINE_CITATIONS and
    # the conversation's earlier Sources read (``citing``), it turns cited
    # handles into [n] markers; when that read fails it strips every handle.
    # With the flag off it strips only the handles an earlier flag-on turn of
    # this conversation stored, which the model may copy from the thread; any
    # other [S<k>] is the model's own text. Frames and the stored message
    # then carry no citation fields.
    citing = False
    citation_rewriter = CitationRewriter(SourceRegistry(), cite=False)
    # A change to the marker list not yet sent on a token frame.
    citations_dirty = False

    def cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    def _collected_artifact_ids() -> Optional[List[str]]:
        return list(artifact_ctx.collected_artifact_ids) or None

    def _answer_text() -> str:
        """The answer as streamed so far (text the citation filter holds is not in it)."""
        return "".join(accumulated)

    def _held_token_events() -> List[str]:
        """Text the citation filter still holds, as one token frame, and into the answer.

        Called before every terminal frame that follows streamed text, so the
        stored answer is what the user saw.
        """
        rest = citation_rewriter.flush()
        if not rest:
            return []
        accumulated.append(rest)
        if output_format == "json":
            return [f"data: {json.dumps({'type': 'token', 'content': rest})}\n\n"]
        return [f"data: {rest}\n\n"]

    async def _rewrite_file_links() -> str:
        """The answer with its file links rewritten; citation offsets follow the text."""
        passes: List[List[Edit]] = []
        answer = await rewrite_answer_file_links(_answer_text(), passes)
        if citing:
            dropped = citation_rewriter.remap(answer, passes)
            if dropped:
                logger.warning("citations.offsets_dropped count=%d", dropped)
        return answer

    def _citation_fields() -> Dict[str, Any]:
        """``citations`` and ``sources_retrieved`` for the frame and the row; {} when off."""
        return citation_payload(citation_rewriter) if citing else {}

    def _log_citations(outcome: str) -> None:
        if not citing:
            return
        # Counts only: never the answer, a passage or a handle's text.
        logger.info(
            "citations.summary outcome=%s cited=%d markers=%d invalid=%d sources=%d",
            outcome,
            len(citation_rewriter.numbers),
            citation_rewriter.markers_seen - citation_rewriter.invalid,
            citation_rewriter.invalid,
            len(_retrieval_ctx.sources.sources),
        )

    def _retrieval_state() -> tuple[List[Dict[str, Any]], bool]:
        """(documents, use_rag) from the retrieval ToolMessages seen so far."""
        documents, retrieval_calls, _errors = _collect_retrieval_documents(
            graph_messages,
            rag_tool_names,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        return documents, retrieval_calls > 0

    def _final_events(
        answer: str,
        latencies: Dict[str, Any],
        error: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        if output_format == "json":
            final: Dict[str, Any] = {
                "type": "final",
                "answer": answer,
                "latencies": latencies,
                "artifact_ids": _collected_artifact_ids(),
                "trace_id": trace_id,
            }
            if endpoint_metadata:
                final["endpoint"] = endpoint_metadata
            if usage_rows:
                final["usage"] = list(usage_rows)
            final.update(_citation_fields())
            if error is not None:
                # A failure after streamed text: the legacy client ignores
                # it, the AG-UI adapter maps it to RUN_ERROR. The code
                # only: ``message`` is raw exception text.
                final["error"] = {"code": error["code"]}
            return [f"data: {json.dumps(final)}\n\n"]
        return ["data: [DONE]\n\n"]

    def _attribution() -> Dict[str, Any]:
        """Endpoint + model attribution to persist on every terminal path."""
        return {
            "endpoint": endpoint_metadata,
            "prompts": llm_prompts or None,
            "generated_model_name": resolve_generated_model_name(endpoint_metadata),
        }

    try:
        if cancelled():
            await persist_message_state(
                message_id,
                stopped=True,
                artifact_ids=_collected_artifact_ids(),
                trace_id=trace_id,
            )
            yield f"data: {json.dumps({'type': 'stopped'})}\n\n"
            return

        yield f"data: {json.dumps({'type': 'status', 'content': 'Thinking…'})}\n\n"

        if src_config.FEATURE_INLINE_CITATIONS:
            citing = await _seed_conversation_sources(
                _retrieval_ctx.sources, conversation_id, message_id
            )
            _retrieval_ctx.cite = citing
            citation_rewriter = (
                CitationRewriter(_retrieval_ctx.sources)
                if citing
                else CitationRewriter(SourceRegistry())
            )
        elif conversation_id:
            prior_sources = SourceRegistry()
            if await _seed_conversation_sources(
                prior_sources, conversation_id, message_id
            ):
                citation_rewriter = CitationRewriter(prior_sources, cite=False)

        setup_latencies: Dict[str, float] = {}

        agent_graph_type = _resolve_agent_graph_type(request)
        step_start = time.perf_counter()
        tools = await _build_tools(
            request,
            cancel_event=cancel_event,
            conversation_id=None if agent_graph_type == "simple" else conversation_id,
        )
        setup_latencies["setup_mcp_tool_loading_s"] = time.perf_counter() - step_start
        rag_tool_names.update(_rag_tool_names(tools))

        step_start = time.perf_counter()
        checkpointer = await _get_agentic_checkpointer()
        setup_latencies["setup_checkpointer_s"] = time.perf_counter() - step_start

        step_start = time.perf_counter()
        (
            history,
            summary,
            prior,
            new_checkpoint,
            agent,
            checkpoint,
        ) = await _prepare_agent_context(
            agent_graph_type, conversation_id, checkpointer, message_id
        )
        agent = await _with_attached_documents(agent, tools, request)
        setup_latencies["setup_loading_history_s"] = time.perf_counter() - step_start
        resolved_instruction = agent.instruction_text(history=history, summary=summary)
        step_start = time.perf_counter()
        llm, llm_prompts = await _resolve_agentic_llm_client(
            request, user_id=user_id
        )
        setup_latencies["setup_llm_resolve_s"] = time.perf_counter() - step_start
        endpoint_metadata = llm_prompts.pop("endpoint", None)
        switch_frame = _endpoint_switch_frame(endpoint_metadata)
        if switch_frame is not None and output_format == "json":
            yield switch_frame

        # Time the sandbox only when this graph can take one. An extra
        # perf_counter call would shift the setup clocks of every other graph.
        # Skills resolve before that clock: building the tree is not sandbox time.
        skills_root = None
        if _compile_declares_backend(agent):
            skills_root = resolve_skills_root(request.skill_configs)
            step_start = time.perf_counter()
            # A streamed turn always has a conversation, so this sandbox stays
            # up and the release below restarts its idle timer.
            sandbox_lease = await acquire_sandbox(
                conversation_id,
                skills_root,
                on_fresh=_on_fresh_tool_results(user_id, conversation_id),
                user_id=user_id,
            )
            if sandbox_lease is not None:
                setup_latencies["setup_sandbox_s"] = time.perf_counter() - step_start
        # Append, do not rebuild the list. It may be a _MCPToolsWithClient,
        # and a new list would drop the client strong reference.
        if sandbox_lease is not None:
            artifact_ctx.sandbox_backend = sandbox_lease.backend
        attached_files = await _attached_files(
            request, user_id, conversation_id, sandbox_lease
        )
        step_start = time.perf_counter()
        graph = _build_react_graph(
            request.llm_type,
            tools,
            checkpointer,
            agent=agent,
            history=history,
            summary=summary,
            llm=llm,
            llm_idle_timeout=endpoint_timeout(
                (endpoint_metadata or {}).get("answered")
            ),
            llm_run_timeout=AGENTIC_TIMEOUT,
            sandbox_backend=(
                None if sandbox_lease is None else sandbox_lease.backend
            ),
            skills_root=skills_root,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        setup_latencies["setup_graph_compile_s"] = time.perf_counter() - step_start

        config = {"configurable": {"thread_id": conversation_id}}
        skills_reload: Dict[str, Any] = {}
        if skills_root is not None:
            step_start = time.perf_counter()
            skills_reload, skills_metadata = await _skills_reload_input(
                checkpoint, conversation_id, skills_root
            )
            if skills_metadata:
                config["metadata"] = skills_metadata
            setup_latencies["setup_skills_check_s"] = time.perf_counter() - step_start

        gen_start = time.perf_counter()
        # First-answer budget. gen_start stays put: generation_latency uses it.
        # This clock restarts on each announced tool call and each tool result.
        answer_budget_started_at = gen_start
        first_token_latency: Optional[float] = None
        tokens_yielded = 0
        fallback = _FallbackUse()
        node_start_time: float = gen_start
        node_latencies: Dict[str, float] = {}
        # Separate from node_start_time/node_latencies above (kept for
        # metadata.latencies): the timeline computes each trace entry's real
        # started_at_s/latency_s per the agent/tool/final-answer timing rules.
        trace_timeline = _TraceTimeline(gen_start)
        step_frames = _StepFrames(tools)

        turn_buffer: List[str] = []
        # A model call ended (its node update, the node changed, or the chunk
        # id changed): the next answer text starts a new paragraph. The break rides that text, so a
        # Stop or an end before it stores none.
        paragraph_pending = False
        # Chunk id of the model call that streamed last: a new id is a new
        # call in the same node (a node retry, retry or fallback middleware).
        model_call_id: Optional[str] = None
        current_node: Optional[str] = None
        # Tool calls already announced in the current agent step, keyed by
        # call id. One call can reach the stream up to three times (a complete
        # message, the node update, the `[TOOL_CALLS]` text flushed after the
        # update) and must open one chip only. Cleared when a tool answers.
        announced_tool_calls: set = set()

        def _step_model(msg: Any = None, node: Optional[str] = None) -> tuple:
            """(endpoint key, provider model id) behind a step.

            The key is always ours (``main``, ``fallback``, ...); the provider
            id comes from the message, when it carries one.
            """
            meta = getattr(msg, "response_metadata", None) or {}
            name = meta.get("model_name") or meta.get("model")
            provider = name if isinstance(name, str) and name else None
            if fallback.used or node == "agent_fallback":
                return LLMType.Fallback.value, provider
            return (endpoint_metadata or {}).get("answered"), provider

        def _record_tool_result(chunk: Any, node: str) -> Dict[str, Any]:
            """Trace entry and step fields for one ToolMessage; the frame."""
            graph_messages.append(chunk)
            announced_tool_calls.clear()
            started_at_s, entry_latency_s = trace_timeline.tool_step(
                getattr(chunk, "tool_call_id", None),
                at=time.perf_counter(),
            )
            result_fields = step_frames.result(
                getattr(chunk, "name", None),
                getattr(chunk, "tool_call_id", None),
                getattr(chunk, "status", None),
                chunk.content,
                duration_ms=_tool_duration_ms(chunk),
            )
            tool_entry = _serialise_trace_entry(
                chunk,
                node=node,
                latency_s=entry_latency_s,
                started_at_s=started_at_s,
            )
            tool_entry["step"] = dict(result_fields)
            trace_entries.append(tool_entry)
            result_frame: Dict[str, Any] = {
                "type": "tool_result",
                "content": stringify_tool_content(chunk.content)[:200],
                "tool": getattr(chunk, "name", None),
                "id": getattr(chunk, "tool_call_id", None),
            }
            result_frame.update(result_fields)
            return result_frame

        def _tool_call_events(
            tool_calls: List[Any],
            model: Optional[str] = None,
            model_name: Optional[str] = None,
        ) -> List[str]:
            nonlocal answer_budget_started_at
            events: List[str] = []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    tname = tc.get("name", "tool")
                    args = tc.get("args") or {}
                    call_id = tc.get("id")
                else:
                    tname = getattr(tc, "name", "tool")
                    args = getattr(tc, "args", None) or {}
                    call_id = getattr(tc, "id", None)
                key = call_id or f"{tname}:{json.dumps(args, sort_keys=True, default=str)}"
                if key in announced_tool_calls:
                    continue
                announced_tool_calls.add(key)
                # Once per announced call. A repeat of the same call must not
                # restart the budget a second time.
                answer_budget_started_at = time.perf_counter()
                query_used = args.get("query", "")
                label = tool_call_label(tname) if tool_call_label else f"Calling {tname}"
                msg = f"{label}: {query_used}" if query_used else f"{label}…"
                frame = {
                    "type": "tool_call",
                    "content": msg,
                    "tool": tname,
                    "label": label,
                    "query": query_used or None,
                    "id": call_id,
                    "args": _stream_tool_args(args),
                }
                if model is None:
                    model, model_name = _step_model()
                frame.update(
                    step_frames.call(
                        tname,
                        args,
                        call_id,
                        announce_key=key,
                        model=model,
                        model_name=model_name,
                    )
                )
                events.append(f"data: {json.dumps(frame)}\n\n")
            return events

        def _feed_answer_text(tok: str) -> str:
            """``tok`` through the citation filter, with the paragraph break it opens."""
            nonlocal paragraph_pending
            if not paragraph_pending:
                return citation_rewriter.feed(tok)
            # The held text counts: it is streamed text the filter has not
            # sent yet.
            so_far = _answer_text() + citation_rewriter.held
            if not so_far:
                paragraph_pending = False
                return citation_rewriter.feed(tok)
            if not tok.strip():
                # A call that writes only whitespace: the break waits for
                # the next text, so no blank paragraph and no double break.
                return ""
            paragraph_pending = False
            # A marker is written inside one model call: a handle prefix held
            # from the last call is plain text, sent before the break.
            head = citation_rewriter.flush()
            body = tok.lstrip(" \t")
            tail = so_far.rstrip(" \t")
            have = (len(tail) - len(tail.rstrip("\n"))) + (
                len(body) - len(body.lstrip("\n"))
            )
            sep = PARAGRAPH_BREAK[: max(0, len(PARAGRAPH_BREAK) - have)]
            return head + citation_rewriter.feed(sep + body)

        def _flush_turn_buffer_to_events() -> List[str]:
            nonlocal tokens_yielded, first_token_latency, citations_dirty
            if not turn_buffer:
                return []
            items = list(turn_buffer)
            joined = "".join(items)
            turn_buffer.clear()

            events: List[str] = []
            answer_items = items

            if has_text_tool_call and has_text_tool_call(joined):
                # A turn can contain one or more `[TOOL_CALLS]name{...}` segments
                # (each call gets its own marker) directly followed by real
                # answer prose in the SAME turn — split_tool_calls_and_answer_text
                # walks every recognized call and returns whatever text is left
                # over so it's emitted as an answer instead of silently dropped
                # or, worse, leaking the raw "[TOOL_CALLS]..." syntax verbatim
                # into `accumulated`/the persisted answer.
                calls, answer_text = (
                    split_tool_calls_and_answer_text(joined)
                    if split_tool_calls_and_answer_text
                    else (parse_text_tool_calls(joined) if parse_text_tool_calls else [], "")
                )
                if not calls:
                    return []
                events.extend(_tool_call_events(calls))
                if not answer_text:
                    return events
                # Streaming granularity is inherently lost for this leftover
                # text: we can't know the tool-call syntax has ended until the
                # whole turn is buffered, so it's emitted as a single chunk
                # rather than token-by-token.
                answer_items = [answer_text]

            if tokens_yielded == 0:
                elapsed = time.perf_counter() - answer_budget_started_at
                if elapsed > AGENTIC_TIMEOUT:
                    raise _AgenticBudgetTimeout(
                        "No final-answer token received within AGENTIC_TIMEOUT"
                    )
                first_token_latency = time.perf_counter() - total_start
                add_span_event(
                    root_span,
                    "gen_ai.first_token",
                    {"eve.time_to_first_token_s": first_token_latency},
                )

            for tok in answer_items:
                if not tok:
                    continue
                version = citation_rewriter.version
                tok = _feed_answer_text(tok)
                if citing and citation_rewriter.version != version:
                    citations_dirty = True
                if not tok:
                    # Held, or a marker that only added an offset: the change
                    # rides on the next token that carries text.
                    continue
                tokens_yielded += 1
                accumulated.append(tok)
                if output_format == "json":
                    token_frame: Dict[str, Any] = {"type": "token", "content": tok}
                    if citations_dirty:
                        # The whole list so far, only on a token after a
                        # marker changed it (sources standard, section 3).
                        token_frame["citations"] = citation_rewriter.citations()
                        citations_dirty = False
                    events.append(f"data: {json.dumps(token_frame)}\n\n")
                else:
                    events.append(f"data: {tok}\n\n")
            return events

        graph_exc: Optional[Exception] = None
        try:
            with agent_span(
                "agentic_generation_stream",
                conversation_id=conversation_id,
                user_id=user_id,
                message_id=message_id,
                attributes={
                    "eve.pipeline": "agentic",
                    "eve.stream": True,
                    "eve.agent_graph": agent_graph_type,
                },
                stop_event=cancel_event,
            ) as root_span, cancel_is_stop():
                trace_id = span_trace_id(root_span) or trace_id
                set_llm_attributes(root_span, endpoint_metadata, include_answered=False)
                # aclosing: a Stop returns from inside the loop, and the stream
                # must close in this task, where the Stop binding is visible to
                # the spans it ends, not in the loop's finaliser.
                async with contextlib.aclosing(
                    graph.astream(
                        {
                            "messages": _build_initial_messages(
                                request,
                                tools,
                                history,
                                prior=prior,
                                message_id=message_id,
                                new_checkpoint=new_checkpoint,
                                agent_graph_type=agent_graph_type,
                                attached_files=attached_files,
                            ),
                            **skills_reload,
                        },
                        config=config,
                        stream_mode=["messages", "updates", "custom"],
                    )
                ) as graph_stream:
                    async for mode, payload in graph_stream:
                        if cancelled():
                            # The event that trips the check is often the
                            # result of the call the Stop waited on: that call
                            # finished, so it is paired before the rest is cut.
                            if mode == "messages" and ToolMessage:
                                stop_chunk = payload[0] if isinstance(payload, tuple) else None
                                if isinstance(stop_chunk, ToolMessage):
                                    _record_tool_result(
                                        stop_chunk,
                                        (payload[1] or {}).get("langgraph_node", ""),
                                    )
                            # The cut calls go to the stored trace, not to the
                            # stream: the Stop route already closed it.
                            step_frames.cancel_pending()
                            held_events = _held_token_events()
                            cancel_documents, cancel_use_rag = _retrieval_state()
                            await persist_message_state(
                                message_id,
                                trace_id=trace_id,
                                stopped=True,
                                output=_answer_text(),
                                documents=cancel_documents,
                                use_rag=cancel_use_rag,
                                trace=trace_entries or None,
                                artifact_ids=_collected_artifact_ids(),
                                **_citation_fields(),
                            )
                            _log_citations("stopped")
                            for event in held_events:
                                yield event
                            yield f"data: {json.dumps({'type': 'stopped', **_citation_fields()})}\n\n"
                            return

                        if mode == "custom":
                            event = _summarizing_event(payload)
                            if event is not None:
                                yield event
                            continue

                        if mode == "updates":
                            fallback_before = fallback.used
                            fallback.observe(payload)
                            if (
                                fallback.used
                                and not fallback_before
                                and output_format == "json"
                            ):
                                yield f"data: {json.dumps({'type': 'model_fallback', 'from': (endpoint_metadata or {}).get('answered'), 'to': LLMType.Fallback.value, 'reason': 'agent_error'})}\n\n"
                            # An agent step is traced from its node update, not from
                            # the streamed chunks: the first chunk carrying a tool
                            # call has no arguments yet when the provider streams
                            # them, and no chunk carries the model or usage.
                            for update_node, node_output in payload.items():
                                msgs = (
                                    node_output.get("messages", [])
                                    if isinstance(node_output, dict)
                                    else []
                                )
                                if not isinstance(msgs, list):
                                    msgs = [msgs]
                                if AIMessage and any(
                                    isinstance(m, AIMessage) for m in msgs
                                ):
                                    # A new agent step: a call the last tools
                                    # step passed by is a ghost, not pending.
                                    step_frames.drop_unanswered()
                                    # That model call is over: later answer
                                    # text comes from another one.
                                    paragraph_pending = True
                                for msg in msgs:
                                    if ToolMessage and isinstance(msg, ToolMessage):
                                        answer_budget_started_at = time.perf_counter()
                                    if AIMessage and isinstance(msg, AIMessage):
                                        row = _usage_row(
                                            msg, *_step_model(msg, update_node)
                                        )
                                        if row is not None:
                                            usage_rows.append(row)
                                    if not (
                                        AIMessage
                                        and isinstance(msg, AIMessage)
                                        and getattr(msg, "tool_calls", None)
                                    ):
                                        continue
                                    started_at_s, entry_latency_s = (
                                        trace_timeline.agent_step(
                                            msg.tool_calls, at=time.perf_counter()
                                        )
                                    )
                                    agent_entry = _serialise_trace_entry(
                                        msg,
                                        node=update_node,
                                        latency_s=entry_latency_s,
                                        started_at_s=started_at_s,
                                    )
                                    trace_entries.append(agent_entry)
                                    # The update carries the complete arguments,
                                    # so the tool call is announced from here.
                                    call_events = _tool_call_events(
                                        msg.tool_calls,
                                        *_step_model(msg, update_node),
                                    )
                                    step_frames.annotate_trace_calls(agent_entry)
                                    for event in call_events:
                                        yield event
                            continue

                        chunk, metadata = payload
                        if _is_internal_model_call(metadata):
                            continue
                        node = metadata.get("langgraph_node", "")
                        if node != current_node:
                            if current_node:
                                elapsed_s = time.perf_counter() - node_start_time
                                latency_key = _latency_node_name(current_node)
                                if latency_key is not None:
                                    node_latencies[latency_key] = (
                                        node_latencies.get(latency_key, 0.0) + elapsed_s
                                    )
                            for event in _flush_turn_buffer_to_events():
                                yield event
                            # One model call never spans two nodes.
                            paragraph_pending = True
                            node_start_time = time.perf_counter()
                            current_node = node

                        if ToolMessage and isinstance(chunk, ToolMessage):
                            answer_budget_started_at = time.perf_counter()
                            result_frame = _record_tool_result(chunk, node)
                            yield f"data: {json.dumps(result_frame)}\n\n"
                            continue

                        if AIMessage and isinstance(chunk, AIMessage):
                            chunk_id = getattr(chunk, "id", None)
                            if chunk_id:
                                if model_call_id and chunk_id != model_call_id:
                                    for event in _flush_turn_buffer_to_events():
                                        yield event
                                    paragraph_pending = True
                                model_call_id = chunk_id
                            if getattr(chunk, "tool_calls", None):
                                # Kept only so a ToolMessage without a name can be
                                # traced back to the tool it answered.
                                graph_messages.append(chunk)
                                # A streamed chunk is partial: the first one has
                                # the tool name and no arguments yet, so the call
                                # waits for the node update. A complete message
                                # (a model that does not stream) is announced now.
                                if not (
                                    AIMessageChunk and isinstance(chunk, AIMessageChunk)
                                ):
                                    for event in _tool_call_events(
                                        chunk.tool_calls,
                                        *_step_model(chunk, node),
                                    ):
                                        yield event
                                continue

                            content = _content_text(chunk.content)
                            if not content:
                                continue

                            turn_buffer.append(content)
                            joined = "".join(turn_buffer)
                            if might_be_incomplete_text_tool_call and (
                                might_be_incomplete_text_tool_call(joined)
                            ):
                                continue
                            for event in _flush_turn_buffer_to_events():
                                yield event

                if current_node:
                    elapsed_s = time.perf_counter() - node_start_time
                    latency_key = _latency_node_name(current_node)
                    if latency_key is not None:
                        node_latencies[latency_key] = (
                            node_latencies.get(latency_key, 0.0) + elapsed_s
                        )
                for event in _flush_turn_buffer_to_events():
                    yield event
                # Still inside the root: the held text goes out first, so the
                # check reads the answer the user saw and the row stores (the
                # file link rewrite below never empties it). A
                # Stop already set ends the turn stopped at the next await, so
                # the trace is left to the Stop.
                for event in _held_token_events():
                    yield event
                if not cancelled() and is_empty_answer(
                    _answer_text(), _collected_artifact_ids()
                ):
                    _mark_empty_answer(root_span)
        except Exception as exc:
            # Whitespace or an unrun text call from the fallback is not an
            # answer: the primary's failure stands.
            has_answer = not is_empty_answer(
                _answer_text() + citation_rewriter.held, _collected_artifact_ids()
            )
            if not _recoverable_after_agent_fallback(
                fallback_used=fallback.used, has_answer=has_answer
            ):
                raise
            graph_exc = exc
            logger.warning(
                "Agent graph raised after successful agent_fallback; "
                "persisting streamed answer: %s",
                exc,
            )

        # Empty unless the graph raised after a recovered fallback: the normal
        # end flushed inside the root above.
        for event in _held_token_events():
            yield event

        stream_end = time.perf_counter()
        gen_latency = stream_end - gen_start
        answer = await _rewrite_file_links()
        # Whitespace, or a text tool call the turn buffer dropped, and no
        # artifact: the user got nothing, so the turn is a failed one.
        empty_answer = is_empty_answer(answer, _collected_artifact_ids())
        if empty_answer:
            answer = ""
        total_latency = time.perf_counter() - total_start

        latencies = fallback.merge_latencies(
            _merge_retrieval_latencies(
                _build_agentic_latencies(
                    total_latency=total_latency,
                    generation_latency=gen_latency,
                    node_latencies=node_latencies,
                    setup_latencies=setup_latencies,
                    first_token_latency=first_token_latency,
                    include_first_token=True,
                ),
                rag_tool_names,
            )
        )

        if answer:
            answer_node = current_node or "agent"
            # The final answer follows the same rule as an agent step: it
            # starts when the previous recorded event ended and ends now, at
            # stream completion.
            started_at_s, entry_latency_s = trace_timeline.agent_step(
                None, at=stream_end
            )
            trace_entries.append(
                {
                    "role": "assistant",
                    "node": answer_node,
                    "content": answer,
                    "started_at_s": started_at_s,
                    "latency_s": entry_latency_s,
                }
            )

        if graph_exc is not None:
            await persist_policy_event(
                policy=_policy_for_uncaught(graph_exc),
                description=(
                    "Primary agent node failed after agent_fallback produced an answer"
                ),
                node=_node_from_exception(graph_exc),
                source="runner",
                error=graph_exc,
                graph=agent_graph_type,
            )

        if fallback.used:
            endpoint_metadata = _record_in_graph_fallback(endpoint_metadata)
            # The footer prefers agentic_llm_resolved when a fallback ran; it
            # must name the model that answered, not the one that died.
            llm_prompts["agentic_llm_resolved"] = LLMType.Fallback.value
        documents, use_rag = _retrieval_state()
        await persist_message_state(
            message_id,
            trace_id=trace_id,
            output=answer,
            documents=documents,
            use_rag=use_rag,
            latencies=latencies,
            prompts={
                "query": request.query,
                "instruction": resolved_instruction,
                "agent_prompts": dict(agent.prompts),
                "agent_graph_type": agent_graph_type,
                "used_fallback_llm": fallback.used,
                "retrieval_call_collections": _retrieval_call_collections(rag_tool_names),
                **llm_prompts,
            },
            endpoint=endpoint_metadata,
            generated_model_name=resolve_generated_model_name(endpoint_metadata),
            trace=trace_entries if trace_entries else None,
            artifact_ids=_collected_artifact_ids(),
            clear_error=bool(answer),
            **({"error": build_empty_answer_payload()} if empty_answer else {}),
            **_citation_fields(),
        )
        _log_citations("answer")
        if empty_answer:
            _log_empty_answer(conversation_id, message_id)

        _schedule_after_turn(agent_graph_type, graph, conversation_id, background_tasks)

        # Set before the terminal frame either way: the turn is persisted, and a
        # Stop that lands on the publish must not replace it.
        final_emitted = True
        if empty_answer:
            # Nothing visible went out, so the error frame, not a final: the
            # client renders it with Retry, as for any failure before text.
            error_info = build_empty_answer_payload()
            yield f"data: {json.dumps({'type': 'error', 'code': error_info['code'], 'message': error_info['message']})}\n\n"
            return
        for event in _final_events(answer, latencies):
            yield event

    except (asyncio.CancelledError, GeneratorExit) as exc:
        # GeneratorExit: the Stop landed while the consumer was publishing a
        # chunk, and it closed this generator. Only a Stop sets the event; any
        # other close is not a stop and is left to the consumer. Past the final
        # event the answer is persisted: a Stop must not replace it.
        stop = cancelled()
        if final_emitted or (isinstance(exc, GeneratorExit) and not stop):
            raise
        logger.info("Agentic generation cancelled")
        if step_frames is not None:
            # Same worker Stop: the route cancelled this task mid-call.
            step_frames.cancel_pending()
        cancelled_documents, cancelled_use_rag = _retrieval_state()
        await persist_message_state(
            message_id,
            trace_id=trace_id,
            output=_answer_text(),
            documents=cancelled_documents,
            use_rag=cancelled_use_rag,
            trace=trace_entries or None,
            stopped=stop,
            error=None if stop else build_error_payload(exc),
            artifact_ids=_collected_artifact_ids(),
            **_citation_fields(),
        )
        # No frame can follow here: text the filter still holds was never
        # shown, so it stays out of the stored answer.
        _log_citations("stopped" if stop else "cancelled")
        if not stop:
            # Worker shutdown or the deadline: a failed turn with its partial
            # output, and the cancel goes on to the caller.
            raise
        return

    except TimeoutError as exc:
        logger.warning("Agentic generation timed out: %s", exc)
        _record_endpoint_failure(endpoint_metadata, exc)
        await persist_policy_event(
            policy=_policy_for_uncaught(exc),
            description="Agentic generation timed out",
            source="runner",
            node=_node_from_exception(exc),
            error=exc,
            graph=agent_graph_type,
        )
        held_events = _held_token_events()
        answer = await _rewrite_file_links()
        error_info = build_error_payload(exc)
        timeout_documents, timeout_use_rag = _retrieval_state()
        with contextlib.suppress(Exception):
            await persist_message_state(
                message_id,
                trace_id=trace_id,
                output=answer,
                documents=timeout_documents,
                use_rag=timeout_use_rag,
                error=error_info,
                artifact_ids=_collected_artifact_ids(),
                **_attribution(),
                **_citation_fields(),
            )
        _log_citations("timeout")
        if final_emitted:
            return
        if answer:
            for event in held_events:
                yield event
            for event in _final_events(answer, {}, error=error_info):
                yield event
        else:
            yield f"data: {json.dumps({'type': 'error', 'code': error_info['code'], 'message': error_info['message']})}\n\n"

    except Exception as exc:
        logger.error("Agentic streaming error: %s", exc)
        _record_endpoint_failure(endpoint_metadata, exc)
        await persist_policy_event(
            policy=_policy_for_uncaught(exc),
            description="Agentic streaming error",
            source="runner",
            node=_node_from_exception(exc),
            error=exc,
            graph=agent_graph_type,
        )
        error_info = build_error_payload(exc)
        held_events = _held_token_events()
        answer = _answer_text()
        error_documents, error_use_rag = _retrieval_state()
        with contextlib.suppress(Exception):
            await persist_message_state(
                message_id,
                trace_id=trace_id,
                output=answer,
                documents=error_documents,
                use_rag=error_use_rag,
                error=error_info,
                artifact_ids=_collected_artifact_ids(),
                **_attribution(),
                **_citation_fields(),
            )
        _log_citations("error")
        if final_emitted:
            return
        if answer:
            for event in held_events:
                yield event
            for event in _final_events(answer, {}, error=error_info):
                yield event
        else:
            yield f"data: {json.dumps({'type': 'error', 'code': error_info['code'], 'message': error_info['message']})}\n\n"

    finally:
        if sandbox_lease is not None:
            sandbox_lease.session.start_workspace_sync(user_id, conversation_id)
        await release_sandbox(sandbox_lease)
        reset_retrieval_context(retrieval_token)
        reset_artifact_context(artifact_token)


# ─── SSE wrappers ─────────────────────────────────────────────────────────────


async def generate_answer_agentic_stream(
    request: GenerationRequest,
    conversation_id: str,
    message_id: str,
    user_id: str,
    background_tasks: Optional[BackgroundTasks] = None,
    cancel_event: Optional[asyncio.Event] = None,
):
    """Plain-text SSE wrapper around the agentic stream helper."""
    async with contextlib.aclosing(
        generate_answer_agentic_stream_helper(
            request,
            conversation_id,
            message_id,
            user_id,
            "plain",
            background_tasks,
            cancel_event,
        )
    ) as chunks:
        async for chunk in chunks:
            yield chunk


async def generate_answer_agentic_json_stream(
    request: GenerationRequest,
    conversation_id: str,
    message_id: str,
    user_id: str,
    background_tasks: Optional[BackgroundTasks] = None,
    cancel_event: Optional[asyncio.Event] = None,
):
    """JSON SSE wrapper around the agentic stream helper."""
    async with contextlib.aclosing(
        generate_answer_agentic_stream_helper(
            request,
            conversation_id,
            message_id,
            user_id,
            "json",
            background_tasks,
            cancel_event,
        )
    ) as chunks:
        async for chunk in chunks:
            yield chunk


# ─── Bus-decoupled entry point ────────────────────────────────────────────────


async def run_agentic_generation_to_bus(
    request: GenerationRequest,
    conversation_id: str,
    message_id: str,
    user_id: str,
    background_tasks: Optional[BackgroundTasks] = None,
    cancel_event: Optional[asyncio.Event] = None,
    subscriber_ready: Optional[asyncio.Event] = None,
    deadline_seconds: Optional[float] = None,
):
    """Run agentic generation in background, publishing chunks to stream bus.

    Neither bus implementation buffers: anything published before a subscriber
    attaches is dropped. ``subscriber_ready`` lets the route hold generation
    until its consumer is listening, so the events emitted at t=0 survive. The
    timeout is a guard against a response that is never consumed (client gone
    between request and response start): after it, generation runs anyway,
    because persistence must happen regardless of the SSE channel.

    ``deadline_seconds`` bounds the whole run (None: no bound), same contract as
    ``run_generation_to_bus``: on expiry the turn fails with a timeout error and
    the task ends, which frees its load shedding slot.
    """
    bus = get_stream_bus()
    deadline = asyncio.timeout(deadline_seconds)
    terminal_sent = False
    try:
        # Inside the try: a Stop that lands while waiting must still reach the
        # handler below and the cleanup in finally.
        if subscriber_ready is not None:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(subscriber_ready.wait(), timeout=5.0)
        async with deadline:
            # aclosing: a cancel that lands on publish closes the generator in
            # this task, so its finally resets its context vars where it set them.
            async with contextlib.aclosing(
                generate_answer_agentic_json_stream(
                    request=request,
                    conversation_id=conversation_id,
                    message_id=message_id,
                    background_tasks=background_tasks,
                    cancel_event=cancel_event,
                    user_id=user_id,
                )
            ) as chunks:
                async for chunk in chunks:
                    terminal_sent = is_terminal_event(chunk)
                    await bus.publish(message_id, chunk)
                    if terminal_sent:
                        release_stop_handle(conversation_id, message_id)
        raise_if_deadline_expired(deadline, deadline_seconds)
    except asyncio.CancelledError as exc:
        # Cancelled outside the generator (before it started, or on publish):
        # nothing persisted the turn yet. Idempotent when the generator did.
        # On the publish of the last event the turn is already persisted.
        if not terminal_sent:
            await persist_runner_cancel(message_id, cancel_event, exc)
    except Exception as exc:
        # Outer safety net: it fires exactly when the inner handlers did not,
        # so it must persist the marker itself or the turn stays a blank shell.
        error_info = build_error_payload(deadline_error(exc, deadline, deadline_seconds))
        with contextlib.suppress(Exception):
            await persist_message_state(
                message_id, error=error_info, **deadline_overrides(deadline)
            )
        await bus.publish(
            message_id,
            f"data: {json.dumps({'type': 'error', 'code': error_info['code'], 'message': error_info['message']})}\n\n",
        )
    finally:
        try:
            user = await User.find_by_id(user_id)
            message = await Message.find_by_id(message_id)
            if user and message:
                token_count = count_tokens_for_texts(message.input, message.output)
                await consume_tokens_for_user(user, token_count)
        except Exception as consume_error:
            logger.warning(
                "Failed to apply token usage for agentic generation: %s",
                consume_error,
            )
        await bus.close(message_id)
        with contextlib.suppress(Exception):
            from src.services.cancel_manager import get_cancel_manager

            cm = get_cancel_manager()
            cm.clear_mapping_for(conversation_id, message_id)
            cm.clear(message_id)
