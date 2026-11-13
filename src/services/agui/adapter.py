"""Legacy stream bus frames of one agentic run, adapted to AG-UI 1.0 events.

The AG-UI spec calls this piece the bridge.

Pure: no FastAPI, no I/O, no clock. The route builds one ``AguiAdapter`` per
run, calls ``start()`` before it subscribes to the bus, ``adapt()`` once per
legacy frame (already parsed from its ``data:`` line into a dict), and
``finish()`` when the bus closes. Each call returns the AG-UI events to send, in
order, as ``ag_ui.core`` models; ``encode_sse`` turns them into the wire body
through the SDK ``EventEncoder``. The adapter holds the open text message,
tool calls, steps and activities, closes them before the terminal event, and
drops every frame after the first terminal one.

What a standard AG-UI client needs (decision 0008): each model call is its own
assistant message (the first keeps the answer messageId, the next ones
``<messageId>:assistant:<n>``), its tool calls point at it by
``parentMessageId``, and a new call starts at the first text or tool call after
a tool result; text deltas pass through untouched, so the texts concatenate to
the stored answer. Tool arguments and results stay valid JSON when cut.

Input rejected before the run starts never reaches the adapter: it is an
HTTP error with no event stream, shaped by ``pre_stream_error``.

Logging: event types, ids and counts only. No user text, no tool arguments,
no tool results, no tokens, no raw exception text.

Contract: ``CONTRACT.md`` of the EVE AG-UI program, sections 1.6, 1.7, 2, 3 and
the decisions of section 8.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Tuple

from ag_ui.core import (
    ActivitySnapshotEvent,
    BaseEvent,
    CustomEvent,
    RunErrorEvent,
    RunFinishedCancelledOutcome,
    RunFinishedEvent,
    RunFinishedSuccessOutcome,
    RunStartedEvent,
    StepFinishedEvent,
    StepStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    TokenUsage,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
    aggregate_token_usage,
    token_usage_from_langchain_metadata,
)
from ag_ui.encoder import EventEncoder

logger = logging.getLogger(__name__)

# One legacy frame as the runner publishes it, parsed from its ``data:`` line.
LegacyFrame = Mapping[str, Any]
# An AG-UI event model (``ag_ui.core``).
AguiEvent = BaseEvent

PROTOCOL_VERSION = "1.0"
STEP_THINKING = "thinking"
ACTIVITY_SUMMARIZING = "eve.summarizing"
CUSTOM_SOURCES = "eve.sources"
CUSTOM_ARTIFACTS = "eve.artifacts"
CUSTOM_NOTICE = "eve.notice"
# Inline citations (sources standard, section 3) ride the assistant message:
# the marker list on TEXT_MESSAGE_CONTENT when one closes, and the final lists
# on the last TEXT_MESSAGE_END. When the text message was already closed (a
# tool call after the last text) they ride RUN_FINISHED instead.
META_CITATIONS = "eve.citations"
META_SOURCES_RETRIEVED = "eve.sourcesRetrieved"
NOTICE_INPUT_PART_SKIPPED = "input_part_skipped"
_NOTICE_MESSAGES = {
    NOTICE_INPUT_PART_SKIPPED: "Some attachments could not be used and were left out.",
}

# RUN_ERROR copy per code (contract 1.7). The message is fixed user copy, never
# the exception text the legacy frame carries.
RUN_ERROR_MESSAGES: Dict[str, str] = {
    "timeout": "The model did not answer in time. Retry in a moment.",
    "empty_answer": "The model returned an empty answer. Retry.",
    "upstream_error": "Something went wrong while answering. Retry.",
    "stream_incomplete": "The answer stream was interrupted. Reload the conversation.",
}
_FALLBACK_ERROR_CODE = "upstream_error"

_TOOL_STATUSES = frozenset({"ok", "error", "timeout", "denied"})

# Tool arguments and results travel as previews (contract D4) and must stay
# valid JSON for a standard client: long strings are shortened inside the value
# with TRUNCATION_MARK, long arrays keep their head plus one count string, and a
# result the runner cut mid-document is closed at its last complete value.
# The string bound leaves room for the runner's own cut (500 chars plus the mark).
TRUNCATION_MARK = "…"
ARG_STRING_MAX_CHARS = 500
RESULT_STRING_MAX_CHARS = 500
PREVIEW_ARRAY_MAX_ITEMS = 20
PREVIEW_OBJECT_MAX_KEYS = 50
PREVIEW_MAX_DEPTH = 8
RESULT_TEXT_MAX_CHARS = 2000
# The runner sends the first 200 chars of every tool result (runner.py,
# ``tool_result`` frame): a result of that length or more was cut there.
RUNNER_RESULT_PREVIEW_CHARS = 200
# A cut JSON result longer than this is cut here before it is closed, so the
# close stays bounded whatever the runner sends.
RESULT_CLOSE_MAX_CHARS = 20_000
# Candidates tried when closing a cut document: the first one parses unless the
# document had an invalid escape, and then the earlier ones fail too.
_CLOSE_MAX_ATTEMPTS = 4
# A UTF-16 half with no partner: Python keeps a pair as one code point, so any
# char in this range is lone, and UTF-8 cannot encode it.
_LONE_SURROGATE = re.compile("[\ud800-\udfff]")

# Assistant message id of the n-th model call of the run, from the second on;
# the first keeps the answer messageId (decision 0008).
_SEGMENT_ID = "{message_id}:assistant:{n}"

# HTTP errors answered instead of the stream (contract 1.6).
_PRE_STREAM_ERRORS: Dict[str, tuple] = {
    "thread_mismatch": (400, "The thread id does not match the conversation."),
    "protocol_version_unsupported": (400, "This protocol version is not supported."),
    "no_user_message": (400, "The request has no user message."),
    "agent_not_allowed": (400, "This agent is not available."),
    "collection_forbidden": (403, "One of the selected collections is not available to you."),
    "not_found": (404, "Conversation not found."),
    "artifact_not_found": (404, "An attached file was not found."),
    "custom_model_not_found": (404, "The selected custom model was not found."),
    "custom_model_unusable": (400, "The selected custom model has no stored credentials."),
    "run_id_conflict": (409, "This run id was already used in this conversation."),
    "interrupt_uncovered": (409, "The resume does not answer every pending approval."),
    "interrupt_expired": (409, "The pending approval has expired."),
    "rate_limited": (429, "Too many requests. Retry after a short wait."),
    "overloaded": (429, "The service is busy. Retry in a few seconds."),
    "budget_exhausted": (429, "Your monthly token budget is spent."),
    "limiter_unavailable": (503, "The service is temporarily unavailable. Retry in a moment."),
    "internal_error": (500, "Something went wrong. Retry."),
}
_DEFAULT_RETRY_AFTER = {"overloaded": 10}

_ENCODER = EventEncoder()


@dataclass(frozen=True)
class RunContext:
    """Ids and labels of one run, known to the route before the first frame."""

    thread_id: str
    run_id: str
    message_id: str
    trace_id: Optional[str] = None
    graph: Optional[str] = None
    # Input parts the route skipped (contract 1.4): their ids, None for a part
    # without one. Reported once, right after RUN_STARTED.
    skipped_parts: Tuple[Optional[str], ...] = ()


@dataclass(frozen=True)
class PreStreamError:
    """An HTTP error answered instead of opening the event stream."""

    status_code: int
    body: Dict[str, Any]
    headers: Dict[str, str] = field(default_factory=dict)


def pre_stream_error(
    code: str,
    *,
    retry_after: Optional[int] = None,
) -> PreStreamError:
    """HTTP status, ``{"detail": {"code", "message"}}`` body and headers for ``code``.

    An unknown code is answered as ``internal_error``: the caller never leaks a
    code the contract does not list.
    """
    if code not in _PRE_STREAM_ERRORS:
        code = "internal_error"
    status_code, message = _PRE_STREAM_ERRORS[code]
    if retry_after is None:
        retry_after = _DEFAULT_RETRY_AFTER.get(code)
    headers: Dict[str, str] = {}
    if retry_after is not None and not isinstance(retry_after, bool) and retry_after >= 0:
        headers["Retry-After"] = str(int(retry_after))
    return PreStreamError(
        status_code=status_code,
        body={"detail": {"code": code, "message": message}},
        headers=headers,
    )


def encode_sse(events: Iterable[AguiEvent]) -> str:
    """The SSE body for ``events``: one ``data:`` line per event, LF only.

    Never raises: an event that does not serialize (a lone UTF-16 half in a
    tool result or a token) goes out with the offending chars replaced by
    U+FFFD, so the stream and the run stay in step.
    """
    return "".join(_encode_event(event) for event in events)


def _scrub(value: Any) -> Any:
    """``value`` with every lone surrogate replaced by U+FFFD, recursively."""
    if isinstance(value, str):
        return _LONE_SURROGATE.sub("\ufffd", value)
    if isinstance(value, Mapping):
        return {_scrub(k): _scrub(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(v) for v in value]
    return value


def _encode_event(event: AguiEvent) -> str:
    try:
        return _ENCODER.encode(event)
    except Exception as exc:
        logger.warning(
            "agui.encode_fallback event_type=%s error_type=%s",
            getattr(event, "type", "?"),
            type(exc).__name__,
        )
    try:
        data = _scrub(event.model_dump(mode="python", by_alias=True))
    except Exception as exc:
        logger.error("agui.encode_failed error_type=%s", type(exc).__name__)
        return ": encode_failed\n\n"
    try:
        return _ENCODER.encode(type(event).model_validate(data))
    except Exception:
        return "data: " + json.dumps(
            data, ensure_ascii=False, separators=(",", ":"), default=str
        ) + "\n\n"


def normalize_error_code(code: Any) -> str:
    """A code from the 1.7 table; anything else is ``upstream_error``."""
    return code if isinstance(code, str) and code in RUN_ERROR_MESSAGES else _FALLBACK_ERROR_CODE


def _str_or_none(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def _bound_preview(
    value: Any, *, max_chars: int, depth: int = 0
) -> Tuple[Any, bool]:
    """``value`` with long strings, arrays and objects cut; and whether anything was.

    A string is cut inside its value and ends with ``TRUNCATION_MARK``; an array
    keeps ``PREVIEW_ARRAY_MAX_ITEMS`` items plus a ``"… N more"`` string; an
    object keeps ``PREVIEW_OBJECT_MAX_KEYS`` keys plus a ``"…"`` key with the
    count. The result is always JSON-serialisable.
    """
    if isinstance(value, str):
        # A lone UTF-16 half (a pair the runner's cut split) cannot be encoded.
        value = _LONE_SURROGATE.sub("", value)
        # One char of slack: a string the runner already cut (max plus the mark)
        # is not cut again.
        if len(value) <= max_chars + len(TRUNCATION_MARK):
            return value, False
        return value[:max_chars] + TRUNCATION_MARK, True
    if isinstance(value, float) and not math.isfinite(value):
        return str(value), False  # NaN and infinity are not JSON
    if value is None or isinstance(value, (bool, int, float)):
        return value, False
    if depth >= PREVIEW_MAX_DEPTH:
        return TRUNCATION_MARK, True
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        cut = False
        items = list(value.items())
        for key, item in items[:PREVIEW_OBJECT_MAX_KEYS]:
            out[_LONE_SURROGATE.sub("", str(key))], item_cut = _bound_preview(item, max_chars=max_chars, depth=depth + 1)
            cut = cut or item_cut
        if len(items) > PREVIEW_OBJECT_MAX_KEYS:
            out[TRUNCATION_MARK] = f"{len(items) - PREVIEW_OBJECT_MAX_KEYS} more keys"
            cut = True
        return out, cut
    if isinstance(value, (list, tuple)):
        head: List[Any] = []
        cut = False
        for item in list(value)[:PREVIEW_ARRAY_MAX_ITEMS]:
            bounded, item_cut = _bound_preview(item, max_chars=max_chars, depth=depth + 1)
            head.append(bounded)
            cut = cut or item_cut
        if len(value) > PREVIEW_ARRAY_MAX_ITEMS:
            head.append(f"{TRUNCATION_MARK} {len(value) - PREVIEW_ARRAY_MAX_ITEMS} more")
            cut = True
        return head, cut
    return _bound_preview(str(value), max_chars=max_chars, depth=depth)


# A complete JSON scalar, and the start of one that may still go on.
_JSON_SCALAR = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?|true|false|null")
_JSON_SCALAR_PREFIX = re.compile(
    r"-?(?:0|[1-9][0-9]*)?(?:\.[0-9]*)?(?:[eE][+-]?[0-9]*)?|t(?:r(?:ue?)?)?|f(?:a(?:l(?:se?)?)?)?|n(?:u(?:ll?)?)?"
)


# The rest of a JSON string body: up to its closing quote, or to the end of a
# cut document (a lone backslash at the very end is left out).
_STRING_BODY = re.compile(r'(?:[^"\\]|\\.)*', re.S)
# A scalar runs to the next separator or space.
_SCALAR_RUN = re.compile(r"[^,\]}\s]*")
_SPACES = re.compile(r"\s*")
_PARTIAL_UNICODE_ESCAPE = re.compile(r"\\u[0-9a-fA-F]{0,3}\Z")
_HIGH_SURROGATE_ESCAPE = re.compile(r"\\u[dD][89abAB][0-9a-fA-F]{2}\Z")


def _is_escape_start(body: str, at: int) -> bool:
    """True when the backslash at ``at`` starts an escape (is not itself escaped)."""
    k = at
    while k > 0 and body[k - 1] == "\\":
        k -= 1
    return (at - k) % 2 == 0


def _complete_escapes(body: str) -> str:
    """``body`` (the inside of a JSON string) cut back to a code point boundary.

    Drops a trailing partial escape (``\\``, ``\\u00``) and then a trailing
    high-surrogate escape whose low half was cut off: json.loads would turn it
    into a lone surrogate that UTF-8 cannot encode.
    """
    trailing = len(body) - len(body.rstrip("\\"))
    if trailing % 2:
        body = body[:-1]
    match = _PARTIAL_UNICODE_ESCAPE.search(body)
    if match and _is_escape_start(body, match.start()):
        body = body[: match.start()]
    match = _HIGH_SURROGATE_ESCAPE.search(body)
    if match and _is_escape_start(body, match.start()):
        body = body[: match.start()]
    return body


def _closers(node: Optional[Tuple[str, Any]]) -> str:
    out: List[str] = []
    while node is not None:
        out.append("}" if node[0] == "{" else "]")
        node = node[1]
    return "".join(out)


def close_json_prefix(text: str) -> Optional[str]:
    """A JSON document cut at an arbitrary char, closed as valid JSON; None if not.

    ``text`` must be a prefix of a JSON document: anything that breaks the JSON
    grammar before its end (prose in brackets, words after the document) gives
    None. A string value cut inside keeps its text plus ``TRUNCATION_MARK``; a
    cut key, a key without its value, or a scalar that may go on (``tru``,
    ``12``) is dropped back to the last complete value; open arrays and objects
    are closed.

    Linear in ``len(text)``: string bodies, scalars and spaces are skipped with
    one regex match each, a cut point keeps a reference to the (shared, linked)
    stack of open containers instead of a copy, and only the candidates tried
    are built.
    """
    # Each stack entry: [container, expected]; expected is what may come next:
    # "key" (or the close), "colon", "value" (or the close) or "comma".
    stack: List[List[str]] = []
    # The open containers again, as a linked list (kind, parent): a cut point
    # keeps the node of its moment, so closing it never copies the stack.
    node: Optional[Tuple[str, Any]] = None
    cuts: List[Tuple[int, Optional[Tuple[str, Any]]]] = []
    open_string: Optional[int] = None
    string_is_key = False
    scalar_tail: Optional[int] = None
    started = False

    def value_done(end: int) -> None:
        if stack:
            stack[-1][1] = "comma"
            cuts.append((end, node))

    def value_allowed() -> bool:
        return (not stack and not started) or (bool(stack) and stack[-1][1] == "value")

    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i = _SPACES.match(text, i).end()
            continue
        if c == '"':
            is_key = bool(stack) and stack[-1][0] == "{" and stack[-1][1] == "key"
            if not is_key and not value_allowed():
                return None
            started = True
            end = _STRING_BODY.match(text, i + 1).end()
            if end >= n or text[end] != '"':
                open_string, string_is_key = i, is_key
                break
            if is_key:
                stack[-1][1] = "colon"
            else:
                value_done(end + 1)
            i = end + 1
            continue
        if c in "{[":
            if not value_allowed():
                return None
            started = True
            stack.append([c, "key" if c == "{" else "value"])
            node = (c, node)
            cuts.append((i + 1, node))
        elif c in "}]":
            want = "{" if c == "}" else "["
            if not stack or stack[-1][0] != want or stack[-1][1] not in ("key", "value", "comma"):
                return None
            stack.pop()
            node = node[1] if node is not None else None
            if not stack:
                if text[i + 1 :].strip():
                    return None  # words after a complete document
                cuts.append((i + 1, None))
                break
            value_done(i + 1)
        elif c == ",":
            if not stack or stack[-1][1] != "comma":
                return None
            stack[-1][1] = "key" if stack[-1][0] == "{" else "value"
        elif c == ":":
            if not stack or stack[-1][1] != "colon":
                return None
            stack[-1][1] = "value"
        else:
            if not value_allowed():
                return None
            started = True
            end = _SCALAR_RUN.match(text, i).end()
            if end >= n:
                scalar_tail = i
                break
            if not _JSON_SCALAR.fullmatch(text[i:end]):
                return None
            value_done(end)
            if not stack:
                return None  # a bare scalar followed by more text
            i = end
            continue
        i += 1

    if scalar_tail is not None and not _JSON_SCALAR_PREFIX.fullmatch(text[scalar_tail:]):
        return None

    def candidates() -> Iterable[str]:
        if open_string is not None and not string_is_key:
            body = _complete_escapes(text[open_string + 1 :])
            yield text[: open_string + 1] + body + TRUNCATION_MARK + '"' + _closers(node)
        for end, at in reversed(cuts):
            yield text[:end] + _closers(at)

    for attempt, candidate in enumerate(candidates()):
        if attempt >= _CLOSE_MAX_ATTEMPTS:
            break
        try:
            json.loads(candidate)
        except ValueError:
            continue
        return candidate
    return None


def _preview_args(args: Mapping[str, Any]) -> Tuple[str, bool]:
    """``TOOL_CALL_ARGS`` text for ``args`` and whether the adapter cut it."""
    bounded, cut = _bound_preview(dict(args), max_chars=ARG_STRING_MAX_CHARS)
    return json.dumps(bounded, ensure_ascii=False, default=str), cut


def _preview_result(content: Any, runner_cut: bool = False) -> Tuple[str, bool]:
    """``TOOL_CALL_RESULT.content`` and whether it is cut.

    JSON stays JSON: a complete document is sent as is unless it is too long; a
    document the runner cut (its 200-char preview) is closed at its last
    complete value. Any other text is sent as text, without lone UTF-16
    halves. ``runner_cut``: the runner already cut the content, so it is
    flagged whatever happens here.
    """
    if content is None:
        return "", runner_cut
    if not isinstance(content, str):
        bounded, cut = _bound_preview(content, max_chars=RESULT_STRING_MAX_CHARS)
        return json.dumps(bounded, ensure_ascii=False, default=str), cut or runner_cut
    head = content.lstrip()[:1]
    if head in ("{", "["):
        try:
            parsed = json.loads(content)
        except ValueError:
            closed = close_json_prefix(content[:RESULT_CLOSE_MAX_CHARS])
            if closed is not None:
                bounded, _ = _bound_preview(json.loads(closed), max_chars=RESULT_STRING_MAX_CHARS)
                return json.dumps(bounded, ensure_ascii=False), True
        else:
            bounded, cut = _bound_preview(parsed, max_chars=RESULT_STRING_MAX_CHARS)
            if not cut:
                return content, runner_cut
            return json.dumps(bounded, ensure_ascii=False), True
    content = _LONE_SURROGATE.sub("", content)
    if len(content) > RESULT_TEXT_MAX_CHARS:
        return content[:RESULT_TEXT_MAX_CHARS] + TRUNCATION_MARK, True
    return content, runner_cut


def _frame_call_id(frame: LegacyFrame) -> Optional[str]:
    """The runner's run-unique ``tool_call_id`` (B-S2), else the raw model id."""
    return _str_or_none(frame.get("tool_call_id")) or _str_or_none(frame.get("id"))


def _usage(rows: Any) -> Optional[List[TokenUsage]]:
    """``RUN_FINISHED.usage`` from ``final.usage``.

    Each row is the LangChain ``usage_metadata`` of one model call (the dict the
    runner already keeps per trace entry, runner.py:192) plus ``provider`` and
    ``model`` labels. Rows are summed per provider and model; a count no row
    reported stays unset, never zero.
    """
    if not isinstance(rows, list):
        return None
    entries: List[TokenUsage] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        entry = token_usage_from_langchain_metadata(
            row,
            provider=_str_or_none(row.get("provider")),
            model=_str_or_none(row.get("model")),
        )
        if entry is not None:
            entries.append(entry)
    return aggregate_token_usage(entries) or None


class AguiAdapter:
    """Stateful adapter for the frames of exactly one run."""

    def __init__(self, context: RunContext) -> None:
        self.context = context
        self._started = False
        self._terminated = False
        self._thinking_open = False
        self._text_open = False
        self._text_streamed = False
        self._tool_result_seen = False
        # Calls started and not ended (calls end on their own frame, so this is
        # empty in practice), and calls ended with no result yet, by minted id.
        self._open_tools: List[str] = []
        self._pending_results: List[str] = []
        # Raw legacy call id (None included) to the minted ids still waiting for
        # their result, oldest first, and to the last minted id seen.
        self._pending_by_raw: Dict[Optional[str], Deque[str]] = {}
        self._last_by_raw: Dict[Optional[str], str] = {}
        self._used_tool_ids: set = set()
        self._tool_count = 0
        self._summary_count = 0
        self._summary_open: Optional[ActivitySnapshotEvent] = None
        self._emitted = 0
        self._last_started: Optional[str] = None
        # One assistant message per model call (decision 0008): the n-th call's
        # text and tool calls share its id. A call ends where the tools ran: the
        # first text or tool call after a tool result opens the next one.
        self._segment = 1
        self._segment_used = False
        self._result_since_segment = False
        self._text_message_id = context.message_id
        # Text message ids already ended: a message is never reopened.
        self._closed_text_ids: set = set()

    # ── public interface ────────────────────────────────────────────────────

    @property
    def terminated(self) -> bool:
        """True once a terminal event (``RUN_FINISHED`` or ``RUN_ERROR``) was returned."""
        return self._terminated

    def start(self) -> List[AguiEvent]:
        """``RUN_STARTED`` with the run ids and the ``eve.*`` metadata."""
        if self._started:
            return []
        self._started = True
        ctx = self.context
        metadata: Dict[str, Any] = {}
        if ctx.trace_id:
            metadata["eve.traceId"] = ctx.trace_id
        metadata["eve.conversationId"] = ctx.thread_id
        metadata["eve.messageId"] = ctx.message_id
        if ctx.graph:
            metadata["eve.graph"] = ctx.graph
        events: List[AguiEvent] = [
            RunStartedEvent(
                thread_id=ctx.thread_id,
                run_id=ctx.run_id,
                protocol_version=PROTOCOL_VERSION,
                metadata=metadata,
            )
        ]
        if ctx.skipped_parts:
            events.append(
                CustomEvent(
                    name=CUSTOM_NOTICE,
                    value={
                        "code": NOTICE_INPUT_PART_SKIPPED,
                        "message": _NOTICE_MESSAGES[NOTICE_INPUT_PART_SKIPPED],
                        "partIds": [p for p in ctx.skipped_parts if p],
                        "count": len(ctx.skipped_parts),
                    },
                )
            )
        return self._out(events)

    def adapt(self, frame: LegacyFrame) -> List[AguiEvent]:
        """Events for one legacy frame; empty after the terminal event."""
        if self._terminated or not isinstance(frame, Mapping):
            return []
        events: List[AguiEvent] = self.start()
        kind = frame.get("type")
        handler = self._HANDLERS.get(kind) if isinstance(kind, str) else None
        if handler is None:
            # `partial`, and frame types of later ticks (subagent, todos,
            # interrupt), are not emitted yet.
            logger.debug("agui.frame_dropped type=%s", kind if isinstance(kind, str) else type(kind).__name__)
            return self._out(events)
        events.extend(handler(self, frame))
        return self._out(events)

    def finish(self) -> List[AguiEvent]:
        """Events for the bus closing: ``RUN_ERROR stream_incomplete`` if no terminal frame came."""
        if self._terminated:
            return []
        events: List[AguiEvent] = self.start()
        events.extend(self._run_error("stream_incomplete"))
        return self._out(events)

    # ── frame handlers ──────────────────────────────────────────────────────

    def _on_status(self, frame: LegacyFrame) -> List[AguiEvent]:
        return self._open_thinking()

    def _on_token(self, frame: LegacyFrame) -> List[AguiEvent]:
        content = frame.get("content")
        if not isinstance(content, str) or not content:
            return []
        events = self._close_thinking()
        citations = frame.get("citations")
        metadata = {META_CITATIONS: citations} if isinstance(citations, list) else None
        events.extend(self._text_delta(content, metadata))
        return events

    def _on_tool_call(self, frame: LegacyFrame) -> List[AguiEvent]:
        events = self._close_thinking()
        events.extend(self._close_text())
        name = _str_or_none(frame.get("tool")) or "tool"
        raw_id = _frame_call_id(frame)
        tool_call_id = self._mint_tool_call_id(raw_id, name)
        self._last_started = tool_call_id
        metadata: Dict[str, Any] = {}
        server = _str_or_none(frame.get("server"))
        if server:
            metadata["eve.server"] = server
        label = _str_or_none(frame.get("label"))
        if label:
            metadata["eve.label"] = label
        args = frame.get("args")
        if not isinstance(args, Mapping):
            args = {}
        delta, cut = _preview_args(args)
        if cut or frame.get("args_truncated") is True:
            metadata["eve.argsTruncated"] = True
        events.extend(
            self._start_tool_call(
                tool_call_id, name, metadata or None, self._segment_message_id()
            )
        )
        events.append(ToolCallArgsEvent(tool_call_id=tool_call_id, delta=delta))
        events.extend(self._end_tool_call(tool_call_id))
        self._pending_by_raw.setdefault(raw_id, deque()).append(tool_call_id)
        return events

    def _on_tool_result(self, frame: LegacyFrame) -> List[AguiEvent]:
        # The model call that asked for the tools is over: its text ends here.
        events: List[AguiEvent] = self._close_text()
        raw_id = _frame_call_id(frame)
        pending = self._pending_by_raw.get(raw_id)
        if pending:
            tool_call_id = pending.popleft()
        else:
            # A result for a call the stream never announced: open and close the
            # call first, so the result names a started call (contract 5,
            # result_ref).
            name = _str_or_none(frame.get("tool")) or "tool"
            tool_call_id = self._mint_tool_call_id(raw_id, name)
            events.extend(self._close_thinking())
            events.extend(
                self._start_tool_call(tool_call_id, name, None, self._current_message_id())
            )
            events.extend(self._end_tool_call(tool_call_id))
            logger.debug("agui.result_without_call tool_call_id=%s", tool_call_id)
        if tool_call_id in self._pending_results:
            self._pending_results.remove(tool_call_id)
        self._last_by_raw[raw_id] = tool_call_id
        self._tool_result_seen = True
        self._result_since_segment = True

        status = frame.get("status")
        if not isinstance(status, str) or not status:
            status = "ok"
        elif status not in _TOOL_STATUSES:
            status = "error"
        metadata: Dict[str, Any] = {"eve.status": status}
        server = _str_or_none(frame.get("server"))
        if server:
            metadata["eve.server"] = server
        duration = frame.get("duration_ms")
        if isinstance(duration, int) and not isinstance(duration, bool) and duration >= 0:
            metadata["eve.durationMs"] = duration
        raw_content = frame.get("content")
        runner_cut = frame.get("content_truncated") is True or (
            isinstance(raw_content, str) and len(raw_content) >= RUNNER_RESULT_PREVIEW_CHARS
        )
        content, cut = _preview_result(raw_content, runner_cut)
        if cut:
            metadata["eve.contentTruncated"] = True
        events.append(
            ToolCallResultEvent(
                message_id=self._tool_result_message_id(tool_call_id),
                tool_call_id=tool_call_id,
                role="tool",
                content=content,
                metadata=metadata,
            )
        )
        events.extend(self._open_thinking())
        return events

    def _on_sources(self, frame: LegacyFrame) -> List[AguiEvent]:
        documents = frame.get("documents")
        if not isinstance(documents, list):
            return []
        raw_id = _str_or_none(frame.get("tool_call_id"))
        sources = frame.get("sources")
        tool_call_id = self._last_by_raw.get(raw_id)
        if tool_call_id is None:
            pending = self._pending_by_raw.get(raw_id)
            tool_call_id = pending[0] if pending else None
        if tool_call_id is None:
            # An id that names no started call: the last started call, since
            # sources follow their call. None started: dropped (contract 2.3
            # requires a toolCallId of a started call).
            tool_call_id = self._last_started
        if tool_call_id is None:
            logger.debug("agui.sources_dropped documents=%d", len(documents))
            return []
        value: Dict[str, Any] = {"toolCallId": tool_call_id, "documents": documents}
        if isinstance(sources, list):
            # Sources standard section 3: the descriptors beside the legacy list.
            value["sources"] = sources
        return [CustomEvent(name=CUSTOM_SOURCES, value=value)]

    def _on_summarizing(self, frame: LegacyFrame) -> List[AguiEvent]:
        phase = frame.get("phase")
        if phase not in ("started", "finished"):
            return []
        if phase == "started" or self._summary_open is None:
            if self._summary_open is not None:
                # A second start while one runs: the first ends here.
                events = self._close_summary()
            else:
                events = []
            self._summary_count += 1
        else:
            events = []
        content: Dict[str, Any] = {"phase": phase}
        first = _str_or_none(frame.get("first_message_id"))
        last = _str_or_none(frame.get("last_message_id"))
        if first:
            content["firstMessageId"] = first
        if last:
            content["lastMessageId"] = last
        snapshot = ActivitySnapshotEvent(
            message_id=f"{self.context.message_id}:summarizing:{self._summary_count}",
            activity_type=ACTIVITY_SUMMARIZING,
            content=content,
        )
        self._summary_open = snapshot if phase == "started" else None
        events.append(snapshot)
        return events

    def _citation_meta(self, frame: LegacyFrame) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """``(on the text end, on RUN_FINISHED)``: the citation keys go to the first
        while the text message is open, else to the second."""
        meta: Dict[str, Any] = {}
        if isinstance(frame.get("citations"), list):
            meta[META_CITATIONS] = frame["citations"]
        if isinstance(frame.get("sources_retrieved"), list):
            meta[META_SOURCES_RETRIEVED] = frame["sources_retrieved"]
        if self._text_open:
            return meta, {}
        return {}, meta

    def _on_final(self, frame: LegacyFrame) -> List[AguiEvent]:
        answer = frame.get("answer")
        answer = answer if isinstance(answer, str) else ""
        endpoint = frame.get("endpoint")
        endpoint = endpoint if isinstance(endpoint, Mapping) else None
        error = frame.get("error")
        error_code = (
            normalize_error_code(error.get("code")) if isinstance(error, Mapping) else None
        )
        usage = _usage(frame.get("usage"))

        events = self._close_thinking()
        if not self._text_streamed and answer:
            # A text-format turn flushed at the end, or a final with no token
            # frame before it: the answer still travels as standard text.
            events.extend(self._text_delta(answer))
        text_meta, run_meta = self._citation_meta(frame)
        events.extend(self._close_text(endpoint, text_meta))

        artifact_ids = frame.get("artifact_ids")
        if isinstance(artifact_ids, list):
            ids = [a for a in artifact_ids if isinstance(a, str) and a]
            if ids:
                events.append(CustomEvent(name=CUSTOM_ARTIFACTS, value={"artifactIds": ids}))

        if error_code is not None:
            # Decision 5: a failure after streamed text rides on `final`.
            events.extend(self._run_error(error_code, usage=usage))
            return events
        if not self._text_streamed and not self._tool_result_seen:
            # Decision 6: nothing to show is not a success.
            events.extend(self._run_error("empty_answer", usage=usage))
            return events

        events.extend(self._closing_sequence(success=True))
        metadata: Dict[str, Any] = {}
        latencies = frame.get("latencies")
        if isinstance(latencies, Mapping):
            metadata["eve.latencies"] = dict(latencies)
        metadata.update(run_meta)
        events.append(
            RunFinishedEvent(
                thread_id=self.context.thread_id,
                run_id=self.context.run_id,
                outcome=RunFinishedSuccessOutcome(),
                usage=usage,
                metadata=metadata or None,
            )
        )
        self._terminate("success")
        return events

    def _on_stopped(self, frame: LegacyFrame) -> List[AguiEvent]:
        text_meta, run_meta = self._citation_meta(frame)
        events = self._closing_sequence(success=False, text_meta=text_meta)
        events.append(
            RunFinishedEvent(
                thread_id=self.context.thread_id,
                run_id=self.context.run_id,
                outcome=RunFinishedCancelledOutcome(),
                metadata={"eve.stopped": True, **run_meta},
            )
        )
        self._terminate("cancelled")
        return events

    def _on_error(self, frame: LegacyFrame) -> List[AguiEvent]:
        return self._run_error(normalize_error_code(frame.get("code")))

    _HANDLERS = {
        "status": _on_status,
        "token": _on_token,
        "tool_call": _on_tool_call,
        "tool_result": _on_tool_result,
        "sources": _on_sources,
        "summarizing": _on_summarizing,
        "final": _on_final,
        "stopped": _on_stopped,
        "error": _on_error,
    }

    # ── building blocks ─────────────────────────────────────────────────────

    def _out(self, events: List[AguiEvent]) -> List[AguiEvent]:
        self._emitted += len(events)
        return events

    def _terminate(self, outcome: str, code: Optional[str] = None) -> None:
        self._terminated = True
        logger.debug(
            "agui.terminal outcome=%s code=%s events=%d tool_calls=%d",
            outcome,
            code or "-",
            self._emitted,
            self._tool_count,
        )

    def _run_error(self, code: str, *, usage: Optional[List[TokenUsage]] = None) -> List[AguiEvent]:
        code = normalize_error_code(code)
        events = self._closing_sequence(success=False)
        events.append(
            RunErrorEvent(
                message=RUN_ERROR_MESSAGES[code],
                code=code,
                usage=usage,
                metadata={"eve.error.code": code},
            )
        )
        self._terminate("error", code)
        return events

    def _closing_sequence(
        self, *, success: bool, text_meta: Optional[Mapping[str, Any]] = None
    ) -> List[AguiEvent]:
        """Contract 2.4: close what is open before the terminal event."""
        events: List[AguiEvent] = []
        for tool_call_id in list(self._open_tools):
            events.extend(self._end_tool_call(tool_call_id))
        if success:
            for tool_call_id in list(self._pending_results):
                events.append(
                    ToolCallResultEvent(
                        message_id=self._tool_result_message_id(tool_call_id),
                        tool_call_id=tool_call_id,
                        role="tool",
                        content="",
                        metadata={"eve.status": "not_run"},
                    )
                )
            self._pending_results.clear()
        events.extend(self._close_text(None, text_meta))
        events.extend(self._close_thinking())
        events.extend(self._close_summary())
        return events

    def _open_thinking(self) -> List[AguiEvent]:
        if self._thinking_open:
            return []
        self._thinking_open = True
        return [StepStartedEvent(step_name=STEP_THINKING)]

    def _close_thinking(self) -> List[AguiEvent]:
        if not self._thinking_open:
            return []
        self._thinking_open = False
        return [StepFinishedEvent(step_name=STEP_THINKING)]

    def _text_delta(
        self, delta: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[AguiEvent]:
        events: List[AguiEvent] = []
        if not self._text_open:
            self._text_open = True
            message_id = self._segment_message_id()
            if message_id in self._closed_text_ids:
                # Text after a tool call inside one model call (the text-format
                # flush): its message ended at the call, so the text goes on
                # under the next assistant message id.
                self._segment += 1
                message_id = self._current_message_id()
            self._text_message_id = message_id
            events.append(TextMessageStartEvent(message_id=self._text_message_id, role="assistant"))
        self._text_streamed = True
        events.append(
            TextMessageContentEvent(
                message_id=self._text_message_id, delta=delta, metadata=metadata
            )
        )
        return events

    def _close_text(
        self,
        endpoint: Optional[Mapping[str, Any]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> List[AguiEvent]:
        if not self._text_open:
            return []
        self._text_open = False
        self._closed_text_ids.add(self._text_message_id)
        metadata: Dict[str, Any] = {}
        if endpoint is not None:
            metadata["eve.endpoint"] = dict(endpoint)
        if extra:
            metadata.update(extra)
        return [TextMessageEndEvent(message_id=self._text_message_id, metadata=metadata or None)]

    def _close_summary(self) -> List[AguiEvent]:
        if self._summary_open is None:
            return []
        opened = self._summary_open
        self._summary_open = None
        return [
            ActivitySnapshotEvent(
                message_id=opened.message_id,
                activity_type=opened.activity_type,
                content={**opened.content, "phase": "finished"},
            )
        ]

    def _mint_tool_call_id(self, raw_id: Optional[str], name: str) -> str:
        """Contract 2.1: the model id, suffixed ``~2``, ``~3`` when null or reused."""
        self._tool_count += 1
        base = raw_id or f"call_{self._tool_count}_{name}"
        candidate = base
        n = 1
        while candidate in self._used_tool_ids:
            n += 1
            candidate = f"{base}~{n}"
        self._used_tool_ids.add(candidate)
        return candidate

    def _current_message_id(self) -> str:
        """The assistant message id of the model call in progress."""
        self._segment_used = True
        if self._segment == 1:
            return self.context.message_id
        return _SEGMENT_ID.format(message_id=self.context.message_id, n=self._segment)

    def _segment_message_id(self) -> str:
        """The id for new text or a new tool call: a new model call after a tool result."""
        if self._result_since_segment and self._segment_used:
            self._segment += 1
            self._segment_used = False
        self._result_since_segment = False
        return self._current_message_id()

    def _start_tool_call(
        self,
        tool_call_id: str,
        name: str,
        metadata: Optional[Dict[str, Any]],
        parent_message_id: str,
    ) -> List[AguiEvent]:
        self._open_tools.append(tool_call_id)
        return [
            ToolCallStartEvent(
                tool_call_id=tool_call_id,
                tool_call_name=name,
                parent_message_id=parent_message_id,
                metadata=metadata,
            )
        ]

    def _end_tool_call(self, tool_call_id: str) -> List[AguiEvent]:
        if tool_call_id not in self._open_tools:
            return []
        self._open_tools.remove(tool_call_id)
        self._pending_results.append(tool_call_id)
        return [ToolCallEndEvent(tool_call_id=tool_call_id)]

    def _tool_result_message_id(self, tool_call_id: str) -> str:
        return f"{self.context.message_id}:tool:{tool_call_id}"
