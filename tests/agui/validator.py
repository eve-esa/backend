"""AG-UI transcript validator for the EVE agentic stream.

Reads one raw SSE body, as `POST /conversations/{id}/agui` sends it, and returns
the list of contract violations. An empty list means the transcript conforms.

Three layers, in order:

1. Framing: every non-empty line is a `data:` line or a `:` comment, LF only,
   one JSON object per SSE event (`frame`).
2. Shape: every event validates against the AG-UI 1.0 JSON schema (closed
   objects) and parses with the `ag-ui-protocol` models with no unknown member
   (`schema`). The SDK models keep unknown members on purpose, so the extras
   check is ours.
3. The EVE contract invariants, one code each (`first_event`,
   `terminal_count`, `after_terminal`, `missing_terminal`, `run_ids`,
   `forbidden_type`, `sequence`, `open_at_terminal`, `tool_reopen`,
   `text_reopen`, `tool_parent`, `result_ref`, `pending_on_success`,
   `outcome_cancelled`, `outcome_interrupt`, `run_error`, `custom_name`,
   `activity_type`, `metadata_keys`, `subagent`, `step_name`, `role`).

The schema file next to this module is the AG-UI 1.0 schema with its prose
descriptions removed (they do not take part in validation); see
`schema/README.md`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import jsonschema
from ag_ui.core import Event
from pydantic import BaseModel, TypeAdapter, ValidationError

SCHEMA_PATH = Path(__file__).parent / "schema" / "ag-ui-1.0.schema.json"

PROTOCOL_VERSION = "1.0"

TERMINAL_TYPES = frozenset({"RUN_FINISHED", "RUN_ERROR"})

FORBIDDEN_TYPES = frozenset(
    {"RAW", "STATE_SNAPSHOT", "STATE_DELTA", "MESSAGES_SNAPSHOT", "ACTIVITY_DELTA"}
)

CUSTOM_NAMES = frozenset({"eve.sources", "eve.artifacts", "eve.notice"})

ACTIVITY_TYPES = frozenset({"eve.summarizing", "eve.plan"})

STEP_NAMES = frozenset({"thinking"})

INTERRUPT_REASONS = frozenset({"tool_approval"})

RUN_ERROR_MESSAGES = {
    "timeout": "The model did not answer in time. Retry in a moment.",
    "empty_answer": "The model returned an empty answer. Retry.",
    "upstream_error": "Something went wrong while answering. Retry.",
    "stream_incomplete": "The answer stream was interrupted. Reload the conversation.",
}

TOOL_STATUSES = frozenset({"ok", "error", "timeout", "denied", "not_run"})

GRAPHS = frozenset({"react", "deep"})

RESERVED_METADATA_KEY = "ag-ui"

_TRACE_ID = re.compile(r"^[0-9a-f]{32}$")
_SUMMARIZING_ID = re.compile(r"^(?P<mid>.+):summarizing:(?P<n>[1-9][0-9]*)$")


def _is_str(value: Any) -> bool:
    return isinstance(value, str)


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_obj(value: Any) -> bool:
    return isinstance(value, dict)


def _is_true(value: Any) -> bool:
    return value is True


def _is_graph(value: Any) -> bool:
    return value in GRAPHS


def _is_trace_id(value: Any) -> bool:
    return isinstance(value, str) and bool(_TRACE_ID.match(value))


def _is_tool_status(value: Any) -> bool:
    return value in TOOL_STATUSES


def _is_error_code(value: Any) -> bool:
    return value in RUN_ERROR_MESSAGES


def _is_citation_list(value: Any) -> bool:
    """Sources standard section 3: cited Sources, each with id, n >= 1 and at offsets."""
    return isinstance(value, list) and all(
        isinstance(c, dict)
        and isinstance(c.get("id"), str)
        and _is_int(c.get("n"))
        and c["n"] >= 1
        and isinstance(c.get("at"), list)
        and all(_is_int(a) and a >= 0 for a in c["at"])
        and "retrieval" not in c
        for c in value
    )


def _is_source_list(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(s, dict) and isinstance(s.get("id"), str) and "retrieval" not in s for s in value
    )


# Contract section 3: key -> (events it may appear on, value check, type text).
# `eve.error.code` is allowed on RUN_ERROR only: decision 8.1 turns every
# failure the user did not choose, a late one included, into RUN_ERROR, so the
# "RUN_FINISHED after a late failure" row of section 3 no longer occurs.
EVENT_METADATA_KEYS: dict[str, tuple[frozenset[str], Any, str]] = {
    "eve.traceId": (frozenset({"RUN_STARTED"}), _is_trace_id, "string, 32 hex"),
    "eve.conversationId": (frozenset({"RUN_STARTED"}), _is_str, "string"),
    "eve.messageId": (frozenset({"RUN_STARTED"}), _is_str, "string"),
    "eve.graph": (frozenset({"RUN_STARTED"}), _is_graph, "react or deep"),
    "eve.label": (frozenset({"TOOL_CALL_START"}), _is_str, "string"),
    "eve.server": (frozenset({"TOOL_CALL_START", "TOOL_CALL_RESULT"}), _is_str, "string"),
    "eve.argsTruncated": (frozenset({"TOOL_CALL_START"}), _is_true, "true (absent otherwise)"),
    "eve.status": (frozenset({"TOOL_CALL_RESULT"}), _is_tool_status, "a tool status"),
    "eve.durationMs": (frozenset({"TOOL_CALL_RESULT"}), _is_int, "integer"),
    "eve.contentTruncated": (frozenset({"TOOL_CALL_RESULT"}), _is_bool, "boolean"),
    "eve.endpoint": (frozenset({"TEXT_MESSAGE_END"}), _is_obj, "object"),
    "eve.citations": (
        frozenset({"TEXT_MESSAGE_CONTENT", "TEXT_MESSAGE_END"}),
        _is_citation_list,
        "a list of cited Sources with id, n and at",
    ),
    "eve.sourcesRetrieved": (frozenset({"TEXT_MESSAGE_END"}), _is_source_list, "a list of Sources"),
    "eve.latencies": (frozenset({"RUN_FINISHED"}), _is_obj, "object"),
    "eve.stopped": (frozenset({"RUN_FINISHED"}), _is_bool, "boolean"),
    "eve.error.code": (frozenset({"RUN_ERROR"}), _is_error_code, "a section 1.7 code"),
}

INTERRUPT_METADATA_KEYS: dict[str, tuple[Any, str]] = {
    "eve.tier": (_is_str, "string"),
    "eve.server": (_is_str, "string"),
    "eve.argsPreview": (_is_str, "string"),
}


@dataclass(frozen=True)
class Violation:
    """One contract failure. `line` is the 1-based line of the SSE body."""

    code: str
    line: int | None
    message: str

    def __str__(self) -> str:
        where = f"line {self.line}" if self.line is not None else "transcript"
        return f"[{self.code}] {where}: {self.message}"


@dataclass(frozen=True)
class SseEvent:
    line: int
    data: dict[str, Any]

    @property
    def type(self) -> str | None:
        value = self.data.get("type")
        return value if isinstance(value, str) else None


# ---------------------------------------------------------------------------
# Layer 1: SSE framing
# ---------------------------------------------------------------------------


def parse_sse(body: str) -> tuple[list[SseEvent], list[Violation]]:
    """Split an SSE body into events.

    Each SSE event (lines up to a blank line) must carry exactly one `data:`
    line holding one JSON object: a browser parser joins two adjacent data lines
    into one payload, and drops a last event that no blank line dispatches.
    Comments (`:` lines) are ignored; `event:`, `id:`, `retry:` and any other
    field are framing failures.
    """
    violations: list[Violation] = []
    events: list[SseEvent] = []
    if "\r" in body:
        violations.append(Violation("frame", None, "CR found: the stream must use LF only"))
        body = body.replace("\r\n", "\n").replace("\r", "\n")

    pending: list[tuple[int, str]] = []

    def dispatch() -> None:
        if not pending:
            return
        if len(pending) > 1:
            violations.append(
                Violation(
                    "frame",
                    pending[1][0],
                    f"{len(pending)} data lines in one SSE event: a parser joins them into one payload",
                )
            )
        for lineno, payload in pending:
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError as exc:
                violations.append(Violation("frame", lineno, f"data is not JSON: {exc.msg}"))
                continue
            if not isinstance(obj, dict):
                violations.append(Violation("frame", lineno, "data is not a JSON object"))
                continue
            events.append(SseEvent(lineno, obj))
        pending.clear()

    lines = body.split("\n")
    # A body that ends with "\n" yields a final empty string: not a line.
    if lines and lines[-1] == "":
        lines.pop()
    for lineno, line in enumerate(lines, start=1):
        if line == "":
            dispatch()
        elif line.startswith(":"):
            continue
        elif line.startswith("data:"):
            payload = line[5:]
            if payload.startswith(" "):
                payload = payload[1:]
            pending.append((lineno, payload))
        else:
            name = line.split(":", 1)[0]
            violations.append(Violation("frame", lineno, f"SSE field {name!r} is not allowed"))
    if pending:
        violations.append(
            Violation(
                "frame",
                pending[-1][0],
                "last event is not followed by a blank line: an SSE parser discards it",
            )
        )
        dispatch()
    return events, violations


# ---------------------------------------------------------------------------
# Layer 2: schema and SDK models
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _definition_by_type() -> dict[str, str]:
    schema = _schema()
    defs = schema["$defs"]
    out: dict[str, str] = {}
    for ref in defs["Event"]["oneOf"]:
        name = ref["$ref"].rsplit("/", 1)[-1]
        out[defs[name]["properties"]["type"]["const"]] = name
    return out


@lru_cache(maxsize=None)
def _validator_for(definition: str) -> jsonschema.Draft202012Validator:
    schema = _schema()
    sub = {"$schema": schema["$schema"], "$defs": schema["$defs"], "$ref": f"#/$defs/{definition}"}
    return jsonschema.Draft202012Validator(sub)


@lru_cache(maxsize=1)
def _event_adapter() -> TypeAdapter:
    return TypeAdapter(Event)


def _extra_members(model: Any, path: str = "") -> Iterable[str]:
    if isinstance(model, BaseModel):
        for name in (model.model_extra or {}):
            yield f"{path}{name}"
        for name in type(model).model_fields:
            yield from _extra_members(getattr(model, name), f"{path}{name}.")
    elif isinstance(model, (list, tuple)):
        for i, item in enumerate(model):
            yield from _extra_members(item, f"{path}{i}.")


def check_schema(event: SseEvent) -> list[str]:
    """Return the schema and SDK problems of one event, empty when it is valid."""
    problems: list[str] = []
    definition = _definition_by_type().get(event.type or "")
    if definition is None:
        return [f"unknown event type {event.data.get('type')!r}"]
    for error in sorted(_validator_for(definition).iter_errors(event.data), key=lambda e: list(e.path)):
        where = "/".join(str(p) for p in error.absolute_path) or "(event)"
        problems.append(f"schema {where}: {error.message}")
    try:
        model = _event_adapter().validate_python(event.data)
    except ValidationError as exc:
        for err in exc.errors():
            where = ".".join(str(p) for p in err["loc"])
            problems.append(f"ag-ui-protocol {where}: {err['msg']}")
    else:
        for extra in _extra_members(model):
            problems.append(f"ag-ui-protocol: unknown member {extra}")
    return problems


# ---------------------------------------------------------------------------
# Layer 3: contract invariants
# ---------------------------------------------------------------------------


@dataclass
class _Run:
    thread_id: str | None = None
    run_id: str | None = None
    message_id: str | None = None
    started: bool = False
    # Open items: id -> subagentRunId of the opener (None for the parent agent).
    open_text: dict[str, str | None] = field(default_factory=dict)
    text_roles: dict[str, str] = field(default_factory=dict)
    open_tools: dict[str, int] = field(default_factory=dict)
    tools_started: set[str] = field(default_factory=set)
    tools_ended: set[str] = field(default_factory=set)
    tool_results: set[str] = field(default_factory=set)
    result_message_ids: set[str] = field(default_factory=set)
    open_steps: set[tuple[str | None, str]] = field(default_factory=set)
    subagents_seen: set[str] = field(default_factory=set)
    open_subagents: set[str] = field(default_factory=set)
    last_text_end_line: int | None = None
    endpoint_lines: list[int] = field(default_factory=list)


class _Checker:
    def __init__(
        self,
        events: list[SseEvent],
        thread_id: str | None,
        run_id: str | None,
        require_terminal: bool,
    ) -> None:
        self.events = events
        self.expected_thread = thread_id
        self.expected_run = run_id
        self.require_terminal = require_terminal
        self.run = _Run()
        self.violations: list[Violation] = []

    def fail(self, code: str, event: SseEvent | None, message: str) -> None:
        self.violations.append(Violation(code, event.line if event else None, message))

    # -- driver ------------------------------------------------------------

    def check(self) -> list[Violation]:
        if not self.events:
            if self.require_terminal:
                self.fail("missing_terminal", None, "no events: the run is truncated")
            return self.violations

        first = self.events[0]
        if first.type != "RUN_STARTED":
            self.fail("first_event", first, f"stream begins with {first.type}, not RUN_STARTED")

        terminals = [e for e in self.events if e.type in TERMINAL_TYPES]
        if len(terminals) > 1:
            self.fail(
                "terminal_count",
                terminals[1],
                f"{len(terminals)} terminal events; exactly one is allowed",
            )

        terminal_seen = False
        for event in self.events:
            if terminal_seen:
                self.fail("after_terminal", event, f"{event.type} after the terminal event")
                continue
            self.check_metadata(event)
            if event.type in TERMINAL_TYPES:
                terminal_seen = True
                self.on_terminal(event)
            else:
                self.on_event(event)

        if not terminal_seen and self.require_terminal:
            self.fail("missing_terminal", self.events[-1], "no RUN_FINISHED or RUN_ERROR: the run is truncated")

        self.check_endpoint_placement()
        return self.violations

    # -- per event ---------------------------------------------------------

    def on_event(self, event: SseEvent) -> None:
        etype = event.type or ""
        if etype.endswith("_CHUNK") or etype in FORBIDDEN_TYPES:
            self.fail("forbidden_type", event, f"{etype} is never emitted")
            return
        self.check_attribution(event)
        if etype not in _definition_by_type():
            return  # unknown type: a schema failure, already reported
        handler = getattr(self, f"on_{etype.lower()}", None)
        if handler is not None:
            handler(event)

    def on_run_started(self, event: SseEvent) -> None:
        run = self.run
        if run.started:
            self.fail("first_event", event, "a second RUN_STARTED: one run per response")
            return
        run.started = True
        data = event.data
        run.thread_id = data.get("threadId")
        run.run_id = data.get("runId")
        meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        run.message_id = meta.get("eve.messageId") if isinstance(meta.get("eve.messageId"), str) else None
        if data.get("protocolVersion") != PROTOCOL_VERSION:
            self.fail("first_event", event, f"protocolVersion is {data.get('protocolVersion')!r}, not {PROTOCOL_VERSION!r}")
        if self.expected_thread is not None and run.thread_id != self.expected_thread:
            self.fail("first_event", event, "threadId is not the conversation id")
        if self.expected_run is not None and run.run_id != self.expected_run:
            self.fail("first_event", event, "runId is not the input runId")
        for key in ("eve.traceId", "eve.conversationId", "eve.messageId"):
            if key not in meta:
                self.fail("first_event", event, f"metadata {key} missing")
        if "eve.conversationId" in meta and meta["eve.conversationId"] != run.thread_id:
            self.fail("first_event", event, "eve.conversationId differs from threadId")

    def on_step_started(self, event: SseEvent) -> None:
        name = event.data.get("stepName")
        if name not in STEP_NAMES:
            self.fail("step_name", event, f"stepName {name!r} is not one of {sorted(STEP_NAMES)}")
        key = (event.data.get("subagentRunId"), name)
        if key in self.run.open_steps:
            self.fail("sequence", event, f"step {name!r} started while open")
        self.run.open_steps.add(key)

    def on_step_finished(self, event: SseEvent) -> None:
        name = event.data.get("stepName")
        if name not in STEP_NAMES:
            self.fail("step_name", event, f"stepName {name!r} is not one of {sorted(STEP_NAMES)}")
        key = (event.data.get("subagentRunId"), name)
        if key not in self.run.open_steps:
            self.fail("sequence", event, f"step {name!r} finished but not open")
        self.run.open_steps.discard(key)

    def on_text_message_start(self, event: SseEvent) -> None:
        run = self.run
        mid = event.data.get("messageId")
        role = event.data.get("role", "assistant")
        if role != "assistant":
            self.fail("role", event, f"role {role!r}: only assistant (or absent)")
        if mid in run.open_text:
            self.fail("sequence", event, f"text message {mid} started while open")
            return
        previous = run.text_roles.get(mid)
        if previous is not None and previous != role:
            self.fail("text_reopen", event, f"text message {mid} reopened as {role!r}, opened as {previous!r}")
        run.text_roles[mid] = role
        run.open_text[mid] = event.data.get("subagentRunId")

    def on_text_message_content(self, event: SseEvent) -> None:
        self._continue_text(event, close=False)

    def on_text_message_end(self, event: SseEvent) -> None:
        self._continue_text(event, close=True)
        self.run.last_text_end_line = event.line
        meta = event.data.get("metadata")
        if isinstance(meta, dict) and "eve.endpoint" in meta:
            self.run.endpoint_lines.append(event.line)

    def _continue_text(self, event: SseEvent, close: bool) -> None:
        run = self.run
        mid = event.data.get("messageId")
        if mid not in run.open_text:
            self.fail("sequence", event, f"{event.type} for text message {mid}, which is not open")
            return
        if event.data.get("subagentRunId") != run.open_text[mid]:
            self.fail("subagent", event, f"{event.type} attribution differs from the TEXT_MESSAGE_START of {mid}")
        if close:
            del run.open_text[mid]

    def on_tool_call_start(self, event: SseEvent) -> None:
        run = self.run
        tid = event.data.get("toolCallId")
        if tid in run.open_tools:
            self.fail("sequence", event, f"tool call {tid} started while open")
            return
        if tid in run.tools_started:
            # Reported, then tracked as open so its own ARGS and END do not cascade.
            self.fail("tool_reopen", event, f"tool call {tid} started a second time")
        run.tools_started.add(tid)
        run.open_tools[tid] = event.line
        self.check_tool_parent(event)

    def check_tool_parent(self, event: SseEvent) -> None:
        run = self.run
        parent = event.data.get("parentMessageId")
        if run.message_id is None:
            return
        if parent == run.message_id:
            return
        prefix = f"{run.message_id}:sa:"
        if isinstance(parent, str) and parent.startswith(prefix):
            sid = parent[len(prefix):]
            if sid in run.subagents_seen:
                return
            self.fail("tool_parent", event, f"parentMessageId names subagent {sid}, never started")
            return
        self.fail("tool_parent", event, f"parentMessageId {parent!r} is neither the answer nor a subagent message")

    def on_tool_call_args(self, event: SseEvent) -> None:
        tid = event.data.get("toolCallId")
        if tid not in self.run.open_tools:
            self.fail("sequence", event, f"TOOL_CALL_ARGS for tool call {tid}, which is not open")

    def on_tool_call_end(self, event: SseEvent) -> None:
        run = self.run
        tid = event.data.get("toolCallId")
        if tid not in run.open_tools:
            self.fail("sequence", event, f"TOOL_CALL_END for tool call {tid}, which is not open")
            return
        del run.open_tools[tid]
        run.tools_ended.add(tid)

    def on_tool_call_result(self, event: SseEvent) -> None:
        run = self.run
        tid = event.data.get("toolCallId")
        mid = event.data.get("messageId")
        if tid not in run.tools_ended:
            state = "still open" if tid in run.open_tools else "never started"
            self.fail("result_ref", event, f"result for tool call {tid}, {state}")
        elif tid in run.tool_results:
            self.fail("result_ref", event, f"second result for tool call {tid}")
        if mid in run.result_message_ids:
            self.fail("result_ref", event, f"result messageId {mid} already used")
        if run.message_id is not None and mid != f"{run.message_id}:tool:{tid}":
            self.fail("result_ref", event, f"result messageId {mid!r} is not <messageId>:tool:<toolCallId>")
        run.tool_results.add(tid)
        run.result_message_ids.add(mid)

    def on_custom(self, event: SseEvent) -> None:
        name = event.data.get("name")
        value = event.data.get("value")
        if name not in CUSTOM_NAMES:
            self.fail("custom_name", event, f"CUSTOM name {name!r} is not one of {sorted(CUSTOM_NAMES)}")
            return
        if name == "eve.sources":
            ok = (
                isinstance(value, dict)
                and isinstance(value.get("toolCallId"), str)
                and isinstance(value.get("documents"), list)
            )
            if not ok:
                self.fail("custom_name", event, "eve.sources value is not {toolCallId, documents}")
            elif value["toolCallId"] not in self.run.tools_started:
                self.fail("custom_name", event, f"eve.sources names tool call {value['toolCallId']}, never started")
        elif name == "eve.artifacts":
            ids = value.get("artifactIds") if isinstance(value, dict) else None
            if not (isinstance(ids, list) and all(isinstance(i, str) for i in ids)):
                self.fail("custom_name", event, "eve.artifacts value is not {artifactIds: [string]}")
        elif name == "eve.notice":
            ok = isinstance(value, dict) and isinstance(value.get("code"), str) and isinstance(value.get("message"), str)
            if not ok:
                self.fail("custom_name", event, "eve.notice value is not {code, message}")

    def on_activity_snapshot(self, event: SseEvent) -> None:
        atype = event.data.get("activityType")
        mid = event.data.get("messageId")
        if atype not in ACTIVITY_TYPES:
            self.fail("activity_type", event, f"activityType {atype!r} is not one of {sorted(ACTIVITY_TYPES)}")
            return
        answer = self.run.message_id
        if answer is None:
            return
        if atype == "eve.plan":
            if mid != f"{answer}:plan":
                self.fail("activity_type", event, f"eve.plan messageId {mid!r} is not <messageId>:plan")
        else:
            match = _SUMMARIZING_ID.match(mid) if isinstance(mid, str) else None
            if match is None or match.group("mid") != answer:
                self.fail("activity_type", event, f"eve.summarizing messageId {mid!r} is not <messageId>:summarizing:<n>")

    def on_subagent_started(self, event: SseEvent) -> None:
        run = self.run
        sid = event.data.get("subagentRunId")
        if sid in run.subagents_seen:
            self.fail("subagent", event, f"subagent {sid} started a second time")
            return
        parent_call = event.data.get("parentToolCallId")
        if parent_call is not None and parent_call not in run.tools_started:
            self.fail("subagent", event, f"parentToolCallId {parent_call} names no started tool call")
        parent_sa = event.data.get("parentSubagentRunId")
        if parent_sa is not None and parent_sa not in run.open_subagents:
            self.fail("subagent", event, f"parentSubagentRunId {parent_sa} is not an open subagent")
        run.subagents_seen.add(sid)
        run.open_subagents.add(sid)

    def on_subagent_finished(self, event: SseEvent) -> None:
        self._close_subagent(event)

    def on_subagent_error(self, event: SseEvent) -> None:
        self._close_subagent(event)

    def _close_subagent(self, event: SseEvent) -> None:
        sid = event.data.get("subagentRunId")
        if sid not in self.run.open_subagents:
            self.fail("subagent", event, f"{event.type} for subagent {sid}, which is not open")
            return
        self.run.open_subagents.discard(sid)

    def check_attribution(self, event: SseEvent) -> None:
        """An event attributed to a subagent needs that subagent open."""
        if event.type in {"SUBAGENT_STARTED", "SUBAGENT_FINISHED", "SUBAGENT_ERROR"}:
            return
        sid = event.data.get("subagentRunId")
        if sid is None:
            return
        if sid not in self.run.subagents_seen:
            self.fail("subagent", event, f"{event.type} attributed to subagent {sid} before SUBAGENT_STARTED")
        elif sid not in self.run.open_subagents:
            self.fail("subagent", event, f"{event.type} attributed to subagent {sid} after it ended")

    # -- terminal ----------------------------------------------------------

    def on_terminal(self, event: SseEvent) -> None:
        run = self.run
        if not run.started:
            self.fail("first_event", event, "terminal event with no RUN_STARTED")
        for mid in run.open_text:
            self.fail("open_at_terminal", event, f"text message {mid} still open")
        for tid in run.open_tools:
            self.fail("open_at_terminal", event, f"tool call {tid} still open")
        for sid, name in sorted(run.open_steps, key=lambda k: (k[0] or "", k[1] or "")):
            owner = f" of subagent {sid}" if sid else ""
            self.fail("open_at_terminal", event, f"step {name!r}{owner} still open")
        for sid in sorted(run.open_subagents):
            self.fail("open_at_terminal", event, f"subagent {sid} still open")
        if event.type == "RUN_FINISHED":
            self.on_run_finished(event)
        else:
            self.on_run_error(event)

    def on_run_finished(self, event: SseEvent) -> None:
        run = self.run
        data = event.data
        if run.started and (data.get("threadId") != run.thread_id or data.get("runId") != run.run_id):
            self.fail("run_ids", event, "RUN_FINISHED threadId or runId differs from RUN_STARTED")
        outcome = data.get("outcome")
        outcome_type = outcome.get("type") if isinstance(outcome, dict) else "success"
        meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        stopped = meta.get("eve.stopped", None)

        if outcome_type == "cancelled":
            if stopped is not True:
                self.fail("outcome_cancelled", event, "outcome cancelled without eve.stopped: true")
            if "result" in data:
                self.fail("outcome_cancelled", event, "a cancelled run carries a result")
        elif "eve.stopped" in meta:
            self.fail(
                "outcome_cancelled",
                event,
                f"eve.stopped: {stopped!r} with outcome {outcome_type}: a Stop is reported only as cancelled",
            )

        if outcome_type == "success":
            for tid in sorted(run.tools_started - run.tool_results):
                self.fail("pending_on_success", event, f"tool call {tid} has no result on a success outcome")
        elif outcome_type == "interrupt":
            self.check_interrupts(event, outcome)

    def check_interrupts(self, event: SseEvent, outcome: dict[str, Any]) -> None:
        run = self.run
        interrupts = outcome.get("interrupts")
        if not isinstance(interrupts, list) or not interrupts:
            self.fail("outcome_interrupt", event, "outcome interrupt without interrupts")
            return
        seen: set[Any] = set()
        for item in interrupts:
            if not isinstance(item, dict):
                self.fail("outcome_interrupt", event, "interrupt is not an object")
                continue
            iid = item.get("id")
            tid = item.get("toolCallId")
            if iid in seen:
                self.fail("outcome_interrupt", event, f"interrupt id {iid} repeated")
            seen.add(iid)
            if tid not in run.tools_started:
                self.fail("outcome_interrupt", event, f"interrupt {iid} names tool call {tid!r}, not a call of this run")
            elif tid in run.tool_results:
                self.fail("outcome_interrupt", event, f"interrupt {iid} names tool call {tid}, which has a result")
            if tid is not None and iid != f"int-{tid}":
                self.fail("outcome_interrupt", event, f"interrupt id {iid!r} is not int-<toolCallId>")
            if item.get("reason") not in INTERRUPT_REASONS:
                self.fail("outcome_interrupt", event, f"interrupt reason {item.get('reason')!r} is not tool_approval")
            self.check_interrupt_metadata(event, item)

    def check_interrupt_metadata(self, event: SseEvent, item: dict[str, Any]) -> None:
        if "metadata" not in item:
            return
        meta = item["metadata"]
        if not isinstance(meta, dict):
            self.fail("metadata_keys", event, "interrupt metadata is not an object")
            return
        for key, value in meta.items():
            spec = INTERRUPT_METADATA_KEYS.get(key)
            if spec is None:
                self.fail("metadata_keys", event, f"interrupt metadata key {key!r} is not in the contract")
            elif not spec[0](value):
                self.fail("metadata_keys", event, f"interrupt metadata {key} must be {spec[1]}")

    def on_run_error(self, event: SseEvent) -> None:
        data = event.data
        code = data.get("code")
        meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        if code is None:
            self.fail("run_error", event, "RUN_ERROR without code")
            return
        if code not in RUN_ERROR_MESSAGES:
            self.fail("run_error", event, f"code {code!r} is not in the section 1.7 table")
        if meta.get("eve.error.code") != code:
            self.fail("run_error", event, "metadata eve.error.code differs from code")
        expected = RUN_ERROR_MESSAGES.get(code)
        if expected is not None and data.get("message") != expected:
            self.fail("run_error", event, f"message is not the fixed copy for {code}")

    # -- metadata ----------------------------------------------------------

    def check_metadata(self, event: SseEvent) -> None:
        if "metadata" not in event.data:
            return
        meta = event.data["metadata"]
        if meta is None:
            self.fail("metadata_keys", event, "metadata: null; omit the key instead")
            return
        if not isinstance(meta, dict):
            return  # a schema failure, already reported
        for key, value in meta.items():
            if key == RESERVED_METADATA_KEY:
                self.fail("metadata_keys", event, "key 'ag-ui' is reserved for the protocol")
                continue
            spec = EVENT_METADATA_KEYS.get(key)
            if spec is None:
                self.fail("metadata_keys", event, f"metadata key {key!r} is not in the contract")
                continue
            allowed, check, type_text = spec
            if event.type not in allowed:
                self.fail("metadata_keys", event, f"metadata {key} is not allowed on {event.type}")
            elif not check(value):
                self.fail("metadata_keys", event, f"metadata {key} must be {type_text}")

    def check_endpoint_placement(self) -> None:
        last = self.run.last_text_end_line
        for line in self.run.endpoint_lines:
            if line != last:
                self.violations.append(
                    Violation("metadata_keys", line, "eve.endpoint on a TEXT_MESSAGE_END that is not the last")
                )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def validate_transcript(
    body: str,
    *,
    thread_id: str | None = None,
    run_id: str | None = None,
    require_terminal: bool = True,
) -> list[Violation]:
    """Validate one AG-UI SSE transcript against the EVE contract.

    `thread_id` and `run_id`, when given, are what RUN_STARTED must echo.
    `require_terminal=False` accepts a transcript cut on purpose (a client test
    of truncation); server tests keep the default.
    """
    events, violations = parse_sse(body)
    for event in events:
        problems = check_schema(event)
        if problems:
            violations.append(Violation("schema", event.line, "; ".join(problems)))
    violations.extend(_Checker(events, thread_id, run_id, require_terminal).check())
    return violations


def violation_codes(violations: Iterable[Violation]) -> set[str]:
    return {v.code for v in violations}
