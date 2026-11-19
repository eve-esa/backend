"""Refuse a tool call that keeps returning the same result in one turn.

A model can loop on one call: on dev (2026-11-19) the deep graph ran the
same ``execute`` command 31 times, each result full of nulls, until the turn
deadline cut it and the user got no answer. Each repeat costs a model call.

Within one turn the guard keys calls by tool name and a hash of the
normalised arguments (JSON with sorted keys, strings trimmed) and keeps a
hash of each run's result, an error included. A call is refused when two
identical calls already ran in a row and returned the same result (or are
still running, as with copies sent in one step). A call whose result
changes, such as polling a job, is never refused. Distinct calls always run.

The refusal is a tool result that starts with ``NOT RUN.`` and says how many
identical calls really ran. From the 5th identical attempt, or once two
refusals were given in the turn, it tells the model to stop calling tools
and answer. If the model still calls refused tools 3 more times
(:data:`END_AFTER_IGNORED`), :attr:`ToolRepeatGuard.end_requested` is set
and the runner ends the turn with the answer so far and :data:`END_NOTE`.

What a call changes resets other counts, after it ran:

- ``execute`` (any result) and ``delete``: the read-only file calls
  (``ls``, ``glob``, ``grep``, ``read_file``) start from zero.
- a successful ``write_file``, ``edit_file`` or ``delete``: every other call
  starts from zero, since a command run again after an edit can give a
  different result. The writing call itself is counted like any other.

A deepagents subagent (the ``task`` tool) runs with its own guard: its
refusals never reach the parent's counts. A subagent that ignores the stop
text is cut and the parent reads a ``task`` result that says so.

The state lives in a context variable opened by each agentic entry point
(:func:`set_tool_repeat_context`), so a turn starts from zero and a call made
outside a turn passes through. Two hooks read it:

- MCP tools, both graphs: :class:`RepeatGuardInterceptor` in
  ``src.services.agents.core.interceptors``.
- deepagents built-in tools (``execute``, the file tools) and ``task``, deep
  graph: the eve-esa/agents deep graph builds its own middleware list and
  takes none from the backend, so :func:`install_builtin_tool_guard` wraps
  ``FilesystemMiddleware.wrap_tool_call`` on the class, once per process.
  MCP calls also pass through that middleware; the name filter keeps them
  from being counted twice.

An error inside the guard never fails a tool call: it is logged and the
call runs as if there were no guard.
"""

from __future__ import annotations

import contextvars
import hashlib
import inspect
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Identical calls with the same result that may run in a row; the next is refused.
MAX_IDENTICAL_RUNS = 2
# From this identical attempt on (runs and refusals), the model is told to stop.
HARD_LIMIT = 5
# After this many first-level refusals in a turn, every refusal is the stop one.
MAX_SOFT_REFUSALS = 2
# Refused calls the model may make after the stop text before the turn ends.
END_AFTER_IGNORED = 3

END_NOTE = (
    "I stopped here because the same tool call kept returning the same result. "
    "This answer uses what I found so far."
)
SUBAGENT_STOPPED = (
    "NOT FINISHED. The subagent kept repeating the same tool call after it was "
    "told to stop, so it was stopped. Do not start it again for the same task: "
    "answer the user with what you have."
)

# deepagents tools served by FilesystemMiddleware. Other names that pass
# through its wrap (MCP tools, history tools, write_todos) are left to their
# own hook or not guarded; ``task`` opens a subagent guard.
BUILTIN_TOOLS = frozenset(
    {"execute", "ls", "read_file", "write_file", "edit_file", "glob", "grep", "delete"}
)
SUBAGENT_TOOL = "task"
# Their result depends on the files: another call that changes files resets them.
_READ_ONLY_TOOLS = frozenset({"ls", "read_file", "glob", "grep"})
# Ran: the files may have changed, whatever the result.
_MAY_CHANGE_FILES = frozenset({"execute", "delete"})
# Succeeded: the files changed.
_FILE_MUTATIONS = frozenset({"write_file", "edit_file", "delete"})


class RepeatLoopStopped(Exception):
    """A subagent ignored the stop text: its run is cut at the ``task`` call."""


def _ran_text(ran: int, compared: bool) -> str:
    times = "once" if ran == 1 else f"{ran} times"
    same = " and returned the same result each time" if compared else ""
    return f"This exact call already ran {times} in this turn{same}."


def _soft_message(ran: int, compared: bool) -> str:
    return (
        f"NOT RUN. {_ran_text(ran, compared)} Do not repeat it: change the "
        "command or the approach, or answer the user with what you have."
    )


def _hard_message(ran: int, compared: bool) -> str:
    return (
        f"NOT RUN. {_ran_text(ran, compared)} Stop calling tools now and answer "
        "the user with what you have, saying what you could not find."
    )


def _trimmed(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return {str(k): _trimmed(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_trimmed(v) for v in value]
    return value


def normalise_args(args: Any) -> str:
    """Stable text for *args*: JSON with sorted keys and trimmed strings."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (TypeError, ValueError):
            pass
    return json.dumps(
        _trimmed(args),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def _tool_message_of(result: Any) -> Any:
    """The ToolMessage in *result*: itself, or the one a ``Command`` carries."""
    update = getattr(result, "update", None)
    if isinstance(update, dict):
        for msg in update.get("messages") or []:
            if getattr(msg, "type", None) == "tool":
                return msg
        return None
    return result


def result_outcome(result: Any) -> Tuple[str, bool]:
    """Hash of a tool result and whether it is an error.

    Reads a langchain ``ToolMessage`` (or the one inside a ``Command``), an MCP
    ``CallToolResult``, or an exception raised by the call.
    """
    if isinstance(result, BaseException):
        return _digest(f"raised:{type(result).__name__}:{result}"), True
    msg = _tool_message_of(result)
    if msg is not None and hasattr(msg, "content") and hasattr(msg, "tool_call_id"):
        content = msg.content
        text = content if isinstance(content, str) else json.dumps(content, default=str)
        status = getattr(msg, "status", "success")
        is_error = status == "error" or text.lstrip().startswith("Error")
        return _digest(f"{status}:{text}"), is_error
    if hasattr(result, "isError") and hasattr(result, "content"):
        blocks = [
            b.model_dump() if hasattr(b, "model_dump") else str(b)
            for b in (result.content or [])
        ]
        is_error = bool(result.isError)
        return _digest(json.dumps([is_error, blocks], default=str, sort_keys=True)), is_error
    return _digest(repr(result)), False


@dataclass
class _KeyState:
    ran: int = 0  # identical calls that ran in this turn
    streak: int = 0  # runs in a row with the same result, those in flight included
    done: int = 0  # runs of the streak that returned
    attempts: int = 0  # runs and refusals since the streak started
    last: Optional[str] = None  # result hash of the streak


@dataclass(frozen=True)
class CallTicket:
    """One call that runs: hand it back to :meth:`ToolRepeatGuard.finish`."""

    key: Tuple[str, str]
    tool_name: str


@dataclass
class ToolRepeatGuard:
    """Identical call counts for one turn or one subagent run. Thread and task safe."""

    is_subagent: bool = False
    keys: Dict[Tuple[str, str], _KeyState] = field(default_factory=dict)
    soft_refusals: int = 0
    hard_refusals: int = 0
    end_requested: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def begin(self, tool_name: str, args: Any) -> Tuple[Optional[str], Optional[CallTicket]]:
        """``(None, ticket)`` runs the call; ``(refusal, None)`` returns the refusal.

        Raises :class:`RepeatLoopStopped` in a subagent that ignored the stop text.
        """
        key = (tool_name, _digest(normalise_args(args)))
        with self._lock:
            state = self.keys.setdefault(key, _KeyState())
            state.attempts += 1
            if state.streak < MAX_IDENTICAL_RUNS:
                state.streak += 1
                state.ran += 1
                return None, CallTicket(key=key, tool_name=tool_name)
            compared = state.done >= MAX_IDENTICAL_RUNS
            if state.attempts >= HARD_LIMIT or self.soft_refusals >= MAX_SOFT_REFUSALS:
                self.hard_refusals += 1
                message = _hard_message(state.ran, compared)
                if self.hard_refusals > END_AFTER_IGNORED:
                    self.end_requested = True
            else:
                self.soft_refusals += 1
                message = _soft_message(state.ran, compared)
            attempts, ended = state.attempts, self.end_requested
        logger.info(
            "tool.repeat_blocked tool=%s count=%d subagent=%s end=%s",
            tool_name,
            attempts,
            self.is_subagent,
            ended,
        )
        if ended and self.is_subagent:
            raise RepeatLoopStopped(tool_name)
        return message, None

    def finish(self, ticket: CallTicket, result: Any) -> None:
        """Record what the call returned (a raised exception is a result too)."""
        digest, is_error = result_outcome(result)
        name = ticket.tool_name
        with self._lock:
            state = self.keys.setdefault(ticket.key, _KeyState())
            if state.last is not None and digest != state.last:
                # A new result: the streak starts again from this run.
                state.streak = max(1, state.streak - state.done)
                state.done = 0
                state.attempts = state.streak
            state.last = digest
            state.done = min(state.done + 1, state.streak)
            if name in _FILE_MUTATIONS and not is_error:
                for other in list(self.keys):
                    if other != ticket.key:
                        del self.keys[other]
            elif name in _MAY_CHANGE_FILES:
                for other in list(self.keys):
                    if other[0] in _READ_ONLY_TOOLS:
                        del self.keys[other]


_tool_repeat_guard: contextvars.ContextVar[Optional[ToolRepeatGuard]] = (
    contextvars.ContextVar("tool_repeat_guard", default=None)
)


def set_tool_repeat_context() -> Tuple[ToolRepeatGuard, contextvars.Token]:
    """Open a fresh guard for this turn. Reset with the token in ``finally``."""
    guard = ToolRepeatGuard()
    return guard, _tool_repeat_guard.set(guard)


def reset_tool_repeat_context(token: contextvars.Token) -> None:
    """Undo :func:`set_tool_repeat_context`. A token from another Context is skipped."""
    try:
        _tool_repeat_guard.reset(token)
    except ValueError:
        logger.debug("tool_repeat_guard reset skipped: token from another Context")


def current_tool_repeat_guard() -> Optional[ToolRepeatGuard]:
    """The guard of the running turn (or subagent), ``None`` outside a turn."""
    return _tool_repeat_guard.get()


def _log_guard_error(exc: BaseException) -> None:
    logger.warning("tool.repeat_guard_failed error=%s", type(exc).__name__)


def begin_tool_call(tool_name: str, args: Any) -> Tuple[Optional[str], Optional[CallTicket]]:
    """Refusal text for this call, or a ticket to finish once it ran.

    Outside a turn, and on any error inside the guard, the call runs untracked
    (``(None, None)``). :class:`RepeatLoopStopped` is the one exception raised.
    """
    guard = _tool_repeat_guard.get()
    if guard is None:
        return None, None
    try:
        return guard.begin(tool_name, args)
    except RepeatLoopStopped:
        raise
    except Exception as exc:  # noqa: BLE001 - the guard must never fail a call
        _log_guard_error(exc)
        return None, None


def finish_tool_call(ticket: Optional[CallTicket], result: Any) -> None:
    """Record the result of a call :func:`begin_tool_call` let run."""
    if ticket is None:
        return
    guard = _tool_repeat_guard.get()
    if guard is None:
        return
    try:
        guard.finish(ticket, result)
    except Exception as exc:  # noqa: BLE001
        _log_guard_error(exc)


def check_tool_call(tool_name: str, args: Any) -> Optional[str]:
    """Refusal text for this call, or ``None``; its result is never recorded.

    A call checked here stays in flight: two identical ones run, the third is
    refused, as with copies sent in one step.
    """
    refusal, _ticket = begin_tool_call(tool_name, args)
    return refusal


def mark_refused_span(tool_name: str) -> None:
    """Tag the current span so a trace tells a refused call from one that ran."""
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span is not None and span.is_recording():
            span.set_attribute("eve.tool.refused", True)
            span.set_attribute("eve.tool.refused_name", tool_name)
    except Exception:  # noqa: BLE001 - telemetry never fails a call
        pass


# --- deepagents built-in tools and subagents ---------------------------------

_DEEPAGENTS_WRAP: Optional[Callable[..., Any]] = None
_DEEPAGENTS_AWRAP: Optional[Callable[..., Any]] = None
_EXPECTED_PARAMETERS = ("self", "request", "handler")
_SKIP_LOGGED = False


def _call_of(request: Any) -> Tuple[Optional[str], Any, str]:
    call = getattr(request, "tool_call", None)
    if not isinstance(call, dict):
        return None, {}, ""
    return call.get("name"), call.get("args") or {}, call.get("id") or ""


def _tool_message(text: str, name: str, call_id: str) -> Any:
    from langchain_core.messages import ToolMessage

    return ToolMessage(content=text, name=name, tool_call_id=call_id)


def _builtin_begin(request: Any) -> Tuple[Any, Optional[CallTicket]]:
    """``(refusal ToolMessage, None)`` or ``(None, ticket)`` for *request*."""
    name, args, call_id = _call_of(request)
    if name not in BUILTIN_TOOLS:
        return None, None
    refusal, ticket = begin_tool_call(name, args)
    if refusal is None:
        return None, ticket
    mark_refused_span(name)
    return _tool_message(refusal, name, call_id), None


def _open_subagent_guard() -> Optional[contextvars.Token]:
    if _tool_repeat_guard.get() is None:
        return None
    return _tool_repeat_guard.set(ToolRepeatGuard(is_subagent=True))


def _wrap_tool_call(self: Any, request: Any, handler: Any) -> Any:
    name, _args, call_id = _call_of(request)
    if name == SUBAGENT_TOOL:
        token = _open_subagent_guard()
        try:
            return _DEEPAGENTS_WRAP(self, request, handler)
        except RepeatLoopStopped:
            return _tool_message(SUBAGENT_STOPPED, name, call_id)
        finally:
            if token is not None:
                reset_tool_repeat_context(token)
    refusal, ticket = _builtin_begin(request)
    if refusal is not None:
        return refusal
    try:
        result = _DEEPAGENTS_WRAP(self, request, handler)
    except Exception as exc:
        finish_tool_call(ticket, exc)
        raise
    finish_tool_call(ticket, result)
    return result


async def _awrap_tool_call(self: Any, request: Any, handler: Any) -> Any:
    name, _args, call_id = _call_of(request)
    if name == SUBAGENT_TOOL:
        token = _open_subagent_guard()
        try:
            return await _DEEPAGENTS_AWRAP(self, request, handler)
        except RepeatLoopStopped:
            return _tool_message(SUBAGENT_STOPPED, name, call_id)
        finally:
            if token is not None:
                reset_tool_repeat_context(token)
    refusal, ticket = _builtin_begin(request)
    if refusal is not None:
        return refusal
    try:
        result = await _DEEPAGENTS_AWRAP(self, request, handler)
    except Exception as exc:
        finish_tool_call(ticket, exc)
        raise
    finish_tool_call(ticket, result)
    return result


def _skip(reason: str) -> bool:
    global _SKIP_LOGGED
    if not _SKIP_LOGGED:
        _SKIP_LOGGED = True
        logger.warning("tool.repeat_guard_install_skipped reason=%s", reason)
    return False


def install_builtin_tool_guard() -> bool:
    """Guard deepagents built-in tools in ``FilesystemMiddleware``. Idempotent.

    Returns False, and built-in tools run unguarded as before, when deepagents
    is missing, when ``FilesystemMiddleware`` does not define the
    ``wrap_tool_call`` pair itself, or when the pair does not take
    ``(self, request, handler)``.
    """
    global _DEEPAGENTS_WRAP, _DEEPAGENTS_AWRAP
    try:
        from deepagents.middleware.filesystem import FilesystemMiddleware
    except ImportError:
        return False
    own = vars(FilesystemMiddleware)
    sync_current = own.get("wrap_tool_call")
    async_current = own.get("awrap_tool_call")
    if sync_current is _wrap_tool_call and async_current is _awrap_tool_call:
        return True
    if sync_current is None or async_current is None:
        # Inherited hooks (a deepagents bump moved them to the base): patching
        # the class would make create_agent treat the base hook as overridden.
        return _skip("no_own_wrap_tool_call")
    try:
        sync_params = tuple(inspect.signature(sync_current).parameters)
        async_params = tuple(inspect.signature(async_current).parameters)
    except (TypeError, ValueError):
        return _skip("signature")
    if sync_params != _EXPECTED_PARAMETERS or async_params != _EXPECTED_PARAMETERS:
        return _skip("signature")
    _DEEPAGENTS_WRAP = sync_current
    _DEEPAGENTS_AWRAP = async_current
    FilesystemMiddleware.wrap_tool_call = _wrap_tool_call
    FilesystemMiddleware.awrap_tool_call = _awrap_tool_call
    return True
