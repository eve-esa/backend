"""Refuse a tool call the agent already made twice in the same turn.

A model can loop on one call: on dev (2026-11-19) the deep graph ran the
same ``execute`` command 31 times, each result full of nulls, until the turn
deadline cut it and the user got no answer. Each repeat costs a model call.

Within one turn the guard counts calls by tool name and normalised arguments
(JSON with sorted keys, strings trimmed). The 3rd identical call is not run:
the model reads a tool result that says so and asks for another approach.
From the 5th identical call, or once two such refusals were given in the
turn, the result tells the model to stop calling tools and answer. Distinct
calls always run. Writing or editing a file clears the counts, since a
command run again after an edit can give a different result.

The state lives in a context variable opened by each agentic entry point
(:func:`set_tool_repeat_context`), so a turn starts from zero and a call made
outside a turn passes through. Two hooks read it:

- MCP tools, both graphs: :class:`RepeatGuardInterceptor` in
  ``src.services.agents.core.interceptors``.
- deepagents built-in tools (``execute``, the file tools), deep graph: the
  eve-esa/agents deep graph builds its own middleware list and takes none
  from the backend, so :func:`install_builtin_tool_guard` wraps
  ``FilesystemMiddleware.wrap_tool_call`` on the class, once per process,
  for those tool names only. MCP calls also pass through that middleware;
  the name filter keeps them from being counted twice.
"""

from __future__ import annotations

import contextvars
import inspect
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# The call that is refused first: two identical calls run, the third does not.
SOFT_LIMIT = 3
# From this identical call on, the model is told to answer without tools.
HARD_LIMIT = 5
# After this many first-level refusals in a turn, every refusal is the stop one.
MAX_SOFT_REFUSALS = 2

# deepagents tools served by FilesystemMiddleware. Other names that pass
# through its wrap (MCP tools, history tools, task, write_todos) are left to
# their own hook or not guarded.
BUILTIN_TOOLS = frozenset(
    {"execute", "ls", "read_file", "write_file", "edit_file", "glob", "grep", "delete"}
)
# A successful write or edit can change what a repeated command returns.
_RESETTING_TOOLS = frozenset({"write_file", "edit_file"})


def _soft_message(previous: int) -> str:
    return (
        f"You already ran this exact call {previous} times and got the same result. "
        "Do not repeat it: change the command or the approach, or answer the user "
        "with what you have."
    )


def _hard_message(previous: int) -> str:
    return (
        f"You already ran this exact call {previous} times. This call was not run. "
        "Stop calling tools now and answer the user with what you have, saying "
        "what you could not find."
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


@dataclass
class ToolRepeatGuard:
    """Identical call counts for one turn. Safe across threads and tasks."""

    counts: Dict[Tuple[str, str], int] = field(default_factory=dict)
    soft_refusals: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def check(self, tool_name: str, args: Any) -> Optional[str]:
        """Count this call. ``None`` runs it; a string is the refusal to return."""
        key = (tool_name, normalise_args(args))
        with self._lock:
            if tool_name in _RESETTING_TOOLS:
                self.counts.clear()
            count = self.counts.get(key, 0) + 1
            self.counts[key] = count
            if count < SOFT_LIMIT:
                return None
            if count >= HARD_LIMIT or self.soft_refusals >= MAX_SOFT_REFUSALS:
                message = _hard_message(count - 1)
            else:
                self.soft_refusals += 1
                message = _soft_message(count - 1)
        logger.info("tool.repeat_blocked tool=%s count=%d", tool_name, count)
        return message


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


def check_tool_call(tool_name: str, args: Any) -> Optional[str]:
    """Refusal text for this call in the current turn, or ``None`` to run it."""
    guard = _tool_repeat_guard.get()
    if guard is None:
        return None
    return guard.check(tool_name, args)


# ─── deepagents built-in tools ───────────────────────────────────────────────

_DEEPAGENTS_WRAP: Optional[Callable[..., Any]] = None
_DEEPAGENTS_AWRAP: Optional[Callable[..., Any]] = None
_EXPECTED_PARAMETERS = ("self", "request", "handler")
_SKIP_LOGGED = False


def _builtin_refusal(request: Any) -> Any:
    """A ToolMessage refusing *request*, or ``None`` to run it."""
    call = getattr(request, "tool_call", None) or {}
    name = call.get("name")
    if name not in BUILTIN_TOOLS:
        return None
    message = check_tool_call(name, call.get("args") or {})
    if message is None:
        return None
    from langchain_core.messages import ToolMessage

    return ToolMessage(content=message, name=name, tool_call_id=call.get("id") or "")


def _wrap_tool_call(self: Any, request: Any, handler: Any) -> Any:
    refusal = _builtin_refusal(request)
    if refusal is not None:
        return refusal
    return _DEEPAGENTS_WRAP(self, request, handler)


async def _awrap_tool_call(self: Any, request: Any, handler: Any) -> Any:
    refusal = _builtin_refusal(request)
    if refusal is not None:
        return refusal
    return await _DEEPAGENTS_AWRAP(self, request, handler)


def _skip(reason: str) -> bool:
    global _SKIP_LOGGED
    if not _SKIP_LOGGED:
        _SKIP_LOGGED = True
        logger.warning("tool.repeat_guard_install_skipped reason=%s", reason)
    return False


def install_builtin_tool_guard() -> bool:
    """Guard deepagents built-in tools in ``FilesystemMiddleware``. Idempotent.

    Returns False, and built-in tools run unguarded as before, when deepagents
    is missing or its ``wrap_tool_call`` pair does not take
    ``(self, request, handler)``.
    """
    global _DEEPAGENTS_WRAP, _DEEPAGENTS_AWRAP
    try:
        from deepagents.middleware.filesystem import FilesystemMiddleware
    except ImportError:
        return False
    sync_current = getattr(FilesystemMiddleware, "wrap_tool_call", None)
    async_current = getattr(FilesystemMiddleware, "awrap_tool_call", None)
    if sync_current is _wrap_tool_call and async_current is _awrap_tool_call:
        return True
    if sync_current is None or async_current is None:
        return _skip("no_wrap_tool_call")
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
