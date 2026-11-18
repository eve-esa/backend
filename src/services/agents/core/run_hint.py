"""Tell the model how the sandbox shell names a file it just wrote.

The file tools accept ``/workspace/count.py`` and store it in the workspace.
The shell starts in that workspace (``prepare_workspace_layout``), where the
file is ``count.py``: ``python3 /workspace/count.py`` fails there. The prompt
says so, and some models still reuse the virtual path. So the tool result the
model reads after a successful ``write_file`` or ``edit_file`` ends with one
line naming the shell form.

The line is added to the request payload, when the model call is built. The
stored tool message, the stream and the UI keep the original text. A request
without the ``execute`` tool (a turn with no sandbox) is unchanged.
"""

from __future__ import annotations

import json
import shlex
from typing import Any, Optional

from src.services.agents.core.sandbox import place_in_workspace

# deepagents' success texts for these tools (deepagents.middleware.filesystem).
# An error result carries the error text instead, so it gets no line.
_SUCCESS_PREFIX = {
    "write_file": "Updated file ",
    "edit_file": "Successfully replaced ",
}
_SHELL_TOOL = "execute"
_WORKSPACE = "/workspace/"
_RUNNERS = {".py": "python3", ".sh": "sh"}
_TAIL = "(the shell starts in the workspace)."


def workspace_relative_path(path: Any) -> Optional[str]:
    """The path of a written file relative to the workspace, or ``None``.

    ``x``, ``./x``, ``workspace/x``, ``/workspace/x`` and a real absolute
    ``<root>/workspace/x`` are all ``x``; sub directories stay. A path in a
    sibling directory (``/skills/``, ``/artifacts/``) or with ``..`` is not in
    the workspace.
    """
    raw = str(path or "").strip()
    if not raw or ".." in raw.split("/"):
        return None
    if raw.startswith("/") and not raw.startswith(_WORKSPACE):
        at = raw.find(_WORKSPACE)
        if at > 0:
            raw = raw[at:]
    placed = place_in_workspace(raw)
    if not placed.startswith(_WORKSPACE):
        return None
    return placed[len(_WORKSPACE) :] or None


def run_hint(path: Any) -> Optional[str]:
    """One line with the shell form of a written file, or ``None``."""
    relative = workspace_relative_path(path)
    if relative is None:
        return None
    quoted = shlex.quote(relative)
    for suffix, runner in _RUNNERS.items():
        if relative.endswith(suffix):
            return f"Run it in the shell as `{runner} {quoted}` {_TAIL}"
    return f"In the shell its path is `{quoted}` {_TAIL}"


def _has_shell(payload: dict) -> bool:
    for tool in payload.get("tools") or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(function, dict) and function.get("name") == _SHELL_TOOL:
            return True
    return False


def _file_calls(messages: list) -> dict:
    """``tool_call_id`` to ``(tool name, file_path)`` for the file tools."""
    calls: dict = {}
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if name not in _SUCCESS_PREFIX:
                continue
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except (TypeError, ValueError):
                continue
            if isinstance(arguments, dict):
                calls[call.get("id")] = (name, arguments.get("file_path"))
    return calls


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def add_run_hints(payload: dict) -> dict:
    """Append the shell form to each successful file write in a chat payload."""
    messages = payload.get("messages") or []
    if not _has_shell(payload):
        return payload
    calls = _file_calls(messages)
    if not calls:
        return payload
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        found = calls.get(message.get("tool_call_id"))
        if found is None:
            continue
        name, path = found
        content = message.get("content")
        text = _text(content)
        if not text.startswith(_SUCCESS_PREFIX[name]):
            continue
        hint = run_hint(path)
        if hint is None or hint in text:
            continue
        if isinstance(content, str):
            message["content"] = f"{content}\n{hint}"
        elif isinstance(content, list):
            content.append({"type": "text", "text": hint})
    return payload
