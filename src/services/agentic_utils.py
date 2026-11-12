"""Shared helpers for the agentic generation pipeline.

Contains text-format tool-call parsing (EVE-Instruct / Mistral), message
reformatting, SSE label generation, and tool input schemas.
"""

import json
import re
from typing import Any, Dict, List, Optional


from src.database.models.message import Message
from src.schemas.generation_request import GenerationRequest

_TEXT_TOOL_CALL_MARKER = "[TOOL_CALLS]"


# ─── SSE label generation ─────────────────────────────────────────────────────


def tool_call_label(tool_name: str) -> str:
    """Return a human-readable label for a streaming tool-call event."""
    if "knowledge_base" in tool_name:
        return "Searching knowledge base"
    if "wiley" in tool_name.lower():
        return "Searching Wiley Gateway"
    pretty = tool_name.replace("_", " ").replace("-", " ").strip()
    return f"Calling {pretty}" if pretty else "Calling tool"


def is_agentic_generation_request(
    request: GenerationRequest,
    message: Optional[Message] = None,
) -> bool:
    """Return True when the stored request should use the agentic pipeline."""
    if (
        request.custom_model_id
        or request.agent
        or request.public_mcp_servers
        or request.skills
    ):
        return True
    if message is None:
        return False
    # The shell is stamped at creation. The trace check alone misroutes retries
    # of failed agentic turns: a generation that dies before its first token
    # never persists a trace, and the retry would silently switch pipelines.
    metadata = getattr(message, "metadata", None) or {}
    if metadata.get("pipeline") == "agentic":
        return True
    return bool(getattr(message, "trace", None))


# ─── Text-format tool-call parsing (EVE-Instruct / Mistral) ───────────────────


def _call_args(item: Dict[str, Any]) -> Dict[str, Any]:
    """Arguments from either key the text format uses.

    History is rewritten with ``arguments``. The ``name{...}`` form and some
    model replies use ``args``. A string value is one more JSON object.
    """
    args = item.get("arguments", item.get("args", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return {}
    return args if isinstance(args, dict) else {}


def _relax_json_escapes(text: str) -> tuple[str, List[int]]:
    """Keep valid JSON escapes and double the rest.

    A shell command such as ``\\(`` is not a JSON escape. ``json.loads``
    rejects the whole tool call, and the turn then ends with an empty answer.
    Also returns the offsets in *text* where a backslash was inserted.
    """
    inserted: List[int] = []

    def _escape(match: "re.Match[str]") -> str:
        if match.group(1) in '"\\/bfnrtu':
            return match.group(0)
        inserted.append(match.start())
        return "\\\\" + match.group(1)

    return re.sub(r"\\(.)", _escape, text), inserted


def _decode_json_value(text: str) -> tuple[Any, int]:
    """Decode one JSON value. Return ``(value, end)`` with *end* in *text*.

    The second try repairs invalid escapes, which lengthens the string, so
    the end is mapped back to *text* before returning.
    """
    try:
        return json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        repaired, inserted = _relax_json_escapes(text)
        if not inserted:
            raise
        value, end = json.JSONDecoder().raw_decode(repaired)
        shift = sum(1 for k, pos in enumerate(inserted) if pos + k < end)
        return value, end - shift


def _json_array_tool_calls(
    text: str, *, start_index: int = 0
) -> Optional[tuple[List[Dict[str, Any]], str]]:
    """Parse ``[{"name": "...", "arguments": {...}}]`` and the text after it.

    ``None`` when *text* is not that array. That is the shape written back
    into the thread after a tool call, and the shape the model copies.
    """
    stripped = text.lstrip()
    if not stripped.startswith("["):
        return None
    try:
        payload, end = _decode_json_value(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, list) or not payload:
        return None
    calls: List[Dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            return None
        name = item.get("name")
        if not isinstance(name, str) or not name:
            return None
        calls.append(
            {
                "id": f"call_{start_index + len(calls)}_{name}",
                "name": name,
                "args": _call_args(item),
                "type": "tool_call",
            }
        )
    lead = len(text) - len(stripped)
    return calls, text[:lead] + stripped[end:]


def parse_text_tool_calls(content: str) -> List[Dict[str, Any]]:
    """Parse ``[TOOL_CALLS]`` text into structured tool calls.

    Two shapes. ``[TOOL_CALLS]tool_name{"key": "val"}`` is the Mistral form.
    ``[TOOL_CALLS] [{"name": "tool_name", "arguments": {...}}]`` is the form
    written into history, which the model then copies.

    Returns a list compatible with ``AIMessage.tool_calls``.
    """
    if _TEXT_TOOL_CALL_MARKER not in content:
        return []

    text = content[
        content.index(_TEXT_TOOL_CALL_MARKER) + len(_TEXT_TOOL_CALL_MARKER) :
    ].strip()
    parsed = _json_array_tool_calls(text)
    if parsed is not None:
        return parsed[0]
    calls: List[Dict[str, Any]] = []
    i = 0
    while i < len(text):
        while i < len(text) and text[i] in " \t\n,":
            i += 1
        if i >= len(text):
            break
        m = re.match(r"([A-Za-z_]\w*)", text[i:])
        if not m:
            break
        name = m.group(1)
        i += m.end()
        while i < len(text) and text[i] in " \t":
            i += 1
        if i < len(text) and text[i] == "{":
            depth, j = 0, i
            while j < len(text):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            args = json.loads(text[i : j + 1])
                        except Exception:
                            args = {}
                        calls.append(
                            {
                                "id": f"call_{len(calls)}_{name}",
                                "name": name,
                                "args": args,
                                "type": "tool_call",
                            }
                        )
                        i = j + 1
                        break
                j += 1
            else:
                break
        else:
            calls.append(
                {
                    "id": f"call_{len(calls)}_{name}",
                    "name": name,
                    "args": {},
                    "type": "tool_call",
                }
            )
    return calls


def has_text_tool_call(content: str) -> bool:
    """Return True if *content* contains an EVE-Instruct-style tool call marker."""
    return _TEXT_TOOL_CALL_MARKER in content


def split_tool_calls_and_answer_text(content: str) -> tuple[List[Dict[str, Any]], str]:
    """Split *content* into (parsed tool calls, trailing answer text).

    ``parse_text_tool_calls`` alone stops at the first token that isn't a
    valid ``name{...}``/``name`` call, which is correct for detecting where
    the tool-call syntax ends but silently swallows everything after it
    (including real answer prose) if the caller only looks at the parsed
    calls. This wrapper walks the SAME ``[TOOL_CALLS]name{...}`` segments —
    each call gets its own leading marker in the EVE-Instruct/Mistral
    convention, e.g. ``[TOOL_CALLS]a{}[TOOL_CALLS]b{}rest of the answer`` —
    and returns whatever text follows the last recognized call so it is
    never lost.

    Returns ``([], content)`` unchanged when no marker is present.
    """
    if _TEXT_TOOL_CALL_MARKER not in content:
        return [], content

    calls: List[Dict[str, Any]] = []
    text = content
    while text.startswith(_TEXT_TOOL_CALL_MARKER):
        rest = text[len(_TEXT_TOOL_CALL_MARKER) :]
        stripped = rest.lstrip()
        parsed = _json_array_tool_calls(stripped, start_index=len(calls))
        if parsed is not None:
            more, remainder = parsed
            calls.extend(more)
            text = remainder.lstrip("\n")
            continue
        m = re.match(r"([A-Za-z_]\w*)", stripped)
        if not m:
            break
        name = m.group(1)
        i = m.end()
        while i < len(stripped) and stripped[i] in " \t":
            i += 1
        if i < len(stripped) and stripped[i] == "{":
            depth, j, closed = 0, i, False
            while j < len(stripped):
                if stripped[j] == "{":
                    depth += 1
                elif stripped[j] == "}":
                    depth -= 1
                    if depth == 0:
                        closed = True
                        break
                j += 1
            if not closed:
                break
            try:
                args = json.loads(stripped[i : j + 1])
            except Exception:
                args = {}
            calls.append(
                {
                    "id": f"call_{len(calls)}_{name}",
                    "name": name,
                    "args": args,
                    "type": "tool_call",
                }
            )
            text = stripped[j + 1 :]
        else:
            calls.append(
                {
                    "id": f"call_{len(calls)}_{name}",
                    "name": name,
                    "args": {},
                    "type": "tool_call",
                }
            )
            text = stripped[i:]

    return calls, text


def might_be_incomplete_text_tool_call(content: str) -> bool:
    """Return True while streamed text may still be building a ``[TOOL_CALLS]`` call."""
    if not content:
        return False
    text = content.lstrip()
    if _TEXT_TOOL_CALL_MARKER in text:
        return True
    return _TEXT_TOOL_CALL_MARKER.startswith(text)


# ─── Message history sanitisation ─────────────────────────────────────────────

# These functions depend on LangChain message types which may not be installed.
# Import them lazily so the module stays importable regardless.


# ─── Tool input schemas ───────────────────────────────────────────────────────
