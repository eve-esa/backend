"""Server side snapshot of the conversation a bug report points at.

The browser only sends ``conversation_id``; everything else is read from
Mongo here, so the report carries the whole conversation no matter what the
page had loaded, and nothing in it is taken from the client.

The snapshot is redacted with :func:`redact_value` and capped at
``BUG_REPORT_CONVERSATION_MAX_BYTES`` of JSON. Past the cap, message outputs
are cut from the oldest message first and end with :data:`TRUNCATED_MARKER`;
if every output is cut and it still does not fit, traces go next, then
request settings, then inputs, in the same order.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from bson import ObjectId
from fastapi import HTTPException

from src.config import BUG_REPORT_CONVERSATION_MAX_BYTES
from src.database.models.conversation import Conversation
from src.database.models.message import Message
from src.database.models.user import User
from src.utils.redaction import redact_value

TRUNCATED_MARKER = "[truncated]"

# request_input keys that are not settings: the query repeats ``input`` and
# the rest are server side wiring injected by Message.to_dict.
_REQUEST_DROP_KEYS = {"query", "collection_ids", "private_collections_map", "user_id"}


def _iso(value: Any) -> Any:
    return value.isoformat() if isinstance(value, datetime) else value


def json_size(value: Any) -> int:
    """Bytes of ``value`` as compact UTF-8 JSON."""
    return len(
        json.dumps(value, default=str, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )


def _attachment_names(attachments: Optional[List[Dict[str, Any]]]) -> List[str]:
    names = []
    for item in attachments or []:
        if isinstance(item, dict):
            name = item.get("filename") or item.get("name")
            if name:
                names.append(str(name))
    return names


def _request_settings(request_input: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(request_input, dict):
        return None
    return {k: v for k, v in request_input.items() if k not in _REQUEST_DROP_KEYS}


def _message_entry(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(doc["_id"]),
        "created_at": _iso(doc.get("timestamp")),
        "input": doc.get("input"),
        "output": doc.get("output"),
        "trace_id": doc.get("trace_id"),
        "metadata": doc.get("metadata"),
        "request": _request_settings(doc.get("request_input")),
        "use_rag": doc.get("use_rag"),
        "feedback": doc.get("feedback"),
        "feedback_reason": doc.get("feedback_reason"),
        "hallucination": doc.get("hallucination"),
        "stopped": doc.get("stopped"),
        "artifact_ids": doc.get("artifact_ids") or [],
        "attachments": _attachment_names(doc.get("attachments")),
        "trace": doc.get("trace"),
    }


async def load_owned_conversation(conversation_id: str, user: User) -> Dict[str, Any]:
    """The conversation document if ``user`` owns it, otherwise 404.

    Missing, malformed and foreign ids answer the same, as the conversation
    routes do for a missing one, so the bug report route is not an oracle.
    """
    if not ObjectId.is_valid(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    doc = await Conversation.get_collection().find_one(
        {"_id": ObjectId(conversation_id), "user_id": user.id}
    )
    if doc is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return doc


def _cut(entry: Dict[str, Any], field: str, overflow: int) -> bool:
    """Shorten ``entry[field]`` by at least ``overflow`` bytes where possible.

    Strings keep their head and end with the marker; anything else becomes the
    marker. Returns False when there was nothing left to cut.
    """
    value = entry.get(field)
    if value is None or value == TRUNCATED_MARKER:
        return False
    if isinstance(value, str):
        if value.endswith(TRUNCATED_MARKER) and entry.get(f"{field}_truncated"):
            head = value[: -len(TRUNCATED_MARKER)]
        else:
            head = value
        # Every character is at least one byte, so dropping ``overflow`` chars
        # plus room for the marker frees at least ``overflow`` bytes.
        keep = max(0, len(head) - overflow - len(TRUNCATED_MARKER))
        if keep == 0 and not head:
            return False
        entry[field] = head[:keep] + TRUNCATED_MARKER if keep else TRUNCATED_MARKER
    else:
        entry[field] = TRUNCATED_MARKER
    entry[f"{field}_truncated"] = True
    return True


def fit_to_cap(snapshot: Dict[str, Any], max_bytes: int) -> bool:
    """Shrink ``snapshot`` in place until its JSON fits ``max_bytes``.

    Returns True when anything was cut. Outputs go first, oldest message first,
    then traces, request settings and inputs.
    """
    size = json_size(snapshot)
    if size <= max_bytes:
        return False
    messages = snapshot.get("messages") or []
    for field in ("output", "trace", "request", "input"):
        for entry in messages:
            # A cut string can come out a little over when it holds multibyte
            # characters, so the same message gets another pass.
            while size > max_bytes and _cut(entry, field, size - max_bytes):
                size = json_size(snapshot)
            if size <= max_bytes:
                return True
    return True


async def build_snapshot(
    conversation_id: Optional[str], user: User, max_bytes: Optional[int] = None
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """Read, redact and cap the conversation for a bug report.

    Returns:
        ``(snapshot, truncated)``; ``(None, False)`` without a conversation id.

    Raises:
        HTTPException: 404 when the conversation is missing or not the user's.
    """
    if not conversation_id:
        return None, False
    conversation = await load_owned_conversation(conversation_id, user)
    cursor = (
        Message.get_collection()
        .find({"conversation_id": str(conversation["_id"])})
        .sort([("timestamp", 1), ("_id", 1)])
    )
    messages = [_message_entry(doc) async for doc in cursor]
    snapshot = {
        "id": str(conversation["_id"]),
        "title": conversation.get("name"),
        "created_at": _iso(conversation.get("timestamp")),
        "summary": conversation.get("summary"),
        # Conversations store no model or settings of their own: those live
        # per message, under ``request``.
        "settings": conversation.get("settings"),
        "messages": messages,
    }
    snapshot = redact_value(snapshot)
    # Set before measuring so the flag's own bytes count against the cap.
    snapshot["truncated"] = True
    cap = BUG_REPORT_CONVERSATION_MAX_BYTES if max_bytes is None else max_bytes
    truncated = fit_to_cap(snapshot, cap) if cap > 0 else False
    snapshot["truncated"] = truncated
    return snapshot, truncated
