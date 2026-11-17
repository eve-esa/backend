"""Make tool-call arguments valid JSON before they are sent again.

A model sometimes ends a tool-call argument string with an extra character.
That string is stored on the assistant message. The next request sends it
back, and the provider rejects the whole turn with ``Extra data``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def repair_tool_arguments(raw: str) -> str:
    """Return one JSON object. Trailing junk is dropped.

    A string that is not JSON at all becomes ``{}`` so the request can still
    be sent. The provider then sees an empty argument object instead of a
    body it refuses to parse.
    """
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError as error:
        if error.msg != "Extra data":
            logger.warning(
                "tool_args.unparseable error=%s", error.msg
            )
            return "{}"
    try:
        value, _end = json.JSONDecoder().raw_decode(raw)
    except json.JSONDecodeError as error:
        logger.warning("tool_args.unparseable error=%s", error.msg)
        return "{}"
    logger.warning(
        "tool_args.trimmed_extra_data kept=%s dropped=%s",
        _end,
        len(raw) - _end,
    )
    return json.dumps(value)


def drop_assistant_names(payload: dict) -> dict:
    """Remove ``name`` from assistant messages in a chat payload.

    The deep graph names the agent ``deep``. After each model call, LangChain
    stores that on ``AIMessage.name``, and the OpenAI converter copies it onto
    the next request. Mistral's assistant message has no ``name`` field and
    rejects the body. A tool message keeps ``name``: that value is the tool.
    """
    for message in payload.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "assistant":
            message.pop("name", None)
    return payload


def repair_payload_tool_args(payload: dict) -> dict:
    """Repair ``function.arguments`` on every tool call in a chat payload."""
    for message in payload.get("messages") or []:
        if not isinstance(message, dict):
            continue
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                function["arguments"] = repair_tool_arguments(arguments)
    return payload


def ensure_tool_args_are_json(llm: Any) -> Any:
    """Repair tool arguments on every request this model sends.

    A model without ``_get_request_payload`` is returned unchanged. A second
    call does not wrap the method again.
    """
    if llm is None or getattr(llm, "_eve_tool_args_repaired", False):
        return llm
    original = getattr(llm, "_get_request_payload", None)
    if original is None:
        return llm

    def _get_request_payload(input_: Any, *, stop: Any = None, **kwargs: Any) -> dict:
        payload = original(input_, stop=stop, **kwargs)
        if isinstance(payload, dict):
            return repair_payload_tool_args(drop_assistant_names(payload))
        return payload

    llm._get_request_payload = _get_request_payload
    llm._eve_tool_args_repaired = True
    return llm
