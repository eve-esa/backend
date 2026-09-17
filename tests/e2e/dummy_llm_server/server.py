"""OpenAI-compatible stub LLMs for local fault-tolerance testing.

ChatOpenAI ``base_url`` targets:

- ``POST /v1/chat/completions`` — intent-based (crash / hang / MCP tool call)
- ``POST /crash/v1/chat/completions`` — always HTTP 500
- ``POST /hang/v1/chat/completions`` — never sends a first token
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator, Optional
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="dummy-llm")

_CRASH_BODY = {
    "error": {
        "message": "dummy LLM crashed",
        "type": "server_error",
        "code": "dummy_crash",
    }
}

_HANG_SECONDS = 3600


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text") or ""))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return ""


def _last_user_blob(body: dict) -> str:
    for message in reversed(body.get("messages") or []):
        if message.get("role") == "user":
            return _message_text(message.get("content")).lower()
    return ""


def _last_role(body: dict) -> Optional[str]:
    messages = body.get("messages") or []
    if not messages:
        return None
    return messages[-1].get("role")


def _tool_names(body: dict) -> list[str]:
    names: list[str] = []
    for tool in body.get("tools") or []:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(fn, dict) and fn.get("name"):
            names.append(str(fn["name"]))
        elif isinstance(tool, dict) and tool.get("name"):
            names.append(str(tool["name"]))
    return names


def _pick_fail_tool(body: dict, blob: str) -> Optional[str]:
    fail_names = [n for n in _tool_names(body) if "fail_" in n]
    if not fail_names:
        return None
    for key in ("fail_raise", "fail_structured", "fail_auth_text", "fail_auth"):
        if key in blob:
            for name in fail_names:
                if key in name:
                    return name
    if "mcp" in blob or "fail_" in blob:
        return fail_names[0]
    return None


def _intent(body: dict) -> str:
    """Return crash | hang | tool | text."""
    if _last_role(body) in {"tool", "function"}:
        return "text"
    blob = _last_user_blob(body)
    if any(
        token in blob
        for token in ("hang", "do not answer", "don't answer", "never respond", "stay silent")
    ):
        return "hang"
    if any(
        token in blob for token in ("crash", "break the model", "http 500", "explode")
    ):
        return "crash"
    if _pick_fail_tool(body, blob):
        return "tool"
    return "text"


def _completion_id() -> str:
    return f"chatcmpl-{uuid4().hex[:12]}"


def _text_body(body: dict, content: str) -> dict:
    return {
        "id": _completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model") or "dummy",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _tool_body(body: dict, tool_name: str) -> dict:
    return {
        "id": _completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model") or "dummy",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_dummy_fail",
                            "type": "function",
                            "function": {"name": tool_name, "arguments": "{}"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _chunk(cid: str, model: str, delta: dict, finish_reason: Optional[str] = None) -> dict:
    return {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


async def _sse_text(body: dict, content: str) -> AsyncIterator[bytes]:
    cid = _completion_id()
    model = body.get("model") or "dummy"
    for event in (
        _chunk(cid, model, {"role": "assistant", "content": content}),
        _chunk(cid, model, {}, "stop"),
    ):
        yield f"data: {json.dumps(event)}\n\n".encode()
    yield b"data: [DONE]\n\n"


async def _sse_tool(body: dict, tool_name: str) -> AsyncIterator[bytes]:
    cid = _completion_id()
    model = body.get("model") or "dummy"
    events = (
        _chunk(
            cid,
            model,
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_dummy_fail",
                        "type": "function",
                        "function": {"name": tool_name, "arguments": "{}"},
                    }
                ],
            },
        ),
        _chunk(cid, model, {}, "tool_calls"),
    )
    for event in events:
        yield f"data: {json.dumps(event)}\n\n".encode()
    yield b"data: [DONE]\n\n"


def _respond(body: dict, payload: dict, stream_iter: AsyncIterator[bytes]) -> Any:
    if body.get("stream"):
        return StreamingResponse(stream_iter, media_type="text/event-stream")
    return JSONResponse(payload)


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def smart_completions(request: Request) -> Any:
    body = await request.json()
    intent = _intent(body)
    if intent == "hang":
        await asyncio.sleep(_HANG_SECONDS)
        return JSONResponse(status_code=504, content={"error": {"message": "still hung"}})
    if intent == "crash":
        return JSONResponse(status_code=500, content=_CRASH_BODY)
    if intent == "tool":
        tool_name = _pick_fail_tool(body, _last_user_blob(body)) or "fail_auth_text"
        return _respond(body, _tool_body(body, tool_name), _sse_tool(body, tool_name))
    if _last_role(body) in {"tool", "function"}:
        text = "The dummy MCP tool returned an error, as expected for this fixture."
    else:
        text = (
            "Dummy LLM is up. Ask me to crash, hang / not answer, "
            "or to call fail_auth_text / fail_structured / fail_raise."
        )
    return _respond(body, _text_body(body, text), _sse_text(body, text))


@app.post("/crash/v1/chat/completions")
@app.post("/crash/chat/completions")
async def crash_completions() -> JSONResponse:
    return JSONResponse(status_code=500, content=_CRASH_BODY)


@app.post("/hang/v1/chat/completions")
@app.post("/hang/chat/completions")
async def hang_completions() -> JSONResponse:
    await asyncio.sleep(_HANG_SECONDS)
    return JSONResponse(status_code=504, content={"error": {"message": "still hung"}})


@app.get("/v1/models")
@app.get("/crash/v1/models")
@app.get("/hang/v1/models")
@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "modes": ["smart", "crash", "hang"]}
