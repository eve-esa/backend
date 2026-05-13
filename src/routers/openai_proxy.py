"""OpenAI-compatible gateway: user JWT on ingress, upstream API key on egress.

Requires ``OPENAI_PROXY_UPSTREAM_URL`` to be set; requests fall through to the
FastAPI app (404) when it is absent.
"""

import json
import logging
from typing import Optional

import httpx
from jose import JWTError

from src.config import OPENAI_PROXY_API_KEY, OPENAI_PROXY_UPSTREAM_URL
from src.middlewares.auth import verify_access_token
from src.services.openai_usage import track_usage

logger = logging.getLogger(__name__)

_STRIP_REQUEST_HEADERS = frozenset(
    {b"host", b"connection", b"keep-alive", b"transfer-encoding", b"te", b"trailer", b"upgrade",
     b"content-length"}  # recalculated by httpx after body is potentially rewritten
)
_STRIP_RESPONSE_HEADERS = frozenset(
    {"content-length", "content-encoding", "connection", "keep-alive", "transfer-encoding", "trailer", "upgrade"}
)


def _parse_usage(payload: dict) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """Extract (input_tokens, output_tokens, total_tokens) from an OpenAI usage block."""
    usage = payload.get("usage") or {}
    return usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("total_tokens")


def _parse_sse_chunks(chunks: list[bytes]) -> tuple[tuple, Optional[dict]]:
    """Parse SSE chunks into (usage_tuple, reconstructed_response_object).

    Reconstructs a non-streaming-style response dict by concatenating all
    content deltas, so the stored document mirrors the non-streaming shape.
    """
    text = b"".join(chunks).decode(errors="replace")
    payloads = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        try:
            payloads.append(json.loads(line[6:]))
        except json.JSONDecodeError:
            continue

    if not payloads:
        return (None, None, None), None

    # Usage — scan in reverse for a chunk that carries it
    usage = (None, None, None)
    for p in reversed(payloads):
        if p.get("usage"):
            usage = _parse_usage(p)
            break

    # Reconstruct final message by concatenating content deltas
    content = "".join(
        (choice.get("delta") or {}).get("content") or ""
        for p in payloads
        for choice in p.get("choices") or []
    )
    finish_reason = next(
        (
            choice.get("finish_reason")
            for p in reversed(payloads)
            for choice in p.get("choices") or []
            if choice.get("finish_reason")
        ),
        None,
    )
    first = payloads[0]
    reconstructed = {
        "id": first.get("id"),
        "object": "chat.completion",
        "created": first.get("created"),
        "model": first.get("model"),
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}],
    }
    return usage, reconstructed


class OpenAIProxyDispatcher:
    """
    ASGI middleware. Intercepts ``/v1/*`` requests, authenticates the user via
    JWT, forwards to the configured upstream OpenAI-compatible endpoint
    (``OPENAI_PROXY_UPSTREAM_URL``) replacing the Authorization header with the
    upstream API key, and records token usage to MongoDB.
    All other requests pass through to the FastAPI app unchanged.
    """

    def __init__(self, main_app):
        self.main_app = main_app
        upstream = OPENAI_PROXY_UPSTREAM_URL.rstrip("/")
        self._upstream: str | None = upstream or None
        self._client = httpx.AsyncClient(timeout=120.0) if upstream else None

    async def __call__(self, scope, receive, send):
        if self._upstream and scope["type"] == "http":
            path: str = scope.get("path", "")
            if path == "/v1" or path.startswith("/v1/"):
                try:
                    await self._proxy(scope, receive, send)
                except PermissionError as exc:
                    await self._send_error(send, 401, str(exc))
                except Exception as exc:
                    logger.exception("OpenAI proxy failed: %s", exc)
                    await self._send_error(send, 502, str(exc))
                return

        await self.main_app(scope, receive, send)

    async def _proxy(self, scope, receive, send):
        headers = dict(scope.get("headers", []))
        auth = headers.get(b"authorization", b"").decode()
        if not auth.startswith("Bearer "):
            raise PermissionError("Missing or malformed Authorization header")

        try:
            claims = verify_access_token(auth[7:])
        except JWTError as exc:
            raise PermissionError("Invalid token") from exc
        user_id: str = claims.get("sub", "")
        if not user_id:
            raise PermissionError("Invalid token payload")

        path: str = scope["path"]
        # Strip /v1 prefix so it isn't doubled when the upstream URL already ends with /v1
        upstream_path = path[3:] if path.startswith("/v1") else path
        query = scope.get("query_string", b"").decode()
        url = f"{self._upstream}{upstream_path}" + (f"?{query}" if query else "")
        method: str = scope.get("method", "GET")

        body = b""
        while True:
            event = await receive()
            body += event.get("body", b"")
            if not event.get("more_body", False):
                break

        # Extract model and stream flag; inject stream_options so the upstream
        # includes a usage chunk in the SSE stream.
        model: Optional[str] = None
        is_streaming = False
        req_body: Optional[dict] = None
        if body:
            try:
                req_body = json.loads(body)
                model = req_body.get("model")
                is_streaming = bool(req_body.get("stream", False))
                if is_streaming:
                    req_body.setdefault("stream_options", {})["include_usage"] = True
                    body = json.dumps(req_body).encode()
            except (json.JSONDecodeError, AttributeError):
                pass

        fwd_headers = {
            k.decode(): v.decode()
            for k, v in scope.get("headers", [])
            if k.lower() not in _STRIP_REQUEST_HEADERS
        }
        fwd_headers["authorization"] = f"Bearer {OPENAI_PROXY_API_KEY}" if OPENAI_PROXY_API_KEY else auth

        async with self._client.stream(method, url, headers=fwd_headers, content=body) as resp:
            resp_headers = [
                [k.lower().encode(), v.encode()]
                for k, v in resp.headers.items()
                if k.lower() not in _STRIP_RESPONSE_HEADERS
            ]
            await send({
                "type": "http.response.start",
                "status": resp.status_code,
                "headers": resp_headers,
            })

            if is_streaming:
                chunks: list[bytes] = []
                async for chunk in resp.aiter_bytes():
                    chunks.append(chunk)
                    await send({"type": "http.response.body", "body": chunk, "more_body": True})
                await send({"type": "http.response.body", "body": b""})

                (input_tokens, output_tokens, total_tokens), response_body = _parse_sse_chunks(chunks)
            else:
                chunks = []
                async for chunk in resp.aiter_bytes():
                    chunks.append(chunk)
                full_body = b"".join(chunks)
                await send({"type": "http.response.body", "body": full_body})

                input_tokens = output_tokens = total_tokens = None
                response_body = None
                try:
                    parsed = json.loads(full_body)
                    input_tokens, output_tokens, total_tokens = _parse_usage(parsed)
                    response_body = parsed
                except (json.JSONDecodeError, AttributeError):
                    response_body = full_body.decode(errors="replace")

        await track_usage(
            user_id=user_id,
            path=path,
            method=method,
            model=model,
            request_body=req_body,
            response_body=response_body,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            status_code=resp.status_code,
        )

    @staticmethod
    async def _send_error(send, status: int, detail: str):
        body = json.dumps({"detail": detail}).encode()
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [[b"content-type", b"application/json"]],
        })
        await send({"type": "http.response.body", "body": body})
