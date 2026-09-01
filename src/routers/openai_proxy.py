"""OpenAI-compatible gateway: EVE credential on ingress, upstream API key on egress.

Callers authenticate with ``Authorization: Bearer``, carrying either an ``eve_``
API key or a login JWT (see ``src/middlewares/auth.py``). That credential is
never relayed upstream: a provider is only usable once it has both a base URL
and a key of its own, and a provider missing either one answers 400.

Requires at least one provider upstream URL (``OPENAI_PROXY_UPSTREAM_URL`` for
RunPod / ``eve``, or ``EVE_JSC_BASE_URL`` for JSC); requests fall through to
the FastAPI app (404) when neither is set.

``GET /v1/models`` is answered locally, listing the providers that resolve,
rather than proxied: a GET carries no model name, so a passthrough would always
go to the default provider and could never list ``jsc/*``.

Model names may use LiteLLM-style provider prefixes
(``<provider>/<model-id>``; see OpenAI-compatible providers in LiteLLM docs).
The proxy strips only the provider segment and forwards the model id unchanged:

- ``eve/eve-esa/EVE-Instruct`` or bare ``eve-esa/EVE-Instruct`` -> RunPod (default)
- ``jsc/alias-eve`` -> JSC (Jülich)
"""

import asyncio
import json
import logging
import time
from typing import Optional

import httpx

from src.config import (
    EVE_JSC_API_KEY,
    EVE_JSC_BASE_URL,
    EVE_JSC_MODEL_NAME,
    MAIN_MODEL_API_KEY,
    MAIN_MODEL_NAME,
    OPENAI_PROXY_API_KEY,
    OPENAI_PROXY_UPSTREAM_URL,
)
from src.middlewares.auth import extract_bearer_token, resolve_principal_from_bearer_token
from src.services.oidc import IdentityProviderUnavailable
from src.services.openai_usage import track_usage

logger = logging.getLogger(__name__)

_STRIP_REQUEST_HEADERS = frozenset(
    {b"host", b"connection", b"keep-alive", b"transfer-encoding", b"te", b"trailer", b"upgrade",
     b"content-length",  # recalculated by httpx after body is potentially rewritten
     # Caller credentials. Never relay them to a third-party upstream: the
     # authorization header is replaced with the upstream key below, and the
     # frontend is same-origin with the API, so a browser call to /api/v1/*
     # would otherwise carry EVE session cookies to the provider.
     b"authorization", b"cookie"}
)
_STRIP_RESPONSE_HEADERS = frozenset(
    {"content-length", "content-encoding", "connection", "keep-alive", "transfer-encoding", "trailer", "upgrade"}
)

_DEFAULT_PROVIDER = "eve"
_RUNPOD_PROVIDERS = frozenset({"eve", "runpod"})
_KNOWN_PROVIDERS = _RUNPOD_PROVIDERS | {"jsc"}


def parse_proxy_model(model: Optional[str]) -> tuple[str, Optional[str]]:
    """Return (provider, upstream_model).

    LiteLLM-style ``provider/model-id``: only ``eve``, ``runpod``, and ``jsc``
    are stripped; the remainder is forwarded unchanged (e.g.
    ``eve/eve-esa/EVE-Instruct`` -> ``eve-esa/EVE-Instruct``).
    """
    if not model:
        return _DEFAULT_PROVIDER, None

    slug, sep, rest = model.partition("/")
    if sep and rest and slug.lower() in _KNOWN_PROVIDERS:
        return slug.lower(), rest

    return _DEFAULT_PROVIDER, model


def _jsc_upstream() -> tuple[str, str]:
    upstream = EVE_JSC_BASE_URL.rstrip("/")
    if not upstream:
        raise ValueError("JSC provider is not configured (EVE_JSC_BASE_URL)")
    if not EVE_JSC_API_KEY:
        raise ValueError("JSC provider is not configured (EVE_JSC_API_KEY)")
    return upstream, EVE_JSC_API_KEY


def _runpod_upstream() -> tuple[str, str]:
    upstream = OPENAI_PROXY_UPSTREAM_URL.rstrip("/")
    if not upstream:
        raise ValueError("EVE provider is not configured (OPENAI_PROXY_UPSTREAM_URL)")
    api_key = OPENAI_PROXY_API_KEY or MAIN_MODEL_API_KEY
    if not api_key:
        raise ValueError("EVE provider is not configured (OPENAI_PROXY_API_KEY)")
    return upstream, api_key


def _configured_models() -> dict:
    """An OpenAI ``/v1/models`` listing built from the providers that resolve.

    Not a passthrough. A GET carries no body, so ``parse_proxy_model(None)``
    always picks the default provider, and proxying the call would answer with
    one provider's catalogue while claiming to describe the whole gateway --
    never listing ``jsc/*`` at all. Clients call ``models.list()`` before their
    first completion, so this is the first thing a partner sees.
    """
    created = int(time.time())
    data = []
    for provider, resolver, model_name in (
        ("eve", _runpod_upstream, MAIN_MODEL_NAME),
        ("jsc", _jsc_upstream, EVE_JSC_MODEL_NAME),
    ):
        try:
            resolver()
        except ValueError:
            continue
        if not model_name:
            continue
        data.append({
            "id": f"{provider}/{model_name}",
            "object": "model",
            "created": created,
            "owned_by": provider,
        })
    return {"object": "list", "data": data}


def resolve_proxy_route(model: Optional[str]) -> tuple[str, str, Optional[str]]:
    """Return (upstream_base_url, api_key, upstream_model) for a proxy request."""
    provider, upstream_model = parse_proxy_model(model)
    if provider == "jsc":
        upstream_base, upstream_api_key = _jsc_upstream()
    elif provider in _RUNPOD_PROVIDERS:
        upstream_base, upstream_api_key = _runpod_upstream()
    else:
        raise ValueError(f"Unknown proxy provider: {provider}")
    return upstream_base, upstream_api_key, upstream_model


def _build_forward_body(
    body: bytes,
    req_body: Optional[dict],
    *,
    model: Optional[str],
    upstream_model: Optional[str],
    is_streaming: bool,
) -> bytes:
    if req_body is None:
        return body
    if not is_streaming and upstream_model == model:
        return body
    if upstream_model == model:
        return json.dumps(req_body).encode()
    return json.dumps({**req_body, "model": upstream_model}).encode()


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
    JWT, forwards to a provider-specific upstream OpenAI-compatible endpoint
    (``eve``/RunPod or ``jsc``) replacing the Authorization header with the
    upstream API key, and records token usage to MongoDB.
    All other requests pass through to the FastAPI app unchanged.
    """

    def __init__(self, main_app):
        self.main_app = main_app
        self._proxy_enabled = bool(
            OPENAI_PROXY_UPSTREAM_URL.strip() or EVE_JSC_BASE_URL.strip()
        )
        self._client: Optional[httpx.AsyncClient] = None
        self._client_loop: Optional[asyncio.AbstractEventLoop] = None
        self._log_provider_status()

    @staticmethod
    def _log_provider_status() -> None:
        """Say at startup which providers resolved and which are half-configured.

        A provider with a base URL but no key enables the dispatcher without
        being usable: /v1/* stops falling through to 404 and answers 400
        instead. Log it rather than raising, so one misconfigured provider
        cannot take down chat, auth and health along with itself.
        """
        for provider, resolver in (("eve", _runpod_upstream), ("jsc", _jsc_upstream)):
            try:
                upstream, _ = resolver()
            except ValueError as exc:
                # Absent entirely is a choice; configured-but-incomplete is a bug.
                base = OPENAI_PROXY_UPSTREAM_URL if provider == "eve" else EVE_JSC_BASE_URL
                if base.strip():
                    logger.error("OpenAI proxy: provider %r is unusable: %s", provider, exc)
                continue
            logger.info("OpenAI proxy: provider %r -> %s", provider, upstream)

    def _get_client(self) -> httpx.AsyncClient:
        """Return an httpx client bound to the current event loop.

        pytest-asyncio uses a fresh loop per test; reusing a client created on a
        prior loop causes ``Event loop is closed`` on later requests.
        """
        loop = asyncio.get_running_loop()
        if self._client is not None and self._client_loop is not None:
            if self._client_loop is not loop or self._client_loop.is_closed():
                self._client = None
                self._client_loop = None
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=120.0)
            self._client_loop = loop
        return self._client

    async def __call__(self, scope, receive, send):
        if self._proxy_enabled and scope["type"] == "http":
            path: str = scope.get("path", "")
            if path == "/v1" or path.startswith("/v1/"):
                try:
                    await self._proxy(scope, receive, send)
                except PermissionError as exc:
                    await self._send_error(send, 401, str(exc))
                except IdentityProviderUnavailable as exc:
                    # An unreachable IdP is neither a bad credential nor a bad
                    # upstream: without this branch it fell to the catch-all and
                    # answered 502 with a stack trace in the body.
                    logger.warning("Identity provider unavailable: %s", exc)
                    await self._send_error(send, 503, "Identity provider unavailable")
                except ValueError as exc:
                    await self._send_error(send, 400, str(exc))
                except Exception as exc:
                    logger.exception("OpenAI proxy failed: %s", exc)
                    await self._send_error(send, 502, str(exc))
                return

        await self.main_app(scope, receive, send)

    async def _proxy(self, scope, receive, send):
        headers = dict(scope.get("headers", []))
        token = extract_bearer_token(headers.get(b"authorization", b"").decode())
        if not token:
            raise PermissionError("Missing or malformed Authorization header")

        principal = await resolve_principal_from_bearer_token(token)
        caller_type = principal.caller_type()

        path: str = scope["path"]
        if path == "/v1/models" and scope.get("method", "GET").upper() == "GET":
            await self._send_json(send, 200, _configured_models())
            return

        # Strip /v1 prefix so it isn't doubled when the upstream URL already ends with /v1
        upstream_path = path[3:] if path.startswith("/v1") else path
        query = scope.get("query_string", b"").decode()
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
            except (json.JSONDecodeError, AttributeError):
                pass

        upstream_base, upstream_api_key, upstream_model = resolve_proxy_route(model)
        url = f"{upstream_base}{upstream_path}" + (f"?{query}" if query else "")

        fwd_body = _build_forward_body(
            body,
            req_body,
            model=model,
            upstream_model=upstream_model,
            is_streaming=is_streaming,
        )

        fwd_headers = {
            k.decode(): v.decode()
            for k, v in scope.get("headers", [])
            if k.lower() not in _STRIP_REQUEST_HEADERS
        }
        # resolve_proxy_route guarantees a non-empty key; never fall back to the
        # caller's own EVE credential, which is meaningless upstream and would
        # hand a third party a working token for this API.
        fwd_headers["authorization"] = f"Bearer {upstream_api_key}"

        started = time.monotonic()
        client = self._get_client()
        async with client.stream(method, url, headers=fwd_headers, content=fwd_body) as resp:
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

        latency_ms = (time.monotonic() - started) * 1000
        await track_usage(
            user_id=principal.user_id,
            caller_type=caller_type,
            api_key_id=principal.api_key_id,
            path=path,
            method=method,
            model=model,
            streaming=is_streaming,
            request_body=req_body,
            response_body=response_body,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            status_code=resp.status_code,
            outcome="success" if resp.status_code < 400 else "error",
            latency_ms=latency_ms,
        )

    @staticmethod
    async def _send_error(send, status: int, detail: str):
        await OpenAIProxyDispatcher._send_json(send, status, {"detail": detail})

    @staticmethod
    async def _send_json(send, status: int, payload: dict):
        body = json.dumps(payload).encode()
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [[b"content-type", b"application/json"]],
        })
        await send({"type": "http.response.body", "body": body})
