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

Every other request is allowlisted: only ``POST`` to exactly
``/v1/chat/completions``, ``/v1/completions`` or ``/v1/embeddings`` is
forwarded upstream, everything else answers 404 before any upstream call.
The match is exact on the path as received: the gateway only ever sends EVE's
upstream key to the endpoints it bills. Usage on the three
forwarded paths counts against the caller's monthly token budget
(``src.services.token_rate_limiter``): once the body is parsed, a rough
pre-flight estimate of it is reserved against the cap with one atomic
conditional increment, before any upstream call, so concurrent requests
cannot all read the same stale usage and all get admitted past the cap; the
reservation is then settled against the real charge once the response is in,
in a ``finally``, refunded in full on every no-charge path, so an aborted
stream still pays for what it already produced and a rejected or failed one
pays nothing.

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
from src.database.models.user import User
from src.middlewares.auth import (
    extract_bearer_token,
    resolve_principal_from_bearer_token,
)
from src.services.approval import PENDING_APPROVAL_DETAIL, ApprovalPending
from src.services.oidc import IdentityProviderUnavailable
from src.services.openai_usage import track_usage
from src.services.token_rate_limiter import (
    TokenBudgetExceeded,
    count_tokens_for_texts,
    reserve_token_budget,
    settle_reserved_tokens,
)

logger = logging.getLogger(__name__)

# Only these three POST endpoints are forwarded; GET /v1/models is answered
# locally above this check and never reaches it. Anything else, including a
# path that is not in canonical form, gets a flat 404 with no upstream call.
_BILLABLE_PATHS = frozenset({"/v1/chat/completions", "/v1/completions", "/v1/embeddings"})

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

    # Reconstruct final message by concatenating content deltas. Chat
    # completions carry the delta under choice["delta"]["content"]; legacy
    # completions stream choice["text"] directly with no "delta" wrapper.
    content = "".join(
        (choice.get("delta") or {}).get("content") or choice.get("text") or ""
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


def _message_content_texts(content) -> list[str]:
    """Plain-text parts of a chat message ``content`` field (string or block list)."""
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return [
            part.get("text")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
        ]
    return []


def _request_texts_for_estimate(req_body: Optional[dict]) -> tuple[list[str], int]:
    """Text parts and pre-tokenized array length of a proxy request body.

    Covers the three billable shapes: chat ``messages[].content``, legacy
    ``prompt`` (str or list of str), and embeddings ``input`` (str, list of
    str, or a token id array / list of token id arrays, which is already
    tokenized and counts by its length rather than through the text counter).
    """
    if not isinstance(req_body, dict):
        return [], 0

    texts: list[str] = []
    for message in req_body.get("messages") or []:
        if isinstance(message, dict):
            texts.extend(_message_content_texts(message.get("content")))

    prompt = req_body.get("prompt")
    if isinstance(prompt, str):
        texts.append(prompt)
    elif isinstance(prompt, list):
        texts.extend(item for item in prompt if isinstance(item, str))

    token_array_len = 0
    input_value = req_body.get("input")
    if isinstance(input_value, str):
        texts.append(input_value)
    elif isinstance(input_value, list) and input_value:
        if all(isinstance(v, int) for v in input_value):
            token_array_len += len(input_value)
        else:
            for item in input_value:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, list) and item and all(isinstance(v, int) for v in item):
                    token_array_len += len(item)

    return texts, token_array_len


def _response_texts_for_estimate(response_body) -> list[str]:
    """Plain-text parts of a (possibly SSE-reconstructed) proxy response body."""
    if not isinstance(response_body, dict):
        return []
    texts: list[str] = []
    for choice in response_body.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            texts.append(message["content"])
        if isinstance(choice.get("text"), str):
            texts.append(choice["text"])
    return texts


def _tokens_to_bill(
    *,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    total_tokens: Optional[int],
    req_body: Optional[dict],
    response_body,
) -> int:
    """What ``consume_tokens_for_user`` charges for one proxied call.

    Prefer what the upstream actually reported (``total_tokens``, else
    ``prompt_tokens + completion_tokens``); fall back to counting the request
    and response text ourselves only when the upstream gave nothing usable.
    Always at least 1, so a request that produced no counted text still
    spends something rather than being effectively free.
    """
    reported: Optional[int] = None
    if isinstance(total_tokens, int) and total_tokens >= 0:
        reported = total_tokens
    elif input_tokens is not None or output_tokens is not None:
        reported = int(input_tokens or 0) + int(output_tokens or 0)

    if reported and reported > 0:
        return reported

    req_texts, token_array_len = _request_texts_for_estimate(req_body)
    resp_texts = _response_texts_for_estimate(response_body)
    estimate = count_tokens_for_texts(*req_texts, *resp_texts) + token_array_len
    return max(estimate, 1)


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
                # Mutable so _proxy can tell us it already sent
                # "http.response.start": once that has happened, none of the
                # except branches below may send anything of their own, or
                # a mid-stream failure would try to start the response twice.
                response_started = {"value": False}
                try:
                    await self._proxy(scope, receive, send, response_started)
                except PermissionError as exc:
                    if response_started["value"]:
                        raise
                    await self._send_error(send, 401, str(exc))
                except ApprovalPending:
                    # Same body the REST routes answer with, so a client can
                    # branch on the code whichever door it came through.
                    if response_started["value"]:
                        raise
                    await self._send_json(
                        send, 403, {"detail": PENDING_APPROVAL_DETAIL}
                    )
                except IdentityProviderUnavailable as exc:
                    # An unreachable IdP is neither a bad credential nor a bad
                    # upstream: without this branch it fell to the catch-all and
                    # answered 502 with a stack trace in the body.
                    if response_started["value"]:
                        raise
                    logger.warning("Identity provider unavailable: %s", exc)
                    await self._send_error(send, 503, "Identity provider unavailable")
                except ValueError as exc:
                    if response_started["value"]:
                        raise
                    await self._send_error(send, 400, str(exc))
                except Exception as exc:
                    logger.exception("OpenAI proxy failed: %s", exc)
                    if response_started["value"]:
                        # The response already started (status and maybe some
                        # body sent): a second "http.response.start" is a
                        # protocol violation, and there is no clean error
                        # shape to send mid-body. Re-raise so the ASGI server
                        # aborts the connection the way it would for any other
                        # exception after the response line went out, instead
                        # of silently returning as if nothing had happened.
                        raise
                    await self._send_error(send, 502, str(exc))
                return

        await self.main_app(scope, receive, send)

    async def _proxy(self, scope, receive, send, response_started: dict):
        headers = dict(scope.get("headers", []))
        token = extract_bearer_token(headers.get(b"authorization", b"").decode())
        if not token:
            raise PermissionError("Missing or malformed Authorization header")

        principal = await resolve_principal_from_bearer_token(token)
        caller_type = principal.caller_type()

        path: str = scope["path"]
        method: str = scope.get("method", "GET").upper()

        if path == "/v1/models" and method == "GET":
            await self._send_json(send, 200, _configured_models())
            response_started["value"] = True
            return

        # Allowlist, checked before touching the body or the budget: anything
        # that isn't a POST to exactly one of the three billable paths is a
        # 404, including a non-canonical path (see module docstring) and any
        # other verb on a billable path.
        if method != "POST" or path not in _BILLABLE_PATHS:
            await self._send_json(send, 404, {"detail": "Not Found"})
            response_started["value"] = True
            return

        user = await User.find_by_id(principal.user_id)
        if not user:
            raise PermissionError("User not found")

        # Strip /v1 prefix so it isn't doubled when the upstream URL already ends with /v1
        upstream_path = path[3:] if path.startswith("/v1") else path
        query = scope.get("query_string", b"").decode()

        body = b""
        while True:
            event = await receive()
            body += event.get("body", b"")
            if not event.get("more_body", False):
                break

        # Extract model and stream flag; inject stream_options so the upstream
        # includes a usage chunk in the SSE stream. A caller-supplied
        # "stream_options": null used to reach setdefault's dict-only branch
        # and crash with TypeError -> 502; this coerces anything that isn't
        # already a dict before adding include_usage.
        model: Optional[str] = None
        is_streaming = False
        req_body: Optional[dict] = None
        if body:
            try:
                req_body = json.loads(body)
                model = req_body.get("model")
                is_streaming = bool(req_body.get("stream", False))
                if is_streaming:
                    stream_options = req_body.get("stream_options")
                    stream_options = stream_options if isinstance(stream_options, dict) else {}
                    stream_options["include_usage"] = True
                    req_body["stream_options"] = stream_options
            except (json.JSONDecodeError, AttributeError):
                pass

        # Reserve against the budget now, atomically, before any upstream call:
        # the pre-flight estimate is deliberately rough (same request-text
        # counter _tokens_to_bill falls back to), it only has to bound how far
        # a concurrent batch can overshoot the cap, not predict the exact
        # charge. See reserve_token_budget for why this replaces a plain
        # check-then-act read.
        # Route first: an unconfigured provider raises ValueError (answered 400)
        # outside the try/finally below that settles the reservation, so
        # reserving before this would leave those tokens held for good.
        upstream_base, upstream_api_key, upstream_model = resolve_proxy_route(model)

        req_texts, token_array_len = _request_texts_for_estimate(req_body)
        estimated_tokens = count_tokens_for_texts(*req_texts) + token_array_len
        exceeded, reserved_tokens = await reserve_token_budget(user, estimated_tokens)
        if exceeded is not None:
            await self._send_budget_exceeded(send, exceeded)
            response_started["value"] = True
            return

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
        status: Optional[int] = None
        response_body = None
        input_tokens = output_tokens = total_tokens = None
        try:
            client = self._get_client()
            async with client.stream(method, url, headers=fwd_headers, content=fwd_body) as resp:
                status = resp.status_code
                resp_headers = [
                    [k.lower().encode(), v.encode()]
                    for k, v in resp.headers.items()
                    if k.lower() not in _STRIP_RESPONSE_HEADERS
                ]
                await send({
                    "type": "http.response.start",
                    "status": status,
                    "headers": resp_headers,
                })
                response_started["value"] = True

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

                    try:
                        parsed = json.loads(full_body)
                        input_tokens, output_tokens, total_tokens = _parse_usage(parsed)
                        response_body = parsed
                    except (json.JSONDecodeError, AttributeError):
                        response_body = full_body.decode(errors="replace")
        finally:
            # Only tracked if a response was actually obtained: an upstream we
            # could not even reach (connection error, DNS failure, ...) leaves
            # status None, and that is deliberately not tracked here -- the
            # caller gets the same 502 as before, and there is nothing yet to
            # record. A mid-stream failure after the response started still
            # lands here with whatever partial chunks were collected, which is
            # exactly what lets an aborted stream still be charged for what it
            # produced. The reservation itself, though, always needs settling
            # here regardless of status: a connection error never billed
            # anything, and the whole pre-flight claim must be refunded.
            billed_tokens = 0
            if status is not None and status < 400:
                billed_tokens = _tokens_to_bill(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    req_body=req_body,
                    response_body=response_body,
                )

            try:
                await settle_reserved_tokens(
                    user,
                    reserved_tokens,
                    billed_tokens if (status is not None and status < 400) else None,
                )
            except Exception:
                logger.exception(
                    "Failed to settle reserved token budget for user %s", principal.user_id
                )

            if status is not None:
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
                    status_code=status,
                    outcome="success" if status < 400 else "error",
                    latency_ms=latency_ms,
                    billed_tokens=billed_tokens,
                )

    @staticmethod
    async def _send_budget_exceeded(send, exceeded: TokenBudgetExceeded) -> None:
        message = exceeded.message
        headers = [
            [b"content-type", b"application/json"],
            [b"x-should-retry", b"false"],
        ]
        retry_after = exceeded.retry_after_seconds()
        if retry_after is not None:
            headers.append([b"retry-after", str(retry_after).encode()])
        body = json.dumps({
            "detail": message,
            "error": {
                "message": message,
                "type": "insufficient_quota",
                "code": "token_budget_exceeded",
                "param": None,
            },
        }).encode()
        await send({"type": "http.response.start", "status": 429, "headers": headers})
        await send({"type": "http.response.body", "body": body})

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
