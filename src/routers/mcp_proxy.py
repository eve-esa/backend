"""MCP gateway: user JWT on ingress, Cognito M2M on egress via FastMCP proxy.

Requires ``fastmcp>=3`` so ``create_proxy`` is exported from ``fastmcp.server``; see
https://gofastmcp.com/servers/providers/proxy
"""

import asyncio
import contextvars
import functools
import json
import logging
import time
from collections.abc import AsyncGenerator, Mapping
from contextvars import ContextVar
from typing import Any, NamedTuple

import fastmcp.client.transports.http as fastmcp_http_transport
import httpx
from fastmcp import settings as fastmcp_settings
from fastmcp.client.transports.http import StreamableHttpTransport
from fastmcp.server import create_proxy
from fastmcp.server.middleware.caching import ResponseCachingMiddleware
from fastmcp.server.providers.proxy import ProxyClient
from mcp.client.streamable_http import streamable_http_client
from src.config import MCP_TOOLS_CACHE_TTL
from src.database.models.mcp_server import RESERVED_CREDENTIAL_HEADERS
from src.database.mongo import get_collection
from src.middlewares.auth import (
    Principal,
    extract_bearer_token,
    resolve_principal_from_bearer_token,
)
from src.services.approval import PENDING_APPROVAL_DETAIL, ApprovalPending
from src.services.mcp.auth import CognitoTokenProvider, get_cognito_token_provider
from src.services.mcp.usage import track_usage
from src.services.mcp.user_credentials import (
    CredentialsUnreadable,
    effective_spec,
    headers_for,
)
from src.services.mcp.user_token_scope import (
    announce_user_token_forwarded,
    forwards_user_token,
)
from src.services.oidc import IdentityProviderUnavailable
from src.services.request_rate_limiter import (
    LIMITER_UNAVAILABLE_DETAIL,
    RateLimiterUnavailable,
    RequestRateLimited,
    check_or_raise,
)

logger = logging.getLogger(__name__)

# ``ProxyClient`` opens a fresh upstream session per proxied request and, on
# close, sends a DELETE for the ``Mcp-Session-Id`` AgentCore hands out.
# AgentCore runtimes are stateless and answer 404, logged as a WARNING by
# ``mcp.client.streamable_http`` on every tool call. ``StreamableHttpTransport``
# (fastmcp 3.x) takes no ``terminate_on_close``, so rebind the name it calls.
# The proxy is the only fastmcp client in this process.
fastmcp_http_transport.streamable_http_client = functools.partial(
    streamable_http_client, terminate_on_close=False
)

_proxy_apps: dict[str, Any] = {}
# One owner task per sub-app holds its lifespan: (stop event, task).
_proxy_lifespans: dict[str, tuple[asyncio.Event, asyncio.Task]] = {}
# Total bound for closing every sub-app, then for the cancelled leftovers.
_PROXY_SHUTDOWN_TIMEOUT_S = 5.0
_PROXY_CANCEL_GRACE_S = 1.0
_proxy_build_lock = asyncio.Lock()

# Per-request bearer token of the authenticated caller (an EVE access token or
# ``eve_`` API key). Set by ``MCPProxyDispatcher`` around each proxied call and
# read by ``_DynamicBearerAuth`` so the upstream MCP server can impersonate the
# user on downstream EVE API calls (e.g. ``eve_retrieval`` → ``POST /retrieve``).
# Without it the MCP server falls back to its shared ``EVE_API_KEY`` and
# user-scoped private collections are dropped by the backend's ownership check.
# Set only for servers allowed to receive it (``user_token_scope``), else None.
_user_eve_token_var: ContextVar[str | None] = ContextVar(
    "mcp_proxy_user_eve_token", default=None
)

# Per-request headers built from the caller's own toolkit credentials, set only
# around a ``tools/call`` to an operator row that declares a credential spec.
# Read by ``_DynamicBearerAuth``. The proxy app is shared across users, so the
# value lives for one proxied call and is reset when it ends.
_upstream_credential_headers_var: ContextVar[dict[str, str] | None] = ContextVar(
    "mcp_proxy_upstream_credential_headers", default=None
)

_REDACTED = "[redacted]"

_CREDENTIALS_MISSING_TEXT = (
    "This toolkit needs your own credentials. Add them in Settings, Toolkits, then try again."
)
_CREDENTIALS_UNREADABLE_TEXT = (
    "Your saved credentials for this toolkit could not be read. "
    "Add them again in Settings, Toolkits."
)
_CREDENTIALS_UNAVAILABLE_TEXT = (
    "Your saved credentials for this toolkit could not be read right now. "
    "Try again in a moment."
)


class _CachedServer(NamedTuple):
    url: str
    forward_token: bool
    server_id: str | None
    # ``name``, ``user_id``, ``user_credentials`` and ``user_environment`` of the
    # row, or None when the row declares no credentials. ``effective_spec`` reads
    # it per call, so the feature flag is honoured without waiting for the cache
    # to expire.
    credential_row: dict | None
    expires_at: float


class _ResolvedProxy(NamedTuple):
    proxy_app: Any
    principal: Principal
    user_token: str | None
    server_id: str | None
    credential_row: dict | None


class _ToolCall(NamedTuple):
    request_id: Any
    name: str | None
    arguments: Any


class _UpstreamCredentials(NamedTuple):
    headers: dict[str, str] | None
    # ``user`` when the caller's own values are sent, ``platform`` when an
    # optional spec found none, ``none`` when the call is refused and nothing is
    # sent, None when the row has no spec.
    source: str | None
    # Text of the tool error to answer instead of calling upstream, or None.
    refusal: str | None


# Cache of (server_name, user_id) to the resolved row (see ``_CachedServer``).
# Avoids a MongoDB round-trip on every proxy request when the result is stable.
# TTL matches MCP_TOOLS_CACHE_TTL so both caches expire together.
# Per-server_name locks ensure at most one concurrent cold-miss fetch.
_server_url_cache: dict[tuple[str, str], _CachedServer] = {}
_server_url_locks: dict[str, asyncio.Lock] = {}
_server_url_registry_lock = asyncio.Lock()


async def _get_server_url_lock(server_name: str) -> asyncio.Lock:
    if server_name not in _server_url_locks:
        async with _server_url_registry_lock:
            if server_name not in _server_url_locks:
                _server_url_locks[server_name] = asyncio.Lock()
    return _server_url_locks[server_name]


class _DynamicBearerAuth(httpx.Auth):
    """httpx Auth that fetches a fresh Cognito token on every request.

    This prevents the proxy from getting stuck with a stale cached token after
    the Cognito JWT expires.  ``CognitoTokenProvider.get_token()`` is already
    internally cached and only hits the network when the token approaches
    expiry, so there is no performance penalty on the hot path.

    On a 401 response from AgentCore the auth flow invalidates the token cache
    and retries the request exactly once with a freshly-issued token.  httpx
    supports this two-yield pattern natively via ``async_auth_flow``.
    """

    def __init__(self, provider: CognitoTokenProvider) -> None:
        self._provider = provider

    def auth_flow(self, request: httpx.Request):
        # Sync fallback — should not be reached by httpx.AsyncClient.
        yield request  # pragma: no cover

    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        token = await self._provider.get_token()
        request.headers["Authorization"] = f"Bearer {token}"

        # Forward the caller's EVE credential so the MCP server impersonates
        # the user on its downstream EVE API call (``X-EVE-Token`` → the
        # server's ``_resolve_eve_token``). The Cognito ``Authorization`` above
        # authenticates the proxy to AgentCore; this authenticates the user to
        # EVE. Without it the MCP server uses its shared ``EVE_API_KEY`` and
        # user-scoped private collections are silently dropped.
        user_eve_token = _user_eve_token_var.get()
        if user_eve_token:
            request.headers["X-EVE-Token"] = user_eve_token

        # The caller's own toolkit credentials, set only on a tools/call to a
        # row that declares a spec. Added after the two headers above and never
        # over them, whatever the spec says.
        credential_headers = _upstream_credential_headers_var.get()
        if credential_headers:
            for name, value in credential_headers.items():
                if name.lower() not in RESERVED_CREDENTIAL_HEADERS:
                    request.headers[name] = value

        # Yield the request; httpx sends it and feeds the response back via send().
        response = yield request

        if response.status_code != 401:
            # Happy path — nothing more to do.
            return

        # ------------------------------------------------------------------ #
        # AgentCore returned 401.  This is intermittent and happens when the  #
        # cached Cognito token expires between our local expiry check and the  #
        # actual HTTP round-trip (clock skew / narrow expiry window).          #
        # Strategy: invalidate the in-memory cache, fetch a fresh token from  #
        # Cognito, and retry the request exactly once.                         #
        # ------------------------------------------------------------------ #
        url_hint = str(request.url)[:80]
        logger.warning(
            "[MCP proxy] AgentCore returned 401 Unauthorized for %s – "
            "Cognito token may be stale. Invalidating cache and retrying with "
            "a fresh token (single retry).",
            url_hint,
        )
        self._provider.invalidate()
        token = await self._provider.get_token()
        request.headers["Authorization"] = f"Bearer {token}"
        logger.info(
            "[MCP proxy] Retrying request to %s with freshly-issued Cognito token.",
            url_hint,
        )
        yield request


async def build_proxy_app(agentcore_url: str, provider: CognitoTokenProvider):
    """Return a cached ASGI app that proxies MCP to ``agentcore_url`` with Cognito auth.

    The proxy is cached by URL only.  Authentication is handled dynamically
    via ``_DynamicBearerAuth`` so tokens are always fresh regardless of how
    long the proxy has been alive.
    """
    cache_key = agentcore_url
    if cache_key in _proxy_apps:
        return _proxy_apps[cache_key]

    async with _proxy_build_lock:
        if cache_key in _proxy_apps:
            return _proxy_apps[cache_key]
        transport = StreamableHttpTransport(
            agentcore_url,
            auth=_DynamicBearerAuth(provider),
        )
        backend = ProxyClient(transport)
        proxy = create_proxy(backend, name=f"proxy-{agentcore_url[-8:]}")

        # Cache the `tools/list` JSON-RPC response at the proxy server layer so
        # that the MCP SDK's per-call output-schema validation
        # (`ClientSession._validate_tool_result` → `list_tools()` in
        # mcp/client/session.py) does not fan out to AgentCore on every
        # `tools/call`.  FastMCP's built-in `ProxyProvider._tools_cache` only
        # accelerates internal `_get_tool(name)` lookups; it does not
        # short-circuit incoming `tools/list` requests
        # (https://gofastmcp.com/servers/providers/proxy#component-list-caching).
        # Explicitly disable `call_tool` caching — tool execution must always
        # reach upstream (default TTL is 1 h, which would silently serve stale
        # results).
        proxy.add_middleware(
            ResponseCachingMiddleware(
                list_tools_settings={
                    "enabled": True,
                    "ttl": int(MCP_TOOLS_CACHE_TTL),
                },
                list_resources_settings={"enabled": False},
                list_prompts_settings={"enabled": False},
                read_resource_settings={"enabled": False},
                get_prompt_settings={"enabled": False},
                call_tool_settings={"enabled": False},
            )
        )
        http_app = proxy.http_app(stateless_http=True)
        # The lifespan must be exited by the task and context that entered it
        # (anyio cancel scopes, fastmcp context variables). Entered in the
        # request task and exited from the server lifespan it raised on every
        # rollover, so a dedicated task with a fresh context owns it.
        started = asyncio.get_running_loop().create_future()
        stop = asyncio.Event()
        task = asyncio.create_task(
            _hold_proxy_lifespan(http_app, started, stop),
            name=f"mcp-proxy-lifespan-{cache_key[-8:]}",
            context=contextvars.Context(),
        )
        try:
            await started
        except BaseException:
            stop.set()
            raise
        _proxy_lifespans[cache_key] = (stop, task)
        _proxy_apps[cache_key] = http_app
        task.add_done_callback(functools.partial(_evict_dead_proxy, cache_key))
        logger.info("MCP proxy sub-app started (lifespan): %s", cache_key[:64])
        return http_app


async def _hold_proxy_lifespan(
    http_app: Any, started: asyncio.Future, stop: asyncio.Event
) -> None:
    """Enter ``http_app``'s lifespan, report it started, hold it until ``stop``."""
    try:
        async with http_app.lifespan(http_app):
            if started.done():
                # The request that started it was cancelled: close again.
                return
            started.set_result(None)
            await stop.wait()
    except asyncio.CancelledError:
        if not started.done():
            started.cancel()
        raise
    except Exception as exc:
        if not started.done():
            started.set_exception(exc)
            return
        if started.cancelled():
            return
        raise


def _task_failure(task: asyncio.Task) -> str | None:
    """Exception class name a finished owner task ended with, or ``None``."""
    if task.cancelled():
        return "CancelledError"
    exc = task.exception()
    return type(exc).__name__ if exc is not None else None


def _evict_dead_proxy(cache_key: str, task: asyncio.Task) -> None:
    """Drop a sub-app whose owner task ended outside a shutdown.

    Its task group is gone, so every later call would fail until the worker
    restarts; the next request starts a fresh sub-app instead.
    """
    entry = _proxy_lifespans.get(cache_key)
    if entry is None or entry[1] is not task:
        # Taken by shutdown_mcp_proxy_lifespans, which reports it.
        return
    del _proxy_lifespans[cache_key]
    _proxy_apps.pop(cache_key, None)
    logger.warning(
        "MCP proxy sub-app stopped unexpectedly type=%s server=%s",
        _task_failure(task) or "exited",
        cache_key[:64],
    )


async def shutdown_mcp_proxy_lifespans() -> None:
    """Close all FastMCP ``http_app()`` lifespans (StreamableHTTPSessionManager task groups).

    Never raises: a sub-app that fails to close logs one WARNING with the
    exception class and the others still close.
    """
    async with _proxy_build_lock:
        lifespans = list(_proxy_lifespans.items())
        _proxy_lifespans.clear()
        _proxy_apps.clear()
        if not lifespans:
            return
        for _key, (stop, _task) in lifespans:
            stop.set()
        tasks = {task: key for key, (_stop, task) in lifespans}
        _, pending = await asyncio.wait(set(tasks), timeout=_PROXY_SHUTDOWN_TIMEOUT_S)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.wait(pending, timeout=_PROXY_CANCEL_GRACE_S)
        for task, key in tasks.items():
            if task in pending:
                if task.done() and not task.cancelled():
                    task.exception()  # retrieved, so asyncio does not log it
                failure = "TimeoutError"
            else:
                failure = _task_failure(task)
            if failure is None:
                continue
            logger.warning(
                "MCP proxy sub-app shutdown failed type=%s server=%s",
                failure,
                key[:64],
            )


def _extract_tool_call(body: bytes) -> _ToolCall | None:
    """Return the id, tool name and arguments of a JSON-RPC ``tools/call`` request.

    MCP messages are JSON-RPC 2.0 over the Streamable HTTP transport: a request
    object (or a batch array) carrying ``method`` and ``params``; ``tools/call``
    puts the tool name in ``params.name`` and the bulky args in
    ``params.arguments``.
    See https://modelcontextprotocol.io/specification (Streamable HTTP transport).

    Returns ``None`` for anything that is not a ``tools/call`` (``initialize``,
    ``tools/list``, notifications, ``ping``, GET/SSE opens with no body, …) so
    those housekeeping hops are not logged, only real tool invocations are.
    """
    if not body:
        return None
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return None

    messages = parsed if isinstance(parsed, list) else [parsed]
    for msg in messages:
        if isinstance(msg, dict) and msg.get("method") == "tools/call":
            params = msg.get("params") or {}
            if not isinstance(params, dict):
                params = {}
            return _ToolCall(msg.get("id"), params.get("name"), params.get("arguments"))
    return None


def _redact_values(payload: Any, values: Mapping[str, str] | None) -> Any:
    """``payload`` with every sent credential value replaced by ``[redacted]``.

    An upstream can echo a credential in an error, so the stored usage payload
    must never hold one. The payload is serialised and each value is replaced
    raw, JSON-escaped once and twice (a JSON document inside a JSON string), then
    parsed back. Only stored data is touched, never what streams to the caller.
    """
    secrets = sorted({v for v in (values or {}).values() if v}, key=len, reverse=True)
    if payload is None or not secrets:
        return payload
    text = json.dumps(payload, default=str)
    for secret in secrets:
        once = json.dumps(secret)[1:-1]
        twice = json.dumps(once)[1:-1]
        for variant in dict.fromkeys((twice, once, secret)):
            text = text.replace(variant, _REDACTED)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # A value that spans JSON syntax broke the document: keep the redacted text.
        return text


def _tool_error_body(request_id: Any, text: str) -> dict:
    """An MCP tool result that reports failure, so the model reads ``text``."""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {"content": [{"type": "text", "text": text}], "isError": True},
    }


async def _upstream_credentials(
    *, user_id: str, server_id: str | None, credential_row: Mapping | None, server_name: str
) -> _UpstreamCredentials:
    """Decide what a tools/call sends upstream from the caller's own credentials.

    Never raises for a credential problem: the caller gets a tool error instead
    of a failed transport. No value is ever logged.
    """
    if credential_row is None or server_id is None:
        return _UpstreamCredentials(None, None, None)
    spec = effective_spec(credential_row)
    if spec is None:
        return _UpstreamCredentials(None, None, None)
    try:
        headers = await headers_for(user_id=user_id, server_id=server_id, spec=spec)
    except CredentialsUnreadable:
        # headers_for already logged the row at ERROR.
        logger.warning("mcp.user_credentials.refused server=%s reason=unreadable", server_name)
        return _UpstreamCredentials(None, "none", _CREDENTIALS_UNREADABLE_TEXT)
    except Exception as exc:
        # KMS outage or missing key config: the class only, the message may hold data.
        logger.error(
            "mcp.user_credentials.read_failed server=%s error=%s",
            server_name,
            type(exc).__name__,
        )
        return _UpstreamCredentials(None, "none", _CREDENTIALS_UNAVAILABLE_TEXT)
    if headers:
        return _UpstreamCredentials(headers, "user", None)
    if spec.required:
        logger.info("mcp.user_credentials.refused server=%s reason=missing", server_name)
        return _UpstreamCredentials(None, "none", _CREDENTIALS_MISSING_TEXT)
    return _UpstreamCredentials(None, "platform", None)


def _parse_mcp_messages(raw: bytes) -> list[dict] | None:
    """Parse the JSON-RPC message(s) out of an MCP Streamable HTTP response body.

    Responses arrive as Server-Sent Events (``event: message`` / ``data: {json}``)
    or, less commonly, as a bare JSON object/array. This decodes them into
    structured dicts so the stored payload is queryable (rather than an opaque
    SSE string). Returns ``None`` when nothing could be parsed (opaque/binary
    body) so the caller can fall back to storing the raw bytes verbatim.
    See https://modelcontextprotocol.io/specification (Streamable HTTP transport).
    """
    if not raw:
        return None
    text = raw.decode(errors="replace")
    messages: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            payload = line[5:].strip()
        elif line.startswith("{") or line.startswith("["):
            payload = line
        else:
            continue
        try:
            parsed = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            continue
        for msg in parsed if isinstance(parsed, list) else [parsed]:
            if isinstance(msg, dict):
                messages.append(msg)
    return messages or None


def _is_error_message(messages: list[dict] | None) -> bool | None:
    """Detect an MCP-level failure across parsed JSON-RPC messages.

    A ``tools/call`` can fail in two ways that both return HTTP 200, so HTTP
    status alone is not enough:
      * a JSON-RPC protocol ``error`` object, or
      * a ``result`` whose ``isError`` flag is ``true`` (the tool ran but
        reported failure).
    Returns ``None`` when no message could be parsed.
    """
    if messages is None:
        return None
    for msg in messages:
        if msg.get("error") is not None:
            return True
        result = msg.get("result")
        if isinstance(result, dict) and result.get("isError"):
            return True
    return False


def _tool_call_outcome(*, http_failed: bool, is_error: bool | None) -> str:
    """Map HTTP + MCP parse/error signals to a usage ``outcome`` value."""
    if http_failed or is_error is True:
        return "error"
    if is_error is None:
        return "unknown"
    return "success"


def _replay_receive(body: bytes, original_receive):
    """Build an ASGI ``receive`` that replays an already-consumed request body.

    The dispatcher must read the body to inspect the JSON-RPC method, but the
    downstream FastMCP sub-app still needs to read it. This yields the buffered
    body once, then delegates to ``original_receive`` so genuine
    ``http.disconnect`` events still reach the sub-app (critical for long-lived
    SSE streams, which poll ``receive`` to detect client disconnect).
    """
    sent = False

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return await original_receive()

    return receive


class MCPProxyDispatcher:
    """
    Outermost ASGI middleware. Intercepts ``/mcp/{server_name}/*`` requests,
    authenticates the user, resolves the server from MongoDB, rewrites the path,
    and forwards to a FastMCP proxy sub-app (Cognito on upstream).
    All other requests pass through to the FastAPI app unchanged.
    """

    def __init__(self, main_app):
        self.main_app = main_app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path: str = scope.get("path", "")
            if path.startswith("/mcp/"):
                parts = path.split("/")
                if len(parts) >= 3:
                    server_name = parts[2]
                    # ``proxy.http_app()`` registers Streamable HTTP at ``streamable_http_path``
                    # (default ``/mcp``). Rewriting outer ``/mcp/{name}`` to ``/`` produced 404 and
                    # MCP clients surfaced it as ``Session terminated``.
                    mcp_path = (
                        fastmcp_settings.streamable_http_path.rstrip("/") or "/mcp"
                    )
                    tail = "/".join(parts[3:]) if len(parts) > 3 else ""
                    remaining = f"{mcp_path}/{tail}".rstrip("/") if tail else mcp_path
                    try:
                        resolved = await self._resolve_proxy(scope, server_name)
                    except PermissionError as exc:
                        await self._send_error(send, 401, str(exc))
                        return
                    except ApprovalPending:
                        # Same body the REST routes answer with, so a client can
                        # branch on the code whichever door it came through.
                        await self._send_json(
                            send, 403, {"detail": PENDING_APPROVAL_DETAIL}
                        )
                        return
                    except IdentityProviderUnavailable as exc:
                        # 503 like the catch-all below, but without the stack
                        # trace: a provider outage is expected weather, not a bug.
                        logger.warning("Identity provider unavailable: %s", exc)
                        await self._send_error(
                            send, 503, "Identity provider unavailable"
                        )
                        return
                    except LookupError as exc:
                        await self._send_error(send, 404, str(exc))
                        return
                    except Exception as exc:
                        logger.exception("MCP proxy resolution failed: %s", exc)
                        await self._send_error(send, 503, str(exc))
                        return

                    scope = {**scope, "path": remaining, "raw_path": remaining.encode()}
                    await self._dispatch_and_track(
                        scope, receive, send, resolved.proxy_app, resolved.principal,
                        server_name, resolved.user_token,
                        server_id=resolved.server_id,
                        credential_row=resolved.credential_row,
                    )
                    return

        await self.main_app(scope, receive, send)

    async def _dispatch_and_track(
        self, scope, receive, send, proxy_app, principal: Principal,
        server_name: str, user_token: str | None = None, *,
        server_id: str | None = None, credential_row: Mapping | None = None,
    ):
        """Forward to the proxy sub-app and record one usage event per tool call.

        Buffers the JSON-RPC request body to detect ``tools/call`` (only those are
        logged), wraps ``send`` to capture the upstream status and response body,
        and writes the usage event after the response has been delivered so the
        client is never blocked by the DB write.

        ``user_token`` is the caller's EVE credential, published to the
        per-request ``_user_eve_token_var`` contextvar so ``_DynamicBearerAuth``
        can forward it as ``X-EVE-Token`` to the upstream MCP server (see the
        docstring there for why user-scoped retrieval depends on it). It is
        ``None`` for servers not allowed to receive it (``_resolve_proxy``).

        On a ``tools/call`` to a row with a credential spec, the caller's own
        credentials go into ``_upstream_credential_headers_var`` for the same
        lifetime, or the caller gets a tool error and upstream is not called.
        """
        # Publish the caller's token for the lifetime of this proxied call only:
        # the proxy app is shared across users, so the contextvar must not leak
        # past this request.
        token_reset = _user_eve_token_var.set(user_token)
        credentials_reset = _upstream_credential_headers_var.set(None)
        try:
            # Buffer the request body once so we can both inspect it and replay it
            # downstream (ASGI bodies are single-consumption streams).
            body = b""
            while True:
                event = await receive()
                body += event.get("body", b"")
                if not event.get("more_body", False):
                    break

            tool_call = _extract_tool_call(body)
            downstream_receive = _replay_receive(body, receive)

            if tool_call is None:
                # Housekeeping hop (initialize/tools_list/notification/SSE) — forward
                # without recording a usage event.
                await proxy_app(scope, downstream_receive, send)
                return

            tool_name, arguments = tool_call.name, tool_call.arguments
            # Per-user request rate limit on tools/call only (it is what bills
            # AgentCore), before forwarding: a refusal records no usage row.
            try:
                await check_or_raise(principal, "mcp")
            except RequestRateLimited as limited:
                await self._send_json(
                    send,
                    429,
                    {"detail": limited.detail},
                    extra_headers=[
                        [b"retry-after", str(limited.retry_after_s).encode()]
                    ],
                )
                return
            except RateLimiterUnavailable:
                await self._send_json(send, 503, {"detail": LIMITER_UNAVAILABLE_DETAIL})
                return
            caller_type = principal.caller_type()

            credentials = await _upstream_credentials(
                user_id=principal.user_id,
                server_id=server_id,
                credential_row=credential_row,
                server_name=server_name,
            )
            if credentials.refusal is not None:
                started = time.monotonic()
                error_body = _tool_error_body(tool_call.request_id, credentials.refusal)
                await self._send_json(send, 200, error_body)
                await track_usage(
                    user_id=principal.user_id,
                    caller_type=caller_type,
                    api_key_id=principal.api_key_id,
                    server_name=server_name,
                    operation="tools/call",
                    tool_name=tool_name,
                    status_code=200,
                    is_error=True,
                    outcome="error",
                    latency_ms=(time.monotonic() - started) * 1000,
                    request_payload=arguments,
                    response_payload=error_body,
                    credential_source=credentials.source,
                )
                return
            if credentials.headers:
                _upstream_credential_headers_var.set(credentials.headers)

            status_holder: dict[str, int] = {}
            resp_chunks: list[bytes] = []

            async def wrapped_send(message):
                if message["type"] == "http.response.start":
                    status_holder["status"] = message["status"]
                elif message["type"] == "http.response.body":
                    chunk = message.get("body") or b""
                    if chunk:
                        resp_chunks.append(chunk)
                await send(message)

            started = time.monotonic()
            try:
                await proxy_app(scope, downstream_receive, wrapped_send)
            finally:
                latency_ms = (time.monotonic() - started) * 1000
                status_code = status_holder.get("status")
                raw_response = b"".join(resp_chunks)

                # Decode the SSE/JSON-RPC body into structured objects so the stored
                # payload is queryable for deeper stats; fall back to the raw string
                # if it can't be parsed (so nothing is ever lost).
                messages = _parse_mcp_messages(raw_response)
                if messages is None:
                    response_payload = (
                        raw_response.decode(errors="replace") if raw_response else None
                    )
                elif len(messages) == 1:
                    response_payload = messages[0]
                else:
                    response_payload = messages

                # A tools/call can fail at the MCP level (result.isError / JSON-RPC
                # error) while still returning HTTP 200, so combine both signals.
                is_error = _is_error_message(messages)
                http_failed = status_code is None or status_code >= 400
                outcome = _tool_call_outcome(http_failed=http_failed, is_error=is_error)
                await track_usage(
                    user_id=principal.user_id,
                    caller_type=caller_type,
                    api_key_id=principal.api_key_id,
                    server_name=server_name,
                    operation="tools/call",
                    tool_name=tool_name,
                    status_code=status_code,
                    is_error=is_error,
                    outcome=outcome,
                    latency_ms=latency_ms,
                    request_payload=_redact_values(arguments, credentials.headers),
                    response_payload=_redact_values(response_payload, credentials.headers),
                    credential_source=credentials.source,
                )
        finally:
            _upstream_credential_headers_var.reset(credentials_reset)
            _user_eve_token_var.reset(token_reset)

    async def _resolve_proxy(self, scope, server_name: str):
        headers = dict(scope.get("headers", []))
        token = extract_bearer_token(headers.get(b"authorization", b"").decode())
        if not token:
            raise PermissionError("Missing or malformed Authorization header")

        principal = await resolve_principal_from_bearer_token(token)
        user_id = principal.user_id

        cache_key = (server_name, user_id)
        now = time.monotonic()
        cached = _server_url_cache.get(cache_key)

        if cached and now < cached.expires_at:
            entry = cached
        else:
            lock = await _get_server_url_lock(server_name)
            async with lock:
                # Re-check after acquiring lock — another coroutine may have
                # already refreshed the entry while we were waiting.
                cached = _server_url_cache.get(cache_key)
                if cached and time.monotonic() < cached.expires_at:
                    entry = cached
                else:
                    mcp_servers = get_collection("mcp_servers")
                    # A user row and a global row can share a name. The
                    # caller's own row wins: BSON orders null before every
                    # string or ObjectId, so a descending sort puts it first.
                    server = await mcp_servers.find_one(
                        {
                            "name": server_name,
                            "enabled": True,
                            "deleted_at": None,
                            "$or": [{"user_id": user_id}, {"user_id": None}],
                        },
                        {
                            "config.url": 1,
                            "name": 1,
                            "user_id": 1,
                            "forward_user_token": 1,
                            "user_credentials": 1,
                            "user_environment": 1,
                        },
                        sort=[("user_id", -1)],
                    )
                    if (
                        not server
                        or not server.get("config")
                        or not server["config"].get("url")
                    ):
                        raise LookupError(
                            f"MCP server '{server_name}' not found or not accessible"
                        )
                    forward_token = forwards_user_token(
                        name=server.get("name", server_name),
                        user_id=server.get("user_id"),
                        forward_user_token=server.get("forward_user_token"),
                    )
                    raw_spec = server.get("user_credentials")
                    raw_environment = server.get("user_environment")
                    entry = _CachedServer(
                        url=server["config"]["url"],
                        forward_token=forward_token,
                        server_id=str(server["_id"]) if server.get("_id") is not None else None,
                        credential_row=None
                        if raw_spec is None and raw_environment is None
                        else {
                            "name": server.get("name", server_name),
                            "user_id": server.get("user_id"),
                            "user_credentials": raw_spec,
                            "user_environment": raw_environment,
                        },
                        expires_at=time.monotonic() + MCP_TOOLS_CACHE_TTL,
                    )
                    _server_url_cache[cache_key] = entry
                    logger.debug(
                        "[MCP proxy] cached URL for server=%r user=%s (TTL=%.0fs)",
                        server_name,
                        user_id,
                        MCP_TOOLS_CACHE_TTL,
                    )

        provider = get_cognito_token_provider()
        if provider is None:
            raise RuntimeError("AgentCore authentication is not configured")

        proxy_app = await build_proxy_app(entry.url, provider)
        # The caller's token goes upstream only to a server allowed to receive
        # it (``user_token_scope``). The Cognito client-credentials bearer is
        # sent by ``_DynamicBearerAuth`` on every proxied call, whatever the row.
        user_token = None
        if entry.forward_token:
            announce_user_token_forwarded(server_name)
            user_token = token
        return _ResolvedProxy(
            proxy_app, principal, user_token, entry.server_id, entry.credential_row
        )

    @staticmethod
    async def _send_error(send, status: int, detail: str):
        await MCPProxyDispatcher._send_json(send, status, {"detail": detail})

    @staticmethod
    async def _send_json(send, status: int, payload: dict, extra_headers=None):
        body = json.dumps(payload).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [[b"content-type", b"application/json"]]
                + list(extra_headers or []),
            }
        )
        await send({"type": "http.response.body", "body": body})
