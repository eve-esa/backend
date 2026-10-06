from src.routers.mcp_proxy import (
    _is_error_message,
    _tool_call_outcome,
    _user_eve_token_var,
    _DynamicBearerAuth,
)
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest


def test_is_error_message_json_rpc_error():
    assert _is_error_message([{"error": {"code": -32603, "message": "boom"}}]) is True


def test_is_error_message_result_is_error():
    assert _is_error_message([{"result": {"isError": True, "content": []}}]) is True


def test_is_error_message_success():
    assert _is_error_message([{"result": {"isError": False, "content": []}}]) is False


def test_is_error_message_unparseable_returns_none():
    assert _is_error_message(None) is None


def test_tool_call_outcome_unknown_when_mcp_unparseable_and_http_ok():
    assert _tool_call_outcome(http_failed=False, is_error=None) == "unknown"


def test_tool_call_outcome_error_when_http_failed_even_if_mcp_unknown():
    assert _tool_call_outcome(http_failed=True, is_error=None) == "error"


def test_tool_call_outcome_success_only_when_mcp_confirmed_ok():
    assert _tool_call_outcome(http_failed=False, is_error=False) == "success"


# ── X-EVE-Token propagation ────────────────────────────────────────────────────
# The MCP proxy authenticates to AgentCore with a shared Cognito M2M token, but
# user-scoped EVE endpoints (notably ``eve_retrieval`` → ``POST /retrieve``) must
# run as the caller: ``apply_private_collections_to_request`` keeps only
# collections owned by the requesting user, so a machine ``EVE_API_KEY`` silently
# drops every private collection. The proxy forwards the caller's EVE credential
# as ``X-EVE-Token`` (the MCP server reads it before its ``EVE_API_KEY`` fallback)
# to preserve that identity. These tests lock the wiring.


@pytest.mark.asyncio
async def test_dynamic_bearer_forwards_user_token_as_x_eve_token():
    """The caller's EVE credential is attached as ``X-EVE-Token`` on egress."""
    token_reset = _user_eve_token_var.set("caller-jwt")
    try:
        provider = MagicMock()
        provider.get_token = AsyncMock(return_value="cognito-m2m")
        auth = _DynamicBearerAuth(provider)

        request = httpx.Request("POST", "https://agentcore.example/mcp")
        gen = auth.async_auth_flow(request)
        sent = await gen.asend(None)

        # Cognito M2M authenticates the proxy to AgentCore; X-EVE-Token
        # authenticates the user to the downstream EVE API.
        assert sent.headers["Authorization"] == "Bearer cognito-m2m"
        assert sent.headers["X-EVE-Token"] == "caller-jwt"

        with pytest.raises(StopAsyncIteration):
            await gen.asend(httpx.Response(200))
    finally:
        _user_eve_token_var.reset(token_reset)


@pytest.mark.asyncio
async def test_dynamic_bearer_omits_x_eve_token_when_no_user_token():
    """No caller token (e.g. a housekeeping hop) must not synthesize a header."""
    token_reset = _user_eve_token_var.set(None)
    try:
        provider = MagicMock()
        provider.get_token = AsyncMock(return_value="cognito-m2m")
        auth = _DynamicBearerAuth(provider)

        request = httpx.Request("POST", "https://agentcore.example/mcp")
        gen = auth.async_auth_flow(request)
        sent = await gen.asend(None)

        headers = {k.lower(): v for k, v in sent.headers.items()}
        assert "x-eve-token" not in headers
        assert sent.headers["Authorization"] == "Bearer cognito-m2m"

        with pytest.raises(StopAsyncIteration):
            await gen.asend(httpx.Response(200))
    finally:
        _user_eve_token_var.reset(token_reset)


# ── No session DELETE upstream ─────────────────────────────────────────────────
# AgentCore hands out an ``Mcp-Session-Id`` but is stateless: the DELETE the MCP
# client sends on close answers 404 and logs a WARNING on every proxied call.


@pytest.mark.asyncio
async def test_upstream_client_does_not_delete_the_session_on_close():
    """A stateful upstream (one that issues a session id) gets no DELETE."""
    from fastmcp import Client, FastMCP
    from fastmcp.client.transports.http import StreamableHttpTransport

    upstream = FastMCP("upstream")

    @upstream.tool
    def ping() -> str:
        return "pong"

    app = upstream.http_app(stateless_http=False)
    methods: list[str] = []

    async def record(request: httpx.Request) -> None:
        methods.append(request.method)

    def client_factory(**kwargs) -> httpx.AsyncClient:
        kwargs.pop("follow_redirects", None)
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            event_hooks={"request": [record]},
            **kwargs,
        )

    transport = StreamableHttpTransport(
        "http://upstream.test/mcp", httpx_client_factory=client_factory
    )
    async with app.lifespan(app):
        async with Client(transport) as client:
            await client.list_tools()
            assert transport.get_session_id(), "upstream must issue a session id"

    assert "POST" in methods
    assert "DELETE" not in methods


# The first proxied request starts the sub-app lifespan; the server lifespan
# closes it on shutdown, from another task. Exiting the lifespan outside the
# task and context that entered it raised ``ValueError`` (fastmcp context
# variable reset) and logged an ERROR with a traceback on every rollover.


@pytest.fixture
def proxy_registry():
    from src.routers import mcp_proxy

    yield mcp_proxy
    mcp_proxy._proxy_lifespans.clear()
    mcp_proxy._proxy_apps.clear()


@pytest.mark.no_db
async def test_shutdown_from_another_task_logs_nothing_at_error(proxy_registry, caplog):
    """A real sub-app started in a request task closes from the lifespan task."""
    import asyncio
    import logging

    provider = MagicMock()
    await asyncio.create_task(
        proxy_registry.build_proxy_app("http://agentcore.invalid/a/mcp", provider)
    )
    await asyncio.create_task(
        proxy_registry.build_proxy_app("http://agentcore.invalid/b/mcp", provider)
    )
    (_, task_a), (_, task_b) = proxy_registry._proxy_lifespans.values()

    with caplog.at_level(logging.DEBUG, logger=proxy_registry.__name__):
        await proxy_registry.shutdown_mcp_proxy_lifespans()
        await proxy_registry.shutdown_mcp_proxy_lifespans()

    assert task_a.done() and task_a.exception() is None
    assert task_b.done() and task_b.exception() is None
    assert not proxy_registry._proxy_apps
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.no_db
async def test_one_failing_sub_app_logs_one_warning_and_the_others_close(
    proxy_registry, caplog
):
    import asyncio
    import logging
    from contextlib import asynccontextmanager

    closed: list[str] = []

    def fake_app(name: str, fail: bool):
        @asynccontextmanager
        async def lifespan(_app):
            yield
            if fail:
                raise RuntimeError("boom")
            closed.append(name)

        return MagicMock(lifespan=lifespan)

    for name, fail in (("a", False), ("b", True), ("c", False)):
        started = asyncio.get_running_loop().create_future()
        stop = asyncio.Event()
        task = asyncio.create_task(
            proxy_registry._hold_proxy_lifespan(fake_app(name, fail), started, stop)
        )
        await started
        proxy_registry._proxy_lifespans[f"http://agentcore.invalid/{name}"] = (stop, task)

    with caplog.at_level(logging.DEBUG, logger=proxy_registry.__name__):
        await proxy_registry.shutdown_mcp_proxy_lifespans()

    assert closed == ["a", "c"]
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert warnings[0].levelno == logging.WARNING
    assert "type=RuntimeError" in warnings[0].getMessage()
    assert "/b" in warnings[0].getMessage()
    assert warnings[0].exc_info is None


def _fake_proxies(monkeypatch, proxy_registry, lifespans):
    """``create_proxy`` returns sub-apps whose lifespans come from ``lifespans``, in order."""
    apps = []

    def create_proxy(*_args, **_kwargs):
        app = MagicMock(lifespan=lifespans.pop(0))
        apps.append(app)
        return MagicMock(http_app=MagicMock(return_value=app))

    monkeypatch.setattr(proxy_registry, "create_proxy", create_proxy)
    return apps


@pytest.mark.no_db
async def test_hung_sub_apps_share_one_shutdown_deadline(
    proxy_registry, monkeypatch, caplog
):
    """Three sub-apps that never close cost one timeout in total, not three."""
    import asyncio
    import logging
    import time
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def hangs_on_close(_app):
        yield
        await asyncio.sleep(3600)

    monkeypatch.setattr(proxy_registry, "_PROXY_SHUTDOWN_TIMEOUT_S", 0.3)
    monkeypatch.setattr(proxy_registry, "_PROXY_CANCEL_GRACE_S", 0.3)
    _fake_proxies(monkeypatch, proxy_registry, [hangs_on_close] * 3)
    for name in "abc":
        await proxy_registry.build_proxy_app(f"http://agentcore.invalid/{name}", MagicMock())
    tasks = [task for _stop, task in proxy_registry._proxy_lifespans.values()]

    with caplog.at_level(logging.DEBUG, logger=proxy_registry.__name__):
        began = time.monotonic()
        await proxy_registry.shutdown_mcp_proxy_lifespans()
        elapsed = time.monotonic() - began

    assert elapsed < 0.6  # sequential waits would take 0.9 s
    assert all(task.done() for task in tasks)
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 3
    assert all("type=TimeoutError" in message for message in warnings)


@pytest.mark.no_db
async def test_a_request_cancelled_during_startup_leaves_no_owner_error(
    proxy_registry, monkeypatch, caplog
):
    import asyncio
    import logging
    from contextlib import asynccontextmanager

    entered = asyncio.Event()
    release = asyncio.Event()
    closed = asyncio.Event()

    @asynccontextmanager
    async def slow_start(_app):
        entered.set()
        await release.wait()
        yield
        closed.set()

    _fake_proxies(monkeypatch, proxy_registry, [slow_start])
    request = asyncio.create_task(
        proxy_registry.build_proxy_app("http://agentcore.invalid/a", MagicMock())
    )
    await entered.wait()
    owner = next(
        t for t in asyncio.all_tasks() if t.get_name().startswith("mcp-proxy-lifespan-")
    )
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    with caplog.at_level(logging.DEBUG):
        release.set()
        await asyncio.wait_for(owner, 1)

    assert closed.is_set()
    assert not owner.cancelled() and owner.exception() is None
    assert not proxy_registry._proxy_apps and not proxy_registry._proxy_lifespans
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.no_db
async def test_a_dead_owner_is_evicted_and_the_next_request_starts_a_fresh_sub_app(
    proxy_registry, monkeypatch, caplog
):
    import asyncio
    import logging
    from contextlib import asynccontextmanager

    import anyio

    crash = asyncio.Event()

    async def session_manager():
        await crash.wait()
        raise RuntimeError("task group crashed")

    @asynccontextmanager
    async def crashes_later(_app):
        async with anyio.create_task_group() as tg:
            tg.start_soon(session_manager)
            yield

    @asynccontextmanager
    async def healthy(_app):
        yield

    apps = _fake_proxies(monkeypatch, proxy_registry, [crashes_later, healthy])
    url = "http://agentcore.invalid/a"
    first = await proxy_registry.build_proxy_app(url, MagicMock())
    (_stop, owner), = proxy_registry._proxy_lifespans.values()

    with caplog.at_level(logging.DEBUG, logger=proxy_registry.__name__):
        crash.set()
        await asyncio.wait({owner}, timeout=1)
        await asyncio.sleep(0)  # done callbacks run on the next loop step

    assert url not in proxy_registry._proxy_apps
    assert url not in proxy_registry._proxy_lifespans
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1 and "stopped unexpectedly" in warnings[0]

    second = await proxy_registry.build_proxy_app(url, MagicMock())
    assert second is not first and second is apps[1]
    await proxy_registry.shutdown_mcp_proxy_lifespans()


@pytest.mark.no_db
async def test_a_cancelled_owner_is_reported_at_shutdown(proxy_registry, caplog):
    import asyncio
    import logging

    owner = asyncio.create_task(asyncio.sleep(3600))
    owner.cancel()
    await asyncio.wait({owner})
    proxy_registry._proxy_lifespans["http://agentcore.invalid/a"] = (asyncio.Event(), owner)

    with caplog.at_level(logging.DEBUG, logger=proxy_registry.__name__):
        await proxy_registry.shutdown_mcp_proxy_lifespans()

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1 and "type=CancelledError" in warnings[0]
