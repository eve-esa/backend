"""Request rate limit wired into the routes and the two ASGI dispatchers.

The limiter runs on a store that refuses every call (or one that is down), so
each test proves where the check sits: in enforce mode a covered route answers
429 ``rate_limited`` before any work, in shadow mode it goes on to its own
answer (404 for a conversation that does not exist, 422 for a missing body),
and the dispatchers send their envelope with no upstream call and no usage row.
"""

import json
import logging
from unittest.mock import AsyncMock, patch

import pytest

from server import app as _root_app
from src.database.models.api_key import ApiKey
from src.middlewares.auth import AUTH_TYPE_API_KEY, Principal
from src.routers.mcp_proxy import MCPProxyDispatcher
from src.services import load_shedding
from src.services import request_rate_limiter as rrl
from src.services.auth import generate_api_key
from src.services.load_shedding import GenerationLimiter
from src.services.request_rate_limiter import RequestRateLimiter
from tests.utils.cleaner import cleanup_models
from tests.utils.openai_proxy import enable_proxy, mock_client
from tests.utils.utils import create_test_user_and_token

_proxy = _root_app.main_app
MISSING_ID = "000000000000000000000000"


class RefusingStore:
    """Every bucket is empty: the next token is 5 s away."""

    def __init__(self, error=None) -> None:
        self.calls = 0
        self.error = error

    def register_script(self, _source):
        return self._script

    async def _script(self, keys, args):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return [0, 5000]

    async def aclose(self):
        pass


@pytest.fixture
def rate_limit(monkeypatch):
    def _install(mode: str, store=None, fail_closed: bool = False):
        store = store or RefusingStore()
        limiter = RequestRateLimiter(
            limits={
                cls: {"rate": 60, "burst": 1}
                for cls in ("chat", "retrieve", "proxy", "mcp", "upload", "errlog")
            },
            mode=mode,
            fail_closed=fail_closed,
            client=store,
        )
        monkeypatch.setattr(rrl, "FEATURE_REQUEST_RATE_LIMIT", True)
        monkeypatch.setattr(rrl, "_limiter", limiter)
        return store

    return _install


@pytest.fixture(autouse=True)
def _fresh_generation_limiter(monkeypatch):
    monkeypatch.setattr(load_shedding, "_limiter", GenerationLimiter(40))


@pytest.fixture(autouse=True)
def _reset_openai_proxy_http_client():
    _proxy._client = None
    _proxy._client_loop = None
    yield


@pytest.fixture
async def user_and_token():
    user, token = await create_test_user_and_token()
    yield user, token
    await ApiKey.delete_many({"user_id": user.id})
    await cleanup_models([user])


CONV = f"/conversations/{MISSING_ID}"
MSG = f"{CONV}/messages/{MISSING_ID}"
# (url, route class, request kwargs). The bodies are deliberately incomplete
# or point at a conversation that does not exist, so a request the limiter
# lets through ends in 404 or 422 without reaching a model, Qdrant or S3.
COVERED = [
    (f"{CONV}/messages", "chat", {"json": {"query": "hi"}}),
    (f"{MSG}/retry", "chat", {"json": {"query": "hi"}}),
    (f"{CONV}/stream_messages", "chat", {"json": {"query": "hi"}}),
    (f"{CONV}/generate-agentic", "chat", {"json": {"query": "hi"}}),
    (f"{CONV}/stream-generate-agentic", "chat", {"json": {"query": "hi"}}),
    (f"{MSG}/hallucination", "chat", {"json": {}}),
    (f"{MSG}/stream-hallucination", "chat", {"json": {}}),
    ("/generate", "chat", {"json": []}),
    ("/generate-llm", "chat", {"json": []}),
    ("/retrieve", "retrieve", {"json": []}),
    (f"/collections/{MISSING_ID}/documents", "upload", {}),
    ("/artifacts", "upload", {}),
    ("/log-error", "errlog", {"json": []}),
]


@pytest.mark.parametrize("url, route_class, kwargs", COVERED, ids=[c[0] for c in COVERED])
async def test_every_covered_route_refuses_in_enforce_before_any_work(
    async_client, user_and_token, rate_limit, url, route_class, kwargs
):
    user, token = user_and_token
    store = rate_limit("enforce")
    # Spies on the next two steps of a chat route: a refusal must come first.
    with patch(
        "src.routers.message.acquire_generation_slot_or_raise", new_callable=AsyncMock
    ) as slot, patch(
        "src.routers.message.enforce_token_budget_or_raise", new_callable=AsyncMock
    ) as budget:
        response = await async_client.post(
            url, headers={"Authorization": f"Bearer {token}"}, **kwargs
        )
    slot.assert_not_awaited()
    budget.assert_not_awaited()
    assert response.status_code == 429, response.text
    assert response.json()["detail"] == {
        "code": "rate_limited",
        "message": "Too many requests, retry in 5 s",
    }
    assert response.headers["Retry-After"] == "5"
    assert store.calls == 1


@pytest.mark.parametrize("url, route_class, kwargs", COVERED, ids=[c[0] for c in COVERED])
async def test_every_covered_route_goes_through_in_shadow(
    async_client, user_and_token, rate_limit, caplog, url, route_class, kwargs
):
    user, token = user_and_token
    store = rate_limit("shadow")
    with caplog.at_level(logging.INFO, logger=rrl.__name__):
        response = await async_client.post(
            url, headers={"Authorization": f"Bearer {token}"}, **kwargs
        )
    assert response.status_code in (404, 422), response.text
    assert store.calls == 1
    assert (
        f"rate_limit.limited subject_kind=user class={route_class} mode=shadow retry_after_s=5"
        in caplog.text
    )


async def test_uncovered_routes_never_reach_the_limiter(async_client, user_and_token, rate_limit):
    user, token = user_and_token
    store = rate_limit("enforce")
    headers = {"Authorization": f"Bearer {token}"}
    for method, url in (
        ("get", "/users/me"),
        ("get", "/conversations"),
        ("post", f"{CONV}/stop"),
    ):
        response = await getattr(async_client, method)(url, headers=headers)
        assert response.status_code != 429, (url, response.text)
    assert store.calls == 0


# ── /v1 ─────────────────────────────────────────────────────────────────────


async def _key_for(user_id: str) -> str:
    raw_token, key_hash = generate_api_key()
    await ApiKey.create(user_id=user_id, name="rate-limit-test", key_hash=key_hash)
    return raw_token


async def test_proxy_rate_limited_in_the_openai_envelope_without_upstream_call(
    async_client, user_and_token, rate_limit, monkeypatch
):
    user, _ = user_and_token
    raw_key = await _key_for(user.id)
    store = rate_limit("enforce")
    enable_proxy(monkeypatch, _proxy)
    client, stream_mock = mock_client()
    monkeypatch.setattr(_proxy, "_client", client)
    with patch("src.routers.openai_proxy.track_usage", new_callable=AsyncMock) as track, patch(
        "src.routers.openai_proxy.reserve_token_budget", new_callable=AsyncMock
    ) as reserve:
        resp = await async_client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"Authorization": f"Bearer {raw_key}"},
        )
    assert resp.status_code == 429
    body = resp.json()
    assert body["detail"] == {"code": "rate_limited", "message": "Too many requests, retry in 5 s"}
    assert body["error"] == {
        "message": "Too many requests, retry in 5 s",
        "type": "rate_limit_error",
        "code": "rate_limited",
        "param": None,
    }
    assert resp.headers["retry-after"] == "5"
    assert resp.headers["x-should-retry"] == "true"
    assert store.calls == 1
    stream_mock.assert_not_called()
    reserve.assert_not_awaited()
    track.assert_not_awaited()


async def test_proxy_unknown_path_is_404_before_the_limit(
    async_client, user_and_token, rate_limit, monkeypatch
):
    user, token = user_and_token
    store = rate_limit("enforce")
    enable_proxy(monkeypatch, _proxy)
    resp = await async_client.post(
        "/v1/files", json={}, headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 404
    assert store.calls == 0


async def test_proxy_limiter_unavailable_when_fail_closed(
    async_client, user_and_token, rate_limit, monkeypatch
):
    user, token = user_and_token
    rate_limit("enforce", store=RefusingStore(error=ConnectionError("down")), fail_closed=True)
    enable_proxy(monkeypatch, _proxy)
    client, stream_mock = mock_client()
    monkeypatch.setattr(_proxy, "_client", client)
    resp = await async_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "limiter_unavailable"
    stream_mock.assert_not_called()


# ── MCP ─────────────────────────────────────────────────────────────────────


def _rpc(method: str) -> bytes:
    params = {"name": "echo", "arguments": {"x": 1}} if method == "tools/call" else {}
    return json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()


async def _dispatch(body: bytes):
    sent, upstream = [], []

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        sent.append(message)

    async def proxy_app(scope, receive_, send_):
        upstream.append(scope["path"])
        await send_({"type": "http.response.start", "status": 200, "headers": []})
        await send_({"type": "http.response.body", "body": b"{}"})

    dispatcher = MCPProxyDispatcher(main_app=None)
    principal = Principal("6ac1aaaaaaaaaaaaaaaaaaaa", AUTH_TYPE_API_KEY, "key-1")
    scope = {"type": "http", "path": "/mcp/dummy/mcp", "method": "POST", "headers": []}
    with patch("src.routers.mcp_proxy.track_usage", new_callable=AsyncMock) as track:
        await dispatcher._dispatch_and_track(
            scope, receive, send, proxy_app, principal, "dummy", user_token="eve_x"
        )
    return sent, upstream, track


@pytest.mark.no_db
async def test_mcp_rate_limited_tools_call_is_not_forwarded_nor_tracked(rate_limit):
    store = rate_limit("enforce")
    sent, upstream, track = await _dispatch(_rpc("tools/call"))
    assert sent[0]["status"] == 429
    assert [b"retry-after", b"5"] in sent[0]["headers"]
    assert json.loads(sent[1]["body"])["detail"]["code"] == "rate_limited"
    assert upstream == []
    track.assert_not_awaited()
    assert store.calls == 1


@pytest.mark.no_db
async def test_mcp_tools_list_never_reaches_the_limiter(rate_limit):
    store = rate_limit("enforce")
    sent, upstream, track = await _dispatch(_rpc("tools/list"))
    assert sent[0]["status"] == 200 and upstream == ["/mcp/dummy/mcp"]
    assert store.calls == 0


@pytest.mark.no_db
async def test_mcp_shadow_forwards_and_tracks(rate_limit):
    store = rate_limit("shadow")
    sent, upstream, track = await _dispatch(_rpc("tools/call"))
    assert sent[0]["status"] == 200 and upstream == ["/mcp/dummy/mcp"]
    track.assert_awaited_once()
    assert store.calls == 1
