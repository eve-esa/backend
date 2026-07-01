import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server import app as _root_app
from src.routers.openai_proxy import parse_proxy_model, resolve_proxy_route
from tests.utils.cleaner import cleanup_models
from tests.utils.utils import create_test_user_and_token

# App stack: MCPProxyDispatcher → OpenAIProxyDispatcher → FastAPI
_proxy = _root_app.main_app


@pytest.fixture(autouse=True)
def _reset_openai_proxy_http_client():
    """Avoid cross-test httpx client reuse (pytest-asyncio uses one loop per test)."""
    _proxy._client = None
    _proxy._client_loop = None
    yield


_FAKE_UPSTREAM = "http://fake-upstream"
_FAKE_JSC_UPSTREAM = "http://fake-jsc-upstream"


def _enable_proxy(monkeypatch, *, runpod_url: str = _FAKE_UPSTREAM, jsc_url: str = ""):
    monkeypatch.setattr(_proxy, "_proxy_enabled", True)
    monkeypatch.setattr("src.routers.openai_proxy.OPENAI_PROXY_UPSTREAM_URL", runpod_url)
    monkeypatch.setattr("src.routers.openai_proxy.EVE_JSC_BASE_URL", jsc_url)
    monkeypatch.setattr("src.routers.openai_proxy.OPENAI_PROXY_API_KEY", "fake-runpod-key")
    monkeypatch.setattr("src.routers.openai_proxy.EVE_JSC_API_KEY", "fake-jsc-key")


def _minimal_completion_body() -> bytes:
    return json.dumps(
        {
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    ).encode()


def _mock_client(status: int = 200, body: bytes = b"{}", content_type: str = "application/json"):
    """Return a mock httpx.AsyncClient whose stream() acts as an async context manager."""
    resp = MagicMock()
    resp.status_code = status
    resp.headers = {"content-type": content_type}

    async def aiter_bytes():
        yield body

    resp.aiter_bytes = aiter_bytes

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=resp)
    cm.__aexit__ = AsyncMock(return_value=None)

    client = MagicMock()
    client.stream.return_value = cm
    return client, client.stream


# ── Proxy disabled ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_proxy_disabled_falls_through(async_client, monkeypatch):
    """When no provider upstream URL is set the dispatcher is a no-op."""
    monkeypatch.setattr(_proxy, "_proxy_enabled", False)
    resp = await async_client.get("/v1/models")
    assert resp.status_code == 404


# ── Auth guard ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_auth_header_returns_401(async_client, monkeypatch):
    _enable_proxy(monkeypatch)
    resp = await async_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_wrong_auth_scheme_returns_401(async_client, monkeypatch):
    _enable_proxy(monkeypatch)
    resp = await async_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
        headers={"Authorization": "Token some-opaque-token"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_invalid_jwt_returns_401(async_client, monkeypatch):
    _enable_proxy(monkeypatch)
    resp = await async_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
        headers={"Authorization": "Bearer not.a.valid.jwt"},
    )
    assert resp.status_code == 401


# ── Chat completions (non-streaming) ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_chat_completions_proxied(async_client, monkeypatch):
    user, token = await create_test_user_and_token()
    try:
        upstream_body = json.dumps(
            {
                "id": "chatcmpl-abc",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        ).encode()

        _enable_proxy(monkeypatch)
        client, _ = _mock_client(body=upstream_body)
        monkeypatch.setattr(_proxy, "_client", client)

        with patch("src.routers.openai_proxy.track_usage", new_callable=AsyncMock) as mock_track:
            resp = await async_client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
                headers={"Authorization": f"Bearer {token}"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Hello!"

        mock_track.assert_awaited_once()
        kw = mock_track.call_args.kwargs
        assert kw["input_tokens"] == 10
        assert kw["output_tokens"] == 5
        assert kw["total_tokens"] == 15
        assert kw["model"] == "gpt-4"
        assert kw["status_code"] == 200
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_chat_completions_upstream_error_passed_through(async_client, monkeypatch):
    user, token = await create_test_user_and_token()
    try:
        error_body = json.dumps(
            {"error": {"message": "upstream overloaded", "type": "server_error"}}
        ).encode()

        _enable_proxy(monkeypatch)
        client, _ = _mock_client(status=503, body=error_body)
        monkeypatch.setattr(_proxy, "_client", client)

        with patch("src.routers.openai_proxy.track_usage", new_callable=AsyncMock) as mock_track:
            resp = await async_client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
                headers={"Authorization": f"Bearer {token}"},
            )

        assert resp.status_code == 503
        mock_track.assert_awaited_once()
        assert mock_track.call_args.kwargs["status_code"] == 503
    finally:
        await cleanup_models([user])


# ── Chat completions (streaming) ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_chat_completions_streaming_proxied(async_client, monkeypatch):
    user, token = await create_test_user_and_token()
    try:
        sse_body = (
            b'data: {"id":"chatcmpl-abc","object":"chat.completion.chunk",'
            b'"choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}\n\n'
            b'data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[],'
            b'"usage":{"prompt_tokens":8,"completion_tokens":3,"total_tokens":11}}\n\n'
            b"data: [DONE]\n\n"
        )

        _enable_proxy(monkeypatch)
        client, _ = _mock_client(body=sse_body, content_type="text/event-stream")
        monkeypatch.setattr(_proxy, "_client", client)

        with patch("src.routers.openai_proxy.track_usage", new_callable=AsyncMock) as mock_track:
            resp = await async_client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                },
                headers={"Authorization": f"Bearer {token}"},
            )

        assert resp.status_code == 200

        mock_track.assert_awaited_once()
        kw = mock_track.call_args.kwargs
        assert kw["input_tokens"] == 8
        assert kw["output_tokens"] == 3
        assert kw["total_tokens"] == 11
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_chat_completions_streaming_no_usage_chunk(async_client, monkeypatch):
    """Streaming response without a usage chunk: tokens should be None, not an error."""
    user, token = await create_test_user_and_token()
    try:
        sse_body = (
            b'data: {"id":"chatcmpl-abc","object":"chat.completion.chunk",'
            b'"choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":"stop"}]}\n\n'
            b"data: [DONE]\n\n"
        )

        _enable_proxy(monkeypatch)
        client, _ = _mock_client(body=sse_body, content_type="text/event-stream")
        monkeypatch.setattr(_proxy, "_client", client)

        with patch("src.routers.openai_proxy.track_usage", new_callable=AsyncMock) as mock_track:
            resp = await async_client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                },
                headers={"Authorization": f"Bearer {token}"},
            )

        assert resp.status_code == 200
        kw = mock_track.call_args.kwargs
        assert kw["input_tokens"] is None
        assert kw["total_tokens"] is None
    finally:
        await cleanup_models([user])


# ── Models endpoint ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_models_endpoint_proxied(async_client, monkeypatch):
    user, token = await create_test_user_and_token()
    try:
        models_body = json.dumps(
            {
                "object": "list",
                "data": [{"id": "gpt-4", "object": "model", "created": 0, "owned_by": "openai"}],
            }
        ).encode()

        _enable_proxy(monkeypatch)
        client, _ = _mock_client(body=models_body)
        monkeypatch.setattr(_proxy, "_client", client)

        with patch("src.routers.openai_proxy.track_usage", new_callable=AsyncMock):
            resp = await async_client.get(
                "/v1/models",
                headers={"Authorization": f"Bearer {token}"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert data["data"][0]["id"] == "gpt-4"
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_models_endpoint_requires_auth(async_client, monkeypatch):
    _enable_proxy(monkeypatch)
    resp = await async_client.get("/v1/models")
    assert resp.status_code == 401


# ── Provider routing ───────────────────────────────────────────────────────────

_PROVIDER_MODEL_CASES = [
    ("eve/eve-esa/EVE-Instruct", "eve", "eve-esa/EVE-Instruct"),
    ("runpod/eve-esa/EVE-Instruct", "runpod", "eve-esa/EVE-Instruct"),
    ("jsc/alias-eve", "jsc", "alias-eve"),
    ("eve-esa/EVE-Instruct", "eve", "eve-esa/EVE-Instruct"),
    ("alias-eve", "eve", "alias-eve"),
    (None, "eve", None),
]


@pytest.mark.parametrize(
    ("model", "expected_provider", "expected_upstream_model"),
    _PROVIDER_MODEL_CASES,
)
def test_parse_proxy_model(model, expected_provider, expected_upstream_model):
    assert parse_proxy_model(model) == (expected_provider, expected_upstream_model)


@pytest.mark.parametrize(
    ("model", "expected_upstream_base", "expected_api_key"),
    [
        ("eve/eve-esa/EVE-Instruct", _FAKE_UPSTREAM, "fake-runpod-key"),
        ("jsc/alias-eve", _FAKE_JSC_UPSTREAM, "fake-jsc-key"),
    ],
)
def test_resolve_proxy_route_selects_upstream(monkeypatch, model, expected_upstream_base, expected_api_key):
    monkeypatch.setattr("src.routers.openai_proxy.OPENAI_PROXY_UPSTREAM_URL", _FAKE_UPSTREAM)
    monkeypatch.setattr("src.routers.openai_proxy.EVE_JSC_BASE_URL", _FAKE_JSC_UPSTREAM)
    monkeypatch.setattr("src.routers.openai_proxy.OPENAI_PROXY_API_KEY", "fake-runpod-key")
    monkeypatch.setattr("src.routers.openai_proxy.EVE_JSC_API_KEY", "fake-jsc-key")
    upstream_base, api_key, _ = resolve_proxy_route(model)
    assert upstream_base == expected_upstream_base
    assert api_key == expected_api_key


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_model", "jsc_url", "expected_upstream", "expected_api_key", "expected_upstream_model"),
    [
        ("eve/eve-esa/EVE-Instruct", "", _FAKE_UPSTREAM, "fake-runpod-key", "eve-esa/EVE-Instruct"),
        ("eve-esa/EVE-Instruct", "", _FAKE_UPSTREAM, "fake-runpod-key", "eve-esa/EVE-Instruct"),
        ("jsc/alias-eve", _FAKE_JSC_UPSTREAM, _FAKE_JSC_UPSTREAM, "fake-jsc-key", "alias-eve"),
    ],
)
async def test_provider_routing(
    async_client,
    monkeypatch,
    request_model,
    jsc_url,
    expected_upstream,
    expected_api_key,
    expected_upstream_model,
):
    user, token = await create_test_user_and_token()
    try:
        _enable_proxy(monkeypatch, jsc_url=jsc_url)
        client, stream_mock = _mock_client(body=_minimal_completion_body())
        monkeypatch.setattr(_proxy, "_client", client)

        with patch("src.routers.openai_proxy.track_usage", new_callable=AsyncMock):
            resp = await async_client.post(
                "/v1/chat/completions",
                json={"model": request_model, "messages": [{"role": "user", "content": "Hi"}]},
                headers={"Authorization": f"Bearer {token}"},
            )

        assert resp.status_code == 200
        call_kwargs = stream_mock.call_args.kwargs
        assert json.loads(call_kwargs["content"]) == {
            "model": expected_upstream_model,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        assert stream_mock.call_args.args[1].startswith(f"{expected_upstream}/chat/completions")
        assert call_kwargs["headers"]["authorization"] == f"Bearer {expected_api_key}"
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_unknown_provider_slug_forwards_unchanged(async_client, monkeypatch):
    """org/model ids that are not proxy providers are passed through to RunPod."""
    user, token = await create_test_user_and_token()
    try:
        _enable_proxy(monkeypatch)
        client, stream_mock = _mock_client(body=_minimal_completion_body())
        monkeypatch.setattr(_proxy, "_client", client)

        with patch("src.routers.openai_proxy.track_usage", new_callable=AsyncMock):
            resp = await async_client.post(
                "/v1/chat/completions",
                json={
                    "model": "unknown/EVE-Instruct",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
                headers={"Authorization": f"Bearer {token}"},
            )

        assert resp.status_code == 200
        assert json.loads(stream_mock.call_args.kwargs["content"])["model"] == "unknown/EVE-Instruct"
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_jsc_provider_not_configured_returns_400(async_client, monkeypatch):
    user, token = await create_test_user_and_token()
    try:
        _enable_proxy(monkeypatch, jsc_url="")
        resp = await async_client.post(
            "/v1/chat/completions",
            json={"model": "jsc/alias-eve", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 400
        assert "JSC provider is not configured" in resp.json()["detail"]
    finally:
        await cleanup_models([user])


# ── Real upstream integration tests ───────────────────────────────────────────
# Skipped unless at least one provider upstream URL is set in the environment.
# Model selection mirrors testing/test-OpenAI-proxy.py: prefer OPENAI_PROXY_TEST_MODEL,
# else jsc/{EVE_JSC_MODEL_NAME} when JSC is configured, else first model from RunPod.

_upstream_configured = pytest.mark.skipif(
    not (
        os.getenv("OPENAI_PROXY_UPSTREAM_URL", "").strip()
        or os.getenv("EVE_JSC_BASE_URL", "").strip()
    ),
    reason="Neither OPENAI_PROXY_UPSTREAM_URL nor EVE_JSC_BASE_URL is set",
)

_runpod_configured = pytest.mark.skipif(
    not os.getenv("OPENAI_PROXY_UPSTREAM_URL", "").strip(),
    reason="OPENAI_PROXY_UPSTREAM_URL not set",
)


def _default_jsc_test_model() -> str:
    name = os.getenv("EVE_JSC_MODEL_NAME", "alias-eve").strip() or "alias-eve"
    return f"jsc/{name}"


async def _resolve_test_model(async_client, token: str) -> str:
    """Return the model id for live proxy integration tests."""
    if model := os.getenv("OPENAI_PROXY_TEST_MODEL", "").strip():
        return model
    if os.getenv("EVE_JSC_BASE_URL", "").strip():
        return _default_jsc_test_model()
    if os.getenv("OPENAI_PROXY_UPSTREAM_URL", "").strip():
        resp = await async_client.get(
            "/v1/models", headers={"Authorization": f"Bearer {token}"}
        )
        if resp.status_code != 200:
            pytest.fail(
                f"/v1/models returned {resp.status_code}: {resp.text[:500]}"
            )
        data = resp.json()
        models = data.get("data")
        if not models:
            pytest.fail(f"/v1/models returned no models: {data!r}")
        return models[0]["id"]
    pytest.skip("No OpenAI proxy upstream configured")


@_upstream_configured
@_runpod_configured
@pytest.mark.asyncio
async def test_real_models_endpoint(async_client):
    user, token = await create_test_user_and_token()
    try:
        resp = await async_client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        assert len(data["data"]) > 0
    finally:
        await cleanup_models([user])


@_upstream_configured
@pytest.mark.asyncio
async def test_real_chat_completions(async_client):
    user, token = await create_test_user_and_token()
    try:
        model = await _resolve_test_model(async_client, token)
        resp = await async_client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
                "max_tokens": 10,
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200, resp.text[:800]
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"]
        assert data["usage"]["total_tokens"] > 0
    finally:
        await cleanup_models([user])


@_upstream_configured
@pytest.mark.asyncio
async def test_real_chat_completions_streaming(async_client):
    user, token = await create_test_user_and_token()
    try:
        model = await _resolve_test_model(async_client, token)
        resp = await async_client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
                "max_tokens": 10,
                "stream": True,
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200, resp.text[:800]
        assert "text/event-stream" in resp.headers.get("content-type", "")

        lines = [l for l in resp.text.splitlines() if l.startswith("data: ") and l != "data: [DONE]"]
        assert len(lines) > 0
        first = json.loads(lines[0][6:])
        assert first["object"] == "chat.completion.chunk"
    finally:
        await cleanup_models([user])


@_upstream_configured
@pytest.mark.asyncio
async def test_real_unknown_model_returns_error(async_client):
    """Proxy must forward the upstream error when the model does not exist."""
    user, token = await create_test_user_and_token()
    try:
        resp = await async_client.post(
            "/v1/chat/completions",
            json={
                "model": "this-model-does-not-exist",
                "messages": [{"role": "user", "content": "Hi"}],
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code != 200
        assert resp.content  # error body was forwarded, not swallowed
    finally:
        await cleanup_models([user])


@_upstream_configured
@pytest.mark.asyncio
async def test_real_missing_messages_returns_error(async_client):
    """Proxy must forward the upstream error when required fields are absent."""
    user, token = await create_test_user_and_token()
    try:
        model = await _resolve_test_model(async_client, token)
        resp = await async_client.post(
            "/v1/chat/completions",
            json={"model": model},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code != 200
        assert resp.content
    finally:
        await cleanup_models([user])
