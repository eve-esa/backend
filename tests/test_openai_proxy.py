import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server import app as _root_app
from tests.utils.cleaner import cleanup_models
from tests.utils.utils import create_test_user_and_token

# App stack: MCPProxyDispatcher → OpenAIProxyDispatcher → FastAPI
_proxy = _root_app.main_app

_FAKE_UPSTREAM = "http://fake-upstream"


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
    return client


# ── Proxy disabled ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_proxy_disabled_falls_through(async_client, monkeypatch):
    """When OPENAI_PROXY_UPSTREAM_URL is not set the dispatcher is a no-op."""
    monkeypatch.setattr(_proxy, "_upstream", None)
    resp = await async_client.get("/v1/models")
    assert resp.status_code == 404


# ── Auth guard ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_auth_header_returns_401(async_client, monkeypatch):
    monkeypatch.setattr(_proxy, "_upstream", _FAKE_UPSTREAM)
    resp = await async_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_wrong_auth_scheme_returns_401(async_client, monkeypatch):
    monkeypatch.setattr(_proxy, "_upstream", _FAKE_UPSTREAM)
    resp = await async_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
        headers={"Authorization": "Token some-opaque-token"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_invalid_jwt_returns_401(async_client, monkeypatch):
    monkeypatch.setattr(_proxy, "_upstream", _FAKE_UPSTREAM)
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

        monkeypatch.setattr(_proxy, "_upstream", _FAKE_UPSTREAM)
        monkeypatch.setattr(_proxy, "_client", _mock_client(body=upstream_body))

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

        monkeypatch.setattr(_proxy, "_upstream", _FAKE_UPSTREAM)
        monkeypatch.setattr(_proxy, "_client", _mock_client(status=503, body=error_body))

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

        monkeypatch.setattr(_proxy, "_upstream", _FAKE_UPSTREAM)
        monkeypatch.setattr(
            _proxy, "_client", _mock_client(body=sse_body, content_type="text/event-stream")
        )

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

        monkeypatch.setattr(_proxy, "_upstream", _FAKE_UPSTREAM)
        monkeypatch.setattr(
            _proxy, "_client", _mock_client(body=sse_body, content_type="text/event-stream")
        )

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

        monkeypatch.setattr(_proxy, "_upstream", _FAKE_UPSTREAM)
        monkeypatch.setattr(_proxy, "_client", _mock_client(body=models_body))

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
    monkeypatch.setattr(_proxy, "_upstream", _FAKE_UPSTREAM)
    resp = await async_client.get("/v1/models")
    assert resp.status_code == 401


# ── Real upstream integration tests ───────────────────────────────────────────
# Skipped unless OPENAI_PROXY_UPSTREAM_URL is set in the environment.

_upstream_configured = pytest.mark.skipif(
    not os.getenv("OPENAI_PROXY_UPSTREAM_URL"),
    reason="OPENAI_PROXY_UPSTREAM_URL not set",
)


async def _get_first_model(async_client, token: str) -> str:
    """Return OPENAI_PROXY_TEST_MODEL if set, otherwise the first model from /v1/models."""
    if model := os.getenv("OPENAI_PROXY_TEST_MODEL"):
        return model
    resp = await async_client.get("/v1/models", headers={"Authorization": f"Bearer {token}"})
    return resp.json()["data"][0]["id"]


@_upstream_configured
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
        model = await _get_first_model(async_client, token)
        resp = await async_client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
                "max_tokens": 10,
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
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
        model = await _get_first_model(async_client, token)
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
        assert resp.status_code == 200
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
        model = await _get_first_model(async_client, token)
        resp = await async_client.post(
            "/v1/chat/completions",
            json={"model": model},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code != 200
        assert resp.content
    finally:
        await cleanup_models([user])
