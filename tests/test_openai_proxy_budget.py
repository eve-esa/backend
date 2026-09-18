"""``/v1`` charged to the caller's monthly token budget.

Complements ``test_openai_proxy.py`` (allowlist, stream_options)
and ``test_token_rate_limiter.py`` (the atomic counter in isolation): this
file is the two wired together through the real dispatcher.
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from server import app as _root_app
from src.database.models.api_key import ApiKey
from src.database.models.user import User
from src.services.auth import generate_api_key
from src.services.token_rate_limiter import count_tokens_for_texts
from tests.utils.cleaner import cleanup_models
from tests.utils.openai_proxy import (
    enable_proxy,
    json_handler,
    mock_client,
    mock_client_failing_mid_stream,
    mock_client_raising,
    seed_active_window,
    sse_handler,
    transport_client,
)
from tests.utils.utils import create_test_user_and_token

# App stack: MCPProxyDispatcher -> OpenAIProxyDispatcher -> FastAPI
_proxy = _root_app.main_app

_PINNED_SETTINGS = {
    "enabled": True,
    "default_group": "eve_free",
    "groups": {"eve_free": {"max_tokens": 100, "period_months": 1}},
}


@pytest.fixture(autouse=True)
def _reset_openai_proxy_http_client():
    """Avoid cross-test httpx client reuse (pytest-asyncio uses one loop per test)."""
    _proxy._client = None
    _proxy._client_loop = None
    yield


@pytest.fixture
def pinned_budget(monkeypatch):
    """Pin the eve_free group to a known, small cap so charges are exact."""
    monkeypatch.setattr(
        "src.services.token_rate_limiter._rate_limit_settings", lambda: _PINNED_SETTINGS
    )
    return _PINNED_SETTINGS


async def _make_api_key(user_id: str) -> str:
    raw_token, key_hash = generate_api_key()
    await ApiKey.create(user_id=user_id, name="budget-test", key_hash=key_hash)
    return raw_token


# ── Exhausted budget: no upstream call ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_exhausted_budget_via_session_returns_429(async_client, monkeypatch, pinned_budget):
    user, token = await create_test_user_and_token()
    try:
        await seed_active_window(user, used_tokens=100)
        enable_proxy(monkeypatch, _proxy)
        client, stream_mock = mock_client()
        monkeypatch.setattr(_proxy, "_client", client)

        with patch("src.routers.openai_proxy.track_usage", new_callable=AsyncMock) as mock_track:
            resp = await async_client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
                headers={"Authorization": f"Bearer {token}"},
            )

        assert resp.status_code == 429
        data = resp.json()
        assert data["detail"].startswith("Token budget exceeded for group 'eve_free'.")
        assert data["error"] == {
            "message": data["detail"],
            "type": "insufficient_quota",
            "code": "token_budget_exceeded",
            "param": None,
        }
        assert resp.headers.get("x-should-retry") == "false"
        assert int(resp.headers["retry-after"]) >= 1
        stream_mock.assert_not_called()
        mock_track.assert_not_awaited()
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_exhausted_budget_via_api_key_returns_429(async_client, monkeypatch, pinned_budget):
    user, _ = await create_test_user_and_token()
    try:
        await seed_active_window(user, used_tokens=100)
        raw_key = await _make_api_key(user.id)
        enable_proxy(monkeypatch, _proxy)
        client, stream_mock = mock_client()
        monkeypatch.setattr(_proxy, "_client", client)

        resp = await async_client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"Authorization": f"Bearer {raw_key}"},
        )

        assert resp.status_code == 429
        stream_mock.assert_not_called()
    finally:
        await ApiKey.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_models_endpoint_not_gated_by_exhausted_budget(async_client, monkeypatch, pinned_budget):
    user, token = await create_test_user_and_token()
    try:
        await seed_active_window(user, used_tokens=100)
        enable_proxy(monkeypatch, _proxy)
        monkeypatch.setattr("src.routers.openai_proxy.MAIN_MODEL_NAME", "eve-esa/EVE-Instruct")

        resp = await async_client.get("/v1/models", headers={"Authorization": f"Bearer {token}"})

        assert resp.status_code == 200
    finally:
        await cleanup_models([user])


# ── Charging: reported usage wins, estimate is the fallback ────────────────────


@pytest.mark.asyncio
async def test_charges_reported_total_non_streaming(async_client, monkeypatch, pinned_budget):
    user, token = await create_test_user_and_token()
    try:
        enable_proxy(monkeypatch, _proxy)
        payload = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        client = transport_client(json_handler(200, payload))
        monkeypatch.setattr(_proxy, "_client", client)

        with patch("src.routers.openai_proxy.track_usage", new_callable=AsyncMock) as mock_track:
            resp = await async_client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
                headers={"Authorization": f"Bearer {token}"},
            )

        assert resp.status_code == 200
        assert mock_track.call_args.kwargs["billed_tokens"] == 15
        refreshed = await User.find_by_id(user.id)
        assert refreshed.rate_limit_tokens_used == 15
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_charges_reported_total_streaming(async_client, monkeypatch, pinned_budget):
    user, token = await create_test_user_and_token()
    try:
        enable_proxy(monkeypatch, _proxy)
        client = transport_client(sse_handler(prompt_tokens=7, completion_tokens=4))
        monkeypatch.setattr(_proxy, "_client", client)

        with patch("src.routers.openai_proxy.track_usage", new_callable=AsyncMock) as mock_track:
            resp = await async_client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
                headers={"Authorization": f"Bearer {token}"},
            )

        assert resp.status_code == 200
        assert mock_track.call_args.kwargs["billed_tokens"] == 11
        refreshed = await User.find_by_id(user.id)
        assert refreshed.rate_limit_tokens_used == 11
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_charges_estimate_when_usage_missing(async_client, monkeypatch, pinned_budget):
    user, token = await create_test_user_and_token()
    try:
        enable_proxy(monkeypatch, _proxy)
        request_text = "Hi"
        response_text = "Hello there"
        payload = {
            "id": "chatcmpl-2",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": response_text}, "finish_reason": "stop"}],
        }
        client = transport_client(json_handler(200, payload))
        monkeypatch.setattr(_proxy, "_client", client)

        resp = await async_client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": request_text}]},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert resp.status_code == 200
        expected = max(count_tokens_for_texts(request_text, response_text), 1)
        refreshed = await User.find_by_id(user.id)
        assert refreshed.rate_limit_tokens_used == expected
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_charges_embeddings(async_client, monkeypatch, pinned_budget):
    user, token = await create_test_user_and_token()
    try:
        enable_proxy(monkeypatch, _proxy)
        payload = {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
            "usage": {"prompt_tokens": 7, "total_tokens": 7},
        }
        client = transport_client(json_handler(200, payload))
        monkeypatch.setattr(_proxy, "_client", client)

        resp = await async_client.post(
            "/v1/embeddings",
            json={"model": "gpt-4", "input": "hello world"},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert resp.status_code == 200
        refreshed = await User.find_by_id(user.id)
        assert refreshed.rate_limit_tokens_used == 7
    finally:
        await cleanup_models([user])


# ── No charge, error paths ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_upstream_error_status_no_charge_but_tracked(async_client, monkeypatch, pinned_budget):
    user, token = await create_test_user_and_token()
    try:
        enable_proxy(monkeypatch, _proxy)
        client, _ = mock_client(status=500, body=json.dumps({"error": "boom"}).encode())
        monkeypatch.setattr(_proxy, "_client", client)

        with patch("src.routers.openai_proxy.track_usage", new_callable=AsyncMock) as mock_track:
            resp = await async_client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
                headers={"Authorization": f"Bearer {token}"},
            )

        assert resp.status_code == 500
        mock_track.assert_awaited_once()
        assert mock_track.call_args.kwargs["billed_tokens"] == 0
        assert mock_track.call_args.kwargs["status_code"] == 500
        refreshed = await User.find_by_id(user.id)
        assert refreshed.rate_limit_tokens_used == 0
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_connect_error_no_charge_no_tracking(async_client, monkeypatch, pinned_budget):
    user, token = await create_test_user_and_token()
    try:
        enable_proxy(monkeypatch, _proxy)
        client, _ = mock_client_raising(httpx.ConnectError("connection refused"))
        monkeypatch.setattr(_proxy, "_client", client)

        with patch("src.routers.openai_proxy.track_usage", new_callable=AsyncMock) as mock_track:
            resp = await async_client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
                headers={"Authorization": f"Bearer {token}"},
            )

        assert resp.status_code == 502
        mock_track.assert_not_awaited()
        refreshed = await User.find_by_id(user.id)
        assert refreshed.rate_limit_tokens_used == 0
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_mid_stream_exception_charges_the_estimate(async_client, monkeypatch, pinned_budget):
    """A client that aborts, or an upstream that drops the connection mid-stream, still pays.

    The finally in ``OpenAIProxyDispatcher._proxy`` runs regardless: no usage
    chunk ever arrived, so the charge falls back to an estimate over whatever
    request text is available, same as ``test_charges_estimate_when_usage_missing``.
    The response had already started (status 200 sent) when the connection
    dropped, so the dispatcher does not try to send a second response of its
    own; it re-raises, and httpx's ASGI test transport surfaces that as the
    original exception out of the client call instead of a response object.
    """
    user, token = await create_test_user_and_token()
    try:
        enable_proxy(monkeypatch, _proxy)
        request_text = "Hi there, please answer at length"
        client, _ = mock_client_failing_mid_stream(exc=ConnectionResetError("reset"))
        monkeypatch.setattr(_proxy, "_client", client)

        with pytest.raises(ConnectionResetError):
            await async_client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4", "messages": [{"role": "user", "content": request_text}], "stream": True},
                headers={"Authorization": f"Bearer {token}"},
            )

        expected = max(count_tokens_for_texts(request_text), 1)
        refreshed = await User.find_by_id(user.id)
        assert refreshed.rate_limit_tokens_used == expected
    finally:
        await cleanup_models([user])


# ── Concurrency and the soft cap ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ten_parallel_calls_bill_exactly_50(async_client, monkeypatch, pinned_budget):
    user, token = await create_test_user_and_token()
    try:
        enable_proxy(monkeypatch, _proxy)
        payload = {
            "id": "chatcmpl-3",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
        client = transport_client(json_handler(200, payload))
        monkeypatch.setattr(_proxy, "_client", client)

        async def _call():
            return await async_client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
                headers={"Authorization": f"Bearer {token}"},
            )

        with patch("src.routers.openai_proxy.track_usage", new_callable=AsyncMock):
            responses = await asyncio.gather(*(_call() for _ in range(10)))

        assert all(resp.status_code == 200 for resp in responses)
        refreshed = await User.find_by_id(user.id)
        assert refreshed.rate_limit_tokens_used == 50
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_soft_cap_lets_the_crossing_request_complete(async_client, monkeypatch, pinned_budget):
    user, token = await create_test_user_and_token()
    try:
        await seed_active_window(user, used_tokens=95)
        enable_proxy(monkeypatch, _proxy)
        client, stream_mock = mock_client(
            body=json.dumps({
                "id": "chatcmpl-4",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            }).encode()
        )
        monkeypatch.setattr(_proxy, "_client", client)

        first = await async_client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert first.status_code == 200
        refreshed = await User.find_by_id(user.id)
        assert refreshed.rate_limit_tokens_used == 115

        second = await async_client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert second.status_code == 429
        assert stream_mock.call_count == 1  # the second request never reached upstream
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_concurrent_calls_bounded_by_one_crossing_request(async_client, monkeypatch, pinned_budget):
    """20 concurrent 30-token requests against a 100-token cap.

    A plain check-then-act (read the used total, forward, charge only once
    the response comes back) lets every one of the 20 in, because they all
    read ``used == 0`` before any of the later charges land: the atomic
    counter itself never loses an increment, so that race is invisible to a
    single-request test, but the cap ends up meaning nothing under caller
    concurrency. The fix must admit only as many as the cap actually allows
    (4, landing exactly on 120: one crossing request's worth over 100, the
    same overshoot a single unlucky request would cause), and refuse the rest
    before they ever reach upstream.
    """
    user, token = await create_test_user_and_token()
    try:
        enable_proxy(monkeypatch, _proxy)
        # Pin the pre-flight estimate to match the upstream-reported total
        # exactly, so the reservation and the real charge never disagree and
        # the math above (4 * 30 = 120) is exact.
        monkeypatch.setattr("src.routers.openai_proxy.count_tokens_for_texts", lambda *a, **k: 30)
        payload = {
            "id": "chatcmpl-conc",
            "object": "chat.completion",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 15, "completion_tokens": 15, "total_tokens": 30},
        }
        upstream_hits = []

        def handler(request: httpx.Request) -> httpx.Response:
            upstream_hits.append(request)
            return httpx.Response(200, json=payload)

        client = transport_client(handler)
        monkeypatch.setattr(_proxy, "_client", client)

        async def _call():
            return await async_client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
                headers={"Authorization": f"Bearer {token}"},
            )

        with patch("src.routers.openai_proxy.track_usage", new_callable=AsyncMock):
            responses = await asyncio.gather(*(_call() for _ in range(20)))

        ok = [r for r in responses if r.status_code == 200]
        refused = [r for r in responses if r.status_code == 429]
        assert len(ok) + len(refused) == 20
        assert len(ok) == 4, f"expected exactly 4 admissions, got {len(ok)}"
        assert len(upstream_hits) == 4, "a refused request must never reach upstream"

        refreshed = await User.find_by_id(user.id)
        assert refreshed.rate_limit_tokens_used == 120
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_rate_limiting_disabled_no_charge(async_client, monkeypatch):
    """No policy resolves (``enabled: false``): nothing is checked or charged."""
    user, token = await create_test_user_and_token()
    try:
        monkeypatch.setattr(
            "src.services.token_rate_limiter._rate_limit_settings", lambda: {"enabled": False}
        )
        enable_proxy(monkeypatch, _proxy)
        client, _ = mock_client(
            body=json.dumps({
                "id": "chatcmpl-5",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            }).encode()
        )
        monkeypatch.setattr(_proxy, "_client", client)

        resp = await async_client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert resp.status_code == 200
        refreshed = await User.find_by_id(user.id)
        assert refreshed.rate_limit_tokens_used == 0
    finally:
        await cleanup_models([user])
