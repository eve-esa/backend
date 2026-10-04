"""Ordered rerank providers: order, fallback, skipping, and no event loop blocking."""

import asyncio
import logging
import time

import httpx
import pytest

from src.services import rerank as rerank_module
from src.services.rerank import (
    RerankProvider,
    configured_providers,
    parse_provider_order,
    rerank_candidates,
)
from src.utils.deepinfra_reranker import DeepInfraReranker
from src.utils.jsc_reranker import JSCReranker

pytestmark = pytest.mark.no_db

DOCS = ["first chunk", "second chunk"]


class _RecordingErrorLogger:
    def __init__(self):
        self.descriptions = []

    async def log_error(self, *, description="", **_kwargs):
        self.descriptions.append(description)


@pytest.fixture
def error_log(monkeypatch):
    recorder = _RecordingErrorLogger()
    monkeypatch.setattr(rerank_module, "get_error_logger", lambda: recorder)
    return recorder


def _result(tag):
    return [{"index": 1, "reranking_score": 0.9, "by": tag}, {"index": 0, "reranking_score": 0.1, "by": tag}]


def _provider(name, label, *, sleep=0.0, fail=False, calls=None, missing=None):
    async def call(query, documents):
        if calls is not None:
            calls.append(name)
        await asyncio.sleep(sleep)
        if fail:
            raise RuntimeError(f"{name} exploded")
        return _result(name)

    return RerankProvider(name, label, missing, call)


async def test_first_provider_in_order_answers_and_the_rest_are_not_called(error_log):
    calls = []
    providers = [_provider("a", "A", calls=calls), _provider("b", "B", calls=calls)]

    result = await rerank_candidates(DOCS, "q", providers=providers)

    assert result == _result("a")
    assert calls == ["a"]
    assert error_log.descriptions == []


async def test_order_setting_is_honoured_and_normalised():
    parse_provider_order.cache_clear()
    assert parse_provider_order("deepinfra,jsc") == ("deepinfra", "jsc")
    assert parse_provider_order(" JSC , jsc,deepinfra ,") == ("jsc", "deepinfra")
    assert [p.name for p in configured_providers("deepinfra,jsc")] == ["deepinfra", "jsc"]


async def test_timeout_falls_to_the_next_provider_within_the_budget(error_log):
    calls = []
    providers = [
        _provider("slow", "Slow", sleep=2.0, calls=calls),
        _provider("fast", "Fast", calls=calls),
    ]

    started = time.perf_counter()
    result = await rerank_candidates(DOCS, "q", providers=providers, timeout=0.1)
    elapsed = time.perf_counter() - started

    assert result == _result("fast")
    assert calls == ["slow", "fast"]
    assert elapsed < 1.0
    assert error_log.descriptions == ["Slow reranker timed out"]


async def test_every_provider_failing_returns_empty_with_todays_descriptions(error_log):
    providers = [
        _provider("jsc", "JSC", fail=True),
        _provider("deepinfra", "DeepInfra", fail=True),
    ]

    assert await rerank_candidates(DOCS, "q", providers=providers) == []
    assert error_log.descriptions == [
        "JSC reranker failed",
        "DeepInfra reranker fallback failed",
    ]


async def test_unknown_provider_is_skipped_with_a_warning(caplog):
    parse_provider_order.cache_clear()
    with caplog.at_level(logging.WARNING, logger="src.services.rerank"):
        assert parse_provider_order("bogus,deepinfra") == ("deepinfra",)
    assert "bogus" in caplog.text
    assert configured_providers("bogus") == []


async def test_unconfigured_provider_is_skipped(error_log, caplog):
    calls = []
    providers = [
        _provider("jsc", "JSC", calls=calls, missing="EVE_JSC_BASE_URL environment variable not set"),
        _provider("deepinfra", "DeepInfra", calls=calls),
    ]

    with caplog.at_level(logging.WARNING, logger="src.services.rerank"):
        result = await rerank_candidates(DOCS, "q", providers=providers)

    assert result == _result("deepinfra")
    assert calls == ["deepinfra"]
    assert "EVE_JSC_BASE_URL environment variable not set" in caplog.text
    assert error_log.descriptions == []


async def test_no_candidates_calls_nothing(error_log):
    calls = []
    assert await rerank_candidates([], "q", providers=[_provider("a", "A", calls=calls)]) == []
    assert calls == []


async def test_cancellation_propagates_and_skips_the_fallback(error_log):
    calls = []
    providers = [
        _provider("slow", "Slow", sleep=5.0, calls=calls),
        _provider("fast", "Fast", calls=calls),
    ]
    task = asyncio.create_task(rerank_candidates(DOCS, "q", providers=providers, timeout=10))
    await asyncio.sleep(0.05)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert calls == ["slow"]
    assert error_log.descriptions == []


async def test_event_loop_keeps_ticking_while_a_rerank_waits(error_log):
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    ticking = asyncio.create_task(ticker())
    try:
        result = await rerank_candidates(
            DOCS, "q", providers=[_provider("slow", "Slow", sleep=0.5)], timeout=2
        )
    finally:
        ticking.cancel()

    assert result == _result("slow")
    # 0.5 s at one tick per 10 ms is about 50; a blocked loop gives 0 or 1.
    assert ticks >= 20


def _mock_client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_jsc_reranker_parses_results_and_sends_the_model():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = request.read()
        return httpx.Response(
            200,
            json={"results": [{"index": 0, "relevance_score": 0.2}, {"index": 1, "relevance_score": 0.8}]},
        )

    async with _mock_client(handler) as client:
        result = await JSCReranker("key", "https://jsc.example/v1/", "m").rerank(client, ["q"], DOCS)

    assert result == [{"index": 1, "reranking_score": 0.8}, {"index": 0, "reranking_score": 0.2}]
    assert seen["url"] == "https://jsc.example/v1/rerank"
    assert seen["auth"] == "Bearer key"
    assert b'"model":"m"' in seen["body"].replace(b" ", b"")


async def test_jsc_reranker_rejects_a_plain_text_200():
    # Blablador answers throttling with 200 text/plain: that is a failure.
    async with _mock_client(lambda request: httpx.Response(200, text="rate limited")) as client:
        with pytest.raises(ValueError):
            await JSCReranker("key", "https://jsc.example/v1").rerank(client, ["q"], DOCS)


async def test_deepinfra_reranker_parses_scores_and_raises_on_http_error():
    async with _mock_client(lambda request: httpx.Response(200, json={"scores": [0.1, 0.7]})) as client:
        result = await DeepInfraReranker("token").rerank(client, ["q"], DOCS)
    assert result == [{"index": 1, "reranking_score": 0.7}, {"index": 0, "reranking_score": 0.1}]

    async with _mock_client(lambda request: httpx.Response(503)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await DeepInfraReranker("token").rerank(client, ["q"], DOCS)


async def test_shared_client_has_the_agreed_timeouts_and_is_reused():
    holder = rerank_module._LoopBoundClient()
    try:
        client = holder.get()
        assert holder.get() is client
        assert client.timeout.connect == 3.0
        assert client.timeout.read == 10.0
    finally:
        await holder.aclose()
