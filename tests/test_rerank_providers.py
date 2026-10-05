"""Ordered rerank providers: order, fallback, skipping, and no event loop blocking."""

import asyncio
import json
import logging
import time

import httpx
import pytest
import pytest_asyncio

from src import config
from src.services import rerank as rerank_module
from src.services.generate_answer import _select_top_k_unique_results
from src.services.rerank import (
    RerankProvider,
    configured_providers,
    parse_provider_order,
    rerank_candidates,
)
from src.utils import jsc_reranker
from src.utils.deepinfra_reranker import DeepInfraReranker
from src.utils.jsc_reranker import JSCReranker

pytestmark = pytest.mark.no_db

DOCS = ["first chunk", "second chunk"]
RETRIEVAL_ORDER = [{"index": 0, "reranking_score": None}, {"index": 1, "reranking_score": None}]


class _RecordingErrorLogger:
    def __init__(self):
        self.descriptions = []
        self.error_types = []

    async def log_error(self, *, description="", error_type=None, **_kwargs):
        self.descriptions.append(description)
        self.error_types.append(error_type)


@pytest.fixture(autouse=True)
def _no_jsc_backoff():
    jsc_reranker.reset_backoff()
    yield
    jsc_reranker.reset_backoff()


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


def test_default_order_reranks_with_deepinfra_first_and_jsc_as_fallback(monkeypatch):
    monkeypatch.setattr(rerank_module, "RERANK_PROVIDER_ORDER", config.DEFAULT_RERANK_PROVIDER_ORDER)

    assert [p.name for p in configured_providers()] == ["deepinfra", "jsc"]


async def test_timeout_falls_to_the_next_provider_within_the_budget(error_log):
    calls = []
    providers = [
        _provider("slow", "Slow", sleep=2.0, calls=calls),
        _provider("fast", "Fast", calls=calls),
    ]

    started = time.perf_counter()
    result = await rerank_candidates(
        DOCS, "q", providers=providers, timeout=1.0, attempt_timeout=0.1
    )
    elapsed = time.perf_counter() - started

    assert result == _result("fast")
    assert calls == ["slow", "fast"]
    assert elapsed < 1.0
    assert error_log.descriptions == ["Slow reranker timed out"]
    assert error_log.error_types == ["TimeoutError"]


async def test_one_deadline_covers_the_whole_provider_loop(error_log):
    calls = []
    providers = [
        _provider("a", "A", sleep=2.0, calls=calls),
        _provider("b", "B", sleep=2.0, calls=calls),
        _provider("c", "C", calls=calls),
    ]

    started = time.perf_counter()
    result = await rerank_candidates(
        DOCS, "q", providers=providers, timeout=0.4, attempt_timeout=0.3
    )
    elapsed = time.perf_counter() - started

    # a takes 0.3, b gets the 0.1 left, c is never tried.
    assert result == RETRIEVAL_ORDER
    assert calls == ["a", "b"]
    assert elapsed < 0.7
    assert error_log.descriptions == ["A reranker timed out", "B reranker fallback timed out"]


async def test_saturated_provider_is_skipped_after_the_acquire_timeout(error_log):
    calls = []
    busy = asyncio.Semaphore(1)
    await busy.acquire()
    slow = _provider("jsc", "JSC", calls=calls)
    providers = [
        RerankProvider(slow.name, slow.label, None, slow.call, semaphore=lambda: busy),
        _provider("deepinfra", "DeepInfra", calls=calls),
    ]

    result = await rerank_candidates(DOCS, "q", providers=providers, acquire_timeout=0.05)

    assert result == _result("deepinfra")
    assert calls == ["deepinfra"]
    assert error_log.descriptions == []


async def test_error_types_keep_the_requests_era_names():
    request = httpx.Request("POST", "https://x.example")
    response = httpx.Response(503, request=request)
    assert rerank_module.error_type_name(TimeoutError()) == "TimeoutError"
    assert rerank_module.error_type_name(httpx.ReadTimeout("t", request=request)) == "TimeoutError"
    assert rerank_module.error_type_name(httpx.ConnectError("c", request=request)) == "RequestException"
    assert (
        rerank_module.error_type_name(httpx.HTTPStatusError("s", request=request, response=response))
        == "RequestException"
    )
    assert rerank_module.error_type_name(json.JSONDecodeError("bad", "x", 0)) == "ValueError"
    assert rerank_module.error_type_name(KeyError("scores")) == "KeyError"


def test_retry_after_is_parsed_and_capped():
    assert jsc_reranker._retry_after_seconds("5") == 5.0
    assert jsc_reranker._retry_after_seconds("3600") == jsc_reranker.RETRY_AFTER_CAP_S
    assert jsc_reranker._retry_after_seconds(None) == jsc_reranker.RETRY_AFTER_DEFAULT_S
    assert jsc_reranker._retry_after_seconds("soon") == jsc_reranker.RETRY_AFTER_DEFAULT_S
    assert jsc_reranker._retry_after_seconds("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0


async def test_every_provider_failing_keeps_retrieval_order_with_todays_descriptions(
    error_log, caplog
):
    providers = [
        _provider("jsc", "JSC", fail=True),
        _provider("deepinfra", "DeepInfra", fail=True),
    ]

    with caplog.at_level(logging.WARNING, logger="src.services.rerank"):
        assert await rerank_candidates(DOCS, "q", providers=providers) == RETRIEVAL_ORDER
    assert caplog.text.count("rerank.skipped reason=all_providers_failed") == 1
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
    holder = rerank_module._LoopBoundClient(concurrency=16)
    try:
        client = holder.get()
        assert holder.get() is client
        assert holder.semaphore() is holder.semaphore()
        assert client.timeout.connect == 3.0
        assert client.timeout.read == 10.0
        pool = client._transport._pool
        assert pool._max_connections == 64
        assert pool._max_keepalive_connections == 32
    finally:
        await holder.aclose()


async def test_stale_client_from_another_loop_is_closed_in_the_background():
    class _DeadLoop:
        def is_closed(self):
            return True

    holder = rerank_module._LoopBoundClient(concurrency=1)
    stale = holder.get()
    holder._loop = _DeadLoop()

    fresh = holder.get()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert fresh is not stale
    assert stale.is_closed
    assert not fresh.is_closed
    await holder.aclose()


# End to end through the real providers, with MockTransport in the module clients.


@pytest_asyncio.fixture
async def wired(monkeypatch, error_log):
    hits = {"jsc": 0, "deepinfra": 0}
    behaviour = {"jsc": None, "deepinfra": None}

    def jsc_handler(request):
        hits["jsc"] += 1
        return behaviour["jsc"](request)

    def deepinfra_handler(request):
        hits["deepinfra"] += 1
        return behaviour["deepinfra"](request)

    monkeypatch.setattr(rerank_module, "JSC_RERANKER_API_KEY", "jsc-key")
    monkeypatch.setattr(rerank_module, "EVE_JSC_BASE_URL", "https://jsc.example/v1")
    monkeypatch.setattr(rerank_module, "DEEPINFRA_API_TOKEN", "di-token")
    monkeypatch.setattr(
        rerank_module, "_jsc_client",
        rerank_module._LoopBoundClient(16, transport=httpx.MockTransport(jsc_handler)),
    )
    monkeypatch.setattr(
        rerank_module, "_deepinfra_client",
        rerank_module._LoopBoundClient(32, transport=httpx.MockTransport(deepinfra_handler)),
    )
    yield hits, behaviour, error_log
    await rerank_module._jsc_client.aclose()
    await rerank_module._deepinfra_client.aclose()


JSC_OK = {"results": [{"index": 0, "relevance_score": 0.3}, {"index": 1, "relevance_score": 0.6}]}
DI_OK = {"scores": [0.9, 0.2]}


async def test_e2e_jsc_answers(wired):
    hits, behaviour, error_log = wired
    behaviour["jsc"] = lambda r: httpx.Response(200, json=JSC_OK)
    behaviour["deepinfra"] = lambda r: httpx.Response(200, json=DI_OK)

    result = await rerank_candidates(DOCS, "q", providers=configured_providers("jsc,deepinfra"))

    assert result == [{"index": 1, "reranking_score": 0.6}, {"index": 0, "reranking_score": 0.3}]
    assert hits == {"jsc": 1, "deepinfra": 0}
    assert error_log.descriptions == []


async def test_e2e_jsc_429_falls_back_and_skips_jsc_for_the_window(wired, caplog):
    hits, behaviour, error_log = wired
    behaviour["jsc"] = lambda r: httpx.Response(429, headers={"Retry-After": "5"}, text="slow down")
    behaviour["deepinfra"] = lambda r: httpx.Response(200, json=DI_OK)
    di_result = [{"index": 0, "reranking_score": 0.9}, {"index": 1, "reranking_score": 0.2}]

    with caplog.at_level(logging.WARNING):
        first = await rerank_candidates(DOCS, "q", providers=configured_providers("jsc,deepinfra"))
        second = await rerank_candidates(DOCS, "q", providers=configured_providers("jsc,deepinfra"))

    assert first == di_result
    assert second == di_result
    assert hits == {"jsc": 1, "deepinfra": 2}
    assert 4.0 < jsc_reranker.backoff_remaining() <= 5.0
    assert caplog.text.count("rate limited (429), skipped for") == 1
    assert error_log.descriptions == ["JSC reranker failed"]
    assert error_log.error_types == ["RequestException"]


async def test_e2e_both_failing_returns_empty(wired):
    hits, behaviour, error_log = wired
    behaviour["jsc"] = lambda r: httpx.Response(500, text="boom")
    behaviour["deepinfra"] = lambda r: httpx.Response(200, text="not json")

    result = await rerank_candidates(DOCS, "q", providers=configured_providers("jsc,deepinfra"))

    assert result == RETRIEVAL_ORDER
    assert hits == {"jsc": 1, "deepinfra": 1}
    assert error_log.descriptions == ["JSC reranker failed", "DeepInfra reranker fallback failed"]
    assert error_log.error_types == ["RequestException", "ValueError"]


async def test_no_configured_provider_keeps_retrieval_order(error_log, caplog):
    providers = [_provider("jsc", "JSC", missing="DEEPINFRA_API_TOKEN environment variable not set")]

    with caplog.at_level(logging.WARNING, logger="src.services.rerank"):
        assert await rerank_candidates(DOCS, "q", providers=providers) == RETRIEVAL_ORDER
    assert "rerank.skipped reason=no_provider_available" in caplog.text


async def test_e2e_jsc_429_and_deepinfra_timeout_still_answer_with_sources(wired, caplog):
    hits, behaviour, error_log = wired

    async def deepinfra_hangs(request):
        await asyncio.sleep(1.0)
        return httpx.Response(200, json=DI_OK)

    behaviour["jsc"] = lambda r: httpx.Response(429, headers={"Retry-After": "5"})
    behaviour["deepinfra"] = deepinfra_hangs
    texts = ["chunk a", "chunk a", "chunk b", "chunk c"]
    formatted = [{"id": f"r{i}", "text": t} for i, t in enumerate(texts)]

    with caplog.at_level(logging.WARNING, logger="src.services.rerank"):
        reranked = await rerank_candidates(
            texts, "q", providers=configured_providers("jsc,deepinfra"), attempt_timeout=0.1
        )
    selected = _select_top_k_unique_results(formatted, reranked, top_k=2)

    assert hits == {"jsc": 1, "deepinfra": 1}
    assert [r["id"] for r in selected] == ["r0", "r2"]
    assert caplog.text.count("rerank.skipped reason=all_providers_failed") == 1
    assert error_log.descriptions == ["JSC reranker failed", "DeepInfra reranker fallback timed out"]
    assert error_log.error_types == ["RequestException", "TimeoutError"]


def _capturing_provider(seen):
    """Records the texts it receives and ranks the longest original first."""

    async def call(query, documents):
        seen.append(list(documents))
        return [
            {"index": i, "reranking_score": 1.0 / (rank + 1)}
            for rank, i in enumerate(sorted(range(len(documents)), key=lambda i: -i))
        ]

    return RerankProvider("cap", "Cap", None, call)


LONG_DOCS = ["a" * 4000, "short", "b" * 1501, "c" * 1500]


async def test_provider_receives_candidates_cut_to_the_cap(error_log, monkeypatch):
    monkeypatch.setattr(rerank_module, "RERANK_MAX_CHARS_PER_CANDIDATE", 1500)
    seen = []

    await rerank_candidates(LONG_DOCS, "q", providers=[_capturing_provider(seen)])

    assert len(seen) == 1
    assert [len(t) for t in seen[0]] == [1500, 5, 1500, 1500]
    assert [t[:1] for t in seen[0]] == ["a", "s", "b", "c"]


async def test_trimmed_result_indexes_still_point_at_the_original_candidates(
    error_log, monkeypatch
):
    monkeypatch.setattr(rerank_module, "RERANK_MAX_CHARS_PER_CANDIDATE", 1500)
    seen = []
    formatted = [{"id": f"r{i}", "text": t} for i, t in enumerate(LONG_DOCS)]

    reranked = await rerank_candidates(LONG_DOCS, "q", providers=[_capturing_provider(seen)])
    selected = _select_top_k_unique_results(formatted, reranked, top_k=4)

    assert [r["index"] for r in reranked] == [3, 2, 1, 0]
    assert [r["id"] for r in selected] == ["r3", "r2", "r1", "r0"]
    assert selected[3]["text"] == "a" * 4000


def test_trim_candidates_accepts_an_empty_list():
    assert rerank_module.trim_candidates([], 1500) == []


async def test_cap_zero_sends_the_full_texts(error_log, monkeypatch):
    monkeypatch.setattr(rerank_module, "RERANK_MAX_CHARS_PER_CANDIDATE", 0)
    seen = []

    await rerank_candidates(LONG_DOCS, "q", providers=[_capturing_provider(seen)])

    assert seen == [LONG_DOCS]


async def test_one_candidates_line_per_turn_carries_the_counts(error_log, monkeypatch, caplog):
    monkeypatch.setattr(rerank_module, "RERANK_MAX_CHARS_PER_CANDIDATE", 1500)
    providers = [_provider("a", "A", fail=True), _capturing_provider([])]

    with caplog.at_level(logging.INFO, logger="src.services.rerank"):
        await rerank_candidates(LONG_DOCS, "q", providers=providers)

    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("rerank.candidates")]
    assert lines == ["rerank.candidates n=4 chars_p50=1500 chars_max=4000 truncated=2"]
