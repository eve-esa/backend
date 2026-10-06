"""Embedding provider order: first provider that answers wins, the rest are fallbacks."""

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest

import src.core.vector_store_manager as vsm
from src.constants import DEEPINFRA_DEFAULT_EMBEDDING_MODEL, JSC_DEFAULT_EMBEDDING_MODEL


class _FakeOpenAI:
    """Records (base_url, model, read timeout) per call; fails for the urls in `failing`.

    A url in `rate_limited` answers 429 once, then succeeds; `delay` holds every call.
    """

    calls: list = []
    built: list = []
    failing: set = set()
    rate_limited: set = set()
    delay = 0.0
    in_flight = 0
    max_in_flight = 0
    size = vsm.EMBEDDING_SIZE

    def __init__(self, api_key, base_url, timeout, max_retries):
        _FakeOpenAI.built.append((base_url, timeout, max_retries))
        self.base_url = base_url
        self.embeddings = SimpleNamespace(create=self._create)

    async def close(self):
        pass

    async def _create(self, input, model, timeout):
        _FakeOpenAI.calls.append((self.base_url, model, timeout.read))
        _FakeOpenAI.in_flight += 1
        _FakeOpenAI.max_in_flight = max(_FakeOpenAI.max_in_flight, _FakeOpenAI.in_flight)
        try:
            await asyncio.sleep(_FakeOpenAI.delay)
        finally:
            _FakeOpenAI.in_flight -= 1
        if self.base_url in _FakeOpenAI.failing:
            raise RuntimeError(f"{self.base_url} down")
        if self.base_url in _FakeOpenAI.rate_limited:
            _FakeOpenAI.rate_limited.discard(self.base_url)
            response = httpx.Response(
                429,
                headers={"retry-after": "0.05"},
                request=httpx.Request("POST", f"http://{self.base_url}/embeddings"),
            )
            raise openai.RateLimitError("slow down", response=response, body=None)
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[float(len(text))] * _FakeOpenAI.size)
                for text in input
            ]
        )


def _configure_providers(monkeypatch):
    monkeypatch.setattr(vsm, "_embedding_clients", {})
    monkeypatch.setattr(vsm, "DEEPINFRA_EMBEDDING_API_KEY", "di-key")
    monkeypatch.setattr(vsm, "JSC_EMBEDDING_API_KEY", "jsc-key")
    monkeypatch.setattr(vsm, "EMBEDDING_PROVIDER_ORDER", ["deepinfra", "jsc"])
    error_logger = MagicMock(log_error_sync=AsyncMock())
    monkeypatch.setattr(vsm, "get_error_logger", lambda: error_logger)
    # No Qdrant client: only the embedding methods are exercised.
    return object.__new__(vsm.VectorStoreManager)


@pytest.fixture
def manager(monkeypatch):
    _FakeOpenAI.calls = []
    _FakeOpenAI.built = []
    _FakeOpenAI.failing = set()
    _FakeOpenAI.rate_limited = set()
    _FakeOpenAI.delay = 0.0
    _FakeOpenAI.in_flight = _FakeOpenAI.max_in_flight = 0
    _FakeOpenAI.size = vsm.EMBEDDING_SIZE
    monkeypatch.setattr(vsm, "AsyncOpenAI", _FakeOpenAI)
    monkeypatch.setattr(vsm, "DEEPINFRA_EMBEDDING_URL", "deepinfra")
    monkeypatch.setattr(vsm, "EVE_JSC_BASE_URL", "jsc")
    return _configure_providers(monkeypatch)


async def test_deepinfra_answers_first_and_jsc_is_not_called(manager):
    vector, fallback_error = await manager.generate_query_vector(
        "sentinel", JSC_DEFAULT_EMBEDDING_MODEL
    )

    assert vector[0] == 8.0 and len(vector) == vsm.EMBEDDING_SIZE
    assert fallback_error is None
    assert _FakeOpenAI.calls == [("deepinfra", DEEPINFRA_DEFAULT_EMBEDDING_MODEL, 10.0)]


async def test_falls_back_to_jsc_and_reports_the_first_error(manager):
    _FakeOpenAI.failing = {"deepinfra"}

    vector, fallback_error = await manager.generate_query_vector(
        "sentinel", JSC_DEFAULT_EMBEDDING_MODEL
    )

    assert vector[0] == 8.0 and len(vector) == vsm.EMBEDDING_SIZE
    assert fallback_error == "deepinfra down"
    assert [url for url, *_ in _FakeOpenAI.calls] == ["deepinfra", "jsc"]


def _embedding_failure_levels(caplog):
    return [
        (record.levelname, record.getMessage().split(":")[0])
        for record in caplog.records
        if record.getMessage().startswith("Failed to generate embeddings via")
    ]


async def test_recovered_provider_failure_is_a_warning(manager, caplog):
    _FakeOpenAI.failing = {"deepinfra"}

    with caplog.at_level("WARNING", logger=vsm.logger.name):
        await manager.generate_query_vector("sentinel", JSC_DEFAULT_EMBEDDING_MODEL)

    assert _embedding_failure_levels(caplog) == [
        ("WARNING", "Failed to generate embeddings via deepinfra")
    ]


async def test_last_provider_failure_is_an_error(manager, caplog):
    _FakeOpenAI.failing = {"deepinfra", "jsc"}

    with caplog.at_level("WARNING", logger=vsm.logger.name):
        with pytest.raises(RuntimeError):
            await manager.generate_query_vector("sentinel", JSC_DEFAULT_EMBEDDING_MODEL)

    assert _embedding_failure_levels(caplog) == [
        ("WARNING", "Failed to generate embeddings via deepinfra"),
        ("ERROR", "Failed to generate embeddings via jsc"),
    ]


async def test_order_is_configurable(manager, monkeypatch):
    monkeypatch.setattr(vsm, "EMBEDDING_PROVIDER_ORDER", ["jsc", "deepinfra"])

    await manager.generate_query_vector("sentinel", JSC_DEFAULT_EMBEDDING_MODEL)

    assert _FakeOpenAI.calls == [("jsc", vsm.JSC_EMBEDDING_MODEL_NAME, 10.0)]


async def test_missing_key_counts_as_a_failure(manager, monkeypatch):
    monkeypatch.setattr(vsm, "DEEPINFRA_EMBEDDING_API_KEY", "")

    _, fallback_error = await manager.generate_query_vector(
        "sentinel", JSC_DEFAULT_EMBEDDING_MODEL
    )

    assert fallback_error == "deepinfra embedding API key is not set"
    assert [url for url, *_ in _FakeOpenAI.calls] == ["jsc"]


async def test_every_provider_failing_names_them_all(manager):
    _FakeOpenAI.failing = {"deepinfra", "jsc"}

    with pytest.raises(RuntimeError, match="via deepinfra and jsc: jsc down"):
        await manager.generate_query_vector("sentinel", JSC_DEFAULT_EMBEDDING_MODEL)


async def test_unknown_names_only_leave_nothing_to_call(manager, monkeypatch):
    monkeypatch.setattr(vsm, "EMBEDDING_PROVIDER_ORDER", ["nope"])

    with pytest.raises(RuntimeError, match="No embedding provider configured"):
        await manager.generate_query_vector("sentinel", JSC_DEFAULT_EMBEDDING_MODEL)


async def test_batch_uses_the_same_order(manager):
    _FakeOpenAI.failing = {"deepinfra"}

    vectors, fallback_error = await manager.generate_batch_embeddings(
        ["a", "bb"], JSC_DEFAULT_EMBEDDING_MODEL
    )

    assert [vector[0] for vector in vectors] == [1.0, 2.0]
    assert fallback_error == "deepinfra down"
    assert [read for *_, read in _FakeOpenAI.calls] == [60.0, 60.0]


async def test_batch_with_no_texts_calls_nothing(manager):
    assert await manager.generate_batch_embeddings([], JSC_DEFAULT_EMBEDDING_MODEL) == (
        [],
        None,
    )
    assert _FakeOpenAI.calls == []


async def test_wrong_vector_size_counts_as_a_failure(manager, monkeypatch):
    # Blablador only serves 4096-d models now: its vectors must not reach Qdrant.
    monkeypatch.setattr(vsm, "EMBEDDING_PROVIDER_ORDER", ["jsc"])
    _FakeOpenAI.size = 4096

    with pytest.raises(RuntimeError, match=r"jsc returned \[4096\]-d vectors"):
        await manager.generate_query_vector("sentinel", JSC_DEFAULT_EMBEDDING_MODEL)


async def test_one_client_per_provider_without_sdk_retries(manager):
    _FakeOpenAI.failing = {"deepinfra"}

    for _ in range(3):
        await manager.generate_query_vector("sentinel", JSC_DEFAULT_EMBEDDING_MODEL)

    assert _FakeOpenAI.built == [
        ("deepinfra", vsm.EMBEDDING_TIMEOUT, 0),
        ("jsc", vsm.EMBEDDING_TIMEOUT, 0),
    ]
    assert vsm.EMBEDDING_TIMEOUT.connect == 3.0 and vsm.EMBEDDING_TIMEOUT.read == 10.0


async def test_deadline_bounds_the_whole_call(manager, monkeypatch):
    # httpx bounds connect and each read; a provider trickling bytes would pass both.
    monkeypatch.setattr(vsm, "EMBEDDING_DEADLINE_S", 0.1)
    _FakeOpenAI.delay = 0.3

    started = time.perf_counter()
    with pytest.raises(RuntimeError, match="jsc did not answer within 0.1 s"):
        await manager.generate_query_vector("sentinel", JSC_DEFAULT_EMBEDDING_MODEL)

    assert [url for url, *_ in _FakeOpenAI.calls] == ["deepinfra", "jsc"]
    assert time.perf_counter() - started < 0.5


async def test_rate_limit_is_retried_once_on_the_same_provider(manager):
    _FakeOpenAI.rate_limited = {"deepinfra"}

    vector, fallback_error = await manager.generate_query_vector(
        "sentinel", JSC_DEFAULT_EMBEDDING_MODEL
    )

    assert fallback_error is None and len(vector) == vsm.EMBEDDING_SIZE
    assert [url for url, *_ in _FakeOpenAI.calls] == ["deepinfra", "deepinfra"]


async def test_rate_limit_wait_follows_retry_after_with_a_cap():
    def limited(headers):
        response = httpx.Response(
            429, headers=headers, request=httpx.Request("POST", "http://x/embeddings")
        )
        return openai.RateLimitError("slow down", response=response, body=None)

    assert vsm._rate_limit_wait_s(limited({"retry-after": "0.5"})) == 0.5
    assert vsm._rate_limit_wait_s(limited({"retry-after": "30"})) == 2.0
    assert vsm._rate_limit_wait_s(limited({"retry-after": "Wed, 21 Oct"})) == 1.0
    assert vsm._rate_limit_wait_s(limited({})) == 1.0


async def test_concurrency_per_provider_is_capped(manager, monkeypatch):
    monkeypatch.setattr(vsm, "EMBEDDING_MAX_CONCURRENCY", 2)
    _FakeOpenAI.delay = 0.05

    await asyncio.gather(
        *(
            manager.generate_query_vector("sentinel", JSC_DEFAULT_EMBEDDING_MODEL)
            for _ in range(6)
        )
    )

    assert len(_FakeOpenAI.calls) == 6
    assert _FakeOpenAI.max_in_flight == 2


# Real AsyncOpenAI against a local HTTP server running in its own thread, so a client
# that blocked the event loop would show up as a stopped ticker, and a provider that
# never answers exercises the real httpx timeout.


class _EmbeddingHandler(BaseHTTPRequestHandler):
    # First path segment picks the behaviour: /<delay seconds>/v1/embeddings
    def do_POST(self):
        delay = float(self.path.strip("/").split("/")[0])
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        time.sleep(delay)
        payload = json.dumps(
            {
                "object": "list",
                "model": body["model"],
                "data": [
                    {"object": "embedding", "index": i, "embedding": [0.5] * vsm.EMBEDDING_SIZE}
                    for i, _ in enumerate(body["input"])
                ],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            }
        ).encode()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the client gave up first, which is what the timeout test wants

    def log_message(self, *args):
        pass


@pytest.fixture
def provider_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _EmbeddingHandler)
    # Joined on close, so no handler thread outlives its test.
    server.daemon_threads = False
    server.block_on_close = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


@pytest.fixture
async def real_manager(monkeypatch, provider_server):
    yield _configure_providers(monkeypatch)
    for entry in vsm._embedding_clients.values():
        await entry.client.close()


async def test_slow_provider_does_not_block_the_event_loop(
    real_manager, provider_server, monkeypatch
):
    monkeypatch.setattr(vsm, "DEEPINFRA_EMBEDDING_URL", f"{provider_server}/0.5/v1")
    monkeypatch.setattr(vsm, "EVE_JSC_BASE_URL", f"{provider_server}/0/v1")
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.create_task(ticker())
    started = time.perf_counter()
    try:
        vector, fallback_error = await real_manager.generate_query_vector(
            "sentinel", JSC_DEFAULT_EMBEDDING_MODEL
        )
    finally:
        task.cancel()
    elapsed = time.perf_counter() - started

    assert fallback_error is None and len(vector) == vsm.EMBEDDING_SIZE
    assert elapsed >= 0.5
    # About 50 ticks fit in 0.5 s; a blocking client leaves the ticker at 0 or 1.
    assert ticks >= 20


async def test_provider_past_the_timeout_falls_to_the_next(
    real_manager, provider_server, monkeypatch
):
    monkeypatch.setattr(vsm, "EMBEDDING_TIMEOUT", httpx.Timeout(0.1))
    monkeypatch.setattr(vsm, "DEEPINFRA_EMBEDDING_URL", f"{provider_server}/0.3/v1")
    monkeypatch.setattr(vsm, "EVE_JSC_BASE_URL", f"{provider_server}/0/v1")

    started = time.perf_counter()
    vector, fallback_error = await real_manager.generate_query_vector(
        "sentinel", JSC_DEFAULT_EMBEDDING_MODEL
    )

    assert len(vector) == vsm.EMBEDDING_SIZE
    assert fallback_error == "Request timed out."
    # No SDK retries: one 0.1 s wait on the dead provider, then the next one answers.
    assert time.perf_counter() - started < 1.0
