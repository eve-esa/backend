"""Qdrant reads run on the async client: no event loop stall, cached aliases."""

import asyncio
import logging
import socket
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

import src.core.vector_store_manager as vsm
from src.core.vector_store_manager import (
    VectorStoreManager,
    _is_timeout_error,
    aclose_qdrant_read_clients,
    invalidate_qdrant_read_cache,
)

pytestmark = pytest.mark.no_db

WILEY = "Wiley AI Gateway"  # public, and never inspected for an env payload


@pytest.fixture(autouse=True)
def _cold_cache():
    invalidate_qdrant_read_cache()
    yield
    invalidate_qdrant_read_cache()


def _manager(
    search_delay_s: float = 0.0, delays: dict[str, float] | None = None
) -> VectorStoreManager:
    manager = VectorStoreManager.__new__(VectorStoreManager)
    manager.client = MagicMock()
    manager.qdrant_url = "http://qdrant.test:6333"
    manager.embeddings_model = "test-model"
    manager._env_payload_cache = {}
    manager.ensure_private_collection = MagicMock()

    async def _query_points(**kwargs):
        await asyncio.sleep((delays or {}).get(kwargs["collection_name"], search_delay_s))
        point = SimpleNamespace(id="p1", score=0.9, payload={"text": "chunk"})
        return SimpleNamespace(points=[point])

    aclient = MagicMock()
    aclient.get_aliases = AsyncMock(
        return_value=SimpleNamespace(
            aliases=[SimpleNamespace(alias_name="Wiley", collection_name=WILEY)]
        )
    )
    aclient.get_collection = AsyncMock(
        return_value=SimpleNamespace(payload_schema={})
    )
    aclient.query_points = AsyncMock(side_effect=_query_points)
    manager.aclient = aclient
    manager.generate_query_vector = AsyncMock(return_value=([0.1, 0.2], None))
    return manager


async def test_event_loop_keeps_ticking_during_a_slow_search():
    manager = _manager(search_delay_s=0.5)
    ticks = 0
    stop = asyncio.Event()

    async def _ticker():
        nonlocal ticks
        while not stop.is_set():
            await asyncio.sleep(0.01)
            ticks += 1

    ticker = asyncio.create_task(_ticker())
    results, _ = await manager.retrieve_documents_with_latencies(
        collection_names=[WILEY], query="sar", k=3, score_threshold=0.0
    )
    stop.set()
    await ticker

    assert len(results) == 1
    # 0.5 s of search at one tick per 10 ms: a blocked loop would tick ~once.
    assert ticks >= 25


async def test_concurrent_searches_are_not_capped_by_a_thread_pool():
    manager = _manager(search_delay_s=0.5)
    started = time.monotonic()
    outcomes = await asyncio.gather(
        *(
            manager.retrieve_documents_from_query(
                collection_names=[WILEY], query=f"q{i}", k=3, score_threshold=0.0
            )
            for i in range(50)
        )
    )
    elapsed = time.monotonic() - started

    assert all(len(results) == 1 for results in outcomes)
    # Fifty searches of 0.5 s each overlap; a pool of about six would need ~4 s.
    assert elapsed < 2.0
    # The alias lookup is shared: one Qdrant call for all fifty turns.
    assert manager.aclient.get_aliases.await_count == 1


async def test_alias_cache_serves_repeat_calls_then_expires(monkeypatch):
    monkeypatch.setattr(vsm, "QDRANT_READ_CACHE_TTL_S", 0.2)
    manager = _manager()

    first, _ = await manager.list_public_collections()
    second, _ = await manager.list_public_collections()
    assert first == second
    assert manager.aclient.get_aliases.await_count == 1

    await asyncio.sleep(0.3)
    await manager.list_public_collections()
    assert manager.aclient.get_aliases.await_count == 2


async def test_collection_write_invalidates_the_cache():
    manager = _manager()
    await manager.list_public_collections()
    assert manager.aclient.get_aliases.await_count == 1

    manager.embeddings_size = 2560
    manager.client.collection_exists.return_value = False
    manager.client.create_collection.return_value = True
    assert manager.create_collection("new-public-collection") is True

    await manager.list_public_collections()
    assert manager.aclient.get_aliases.await_count == 2


async def test_alias_failure_is_not_cached(caplog):
    manager = _manager()
    ok = manager.aclient.get_aliases.return_value
    manager.aclient.get_aliases.side_effect = [RuntimeError("qdrant down"), ok]

    with caplog.at_level(logging.WARNING, logger=vsm.__name__):
        await manager.list_public_collections()
    assert "Failed to load Qdrant aliases for public collections" in caplog.text

    await manager.list_public_collections()
    assert manager.aclient.get_aliases.await_count == 2


async def test_env_payload_lookup_is_shared_across_managers():
    first = _manager()
    await first._search_across_collections(
        collection_names=["wikipedia-512"],
        query_vector=[0.1],
        score_threshold=0.0,
        query_filter=None,
        limit_per_collection=3,
    )
    second = _manager()
    await second._search_across_collections(
        collection_names=["wikipedia-512"],
        query_vector=[0.1],
        score_threshold=0.0,
        query_filter=None,
        limit_per_collection=3,
    )
    first.aclient.get_collection.assert_awaited_once()
    second.aclient.get_collection.assert_not_awaited()


@pytest.fixture
def hanging_qdrant():
    """A TCP server that accepts and never answers: every read times out."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(64)
    held = []

    def _accept():
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return
            held.append(conn)

    threading.Thread(target=_accept, daemon=True).start()
    yield f"http://127.0.0.1:{server.getsockname()[1]}"
    server.close()
    for conn in held:
        conn.close()


async def test_search_timeout_raises_what_the_sync_client_raised(
    monkeypatch, hanging_qdrant, caplog
):
    monkeypatch.setattr(vsm, "QDRANT_READ_TIMEOUT_S", 1)
    manager = VectorStoreManager.__new__(VectorStoreManager)
    manager.qdrant_url = hanging_qdrant
    manager.qdrant_api_key = ""
    manager._env_payload_cache = {}
    try:
        aclient, _ = await manager._read_handle()
        sync_client = QdrantClient(hanging_qdrant, timeout=1, check_compatibility=False)

        async def _async_search():
            await aclient.query_points(collection_name=WILEY, query=[0.1], timeout=1)

        def _sync_search():
            sync_client.query_points(collection_name=WILEY, query=[0.1], timeout=1)

        async_error, sync_error = await asyncio.gather(
            _async_search(), asyncio.to_thread(_sync_search), return_exceptions=True
        )
        assert type(async_error) is type(sync_error) is ResponseHandlingException
        assert _is_timeout_error(async_error) and _is_timeout_error(sync_error)

        # The search path skips the collection, as before, and says why.
        started = time.monotonic()
        with caplog.at_level(logging.WARNING, logger=vsm.__name__):
            results = await manager._search_across_collections(
                collection_names=[WILEY],
                query_vector=[0.1],
                score_threshold=0.0,
                query_filter=None,
                limit_per_collection=3,
            )
        assert results == []
        assert f"qdrant.timeout collection={WILEY}" in caplog.text
        # Alias lookup plus search, each bounded by the read timeout.
        assert time.monotonic() - started < 5
    finally:
        await aclose_qdrant_read_clients()


async def test_reads_never_exceed_the_slot_count(monkeypatch):
    monkeypatch.setattr(vsm, "QDRANT_READ_CONCURRENCY", 4)
    manager = _manager()
    in_flight = peak = 0

    async def _query_points(**kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        return SimpleNamespace(points=[])

    manager.aclient.query_points.side_effect = _query_points
    await asyncio.gather(
        *(
            manager.retrieve_documents_from_query(
                collection_names=[WILEY], query=f"q{i}", k=3, score_threshold=0.0
            )
            for i in range(20)
        )
    )
    assert peak == 4


async def test_waiting_for_a_slot_is_bounded_by_the_read_timeout(monkeypatch, caplog):
    monkeypatch.setattr(vsm, "QDRANT_READ_CONCURRENCY", 1)
    monkeypatch.setattr(vsm, "QDRANT_READ_TIMEOUT_S", 0.2)
    manager = _manager(search_delay_s=1.0)
    with caplog.at_level(logging.WARNING, logger=vsm.__name__):
        first, second = await asyncio.gather(
            manager.retrieve_documents_from_query(
                collection_names=[WILEY], query="a", k=3, score_threshold=0.0
            ),
            manager.retrieve_documents_from_query(
                collection_names=[WILEY], query="b", k=3, score_threshold=0.0
            ),
        )
    assert sorted([len(first), len(second)]) == [0, 1]
    assert f"qdrant.timeout collection={WILEY}" in caplog.text


async def test_waiter_after_an_invalidation_does_not_get_the_old_value():
    manager = _manager()
    old = SimpleNamespace(aliases=[SimpleNamespace(alias_name="Old", collection_name=WILEY)])
    new = SimpleNamespace(aliases=[SimpleNamespace(alias_name="New", collection_name=WILEY)])
    responses = [old, new]

    async def _get_aliases():
        response = responses.pop(0)
        if response is old:
            await asyncio.sleep(0.2)
        return response

    manager.aclient.get_aliases.side_effect = _get_aliases
    early = asyncio.create_task(manager._alias_map())
    await asyncio.sleep(0.05)  # the first fetch is in flight
    invalidate_qdrant_read_cache()
    late = await manager._alias_map()

    assert late == {WILEY: "New"}
    assert await early == {WILEY: "Old"}
    # The stale answer is not cached either.
    assert await manager._alias_map() == {WILEY: "New"}
    assert manager.aclient.get_aliases.await_count == 2


async def test_unreadable_payload_schema_skips_the_collection(caplog):
    manager = _manager()
    manager.aclient.get_collection.side_effect = RuntimeError("schema lookup failed")
    with caplog.at_level(logging.WARNING, logger=vsm.__name__):
        results = await manager._search_across_collections(
            collection_names=["wikipedia-512", WILEY],
            query_vector=[0.1],
            score_threshold=0.0,
            query_filter=None,
            limit_per_collection=3,
        )
    searched = [c.kwargs["collection_name"] for c in manager.aclient.query_points.call_args_list]
    # Never searched without the env filter; the other collection still answers.
    assert searched == [WILEY]
    assert len(results) == 1
    assert "wikipedia-512" in caplog.text and "skipping the collection" in caplog.text


async def test_collections_run_concurrently_under_one_budget(monkeypatch, caplog):
    monkeypatch.setattr(vsm, "QDRANT_RETRIEVAL_BUDGET_S", 0.5)
    manager = _manager(delays={WILEY: 0.3, "wikipedia-512": 0.3, "slow-512": 5.0})
    started = time.monotonic()
    with caplog.at_level(logging.WARNING, logger=vsm.__name__):
        results = await manager._search_across_collections(
            collection_names=[WILEY, "wikipedia-512", "slow-512"],
            query_vector=[0.1],
            score_threshold=0.0,
            query_filter=None,
            limit_per_collection=3,
        )
    elapsed = time.monotonic() - started

    # Two 0.3 s searches overlap; the slow one is cut at the budget and skipped.
    assert len(results) == 2
    assert elapsed < 1.0
    assert "qdrant.timeout collection=slow-512" in caplog.text


async def test_private_collection_existence_is_checked_async_and_cached():
    manager = _manager()
    manager.aclient.collection_exists = AsyncMock(return_value=True)
    manager._ensure_private_payload_indexes = MagicMock()
    private_id = "64b64b64b64b64b64b64b64b"
    for _ in range(3):
        await manager._search_across_collections(
            collection_names=[private_id],
            query_vector=[0.1],
            score_threshold=0.0,
            query_filter=None,
            limit_per_collection=3,
            private_collections_map={private_id: "Docs"},
            user_id="user-1",
        )
    manager.aclient.collection_exists.assert_awaited_once()
    manager.ensure_private_collection.assert_not_called()
    manager._ensure_private_payload_indexes.assert_called_once()

    manager.client.collection_exists.return_value = False
    manager.client.create_collection.return_value = True
    manager.embeddings_size = 2560
    assert manager.create_collection("another-public") is True
    await manager._search_across_collections(
        collection_names=[private_id],
        query_vector=[0.1],
        score_threshold=0.0,
        query_filter=None,
        limit_per_collection=3,
        private_collections_map={private_id: "Docs"},
        user_id="user-1",
    )
    assert manager.aclient.collection_exists.await_count == 2
