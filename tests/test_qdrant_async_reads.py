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


def _manager(search_delay_s: float = 0.0) -> VectorStoreManager:
    manager = VectorStoreManager.__new__(VectorStoreManager)
    manager.client = MagicMock()
    manager.qdrant_url = "http://qdrant.test:6333"
    manager.embeddings_model = "test-model"
    manager._env_payload_cache = {}
    manager.ensure_private_collection = MagicMock()

    async def _query_points(**kwargs):
        await asyncio.sleep(search_delay_s)
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
        aclient = await manager._read_client()
        sync_client = QdrantClient(hanging_qdrant, timeout=1, check_compatibility=False)

        async def _async_search():
            await aclient.query_points(collection_name=WILEY, query=[0.1], timeout=1)

        def _sync_search():
            sync_client.query_points(collection_name=WILEY, query=[0.1], timeout=1)

        async_error, sync_error = await asyncio.gather(
            _async_search(), asyncio.to_thread(_sync_search), return_exceptions=True
        )
        assert type(async_error) is type(sync_error) is ResponseHandlingException

        # The search path handles it as before: logged, collection skipped.
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
        assert f"Failed to search collection '{WILEY}'" in caplog.text
        # Alias lookup plus search, each bounded by the read timeout.
        assert time.monotonic() - started < 5
    finally:
        await aclose_qdrant_read_clients()
