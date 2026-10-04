"""The citation minimum applies to every public collection, never to private ones."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

from src.config import PRIVATE_COLLECTION_NAME
from tests.test_qdrant_multitenancy import (
    PRIVATE_ID,
    _manager_with_mock_client,
)

pytestmark = pytest.mark.no_db

EVE = "qwen-512-filtered"
WIKI = "wikipedia-512"
KB = "esa-rag-scraped-qwen3-newpipeline"
NO_COUNT = "wikipedia-no-count"

YEAR = FieldCondition(key="year", range=Range(gte=2015, lte=2024))
CITATIONS = FieldCondition(key="n_citations", range=Range(gte=10))
JOURNAL = FieldCondition(key="journal", match=MatchValue(value="Nature"))


def _must_keys(query_filter) -> list:
    if query_filter is None:
        return []
    return [getattr(c, "key", None) for c in (query_filter.must or [])]


def _manager(with_citations: set[str]):
    """Mock manager whose payload schema lists n_citations on ``with_citations``."""
    manager = _manager_with_mock_client()

    def _get_collection(name: str):
        schema = {"n_citations": object()} if name in with_citations else {}
        return SimpleNamespace(payload_schema=schema)

    manager.aclient.get_collection = AsyncMock(side_effect=_get_collection)
    return manager


async def _filters_by_collection(query_filter, with_citations=frozenset({WIKI, KB})):
    manager = _manager(set(with_citations))
    await manager._search_across_collections(
        collection_names=[EVE, WIKI, KB, PRIVATE_ID],
        query_vector=[0.1],
        score_threshold=0.0,
        query_filter=query_filter,
        limit_per_collection=3,
        private_collections_map={PRIVATE_ID: "My docs"},
        user_id="user-1",
    )
    calls = manager.aclient.query_points.call_args_list
    public = {
        c.kwargs["collection_name"]: c.kwargs["query_filter"]
        for c in calls
        if c.kwargs["collection_name"] != PRIVATE_COLLECTION_NAME
    }
    private = [
        c.kwargs["query_filter"]
        for c in calls
        if c.kwargs["collection_name"] == PRIVATE_COLLECTION_NAME
    ]
    return public, private


async def test_citation_minimum_reaches_every_public_collection():
    public, private = await _filters_by_collection(
        Filter(must=[YEAR, CITATIONS, JOURNAL])
    )
    assert set(public) == {EVE, WIKI, KB}
    assert set(_must_keys(public[EVE])) == {"year", "n_citations", "journal"}
    for name in (WIKI, KB):
        assert sorted(_must_keys(public[name])) == ["n_citations", "year"]
        cond = next(c for c in public[name].must if c.key == "n_citations")
        assert cond.range.gte == 10
    assert len(private) == 1
    assert "n_citations" not in _must_keys(private[0])
    assert "year" in _must_keys(private[0])


async def test_collection_without_citation_index_is_skipped_not_scanned():
    public, private = await _filters_by_collection(
        Filter(must=[YEAR, CITATIONS]), with_citations={KB}
    )
    # wikipedia-512 lists no n_citations index: no point can pass the range,
    # so it is never queried. EVE keeps its own filter and is always queried.
    assert set(public) == {EVE, KB}
    assert len(private) == 1


async def test_without_citation_minimum_nothing_changes():
    # Regression guard: passes on main by design. No schema lists n_citations,
    # and without a minimum no collection is skipped.
    public, private = await _filters_by_collection(
        Filter(must=[YEAR, JOURNAL]), with_citations=frozenset()
    )
    assert set(public) == {EVE, WIKI, KB}
    assert set(_must_keys(public[EVE])) == {"year", "journal"}
    assert _must_keys(public[WIKI]) == ["year"]
    assert _must_keys(public[KB]) == ["year"]
    assert sorted(_must_keys(private[0])) == ["collection_id", "user_id", "year"]

    public, private = await _filters_by_collection(None, with_citations=frozenset())
    assert public[WIKI] is None
    assert public[KB] is None
    assert sorted(_must_keys(private[0])) == ["collection_id", "user_id"]


def test_qdrant_range_excludes_points_without_the_field():
    """Qdrant semantics the fix relies on: a missing field fails a ``must`` range.

    A string count fails it too: EVE open access stores n_citations as a string
    today, so its documents cannot pass a minimum until ingestion casts it.
    """
    client = QdrantClient(":memory:")
    client.create_collection(
        WIKI, vectors_config=VectorParams(size=2, distance=Distance.COSINE)
    )
    client.upsert(
        WIKI,
        points=[
            PointStruct(id=1, vector=[1.0, 0.0], payload={"year": 2020}),
            PointStruct(id=2, vector=[1.0, 0.0], payload={"year": 2020, "n_citations": 50}),
            PointStruct(id=3, vector=[1.0, 0.0], payload={"year": 2020, "n_citations": 3}),
            PointStruct(id=4, vector=[1.0, 0.0], payload={"year": 2020, "n_citations": "50"}),
        ],
    )
    hits = client.query_points(
        WIKI, query=[1.0, 0.0], query_filter=Filter(must=[YEAR, CITATIONS]), limit=10
    ).points
    assert [p.id for p in hits] == [2]
    unfiltered = client.query_points(
        WIKI, query=[1.0, 0.0], query_filter=Filter(must=[YEAR]), limit=10
    ).points
    assert sorted(p.id for p in unfiltered) == [1, 2, 3, 4]
