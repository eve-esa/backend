"""ATTACHED-DOCS-RETRIEVAL (2026-11-27): a conversation's attached documents are searched with
no score threshold, and no chunk is only a page number or a paper id.

Revert together with the ATTACHED-DOCS-RETRIEVAL lines in src/core/vector_store_manager.py and
src/services/docling_chunks.py.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.vector_store_manager import VectorStoreManager, invalidate_qdrant_read_cache
from src.services.docling_chunks import (
    MIN_CHUNK_WORDS,
    Chunk,
    _merge_tiny_chunks,
    chunks_from_docling,
)

pytestmark = pytest.mark.no_db

LONG = " ".join(["Landsat"] * 30)


# --- chunking ---------------------------------------------------------------------------------


def test_a_paper_id_alone_under_a_heading_joins_the_next_chunk():
    chunks = [
        Chunk(text=LONG, section="Intro", pages=[1]),
        Chunk(text="Paper L 4\nE73-10202", section="ABSTRACT", pages=[5]),
        Chunk(text="Land use mapping of the Central Valley " + LONG, section="LAND USE", pages=[5]),
    ]
    out = _merge_tiny_chunks(chunks)
    assert len(out) == 2
    assert out[1].text.startswith("Paper L 4\nE73-10202\nLand use mapping")
    assert out[1].section == "LAND USE" and out[1].pages == [5]


def test_a_tiny_last_chunk_joins_the_one_before():
    out = _merge_tiny_chunks([Chunk(text=LONG, section="A", pages=[1]), Chunk(text="p. 9", section="B", pages=[2])])
    assert len(out) == 1 and out[0].text.endswith("\np. 9") and out[0].pages == [1, 2]


def test_tables_are_never_merged():
    table = Chunk(text="| a |\n|---|\n| 1 |", section="T", pages=[3], element_type="table")
    out = _merge_tiny_chunks([table, Chunk(text=LONG, section="T", pages=[3])])
    assert out[0] is table and len(out) == 2


def test_no_parsed_chunk_is_under_the_minimum():
    parsed = json.loads((Path(__file__).parent / "fixtures" / "docling_document.json").read_text())
    for chunk in chunks_from_docling(parsed):
        assert chunk.element_type == "table" or len(chunk.text.split()) >= MIN_CHUNK_WORDS


# --- retrieval ----------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _cold_cache():
    invalidate_qdrant_read_cache()
    yield
    invalidate_qdrant_read_cache()


def _manager(calls: list) -> VectorStoreManager:
    manager = VectorStoreManager.__new__(VectorStoreManager)
    manager.client = MagicMock()
    manager.qdrant_url = "http://qdrant.test:6333"
    manager.embeddings_model = "test-model"
    manager._env_payload_cache = {}
    manager.ensure_private_collection = MagicMock()

    async def query_points(**kwargs):
        calls.append(kwargs)
        point = SimpleNamespace(id="p1", score=0.5, payload={"text": "chunk", "collection_id": "x"})
        return SimpleNamespace(points=[point])

    aclient = MagicMock()
    aclient.get_aliases = AsyncMock(return_value=SimpleNamespace(aliases=[]))
    aclient.get_collection = AsyncMock(return_value=SimpleNamespace(payload_schema={}))
    aclient.query_points = AsyncMock(side_effect=query_points)
    manager.aclient = aclient
    manager._ensure_private_collection_ready = AsyncMock()
    return manager


async def test_attached_documents_are_searched_without_the_threshold():
    calls: list = []
    manager = _manager(calls)
    attached, other = "6b090b07c755a9689cd90846", "6b090b07c755a9689cd90999"
    await manager._search_across_collections(
        collection_names=[attached, other],
        query_vector=[0.1],
        score_threshold=0.6,
        query_filter=None,
        limit_per_collection=5,
        private_collections_map={attached: "Attached documents", other: "My collection"},
        user_id="u1",
        unfiltered_private_ids=[attached],
    )
    by_collection = {
        json.dumps(c["query_filter"].model_dump(), default=str).count(attached) > 0: c["score_threshold"]
        for c in calls
    }
    assert by_collection[True] is None  # the conversation's own documents
    assert by_collection[False] == 0.6  # every other collection keeps the UI threshold
