"""Unit tests for public env filters and private tenant partitioning."""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client.http.models import (
    FieldCondition,
    Filter,
    IsEmptyCondition,
    MatchAny,
    MatchValue,
    KeywordIndexParams,
    KeywordIndexType,
    MinShould,
    Range,
    UpdateResult,
    UpdateStatus,
)

from src.config import PRIVATE_COLLECTION_NAME
from src.constants import (
    ALL_PRIVATE_COLLECTION_NAMES,
    PUBLIC_ENV_PROD,
    PUBLIC_ENV_STAGING,
)
import src.core.vector_store_manager as vsm
from src.core.vector_store_manager import (
    VectorStoreManager,
    build_private_tenant_filter,
    build_public_env_condition,
    build_public_env_filter,
    is_eve_public_collection,
    is_private_qdrant_collection,
    invalidate_qdrant_read_cache,
    is_wiley_public_collection,
    looks_like_mongo_id,
    merge_must_filters,
    split_public_and_private_collections,
)

pytestmark = pytest.mark.no_db


PROD_PUBLIC = "qwen-512-filtered"
WIKI_PUBLIC = "wikipedia-512"
PRIVATE_ID = "64b64b64b64b64b64b64b64b"


def test_prod_env_filter_matches_prod_only():
    condition = build_public_env_condition(is_prod=True)
    assert condition.key == "env"
    assert isinstance(condition.match, MatchValue)
    assert condition.match.value == PUBLIC_ENV_PROD


def test_staging_env_filter_matches_prod_and_staging():
    condition = build_public_env_condition(is_prod=False)
    assert condition.key == "env"
    assert isinstance(condition.match, MatchAny)
    assert set(condition.match.any) == {PUBLIC_ENV_PROD, PUBLIC_ENV_STAGING}


def test_private_tenant_filter_requires_user_and_collection():
    filt = build_private_tenant_filter("user-1", [PRIVATE_ID])
    keys = [cond.key for cond in filt.must]
    assert keys == ["user_id", "collection_id"]
    assert filt.must[0].match.value == "user-1"
    assert filt.must[1].match.value == PRIVATE_ID


def test_private_tenant_filter_match_any_for_multiple_collections():
    filt = build_private_tenant_filter("user-1", [PRIVATE_ID, "aaaaaaaaaaaaaaaaaaaaaaaa"])
    collection_cond = filt.must[1]
    assert collection_cond.key == "collection_id"
    assert isinstance(collection_cond.match, MatchAny)
    assert PRIVATE_ID in collection_cond.match.any


def test_split_keeps_public_names_and_private_mongo_ids():
    public_names, private_ids = split_public_and_private_collections(
        [PROD_PUBLIC, PRIVATE_ID, WIKI_PUBLIC, PRIVATE_COLLECTION_NAME],
        private_collections_map={PRIVATE_ID: "My Docs"},
    )
    assert public_names == [PROD_PUBLIC, WIKI_PUBLIC]
    assert private_ids == [PRIVATE_ID]
    assert PRIVATE_COLLECTION_NAME not in public_names
    assert PRIVATE_ID not in public_names


def test_split_skips_all_private_qdrant_collection_names():
    names = [PROD_PUBLIC, *sorted(ALL_PRIVATE_COLLECTION_NAMES), PRIVATE_ID]
    public_names, private_ids = split_public_and_private_collections(
        names,
        private_collections_map={PRIVATE_ID: "My Docs"},
    )
    assert public_names == [PROD_PUBLIC]
    assert private_ids == [PRIVATE_ID]
    for reserved in ALL_PRIVATE_COLLECTION_NAMES:
        assert reserved not in public_names
        assert reserved not in private_ids


def test_split_treats_map_keys_as_private_even_if_not_object_id():
    public_names, private_ids = split_public_and_private_collections(
        ["not-an-object-id", PROD_PUBLIC],
        private_collections_map={"not-an-object-id": "Legacy"},
    )
    assert public_names == [PROD_PUBLIC]
    assert private_ids == ["not-an-object-id"]


def test_mongo_id_heuristic():
    assert looks_like_mongo_id(PRIVATE_ID)
    assert not looks_like_mongo_id(PROD_PUBLIC)
    assert not looks_like_mongo_id(PRIVATE_COLLECTION_NAME)


PRIVATE_ID_B = "aaaaaaaaaaaaaaaaaaaaaaaa"


def _manager_with_mock_client(env_collections: set[str] | None = None) -> VectorStoreManager:
    # Aliases and payload schemas are cached per process: start each test cold.
    invalidate_qdrant_read_cache()
    manager = VectorStoreManager.__new__(VectorStoreManager)
    # Writes go through the sync client, reads through the async one.
    manager.client = MagicMock()
    aclient = MagicMock()
    aclient.get_aliases = AsyncMock(return_value=SimpleNamespace(aliases=[]))
    aclient.query_points = AsyncMock(return_value=SimpleNamespace(points=[]))
    # Missing, so the search path runs the (mocked) sync ensure.
    aclient.collection_exists = AsyncMock(return_value=False)
    env_collections = env_collections or set()

    def _get_collection(name: str):
        schema = {"env": object()} if name in env_collections else {}
        return SimpleNamespace(payload_schema=schema)

    aclient.get_collection = AsyncMock(side_effect=_get_collection)
    manager.aclient = aclient
    manager._env_payload_cache = {}
    manager.ensure_private_collection = MagicMock()
    return manager


def _filter_has_env(query_filter) -> bool:
    if query_filter is None:
        return False
    for cond in list(query_filter.must or []):
        if getattr(cond, "key", None) == "env":
            return True
        for inner in list(getattr(cond, "should", None) or []):
            if getattr(inner, "key", None) == "env":
                return True
            if isinstance(inner, IsEmptyCondition):
                field = getattr(inner, "is_empty", None)
                if getattr(field, "key", None) == "env":
                    return True
    return False


def _private_collection_id_from_filter(query_filter) -> str:
    collection_cond = next(c for c in query_filter.must if c.key == "collection_id")
    assert isinstance(collection_cond.match, MatchValue)
    return collection_cond.match.value


async def test_private_search_queries_each_collection_with_own_limit():
    manager = _manager_with_mock_client()
    limit = 7

    await manager._search_across_collections(
        collection_names=[PRIVATE_ID, PRIVATE_ID_B],
        query_vector=[0.1, 0.2],
        score_threshold=0.5,
        query_filter=None,
        limit_per_collection=limit,
        private_collections_map={PRIVATE_ID: "Docs A", PRIVATE_ID_B: "Docs B"},
        user_id="user-1",
    )

    calls = manager.aclient.query_points.call_args_list
    assert len(calls) == 2
    seen_ids = []
    for call in calls:
        kwargs = call.kwargs
        assert kwargs["collection_name"] == PRIVATE_COLLECTION_NAME
        assert kwargs["limit"] == limit
        filt = kwargs["query_filter"]
        keys = [cond.key for cond in filt.must]
        assert keys == ["user_id", "collection_id"]
        assert filt.must[0].match.value == "user-1"
        seen_ids.append(_private_collection_id_from_filter(filt))
    assert seen_ids == [PRIVATE_ID, PRIVATE_ID_B]


async def test_private_search_skipped_without_user_id():
    manager = _manager_with_mock_client()

    await manager._search_across_collections(
        collection_names=[PRIVATE_ID, PRIVATE_ID_B],
        query_vector=[0.1, 0.2],
        score_threshold=0.5,
        query_filter=None,
        limit_per_collection=5,
        private_collections_map={PRIVATE_ID: "Docs A", PRIVATE_ID_B: "Docs B"},
        user_id=None,
    )

    manager.aclient.query_points.assert_not_called()
    manager.ensure_private_collection.assert_not_called()


async def test_private_search_ensures_collection_before_query():
    manager = _manager_with_mock_client()
    await manager._search_across_collections(
        collection_names=[PRIVATE_ID],
        query_vector=[0.1, 0.2],
        score_threshold=0.5,
        query_filter=None,
        limit_per_collection=5,
        private_collections_map={PRIVATE_ID: "Docs A"},
        user_id="user-1",
    )
    manager.ensure_private_collection.assert_called_once()
    manager.aclient.query_points.assert_called_once()


async def test_private_ensure_failure_keeps_public_results():
    manager = _manager_with_mock_client()
    manager.ensure_private_collection.side_effect = RuntimeError("cannot create")
    results = await manager._search_across_collections(
        collection_names=["wikipedia-512", PRIVATE_ID],
        query_vector=[0.1],
        score_threshold=0.0,
        query_filter=None,
        limit_per_collection=3,
        private_collections_map={PRIVATE_ID: "Docs A"},
        user_id="user-1",
    )
    assert results == []
    names = [
        call.kwargs["collection_name"]
        for call in manager.aclient.query_points.call_args_list
    ]
    assert names == ["wikipedia-512"]


async def test_private_missing_collection_does_not_abort_public():
    manager = _manager_with_mock_client()

    def _query_points(*args, **kwargs):
        collection_name = kwargs.get("collection_name")
        if collection_name == PRIVATE_COLLECTION_NAME:
            raise RuntimeError(
                f"Not found: Collection `{PRIVATE_COLLECTION_NAME}` doesn't exist!"
            )
        return SimpleNamespace(points=[])

    manager.aclient.query_points.side_effect = _query_points
    results = await manager._search_across_collections(
        collection_names=["wikipedia-512", PRIVATE_ID],
        query_vector=[0.1],
        score_threshold=0.0,
        query_filter=None,
        limit_per_collection=3,
        private_collections_map={PRIVATE_ID: "Docs A"},
        user_id="user-1",
    )
    assert results == []
    names = [
        call.kwargs["collection_name"]
        for call in manager.aclient.query_points.call_args_list
    ]
    assert "wikipedia-512" in names


def test_public_env_filter_allows_untagged_points():
    filt = build_public_env_filter(is_prod=True)
    assert filt.should is not None
    assert len(filt.should) == 2
    assert isinstance(filt.should[1], IsEmptyCondition)


def test_merge_must_filters_preserves_min_should():
    year = FieldCondition(key="year", match=MatchValue(value=2020))
    min_should = MinShould(conditions=[year], min_count=1)
    base = Filter(should=[year], min_should=min_should)
    extra = FieldCondition(key="env", match=MatchValue(value="prod"))
    merged = merge_must_filters(base, [extra])
    assert merged.min_should == min_should
    assert merged.should == [year]
    assert extra in merged.must


async def test_client_filters_apply_to_every_public_collection():
    manager = _manager_with_mock_client()
    year = FieldCondition(key="year", match=MatchValue(value=2020))
    journal = FieldCondition(key="journal", match=MatchValue(value="Nature"))
    citations = FieldCondition(key="n_citations", range=Range(gte=10))
    await manager._search_across_collections(
        collection_names=["qwen-512-filtered", "wikipedia-512"],
        query_vector=[0.1],
        score_threshold=0.0,
        query_filter=Filter(must=[year, journal, citations]),
        limit_per_collection=3,
        private_collections_map={},
        user_id="user-1",
    )
    by_name = {
        call.kwargs["collection_name"]: call.kwargs["query_filter"]
        for call in manager.aclient.query_points.call_args_list
    }
    assert set(by_name) == {
        "qwen-512-filtered",
        "wikipedia-512",
    }
    for name in ("qwen-512-filtered", "wikipedia-512"):
        must_keys = [
            getattr(cond, "key", None) for cond in (by_name[name].must or [])
        ]
        assert set(must_keys) == {"year", "journal", "n_citations"}


async def test_client_filters_apply_to_private_collections():
    manager = _manager_with_mock_client()
    year = FieldCondition(key="year", range=Range(gte=2015, lte=2024))
    citations = FieldCondition(key="n_citations", range=Range(gte=10))
    await manager._search_across_collections(
        collection_names=[PRIVATE_ID],
        query_vector=[0.1],
        score_threshold=0.0,
        query_filter=Filter(must=[year, citations]),
        limit_per_collection=3,
        private_collections_map={PRIVATE_ID: "Docs A"},
        user_id="user-1",
    )
    filt = manager.aclient.query_points.call_args.kwargs["query_filter"]
    must_keys = [cond.key for cond in filt.must]
    assert must_keys == ["user_id", "collection_id", "year", "n_citations"]


async def test_missing_public_collection_raises():
    manager = _manager_with_mock_client()
    manager.aclient.query_points.side_effect = RuntimeError(
        "Not found: Collection `qwen-512-filtered` doesn't exist!"
    )
    with pytest.raises(RuntimeError, match="Failed to search collection"):
        await manager._search_across_collections(
            collection_names=["qwen-512-filtered"],
            query_vector=[0.1],
            score_threshold=0.0,
            query_filter=None,
            limit_per_collection=3,
            private_collections_map={},
            user_id="user-1",
        )


def test_unscoped_delete_on_private_collection_refused():
    manager = _manager_with_mock_client()
    for reserved in sorted(ALL_PRIVATE_COLLECTION_NAMES):
        with pytest.raises(RuntimeError, match="delete_private_docs"):
            manager.delete_docs_by_metadata_filter(
                reserved, {"metadata.document_id": "x"}
            )
    manager.client.delete.assert_not_called()


def test_eve_public_collection_name_helper():
    assert is_eve_public_collection("qwen-512-filtered")
    assert not is_eve_public_collection("EVE open access")
    assert not is_eve_public_collection("EVE open-access")
    assert not is_eve_public_collection("wikipedia-512")
    assert not is_eve_public_collection("qwen-512-filtered-prod")
    assert is_wiley_public_collection("Wiley AI Gateway")
    assert not is_wiley_public_collection("qwen-512-filtered")


async def test_env_filter_applied_only_when_payload_schema_has_env():
    manager = _manager_with_mock_client(env_collections={"qwen-512-filtered"})
    await manager._search_across_collections(
        collection_names=["qwen-512-filtered", "wikipedia-512"],
        query_vector=[0.1],
        score_threshold=0.0,
        query_filter=None,
        limit_per_collection=3,
        private_collections_map={},
        user_id="user-1",
    )
    by_name = {
        call.kwargs["collection_name"]: call.kwargs["query_filter"]
        for call in manager.aclient.query_points.call_args_list
    }
    assert _filter_has_env(by_name["qwen-512-filtered"])
    assert not _filter_has_env(by_name["wikipedia-512"])


async def test_wiley_never_gets_env_filter():
    manager = _manager_with_mock_client(env_collections={"Wiley AI Gateway"})
    await manager._search_across_collections(
        collection_names=["Wiley AI Gateway"],
        query_vector=[0.1],
        score_threshold=0.0,
        query_filter=None,
        limit_per_collection=3,
        private_collections_map={},
        user_id="user-1",
    )
    query_filter = manager.aclient.query_points.call_args.kwargs["query_filter"]
    assert not _filter_has_env(query_filter)
    manager.aclient.get_collection.assert_not_called()


def test_create_collection_does_not_recreate_existing():
    manager = _manager_with_mock_client()
    manager.embeddings_size = 2560
    manager.client.collection_exists.return_value = True
    assert manager.create_collection("wikipedia-512") is True
    manager.client.recreate_collection.assert_not_called()
    manager.client.create_collection.assert_not_called()


def test_create_collection_routes_private_to_ensure():
    manager = _manager_with_mock_client()
    for reserved in sorted(ALL_PRIVATE_COLLECTION_NAMES):
        manager.ensure_private_collection = MagicMock()
        assert manager.create_collection(reserved) is True
        manager.ensure_private_collection.assert_called_once()
    manager.client.recreate_collection.assert_not_called()
    manager.client.create_collection.assert_not_called()


def test_delete_collection_refuses_shared_private():
    manager = _manager_with_mock_client()
    for reserved in sorted(ALL_PRIVATE_COLLECTION_NAMES):
        with pytest.raises(RuntimeError, match="Refusing to delete"):
            manager.delete_collection(reserved)
    manager.client.delete_collection.assert_not_called()


def test_is_private_qdrant_collection_helper():
    for name in ALL_PRIVATE_COLLECTION_NAMES:
        assert is_private_qdrant_collection(name)
    assert not is_private_qdrant_collection(PROD_PUBLIC)
    assert not is_private_qdrant_collection("")


async def test_year_filter_reaches_private_collections():
    manager = _manager_with_mock_client()
    year = FieldCondition(key="year", match=MatchValue(value=2020))
    await manager._search_across_collections(
        collection_names=["wikipedia-512", PRIVATE_ID],
        query_vector=[0.1],
        score_threshold=0.0,
        query_filter=Filter(must=[year]),
        limit_per_collection=3,
        private_collections_map={PRIVATE_ID: "Docs A"},
        user_id="user-1",
    )
    by_name = {
        call.kwargs["collection_name"]: call.kwargs["query_filter"]
        for call in manager.aclient.query_points.call_args_list
    }
    private_keys = [cond.key for cond in by_name[PRIVATE_COLLECTION_NAME].must]
    assert private_keys == ["user_id", "collection_id", "year"]
    assert [cond.key for cond in by_name["wikipedia-512"].must] == ["year"]


def _index_calls(manager) -> dict:
    return {
        call.kwargs["field_name"]: call.kwargs
        for call in manager.client.create_payload_index.call_args_list
    }


def test_private_indexes_include_document_id_keyword():
    # Strict mode refuses a delete filtered on an unindexed metadata.document_id.
    manager = _manager_with_mock_client()
    manager._ensure_private_payload_indexes()
    calls = _index_calls(manager)
    assert set(calls) == {"user_id", "collection_id", "metadata.document_id"}
    doc_call = calls["metadata.document_id"]
    assert doc_call["collection_name"] == PRIVATE_COLLECTION_NAME
    assert doc_call["timeout"] == 10
    schema = doc_call["field_schema"]
    assert isinstance(schema, KeywordIndexParams)
    assert schema.type == KeywordIndexType.KEYWORD
    # The private collection has m=0: no payload HNSW links for this field.
    assert schema.enable_hnsw is False


def test_existing_document_id_index_is_a_success(caplog):
    # Qdrant answers 200 to a second create with the same schema.
    manager = _manager_with_mock_client()
    manager.client.create_payload_index.return_value = UpdateResult(
        operation_id=1, status=UpdateStatus.COMPLETED
    )
    with caplog.at_level(logging.WARNING, logger=vsm.__name__):
        assert manager._ensure_private_document_id_index() is True
        invalidate_qdrant_read_cache()
        assert manager._ensure_private_document_id_index() is True
    assert manager.client.create_payload_index.call_count == 2
    assert "payload_index_failed" not in caplog.text


def test_document_id_index_failure_is_logged_not_raised(caplog):
    manager = _manager_with_mock_client()

    def _create(**kwargs):
        if kwargs["field_name"] == "metadata.document_id":
            raise TimeoutError("qdrant down")

    manager.client.create_payload_index.side_effect = _create
    with caplog.at_level(logging.WARNING, logger=vsm.__name__):
        manager._ensure_private_payload_indexes()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "field=metadata.document_id type=TimeoutError" in warnings[0].getMessage()


def test_document_id_index_is_ensured_once_per_process():
    manager = _manager_with_mock_client()
    assert manager._ensure_private_document_id_index() is True
    assert manager._ensure_private_document_id_index() is True
    manager.client.create_payload_index.assert_called_once()


def test_document_id_index_failure_is_not_retried_for_a_minute(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(vsm.time, "monotonic", lambda: now[0])
    manager = _manager_with_mock_client()
    manager.client.create_payload_index.side_effect = [TimeoutError(), None]
    assert manager._ensure_private_document_id_index() is False
    now[0] += 59
    assert manager._ensure_private_document_id_index() is False
    assert manager.client.create_payload_index.call_count == 1
    now[0] += 2
    assert manager._ensure_private_document_id_index() is True
    assert manager.client.create_payload_index.call_count == 2


def test_document_delete_ensures_the_index_first():
    manager = _manager_with_mock_client()
    manager.client.count.return_value = SimpleNamespace(count=0)
    manager.delete_private_docs(
        user_id="user-1",
        collection_id=PRIVATE_ID,
        metadata={"metadata.document_id": "doc-1"},
    )
    names = [name for name, _, _ in manager.client.mock_calls]
    assert names.index("create_payload_index") < names.index("delete")
    assert "metadata.document_id" in _index_calls(manager)


def test_document_delete_runs_when_the_index_cannot_be_ensured():
    manager = _manager_with_mock_client()
    manager.client.create_payload_index.side_effect = TimeoutError()
    manager.client.count.side_effect = [
        SimpleNamespace(count=1),
        SimpleNamespace(count=0),
    ]
    result = manager.delete_private_docs(
        user_id="user-1",
        collection_id=PRIVATE_ID,
        metadata={"metadata.document_id": "doc-1"},
    )
    assert result.deleted == 1
    manager.client.delete.assert_called_once()


def test_collection_delete_skips_the_document_index():
    manager = _manager_with_mock_client()
    manager.client.count.return_value = SimpleNamespace(count=0)
    manager.delete_points_for_collection("user-1", PRIVATE_ID)
    manager.client.create_payload_index.assert_not_called()
