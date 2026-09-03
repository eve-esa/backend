"""Unit tests for public env filters and private tenant partitioning."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from qdrant_client.http.models import (
    FieldCondition,
    Filter,
    IsEmptyCondition,
    MatchAny,
    MatchValue,
    MinShould,
)

from src.constants import (
    PRIVATE_COLLECTION_NAME,
    PUBLIC_ENV_PROD,
    PUBLIC_ENV_STAGING,
)
from src.core.vector_store_manager import (
    VectorStoreManager,
    build_private_tenant_filter,
    build_public_env_condition,
    build_public_env_filter,
    is_eve_public_collection,
    is_wiley_public_collection,
    looks_like_mongo_id,
    merge_must_filters,
    split_public_and_private_collections,
)

pytestmark = pytest.mark.no_db


PROD_PUBLIC = "qwen-512-filtered"
STAGING_PUBLIC = "satcom-chunks-collection"
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
        [PROD_PUBLIC, PRIVATE_ID, STAGING_PUBLIC, PRIVATE_COLLECTION_NAME],
        private_collections_map={PRIVATE_ID: "My Docs"},
    )
    assert public_names == [PROD_PUBLIC, STAGING_PUBLIC]
    assert private_ids == [PRIVATE_ID]
    assert PRIVATE_COLLECTION_NAME not in public_names
    assert PRIVATE_ID not in public_names


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
    manager = VectorStoreManager.__new__(VectorStoreManager)
    client = MagicMock()
    client.get_aliases.return_value = SimpleNamespace(aliases=[])
    client.query_points.return_value = SimpleNamespace(points=[])
    env_collections = env_collections or set()

    def _get_collection(name: str):
        schema = {"env": object()} if name in env_collections else {}
        return SimpleNamespace(payload_schema=schema)

    client.get_collection.side_effect = _get_collection
    manager.client = client
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


def test_private_search_queries_each_collection_with_own_limit():
    manager = _manager_with_mock_client()
    limit = 7

    manager._search_across_collections(
        collection_names=[PRIVATE_ID, PRIVATE_ID_B],
        query_vector=[0.1, 0.2],
        score_threshold=0.5,
        query_filter=None,
        limit_per_collection=limit,
        private_collections_map={PRIVATE_ID: "Docs A", PRIVATE_ID_B: "Docs B"},
        user_id="user-1",
    )

    calls = manager.client.query_points.call_args_list
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


def test_private_search_skipped_without_user_id():
    manager = _manager_with_mock_client()

    manager._search_across_collections(
        collection_names=[PRIVATE_ID, PRIVATE_ID_B],
        query_vector=[0.1, 0.2],
        score_threshold=0.5,
        query_filter=None,
        limit_per_collection=5,
        private_collections_map={PRIVATE_ID: "Docs A", PRIVATE_ID_B: "Docs B"},
        user_id=None,
    )

    manager.client.query_points.assert_not_called()
    manager.ensure_private_collection.assert_not_called()


def test_private_search_ensures_collection_before_query():
    manager = _manager_with_mock_client()
    manager._search_across_collections(
        collection_names=[PRIVATE_ID],
        query_vector=[0.1, 0.2],
        score_threshold=0.5,
        query_filter=None,
        limit_per_collection=5,
        private_collections_map={PRIVATE_ID: "Docs A"},
        user_id="user-1",
    )
    manager.ensure_private_collection.assert_called_once()
    manager.client.query_points.assert_called_once()


def test_private_ensure_failure_keeps_public_results():
    manager = _manager_with_mock_client()
    manager.ensure_private_collection.side_effect = RuntimeError("cannot create")
    results = manager._search_across_collections(
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
        for call in manager.client.query_points.call_args_list
    ]
    assert names == ["wikipedia-512"]


def test_private_missing_collection_does_not_abort_public():
    manager = _manager_with_mock_client()

    def _query_points(*args, **kwargs):
        collection_name = kwargs.get("collection_name")
        if collection_name == PRIVATE_COLLECTION_NAME:
            raise RuntimeError(
                "Not found: Collection `private-collections` doesn't exist!"
            )
        return SimpleNamespace(points=[])

    manager.client.query_points.side_effect = _query_points
    results = manager._search_across_collections(
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
        for call in manager.client.query_points.call_args_list
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


def test_eve_client_filters_apply_to_prod_and_staging_names():
    manager = _manager_with_mock_client()
    year = FieldCondition(key="year", match=MatchValue(value=2020))
    manager._search_across_collections(
        collection_names=["qwen-512-filtered", "EVE open access", "wikipedia-512"],
        query_vector=[0.1],
        score_threshold=0.0,
        query_filter=Filter(must=[year]),
        limit_per_collection=3,
        private_collections_map={},
        user_id="user-1",
    )
    by_name = {
        call.kwargs["collection_name"]: call.kwargs["query_filter"]
        for call in manager.client.query_points.call_args_list
    }
    assert set(by_name) == {
        "qwen-512-filtered",
        "EVE open access",
        "wikipedia-512",
    }
    for name in ("qwen-512-filtered", "EVE open access"):
        filt = by_name[name]
        must_keys = [
            getattr(cond, "key", None) for cond in (filt.must or [])
        ]
        assert "year" in must_keys
    wiki_filter = by_name["wikipedia-512"]
    wiki_must_keys = [
        getattr(cond, "key", None)
        for cond in ((wiki_filter.must if wiki_filter is not None else None) or [])
    ]
    assert "year" not in wiki_must_keys


def test_missing_public_collection_raises():
    manager = _manager_with_mock_client()
    manager.client.query_points.side_effect = RuntimeError(
        "Not found: Collection `qwen-512-filtered` doesn't exist!"
    )
    with pytest.raises(RuntimeError, match="Failed to search collection"):
        manager._search_across_collections(
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
    with pytest.raises(RuntimeError, match="delete_private_docs"):
        manager.delete_docs_by_metadata_filter(
            PRIVATE_COLLECTION_NAME, {"metadata.document_id": "x"}
        )
    manager.client.delete.assert_not_called()


def test_eve_public_collection_name_helper():
    assert is_eve_public_collection("qwen-512-filtered")
    assert is_eve_public_collection("EVE open access")
    assert is_eve_public_collection("EVE open-access")
    assert not is_eve_public_collection("wikipedia-512")
    assert not is_eve_public_collection("qwen-512-filtered-prod")
    assert is_wiley_public_collection("Wiley AI Gateway")
    assert not is_wiley_public_collection("qwen-512-filtered")


def test_env_filter_applied_only_when_payload_schema_has_env():
    manager = _manager_with_mock_client(env_collections={"qwen-512-filtered"})
    manager._search_across_collections(
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
        for call in manager.client.query_points.call_args_list
    }
    assert _filter_has_env(by_name["qwen-512-filtered"])
    assert not _filter_has_env(by_name["wikipedia-512"])


def test_wiley_never_gets_env_filter():
    manager = _manager_with_mock_client(env_collections={"Wiley AI Gateway"})
    manager._search_across_collections(
        collection_names=["Wiley AI Gateway"],
        query_vector=[0.1],
        score_threshold=0.0,
        query_filter=None,
        limit_per_collection=3,
        private_collections_map={},
        user_id="user-1",
    )
    query_filter = manager.client.query_points.call_args.kwargs["query_filter"]
    assert not _filter_has_env(query_filter)
    manager.client.get_collection.assert_not_called()


def test_create_collection_does_not_recreate_existing():
    manager = _manager_with_mock_client()
    manager.embeddings_size = 2560
    manager.client.collection_exists.return_value = True
    assert manager.create_collection("wikipedia-512") is True
    manager.client.recreate_collection.assert_not_called()
    manager.client.create_collection.assert_not_called()


def test_create_collection_routes_private_to_ensure():
    manager = _manager_with_mock_client()
    manager.ensure_private_collection = MagicMock()
    assert manager.create_collection(PRIVATE_COLLECTION_NAME) is True
    manager.ensure_private_collection.assert_called_once()
    manager.client.recreate_collection.assert_not_called()
    manager.client.create_collection.assert_not_called()


def test_delete_collection_refuses_shared_private():
    manager = _manager_with_mock_client()
    with pytest.raises(RuntimeError, match="Refusing to delete"):
        manager.delete_collection(PRIVATE_COLLECTION_NAME)
    manager.client.delete_collection.assert_not_called()
