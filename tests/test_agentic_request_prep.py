"""The agentic endpoints must honour the collections selected in the UI.

Both agentic create handlers used to append every collection the user owns to
``collection_ids``, so the retrieval tool searched private collections nobody
had selected. Preparation now goes through
``apply_private_collections_to_request``, exactly like the classic endpoints.
"""

from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch

import pytest
from bson import ObjectId

from src.routers.message import _prepare_agentic_request
from src.schemas.generation_request import GenerationRequest

pytestmark = pytest.mark.no_db

_MESSAGE = "src.routers.message"
_USER_ID = "user-1"

_OWNED = [
    SimpleNamespace(id="68b0c0ffee000000000000a1", name="notes"),
    SimpleNamespace(id="68b0c0ffee000000000000a2", name="papers"),
    SimpleNamespace(id="68b0c0ffee000000000000a3", name="drafts"),
]


def _fake_user() -> Any:
    return SimpleNamespace(id=_USER_ID)


async def _find_all(filter_dict: Dict[str, Any] | None = None, **_: Any) -> List[Any]:
    """Stand-in for Collection.find_all that applies the id filter itself."""
    filter_dict = filter_dict or {}
    if filter_dict.get("user_id") != _USER_ID:
        return []
    id_filter = filter_dict.get("_id", {})
    wanted = {str(oid) for oid in id_filter.get("$in", [])}
    return [c for c in _OWNED if c.id in wanted]


async def _prepare(request: GenerationRequest) -> GenerationRequest:
    with patch(f"{_MESSAGE}.IS_PROD", False), patch(
        f"{_MESSAGE}.CollectionModel.find_all", AsyncMock(side_effect=_find_all)
    ), patch(f"{_MESSAGE}.MCPServer.find_all", AsyncMock(return_value=[])):
        return await _prepare_agentic_request(request, _fake_user())


class TestPrivateCollections:
    @pytest.mark.asyncio
    async def test_only_the_selected_private_collection_is_used(self):
        request = GenerationRequest(
            query="q", private_collections=[_OWNED[1].id]
        )

        await _prepare(request)

        assert request.collection_ids == [_OWNED[1].id]
        assert request.private_collections_map == {_OWNED[1].id: "papers"}

    @pytest.mark.asyncio
    async def test_no_selection_means_no_private_collection(self):
        request = GenerationRequest(query="q")

        await _prepare(request)

        assert request.collection_ids == []
        assert request.private_collections_map == {}

    @pytest.mark.asyncio
    async def test_unowned_private_collection_is_dropped(self):
        request = GenerationRequest(
            query="q", private_collections=[str(ObjectId())]
        )

        await _prepare(request)

        assert request.collection_ids == []
        assert request.private_collections_map == {}


class TestPublicCollections:
    @pytest.mark.asyncio
    async def test_unknown_public_collection_is_dropped(self):
        request = GenerationRequest(
            query="q",
            public_collections=["wikipedia-512", "Totally Unknown"],
        )

        await _prepare(request)

        assert request.public_collections == ["wikipedia-512"]
        assert request.collection_ids == ["wikipedia-512"]

    @pytest.mark.asyncio
    async def test_wiley_gateway_is_kept_public_but_stripped_from_collection_ids(self):
        request = GenerationRequest(
            query="q",
            public_collections=["wikipedia-512", "Wiley AI Gateway"],
        )

        await _prepare(request)

        assert request.public_collections == ["wikipedia-512", "Wiley AI Gateway"]
        assert request.collection_ids == ["wikipedia-512"]


class TestIdempotence:
    @pytest.mark.asyncio
    async def test_preparing_twice_does_not_duplicate_collection_ids(self):
        """The retry path re-prepares a request that was already prepared."""
        request = GenerationRequest(
            query="q",
            public_collections=["wikipedia-512"],
            private_collections=[_OWNED[0].id],
        )

        await _prepare(request)
        first = list(request.collection_ids)
        await _prepare(request)

        assert request.collection_ids == first
        assert sorted(request.collection_ids) == sorted(
            ["wikipedia-512", _OWNED[0].id]
        )


class TestCollectionReads:
    @pytest.mark.asyncio
    async def test_only_ids_the_secondary_missed_are_read_on_the_primary(self):
        """A collection created a moment ago may be missing on a secondary."""
        from src.routers.message import apply_private_collections_to_request

        fresh = SimpleNamespace(
            id="68b0c0ffee000000000000a4",
            name="Attached documents",
            conversation_id="c1",
        )
        calls = []

        async def find_all(filter_dict=None, primary=False, **_):
            wanted = {str(oid) for oid in filter_dict["_id"]["$in"]}
            calls.append((primary, wanted))
            # The secondary lags behind: it does not have the fresh row yet.
            rows = [*_OWNED, fresh] if primary else _OWNED
            return [c for c in rows if c.id in wanted]

        request = GenerationRequest(
            query="q", private_collections=[_OWNED[0].id, fresh.id]
        )
        with patch(f"{_MESSAGE}.CollectionModel.find_all", find_all):
            await apply_private_collections_to_request(request, _USER_ID)

        assert calls == [
            (False, {_OWNED[0].id, fresh.id}),
            (True, {fresh.id}),
        ]
        assert request.private_collections_map == {
            _OWNED[0].id: "notes",
            fresh.id: "Attached documents",
        }
        # Only the attached documents skip the year and citation filters.
        assert request.unfiltered_private_collections == [fresh.id]

    @pytest.mark.asyncio
    async def test_the_primary_is_not_read_when_the_secondary_has_every_id(self):
        from src.routers.message import apply_private_collections_to_request

        primaries = []

        async def find_all(filter_dict=None, primary=False, **_):
            primaries.append(primary)
            return await _find_all(filter_dict)

        request = GenerationRequest(query="q", private_collections=[_OWNED[0].id])
        with patch(f"{_MESSAGE}.CollectionModel.find_all", find_all):
            await apply_private_collections_to_request(request, _USER_ID)

        assert primaries == [False]
