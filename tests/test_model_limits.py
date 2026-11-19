"""Context-window lookup: payload fields, catalog, store, and the miss cache."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.core import model_limits
from src.database.models.catalog_provider import CatalogProviderModelEntry
from src.database.models.model_context_window import ModelContextWindow

pytestmark = pytest.mark.no_db


class _MemoryCollection:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    async def find_one(self, query: dict):
        for row in self.rows:
            if all(row.get(key) == value for key, value in query.items()):
                return dict(row)
        return None

    async def insert_one(self, doc: dict):
        from bson import ObjectId

        stored = dict(doc)
        stored["_id"] = ObjectId()
        self.rows.append(stored)
        return SimpleNamespace(inserted_id=stored["_id"])

    async def replace_one(self, query: dict, doc: dict):
        for row in self.rows:
            if row.get("_id") == query.get("_id"):
                row.clear()
                row.update(doc)
                row["_id"] = query["_id"]
                return SimpleNamespace(matched_count=1)
        return SimpleNamespace(matched_count=0)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    model_limits.clear_window_cache_for_tests()
    collection = _MemoryCollection()
    monkeypatch.setattr(
        ModelContextWindow, "get_collection", classmethod(lambda cls: collection)
    )
    yield collection
    model_limits.clear_window_cache_for_tests()


def test_each_window_field_is_read_and_a_bool_is_not():
    assert model_limits.window_from_payload({"max_model_len": 131072}) == 131072
    assert model_limits.window_from_payload({"max_context_length": 262144}) == 262144
    assert model_limits.window_from_payload({"context_length": 8192}) == 8192
    assert model_limits.window_from_payload({"max_input_tokens": 128000}) == 128000
    assert model_limits.window_from_payload({"max_model_len": True}) is None
    assert model_limits.window_from_payload({"max_model_len": 0}) is None


def test_catalog_entry_keeps_an_optional_window():
    entry = CatalogProviderModelEntry(
        catalog_model_id="m",
        display_name="M",
        model_name="m",
        context_window=32000,
    )
    assert entry.context_window == 32000
    assert (
        CatalogProviderModelEntry(
            catalog_model_id="m", display_name="M", model_name="m"
        ).context_window
        is None
    )


@pytest.mark.asyncio
async def test_catalog_override_wins_and_is_stored(_isolated, monkeypatch):
    async def catalog(base_url, model_name):
        return 32000

    async def fetch(*_args):
        raise AssertionError("the endpoint must not be called")

    monkeypatch.setattr(model_limits, "_catalog_window", catalog)
    monkeypatch.setattr(model_limits, "_fetch_window", fetch)
    found = await model_limits.window_for("https://models.test/v1", "custom", "key")
    assert found == 32000
    assert _isolated.rows[0]["source"] == "catalog"
    assert _isolated.rows[0]["context_window"] == 32000


@pytest.mark.asyncio
async def test_endpoint_window_is_stored_and_read_back(_isolated, monkeypatch):
    calls = {"n": 0}

    async def catalog(*_args):
        return None

    async def fetch(*_args):
        calls["n"] += 1
        return 131072

    monkeypatch.setattr(model_limits, "_catalog_window", catalog)
    monkeypatch.setattr(model_limits, "_fetch_window", fetch)
    first = await model_limits.window_for("https://models.test/v1", "alias-eve", "key")
    model_limits._memory.clear()
    second = await model_limits.window_for("https://models.test/v1", "alias-eve", "key")
    assert first == second == 131072
    assert calls["n"] == 1
    assert _isolated.rows[0]["source"] == "endpoint"


@pytest.mark.asyncio
async def test_a_failed_lookup_is_not_repeated(_isolated, monkeypatch):
    calls = {"n": 0}

    async def catalog(*_args):
        return None

    async def fetch(*_args):
        calls["n"] += 1
        raise TimeoutError("endpoint down")

    monkeypatch.setattr(model_limits, "_catalog_window", catalog)
    monkeypatch.setattr(model_limits, "_fetch_window", fetch)
    from src.config import AGENT_CONTEXT_WINDOW_TOKENS

    first = await model_limits.window_for("https://models.test/v1", "alias-eve", "key")
    second = await model_limits.window_for("https://models.test/v1", "alias-eve", "key")
    assert first == second == AGENT_CONTEXT_WINDOW_TOKENS
    assert calls["n"] == 1
    assert _isolated.rows == []


@pytest.mark.asyncio
async def test_a_stale_window_is_returned_and_refreshed(_isolated, monkeypatch):
    scheduled = []

    def schedule(*args):
        scheduled.append(args)

    monkeypatch.setattr(model_limits, "_schedule_refresh", schedule)
    old = datetime.now(timezone.utc) - timedelta(days=8)
    model_limits._memory[("https://models.test/v1", "alias-eve")] = (128000, old)
    found = await model_limits.window_for("https://models.test/v1", "alias-eve", "key")
    assert found == 128000
    assert scheduled == [("https://models.test/v1", "alias-eve", "key")]
