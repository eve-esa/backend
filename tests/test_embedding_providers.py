"""Embedding provider order: first provider that answers wins, the rest are fallbacks."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import src.core.vector_store_manager as vsm
from src.constants import DEEPINFRA_DEFAULT_EMBEDDING_MODEL, JSC_DEFAULT_EMBEDDING_MODEL


class _FakeOpenAI:
    """Records each client's base_url and model; fails for the base urls in `failing`."""

    calls: list = []
    failing: set = set()

    def __init__(self, api_key, base_url):
        self.base_url = base_url
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, input, model):
        _FakeOpenAI.calls.append((self.base_url, model))
        if self.base_url in _FakeOpenAI.failing:
            raise RuntimeError(f"{self.base_url} down")
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[float(len(text))]) for text in input]
        )


@pytest.fixture
def manager(monkeypatch):
    _FakeOpenAI.calls = []
    _FakeOpenAI.failing = set()
    monkeypatch.setattr(vsm, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(vsm, "DEEPINFRA_EMBEDDING_URL", "deepinfra")
    monkeypatch.setattr(vsm, "DEEPINFRA_EMBEDDING_API_KEY", "di-key")
    monkeypatch.setattr(vsm, "EVE_JSC_BASE_URL", "jsc")
    monkeypatch.setattr(vsm, "JSC_EMBEDDING_API_KEY", "jsc-key")
    monkeypatch.setattr(vsm, "EMBEDDING_PROVIDER_ORDER", ["deepinfra", "jsc"])
    error_logger = MagicMock(log_error_sync=AsyncMock())
    monkeypatch.setattr(vsm, "get_error_logger", lambda: error_logger)
    # No Qdrant client: only the embedding methods are exercised.
    return object.__new__(vsm.VectorStoreManager)


async def test_deepinfra_answers_first_and_jsc_is_not_called(manager):
    vector, fallback_error = await manager.generate_query_vector(
        "sentinel", JSC_DEFAULT_EMBEDDING_MODEL
    )

    assert vector == [8.0]
    assert fallback_error is None
    assert _FakeOpenAI.calls == [("deepinfra", DEEPINFRA_DEFAULT_EMBEDDING_MODEL)]


async def test_falls_back_to_jsc_and_reports_the_first_error(manager):
    _FakeOpenAI.failing = {"deepinfra"}

    vector, fallback_error = await manager.generate_query_vector(
        "sentinel", JSC_DEFAULT_EMBEDDING_MODEL
    )

    assert vector == [8.0]
    assert fallback_error == "deepinfra down"
    assert [url for url, _ in _FakeOpenAI.calls] == ["deepinfra", "jsc"]


async def test_order_is_configurable(manager, monkeypatch):
    monkeypatch.setattr(vsm, "EMBEDDING_PROVIDER_ORDER", ["jsc", "deepinfra"])

    await manager.generate_query_vector("sentinel", JSC_DEFAULT_EMBEDDING_MODEL)

    assert _FakeOpenAI.calls == [("jsc", vsm.JSC_EMBEDDING_MODEL_NAME)]


async def test_missing_key_counts_as_a_failure(manager, monkeypatch):
    monkeypatch.setattr(vsm, "DEEPINFRA_EMBEDDING_API_KEY", "")

    _, fallback_error = await manager.generate_query_vector(
        "sentinel", JSC_DEFAULT_EMBEDDING_MODEL
    )

    assert fallback_error == "deepinfra embedding API key is not set"
    assert [url for url, _ in _FakeOpenAI.calls] == ["jsc"]


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

    assert vectors == [[1.0], [2.0]]
    assert fallback_error == "deepinfra down"


async def test_batch_with_no_texts_calls_nothing(manager):
    assert await manager.generate_batch_embeddings([], JSC_DEFAULT_EMBEDDING_MODEL) == (
        [],
        None,
    )
    assert _FakeOpenAI.calls == []
