"""Make the local MongoDB read like a replica set whose secondaries lag.

Staging and prod connect with ``readPreference=secondaryPreferred``, so a read
can miss a write made a moment earlier on the primary. A single local MongoDB
never shows that. Here every read on the named collections that is not pinned
to the primary sees a secondary that has replicated nothing yet; reads through
``with_options(read_preference=ReadPreference.PRIMARY)`` see the real data.
"""

from typing import Any

import pytest
from pymongo import ReadPreference

import src.database.mongo_model as mongo_model

_READ_METHODS = ("find", "find_one", "count_documents")


class _LaggingCollection:
    def __init__(self, inner: Any, *, primary: bool = False) -> None:
        self._inner = inner
        self._primary = primary

    def with_options(self, **kwargs: Any) -> "_LaggingCollection":
        primary = kwargs.get("read_preference") == ReadPreference.PRIMARY
        return _LaggingCollection(self._inner.with_options(**kwargs), primary=primary)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if self._primary or name not in _READ_METHODS:
            return attr
        if name == "find":

            def find_nothing(filter_=None, *args, **kwargs):
                return attr({"_id": {"$in": []}}, *args, **kwargs)

            return find_nothing

        async def read_nothing(*args, **kwargs):
            return 0 if name == "count_documents" else None

        return read_nothing


def lag_secondaries(monkeypatch: pytest.MonkeyPatch, *collection_names: str) -> None:
    """Route MongoModel reads on ``collection_names`` through an empty secondary."""
    real_get_collection = mongo_model.get_collection

    def get_collection(name: str):
        collection = real_get_collection(name)
        return _LaggingCollection(collection) if name in collection_names else collection

    monkeypatch.setattr(mongo_model, "get_collection", get_collection)
