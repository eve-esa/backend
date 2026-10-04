"""Make the local MongoDB reject what Amazon DocumentDB 5.0 rejects.

Local MongoDB accepts aggregation pipeline updates; DocumentDB answers them
with OperationFailure code 14, so a test on MongoDB alone cannot catch one.
"""

from typing import Any

import pytest
from pymongo.errors import OperationFailure

import src.database.mongo_model as mongo_model

_UPDATE_METHODS = ("update_one", "update_many", "find_one_and_update")


class _DocumentDBCollection:
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if name not in _UPDATE_METHODS:
            return attr

        async def guarded(filter_, update, *args, **kwargs):
            if isinstance(update, list):
                raise OperationFailure(
                    "Wrong type for 'q.u', expected an object (pipeline updates "
                    "are not supported by DocumentDB)",
                    code=14,
                )
            return await attr(filter_, update, *args, **kwargs)

        return guarded


def reject_pipeline_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route every MongoModel collection through the DocumentDB guard."""
    real_get_collection = mongo_model.get_collection
    monkeypatch.setattr(
        mongo_model,
        "get_collection",
        lambda name: _DocumentDBCollection(real_get_collection(name)),
    )
