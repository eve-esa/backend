import io

import pytest
from bson import ObjectId

from src.database.models.collection import Collection
from src.database.models.document import Document
from src.database.models.user import User
from src.services.private_document_limit import release_private_document_slots
from tests.test_documents import _stub_vector_and_service
from tests.utils.cleaner import cleanup_models
from tests.utils.documentdb import reject_pipeline_updates
from tests.utils.utils import create_test_user_and_token


async def _set_counter(user_id: str, value) -> None:
    update = (
        {"$unset": {"private_document_count": ""}}
        if value is None
        else {"$set": {"private_document_count": value}}
    )
    await User.get_collection().update_one({"_id": ObjectId(user_id)}, update)


async def _counter(user_id: str):
    doc = await User.get_collection().find_one({"_id": ObjectId(user_id)})
    return doc.get("private_document_count")


@pytest.mark.asyncio
async def test_delete_collection_with_documents_on_documentdb(async_client, monkeypatch):
    """One delete removes the collection, its documents and frees their slots."""
    _stub_vector_and_service(monkeypatch)
    reject_pipeline_updates(monkeypatch)

    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        coll_id = (
            await async_client.post("/collections", json={"name": "DocDB"}, headers=headers)
        ).json()["id"]
        for name in ("one.txt", "two.txt"):
            resp = await async_client.post(
                f"/collections/{coll_id}/documents",
                headers=headers,
                files={
                    "files": (name, io.BytesIO(b"hello"), "text/plain"),
                    "metadata_names": (None, name),
                },
            )
            assert resp.status_code == 200
        assert await _counter(user.id) == 2

        resp = await async_client.delete(f"/collections/{coll_id}", headers=headers)

        assert resp.status_code == 200, resp.text
        assert await Collection.find_by_id(coll_id) is None
        assert await Document.count_documents({"collection_id": coll_id}) == 0
        assert await _counter(user.id) == 0
    finally:
        await Document.delete_many({"user_id": user.id})
        await Collection.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("start", "released", "expected"),
    [(5, 2, 3), (2, 2, 0), (1, 3, 0), (None, 2, 0)],
)
async def test_release_decrements_and_floors_at_zero(monkeypatch, start, released, expected):
    reject_pipeline_updates(monkeypatch)
    user, _ = await create_test_user_and_token()
    try:
        await _set_counter(user.id, start)

        await release_private_document_slots(user.id, released)

        assert await _counter(user.id) == expected
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_release_for_a_missing_user_is_a_no_op(monkeypatch):
    reject_pipeline_updates(monkeypatch)
    await release_private_document_slots(str(ObjectId()), 3)


class _RacingUsers:
    """Users collection where a reserve of `bump` lands right after the first read."""

    def __init__(self, inner, user_id: str, bump: int) -> None:
        self._inner = inner
        self._user_id = user_id
        self._bump = bump
        self.reads = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def find_one(self, *args, **kwargs):
        doc = await self._inner.find_one(*args, **kwargs)
        self.reads += 1
        if self.reads == 1:
            await self._inner.update_one(
                {"_id": ObjectId(self._user_id)},
                {"$inc": {"private_document_count": self._bump}},
            )
        return doc


@pytest.mark.asyncio
async def test_release_keeps_a_reserve_that_lands_before_the_floor(monkeypatch):
    """The floor is a compare-and-set: a missed floor retries the decrement."""
    reject_pipeline_updates(monkeypatch)
    user, _ = await create_test_user_and_token()
    try:
        await _set_counter(user.id, 1)
        racing = _RacingUsers(User.get_collection(), user.id, bump=5)
        monkeypatch.setattr(User, "get_collection", classmethod(lambda cls: racing))

        await release_private_document_slots(user.id, 3)

        monkeypatch.undo()
        assert racing.reads == 1
        # 1, then a reserve of 5 lands (6), the floor misses, the retry takes 3.
        assert await _counter(user.id) == 3
    finally:
        await cleanup_models([user])


class _NeverMatches:
    def __init__(self, inner) -> None:
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def update_one(self, *args, **kwargs):
        class _Result:
            matched_count = 0

        return _Result()


@pytest.mark.asyncio
async def test_release_logs_a_warning_when_attempts_run_out(monkeypatch, caplog):
    reject_pipeline_updates(monkeypatch)
    user, _ = await create_test_user_and_token()
    try:
        await _set_counter(user.id, 1)
        stuck = _NeverMatches(User.get_collection())
        monkeypatch.setattr(User, "get_collection", classmethod(lambda cls: stuck))

        with caplog.at_level("WARNING", logger="src.services.private_document_limit"):
            await release_private_document_slots(user.id, 3)

        monkeypatch.undo()
        assert f"private_document_release_exhausted user_id={user.id} released=3" in caplog.text
        assert await _counter(user.id) == 1
    finally:
        await cleanup_models([user])
