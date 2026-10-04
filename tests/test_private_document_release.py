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
