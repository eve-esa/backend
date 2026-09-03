import asyncio
import io
import pytest

from tests.utils.utils import create_test_user_and_token
from tests.utils.cleaner import cleanup_models
from src.database.models.collection import Collection
from src.database.models.document import Document
from src.database.models.user import User
from src.services.document import DocumentResult


# mocks for testing, if we want to test the vector store we can remove this
def _stub_vector_and_service(monkeypatch):
    """Disable heavy VectorStoreManager operations and stub add_documents."""

    # VectorStoreManager stubs (no-op)
    monkeypatch.setattr(
        "src.routers.collection.VectorStoreManager.ensure_private_collection",
        lambda self: None,
    )
    monkeypatch.setattr(
        "src.routers.collection.VectorStoreManager.delete_points_for_collection",
        lambda self, user_id, collection_id: 0,
    )
    monkeypatch.setattr(
        "src.routers.collection.VectorStoreManager.count_points_for_collection",
        lambda self, user_id, collection_id: 0,
    )
    monkeypatch.setattr(
        "src.routers.document.VectorStoreManager.delete_private_docs",
        lambda *args, **kwargs: None,
    )

    async def fake_add_documents(*args, **kwargs):  # noqa: D401
        files = kwargs.get("files") or (args[2] if len(args) > 2 else [])
        return DocumentResult(
            success=True,
            message="stubbed",
            data={"file_count": len(files)},
        )

    monkeypatch.setattr(
        "src.routers.document.document_service.add_documents", fake_add_documents
    )


@pytest.mark.asyncio
async def test_upload_single_file(async_client, monkeypatch):
    """Uploading one file creates exactly one Document entry."""

    _stub_vector_and_service(monkeypatch)

    user, token = await create_test_user_and_token()
    try:
        # Create collection
        coll_id = (
            await async_client.post(
                "/collections",
                json={"name": "Docs Coll"},
                headers={"Authorization": f"Bearer {token}"},
            )
        ).json()["id"]

        # Upload a single file
        files = {
            "files": ("one.txt", io.BytesIO(b"hello"), "text/plain"),
            "metadata_names": (None, "one.txt"),
        }
        resp = await async_client.post(
            f"/collections/{coll_id}/documents",
            headers={"Authorization": f"Bearer {token}"},
            files=files,
        )
        assert resp.status_code == 200
        assert resp.json()["file_count"] == 1

        docs = await Document.find_all(filter_dict={"collection_id": coll_id})
        assert len(docs) == 1

        # cleanup collection (also deletes docs)
        await async_client.delete(
            f"/collections/{coll_id}", headers={"Authorization": f"Bearer {token}"}
        )
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_private_document_upload_limit(async_client, monkeypatch):
    """Users cannot exceed the total private document upload cap."""

    _stub_vector_and_service(monkeypatch)
    monkeypatch.setattr("src.services.private_document_limit.MAX_PRIVATE_DOCUMENTS", 2)

    user, token = await create_test_user_and_token()
    try:
        coll_id = (
            await async_client.post(
                "/collections",
                json={"name": "Limit Coll"},
                headers={"Authorization": f"Bearer {token}"},
            )
        ).json()["id"]

        files = {
            "files": ("one.txt", io.BytesIO(b"hello"), "text/plain"),
            "metadata_names": (None, "one.txt"),
        }
        resp = await async_client.post(
            f"/collections/{coll_id}/documents",
            headers={"Authorization": f"Bearer {token}"},
            files=files,
        )
        assert resp.status_code == 200

        resp = await async_client.post(
            f"/collections/{coll_id}/documents",
            headers={"Authorization": f"Bearer {token}"},
            files=files,
        )
        assert resp.status_code == 200

        resp = await async_client.post(
            f"/collections/{coll_id}/documents",
            headers={"Authorization": f"Bearer {token}"},
            files=files,
        )
        assert resp.status_code == 400
        assert "Private document limit reached" in resp.json()["detail"]

        docs = await Document.find_all(filter_dict={"user_id": user.id})
        assert len(docs) == 2

        await async_client.delete(
            f"/collections/{coll_id}", headers={"Authorization": f"Bearer {token}"}
        )
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_private_document_upload_limit_concurrent(async_client, monkeypatch):
    """Concurrent uploads cannot exceed the per-user private document cap."""

    _stub_vector_and_service(monkeypatch)
    monkeypatch.setattr("src.services.private_document_limit.MAX_PRIVATE_DOCUMENTS", 2)

    user, token = await create_test_user_and_token()
    try:
        coll_id = (
            await async_client.post(
                "/collections",
                json={"name": "Concurrent Limit Coll"},
                headers={"Authorization": f"Bearer {token}"},
            )
        ).json()["id"]

        files = {
            "files": ("one.txt", io.BytesIO(b"hello"), "text/plain"),
            "metadata_names": (None, "one.txt"),
        }
        resp = await async_client.post(
            f"/collections/{coll_id}/documents",
            headers={"Authorization": f"Bearer {token}"},
            files=files,
        )
        assert resp.status_code == 200

        async def upload_one():
            return await async_client.post(
                f"/collections/{coll_id}/documents",
                headers={"Authorization": f"Bearer {token}"},
                files=files,
            )

        results = await asyncio.gather(upload_one(), upload_one())
        statuses = sorted(resp.status_code for resp in results)
        assert statuses == [200, 400]

        docs = await Document.find_all(filter_dict={"user_id": user.id})
        assert len(docs) == 2

        refreshed_user = await User.find_by_id(user.id)
        assert refreshed_user.private_document_count == 2

        await async_client.delete(
            f"/collections/{coll_id}", headers={"Authorization": f"Bearer {token}"}
        )
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_upload_two_files(async_client, monkeypatch):
    """Uploading two files creates exactly two Document entries."""

    _stub_vector_and_service(monkeypatch)

    user, token = await create_test_user_and_token()
    try:
        coll_id = (
            await async_client.post(
                "/collections",
                json={"name": "Docs Coll 2"},
                headers={"Authorization": f"Bearer {token}"},
            )
        ).json()["id"]

        multipart_files = [
            (
                "files",
                ("one.txt", io.BytesIO(b"hello"), "text/plain"),
            ),
            (
                "files",
                ("two.txt", io.BytesIO(b"world"), "text/plain"),
            ),
            (
                "metadata_names",
                (None, "one.txt"),
            ),
            (
                "metadata_names",
                (None, "two.txt"),
            ),
        ]

        resp = await async_client.post(
            f"/collections/{coll_id}/documents",
            headers={"Authorization": f"Bearer {token}"},
            files=multipart_files,
        )
        assert resp.status_code == 200
        assert resp.json()["file_count"] == 2

        docs = await Document.find_all(filter_dict={"collection_id": coll_id})
        assert len(docs) == 2

        await async_client.delete(
            f"/collections/{coll_id}", headers={"Authorization": f"Bearer {token}"}
        )
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_upload_rolls_back_files_that_were_not_ingested(async_client, monkeypatch):
    """Mongo rows and quota for files that produced no vectors are released."""

    _stub_vector_and_service(monkeypatch)

    async def fake_add_documents(*args, **kwargs):
        document_ids = kwargs.get("document_ids") or []
        return DocumentResult(
            success=True,
            message="partial",
            data={
                "file_count": 1,
                "ingested_document_ids": document_ids[:1],
            },
        )

    monkeypatch.setattr(
        "src.routers.document.document_service.add_documents", fake_add_documents
    )

    user, token = await create_test_user_and_token()
    try:
        coll_id = (
            await async_client.post(
                "/collections",
                json={"name": "Partial Ingest Coll"},
                headers={"Authorization": f"Bearer {token}"},
            )
        ).json()["id"]

        multipart_files = [
            ("files", ("one.txt", io.BytesIO(b"hello"), "text/plain")),
            ("files", ("two.txt", io.BytesIO(b"world"), "text/plain")),
            ("metadata_names", (None, "one.txt")),
            ("metadata_names", (None, "two.txt")),
        ]
        resp = await async_client.post(
            f"/collections/{coll_id}/documents",
            headers={"Authorization": f"Bearer {token}"},
            files=multipart_files,
        )
        assert resp.status_code == 200
        assert resp.json()["file_count"] == 1
        assert "ingested_document_ids" not in resp.json()

        docs = await Document.find_all(filter_dict={"collection_id": coll_id})
        assert len(docs) == 1
        assert docs[0].filename == "one.txt"

        refreshed_user = await User.find_by_id(user.id)
        assert refreshed_user.private_document_count == 1

        await async_client.delete(
            f"/collections/{coll_id}", headers={"Authorization": f"Bearer {token}"}
        )
    finally:
        await cleanup_models([user])
