import io
import pytest

from tests.utils.utils import create_test_user_and_token
from tests.utils.cleaner import cleanup_models
from src.database.models.collection import Collection
from src.database.models.document import Document
from src.services.document import DocumentResult


# mocks for testing, if we want to test the vector store we can remove this
def _stub_vector_and_service(monkeypatch):
    """Disable heavy VectorStoreManager operations and stub add_documents."""

    # VectorStoreManager stubs (no-op)
    monkeypatch.setattr(
        "src.routers.collection.VectorStoreManager.create_collection",
        lambda self, name: None,
    )
    monkeypatch.setattr(
        "src.routers.collection.VectorStoreManager.delete_collection",
        lambda self, name: None,
    )
    monkeypatch.setattr(
        "src.routers.document.VectorStoreManager.delete_docs_by_metadata_filter",
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
