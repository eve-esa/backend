"""POST /conversations/{id}/documents: documents attached to one conversation.

Parsing and chunking run for real; only the Qdrant boundary is replaced by an
in-memory store, so the number of points written can be compared with the
chunk_count the endpoint reports.
"""

import asyncio
import io
import time
from types import SimpleNamespace

import anyio

import pytest
from bson import ObjectId
from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)

from src.database.models.collection import Collection
from src.database.models.conversation import Conversation
from src.database.models.document import Document
from src.database.models.message import Message
from src.database.models.user import User
from src.database.mongo import get_collection
from src.routers.message import apply_private_collections_to_request
from src.schemas.generation_request import GenerationRequest
from src.services.agents.core.runner import _rag_mcp_tool_defaults
from tests.utils.cleaner import cleanup_models
from tests.utils.documentdb import reject_pipeline_updates
from tests.utils.utils import create_test_user_and_token


class FakeVectorStore:
    """Points by (user_id, collection_id), each carrying its document_id."""

    def __init__(self):
        self.points = []
        # Seconds to hang after writing a batch, as an upsert stuck mid-upload would.
        self.delay = 0.0
        self.add_error = None
        self.delete_error = None
        self.add_calls = 0
        # Seconds a write blocks a worker thread that no deadline can interrupt,
        # like the real upsert (anyio.to_thread without abandon).
        self.uncancellable_delay = 0.0
        # Seconds every point delete blocks its thread, as a slow Qdrant would.
        self.delete_delay = 0.0

    def count(self, document_id=None, collection_id=None):
        return sum(
            1
            for p in self.points
            if (document_id is None or p["document_id"] == document_id)
            and (collection_id is None or p["collection_id"] == collection_id)
        )


@pytest.fixture
def store(monkeypatch):
    fake = FakeVectorStore()
    vsm = "src.core.vector_store_manager.VectorStoreManager"

    async def add_document_list(self, document_list, *, user_id, collection_id):
        fake.add_calls += 1
        if fake.add_error:
            raise fake.add_error
        for doc in document_list:
            fake.points.append(
                {
                    "user_id": user_id,
                    "collection_id": collection_id,
                    "document_id": doc.metadata.get("document_id"),
                }
            )
        if fake.uncancellable_delay:
            await anyio.to_thread.run_sync(time.sleep, fake.uncancellable_delay)
        if fake.delay:
            await asyncio.sleep(fake.delay)
        return [str(i) for i in range(len(document_list))]

    def delete_private_docs(self, *, user_id, collection_id, metadata=None, **_):
        if fake.delete_delay:
            time.sleep(fake.delete_delay)
        if fake.delete_error:
            raise fake.delete_error
        document_id = (metadata or {}).get("metadata.document_id")
        fake.points = [
            p
            for p in fake.points
            if not (
                p["user_id"] == user_id
                and p["collection_id"] == collection_id
                and (document_id is None or p["document_id"] == document_id)
            )
        ]

    def delete_points_for_collection(self, user_id, collection_id):
        before = len(fake.points)
        delete_private_docs(self, user_id=user_id, collection_id=collection_id)
        return before - len(fake.points)

    monkeypatch.setattr(f"{vsm}.add_document_list", add_document_list)
    monkeypatch.setattr(f"{vsm}.delete_private_docs", delete_private_docs)
    monkeypatch.setattr(
        f"{vsm}.delete_points_for_collection", delete_points_for_collection
    )
    monkeypatch.setattr("src.config.FEATURE_CONVERSATION_DOCUMENTS", True)
    return fake


def _pdf_bytes(text: str, pad_to: int = 0) -> bytes:
    """A one-page PDF with real text, padded with an attachment to ``pad_to`` bytes."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): writer._add_object(font)}
            )
        }
    )
    content = DecodedStreamObject()
    content.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode())
    page[NameObject("/Contents")] = writer._add_object(content)
    if pad_to:
        writer.add_attachment("padding.bin", b"\0" * pad_to)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


async def _conversation(user: User) -> Conversation:
    return await Conversation.create(user_id=user.id, name="Attach test")


async def _slots(user: User) -> int:
    row = await User.get_collection().find_one({"_id": ObjectId(user.id)})
    return int(row.get("private_document_count") or 0)


async def _cleanup(user: User, conversation: Conversation) -> None:
    collections = await Collection.find_all(filter_dict={"user_id": user.id})
    for collection in collections:
        await Document.delete_many({"collection_id": collection.id})
    await cleanup_models([user, conversation, *collections])


def _txt(name: str, body: bytes = b"EVE reads attached text.") -> tuple:
    return ("files", (name, io.BytesIO(body), "text/plain"))


async def test_flag_off_answers_404(async_client, store, monkeypatch):
    monkeypatch.setattr("src.config.FEATURE_CONVERSATION_DOCUMENTS", False)
    user, token = await create_test_user_and_token()
    conversation = await _conversation(user)
    try:
        resp = await async_client.post(
            f"/conversations/{conversation.id}/documents",
            headers={"Authorization": f"Bearer {token}"},
            files=[_txt("a.txt")],
        )
        assert resp.status_code == 404
        assert await _slots(user) == 0
    finally:
        await _cleanup(user, conversation)


async def test_pdf_upload_attaches_to_the_conversation(async_client, store):
    user, token = await create_test_user_and_token()
    conversation = await _conversation(user)
    try:
        pdf = _pdf_bytes("Sentinel-2 revisit time is five days.", pad_to=2 * 1024 * 1024)
        assert len(pdf) >= 2 * 1024 * 1024

        resp = await async_client.post(
            f"/conversations/{conversation.id}/documents",
            headers={"Authorization": f"Bearer {token}"},
            files=[("files", ("orbit.pdf", io.BytesIO(pdf), "application/pdf"))],
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        [document] = body["documents"]
        assert document["document_id"]
        assert document["chunk_count"] >= 1

        rows = await Collection.find_all(
            filter_dict={"user_id": user.id, "conversation_id": conversation.id}
        )
        assert [row.id for row in rows] == [body["collection_id"]]

        stored = await Document.find_by_id(document["document_id"])
        assert stored.collection_id == body["collection_id"]
        assert stored.chunk_count == document["chunk_count"]
        assert store.count(document_id=document["document_id"]) == document["chunk_count"]
        assert await _slots(user) == 1

        listed = await async_client.get(
            "/collections", headers={"Authorization": f"Bearer {token}"}
        )
        assert listed.status_code == 200
        assert body["collection_id"] not in [c["id"] for c in listed.json()["data"]]
    finally:
        await _cleanup(user, conversation)


async def test_eleventh_document_is_refused(async_client, store):
    user, token = await create_test_user_and_token()
    conversation = await _conversation(user)
    try:
        resp = await async_client.post(
            f"/conversations/{conversation.id}/documents",
            headers={"Authorization": f"Bearer {token}"},
            files=[_txt(f"doc-{i}.txt") for i in range(10)],
        )
        assert resp.status_code == 200, resp.text
        assert await _slots(user) == 10

        resp = await async_client.post(
            f"/conversations/{conversation.id}/documents",
            headers={"Authorization": f"Bearer {token}"},
            files=[_txt("doc-10.txt")],
        )
        assert resp.status_code == 400
        assert resp.json()["detail"]["code"] == "conversation_document_limit"
        assert await _slots(user) == 10
    finally:
        await _cleanup(user, conversation)


async def test_wrong_type_and_oversize_use_no_slot(async_client, store, monkeypatch):
    monkeypatch.setattr("src.routers.document.MAX_CONVERSATION_DOCUMENT_BYTES", 1024)
    user, token = await create_test_user_and_token()
    conversation = await _conversation(user)
    try:
        png = await async_client.post(
            f"/conversations/{conversation.id}/documents",
            headers={"Authorization": f"Bearer {token}"},
            files=[("files", ("map.png", io.BytesIO(b"\x89PNG"), "image/png"))],
        )
        assert png.status_code == 415
        assert png.json()["detail"]["code"] == "conversation_document_type"

        big = await async_client.post(
            f"/conversations/{conversation.id}/documents",
            headers={"Authorization": f"Bearer {token}"},
            files=[_txt("big.txt", b"x" * 1025)],
        )
        assert big.status_code == 413
        assert big.json()["detail"]["code"] == "conversation_document_size"

        assert await _slots(user) == 0
        assert await Collection.count_documents({"user_id": user.id}) == 0
    finally:
        await _cleanup(user, conversation)


async def test_deadline_rolls_everything_back(async_client, store, monkeypatch):
    monkeypatch.setattr("src.routers.document.CONVERSATION_DOCUMENT_DEADLINE_SECONDS", 0.2)
    store.delay = 2.0
    user, token = await create_test_user_and_token()
    conversation = await _conversation(user)
    try:
        resp = await async_client.post(
            f"/conversations/{conversation.id}/documents",
            headers={"Authorization": f"Bearer {token}"},
            files=[_txt("slow.txt")],
        )
        assert resp.status_code == 504
        assert resp.json()["detail"]["code"] == "conversation_document_timeout"
        # The fake wrote its points before hanging: the rollback removed them.
        assert store.add_calls == 1
        assert await _slots(user) == 0
        assert await Document.count_documents({"user_id": user.id}) == 0
        assert store.count() == 0
    finally:
        await _cleanup(user, conversation)


async def test_other_users_conversation_answers_404(async_client, store):
    owner, _ = await create_test_user_and_token()
    other, other_token = await create_test_user_and_token()
    conversation = await _conversation(owner)
    try:
        resp = await async_client.post(
            f"/conversations/{conversation.id}/documents",
            headers={"Authorization": f"Bearer {other_token}"},
            files=[_txt("a.txt")],
        )
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Conversation not found"
        assert await _slots(other) == 0
    finally:
        await _cleanup(owner, conversation)
        await cleanup_models([other])


async def test_chat_searches_attached_documents_only_in_their_conversation(
    async_client, store, monkeypatch
):
    user, token = await create_test_user_and_token()
    conversation = await _conversation(user)
    other_conversation = await _conversation(user)
    try:
        resp = await async_client.post(
            f"/conversations/{conversation.id}/documents",
            headers={"Authorization": f"Bearer {token}"},
            files=[_txt("a.txt")],
        )
        collection_id = resp.json()["collection_id"]

        request = GenerationRequest(query="what does the file say?")
        await apply_private_collections_to_request(request, user.id, conversation.id)
        assert collection_id in request.collection_ids
        assert collection_id in request.private_collections_map
        # The agentic retrieve tool receives private_collections, not collection_ids.
        retrieve = SimpleNamespace(
            name="retrieve",
            args={"query": {}, "private_collections": {}, "collection_ids": {}},
        )
        defaults = _rag_mcp_tool_defaults(retrieve, request)
        assert collection_id in defaults["private_collections"]
        # Preparing the same request again (the retry path) adds no duplicate.
        await apply_private_collections_to_request(request, user.id, conversation.id)
        assert request.private_collections.count(collection_id) == 1

        elsewhere = GenerationRequest(query="what does the file say?")
        await apply_private_collections_to_request(
            elsewhere, user.id, other_conversation.id
        )
        assert collection_id not in elsewhere.collection_ids

        monkeypatch.setattr("src.config.FEATURE_CONVERSATION_DOCUMENTS", False)
        flag_off = GenerationRequest(query="what does the file say?")
        await apply_private_collections_to_request(flag_off, user.id, conversation.id)
        assert collection_id not in flag_off.collection_ids
        assert collection_id not in flag_off.private_collections
    finally:
        await cleanup_models([other_conversation])
        await _cleanup(user, conversation)


async def test_deleting_the_conversation_frees_its_documents(async_client, store):
    user, token = await create_test_user_and_token()
    conversation = await _conversation(user)
    try:
        resp = await async_client.post(
            f"/conversations/{conversation.id}/documents",
            headers={"Authorization": f"Bearer {token}"},
            files=[_txt("a.txt"), _txt("b.txt")],
        )
        assert resp.status_code == 200, resp.text
        collection_id = resp.json()["collection_id"]
        assert await _slots(user) == 2

        deleted = await async_client.delete(
            f"/conversations/{conversation.id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert deleted.status_code == 200
        assert await _slots(user) == 0
        assert await Collection.find_by_id(collection_id) is None
        assert await Document.count_documents({"collection_id": collection_id}) == 0
        assert store.count(collection_id=collection_id) == 0
    finally:
        await cleanup_models([user])


class _LaggingSecondary:
    """A replica that has not received the latest writes: reads find nothing.

    Writes and ``with_options`` (which get_primary_collection uses) go to the
    real collection, so only code that reads on the primary sees fresh rows.
    """

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    async def find_one(self, *args, **kwargs):
        return None

    async def count_documents(self, *args, **kwargs):
        return 0

    def find(self, *args, **kwargs):
        return self._real.find({"_id": None})


async def test_reads_after_writes_use_the_primary(async_client, store, monkeypatch):
    user, token = await create_test_user_and_token()
    conversation = await _conversation(user)
    try:
        for model in (Conversation, Collection, Document):
            monkeypatch.setattr(
                model,
                "get_collection",
                classmethod(lambda cls: _LaggingSecondary(get_collection(cls.collection_name))),
            )

        resp = await async_client.post(
            f"/conversations/{conversation.id}/documents",
            headers={"Authorization": f"Bearer {token}"},
            files=[_txt("a.txt")],
        )
        assert resp.status_code == 200, resp.text
        [document] = resp.json()["documents"]
        assert document["chunk_count"] == 1

        request = GenerationRequest(query="what does the file say?")
        await apply_private_collections_to_request(request, user.id, conversation.id)
        assert resp.json()["collection_id"] in request.private_collections
    finally:
        monkeypatch.undo()
        await _cleanup(user, conversation)


async def test_concurrent_first_uploads_share_one_collection(async_client, store):
    user, token = await create_test_user_and_token()
    conversation = await _conversation(user)
    try:
        first, second = await asyncio.gather(
            *(
                async_client.post(
                    f"/conversations/{conversation.id}/documents",
                    headers={"Authorization": f"Bearer {token}"},
                    files=[_txt(name)],
                )
                for name in ("a.txt", "b.txt")
            )
        )
        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text
        assert first.json()["collection_id"] == second.json()["collection_id"]
        assert first.json()["collection_id"] == conversation.id
        assert (
            await Collection.count_documents({"conversation_id": conversation.id}) == 1
        )
        assert await _slots(user) == 2
    finally:
        await _cleanup(user, conversation)


async def test_qdrant_failure_stops_the_delete(async_client, store):
    user, token = await create_test_user_and_token()
    conversation = await _conversation(user)
    try:
        resp = await async_client.post(
            f"/conversations/{conversation.id}/documents",
            headers={"Authorization": f"Bearer {token}"},
            files=[_txt("a.txt")],
        )
        collection_id = resp.json()["collection_id"]
        store.delete_error = RuntimeError("qdrant unavailable")

        deleted = await async_client.delete(
            f"/conversations/{conversation.id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert deleted.status_code == 500
        assert deleted.json()["detail"]["code"] == "conversation_document_cleanup_failed"
        # Nothing was deleted, so a retry can finish the job.
        assert await Conversation.find_by_id(conversation.id) is not None
        assert await Collection.find_by_id(collection_id) is not None
        assert await Document.count_documents({"collection_id": collection_id}) == 1
        assert store.count(collection_id=collection_id) == 1
        assert await _slots(user) == 1

        store.delete_error = None
        retried = await async_client.delete(
            f"/conversations/{conversation.id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert retried.status_code == 200
        assert await _slots(user) == 0
    finally:
        await _cleanup(user, conversation)


async def test_unexpected_error_hides_its_details(async_client, store):
    user, token = await create_test_user_and_token()
    conversation = await _conversation(user)
    try:
        store.add_error = RuntimeError("connect to 10.30.12.7:6333 refused")
        resp = await async_client.post(
            f"/conversations/{conversation.id}/documents",
            headers={"Authorization": f"Bearer {token}"},
            files=[_txt("a.txt")],
        )
        assert resp.status_code == 500
        assert resp.json()["detail"]["code"] == "conversation_document_failed"
        assert "10.30.12.7" not in resp.text
        assert await _slots(user) == 0
        assert await Document.count_documents({"user_id": user.id}) == 0
    finally:
        await _cleanup(user, conversation)


async def test_too_many_chunks_is_refused_before_embedding(
    async_client, store, monkeypatch
):
    monkeypatch.setattr("src.routers.document.MAX_CONVERSATION_DOCUMENT_CHUNKS", 2)
    user, token = await create_test_user_and_token()
    conversation = await _conversation(user)
    try:
        long_text = ("Sentinel-2 observes land and coastal areas. " * 200).encode()
        resp = await async_client.post(
            f"/conversations/{conversation.id}/documents",
            headers={"Authorization": f"Bearer {token}"},
            files=[_txt("long.txt", long_text)],
        )
        assert resp.status_code == 413
        assert resp.json()["detail"]["code"] == "conversation_document_chunks"
        assert store.add_calls == 0
        assert await _slots(user) == 0
        assert await Document.count_documents({"user_id": user.id}) == 0
    finally:
        await _cleanup(user, conversation)


async def test_collection_upload_refuses_a_conversation_collection(
    async_client, store
):
    user, token = await create_test_user_and_token()
    conversation = await _conversation(user)
    try:
        resp = await async_client.post(
            f"/conversations/{conversation.id}/documents",
            headers={"Authorization": f"Bearer {token}"},
            files=[_txt("a.txt")],
        )
        collection_id = resp.json()["collection_id"]

        bypass = await async_client.post(
            f"/collections/{collection_id}/documents",
            headers={"Authorization": f"Bearer {token}"},
            files={
                "files": ("b.txt", io.BytesIO(b"more text"), "text/plain"),
                "metadata_names": (None, "b.txt"),
            },
        )
        assert bypass.status_code == 400
        assert bypass.json()["detail"]["code"] == "conversation_collection"
        assert await _slots(user) == 1
        assert await Document.count_documents({"collection_id": collection_id}) == 1
    finally:
        await _cleanup(user, conversation)


async def test_chunk_cap_counts_all_files_of_the_request(
    async_client, store, monkeypatch
):
    monkeypatch.setattr("src.routers.document.MAX_CONVERSATION_DOCUMENT_CHUNKS", 3)
    user, token = await create_test_user_and_token()
    conversation = await _conversation(user)
    try:
        # Each file splits into 2 chunks: under the cap alone, 4 together.
        two_chunks = ("Sentinel-2 observes land and coastal areas. " * 30).encode()
        resp = await async_client.post(
            f"/conversations/{conversation.id}/documents",
            headers={"Authorization": f"Bearer {token}"},
            files=[_txt("a.txt", two_chunks), _txt("b.txt", two_chunks)],
        )
        assert resp.status_code == 413, resp.text
        assert resp.json()["detail"]["code"] == "conversation_document_chunks"
        assert store.add_calls == 0
        assert await _slots(user) == 0
        assert await Collection.count_documents({"conversation_id": conversation.id}) == 0
    finally:
        await _cleanup(user, conversation)


async def test_a_write_that_cannot_be_interrupted_still_ends_in_our_504(
    async_client, store, monkeypatch
):
    monkeypatch.setattr("src.routers.document.CONVERSATION_DOCUMENT_DEADLINE_SECONDS", 0.2)
    store.uncancellable_delay = 1.0
    # The pause before the next batch, where the expired deadline takes effect.
    store.delay = 0.05
    user, token = await create_test_user_and_token()
    conversation = await _conversation(user)
    try:
        started = time.monotonic()
        resp = await async_client.post(
            f"/conversations/{conversation.id}/documents",
            headers={"Authorization": f"Bearer {token}"},
            files=[_txt("slow.txt")],
        )
        elapsed = time.monotonic() - started
        assert resp.status_code == 504
        assert resp.json()["detail"]["code"] == "conversation_document_timeout"
        # The deadline waited for the write in flight (it cannot be cut short),
        # stopped at the next pause and rolled the write back.
        assert 1.0 <= elapsed < 3.0
        assert store.count() == 0
        assert await _slots(user) == 0
        assert await Collection.count_documents({"conversation_id": conversation.id}) == 0
    finally:
        await _cleanup(user, conversation)


async def test_rollback_is_bounded_and_keeps_slots_for_rows_it_could_not_remove(
    async_client, store, monkeypatch
):
    monkeypatch.setattr("src.routers.document.CONVERSATION_DOCUMENT_DEADLINE_SECONDS", 0.2)
    monkeypatch.setattr("src.routers.document.CONVERSATION_DOCUMENT_ROLLBACK_SECONDS", 0.3)
    store.delay = 2.0
    store.delete_delay = 2.0
    user, token = await create_test_user_and_token()
    conversation = await _conversation(user)
    try:
        started = time.monotonic()
        resp = await async_client.post(
            f"/conversations/{conversation.id}/documents",
            headers={"Authorization": f"Bearer {token}"},
            files=[_txt("slow.txt")],
        )
        elapsed = time.monotonic() - started
        assert resp.status_code == 504
        assert resp.json()["detail"]["code"] == "conversation_document_timeout"
        # Answered on budget, without waiting out the 2 s delete.
        assert elapsed < 1.5
        # The row the rollback left keeps its slot, so the counter stays true.
        assert await Document.count_documents({"user_id": user.id}) == 1
        assert await _slots(user) == 1
    finally:
        await asyncio.sleep(2.0)  # let the abandoned delete finish before cleanup
        await _cleanup(user, conversation)


async def test_refused_first_upload_leaves_no_collection(async_client, store):
    user, token = await create_test_user_and_token()
    conversation = await _conversation(user)
    try:
        await User.get_collection().update_one(
            {"_id": ObjectId(user.id)}, {"$set": {"private_document_count": 150}}
        )
        resp = await async_client.post(
            f"/conversations/{conversation.id}/documents",
            headers={"Authorization": f"Bearer {token}"},
            files=[_txt("a.txt")],
        )
        assert resp.status_code == 400
        assert "Private document limit reached" in resp.text
        assert await Collection.count_documents({"conversation_id": conversation.id}) == 0
    finally:
        await _cleanup(user, conversation)


async def test_mongo_failure_in_the_delete_answers_with_a_code(
    async_client, store, monkeypatch
):
    user, token = await create_test_user_and_token()
    conversation = await _conversation(user)
    try:
        await async_client.post(
            f"/conversations/{conversation.id}/documents",
            headers={"Authorization": f"Bearer {token}"},
            files=[_txt("a.txt")],
        )

        async def broken_delete_many(*args, **kwargs):
            raise RuntimeError("mongo at 10.30.40.5 refused")

        monkeypatch.setattr(Document, "delete_many", broken_delete_many)
        deleted = await async_client.delete(
            f"/conversations/{conversation.id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert deleted.status_code == 500
        assert deleted.json()["detail"]["code"] == "conversation_document_cleanup_failed"
        assert "10.30.40.5" not in deleted.text
        assert await Conversation.find_by_id(conversation.id) is not None
    finally:
        monkeypatch.undo()
        await _cleanup(user, conversation)


async def test_collections_are_indexed_for_the_chat_lookup():
    from src.database.mongo import async_mongo_manager

    await async_mongo_manager._ensure_indexes()
    indexes = await Collection.get_collection().index_information()
    assert any(
        index["key"] == [("user_id", 1), ("conversation_id", 1)]
        for index in indexes.values()
    )


# GET /conversations/{id}/documents, DELETE .../documents/{doc_id} and the cascade.
# All on the DocumentDB guard from backend #275: local MongoDB accepts pipeline
# updates that DocumentDB answers with code 14.


async def _attach(async_client, token, conversation, *names) -> list[dict]:
    resp = await async_client.post(
        f"/conversations/{conversation.id}/documents",
        headers={"Authorization": f"Bearer {token}"},
        files=[_txt(name) for name in names],
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["documents"]


async def _list(async_client, token, conversation_id):
    return await async_client.get(
        f"/conversations/{conversation_id}/documents",
        headers={"Authorization": f"Bearer {token}"},
    )


async def _delete_doc(async_client, token, conversation_id, document_id):
    return await async_client.delete(
        f"/conversations/{conversation_id}/documents/{document_id}",
        headers={"Authorization": f"Bearer {token}"},
    )


async def _set_slots(user: User, value: int) -> None:
    await User.get_collection().update_one(
        {"_id": ObjectId(user.id)}, {"$set": {"private_document_count": value}}
    )


async def test_list_shows_each_document_with_its_chunks(
    async_client, store, monkeypatch
):
    reject_pipeline_updates(monkeypatch)
    user, token = await create_test_user_and_token()
    conversation = await _conversation(user)
    try:
        attached = await _attach(async_client, token, conversation, "a.txt", "b.txt")

        resp = await _list(async_client, token, conversation.id)

        assert resp.status_code == 200, resp.text
        documents = resp.json()["documents"]
        assert [d["document_id"] for d in documents] == [
            d["document_id"] for d in attached
        ]
        assert [d["name"] for d in documents] == ["a.txt", "b.txt"]
        for document in documents:
            assert document["chunk_count"] > 0
            assert document["chunk_count"] == store.count(
                document_id=document["document_id"]
            )
            assert document["file_type"] == "txt"
            assert document["file_size"] > 0
            assert "vector_ids" not in document
    finally:
        await _cleanup(user, conversation)


async def test_list_is_empty_before_any_upload(async_client, store):
    user, token = await create_test_user_and_token()
    conversation = await _conversation(user)
    try:
        resp = await _list(async_client, token, conversation.id)
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"documents": []}
    finally:
        await _cleanup(user, conversation)


async def test_an_upload_still_running_is_neither_listed_nor_deletable(
    async_client, store
):
    """Its row exists before its points: a delete now would orphan what follows."""
    user, token = await create_test_user_and_token()
    conversation = await _conversation(user)
    try:
        await _attach(async_client, token, conversation, "done.txt")
        collection_id = conversation.id
        running = await Document.create(
            user_id=user.id, collection_id=collection_id, name="running.txt"
        )
        await _set_slots(user, 2)

        listed = (await _list(async_client, token, conversation.id)).json()
        deleted = await _delete_doc(async_client, token, conversation.id, running.id)

        assert [d["name"] for d in listed["documents"]] == ["done.txt"]
        assert deleted.status_code == 404
        assert await Document.find_by_id(running.id) is not None
        assert await _slots(user) == 2
    finally:
        await _cleanup(user, conversation)


async def test_delete_removes_one_document_and_keeps_the_others(
    async_client, store, monkeypatch
):
    reject_pipeline_updates(monkeypatch)
    user, token = await create_test_user_and_token()
    conversation = await _conversation(user)
    try:
        kept, gone = await _attach(async_client, token, conversation, "a.txt", "b.txt")
        assert store.count(document_id=gone["document_id"]) > 0
        assert await _slots(user) == 2

        resp = await _delete_doc(
            async_client, token, conversation.id, gone["document_id"]
        )

        assert resp.status_code == 200, resp.text
        assert store.count(document_id=gone["document_id"]) == 0
        assert store.count(document_id=kept["document_id"]) == kept["chunk_count"]
        assert await Document.find_by_id(gone["document_id"]) is None
        assert await _slots(user) == 1
        listed = (await _list(async_client, token, conversation.id)).json()
        assert [d["name"] for d in listed["documents"]] == ["a.txt"]
        assert await Collection.find_by_id(conversation.id) is not None
    finally:
        await _cleanup(user, conversation)


async def test_deleting_the_last_document_drops_the_collection(async_client, store):
    """Otherwise every chat turn in the conversation searches an empty collection."""
    user, token = await create_test_user_and_token()
    conversation = await _conversation(user)
    try:
        [only] = await _attach(async_client, token, conversation, "a.txt")

        resp = await _delete_doc(
            async_client, token, conversation.id, only["document_id"]
        )

        assert resp.status_code == 200, resp.text
        assert await Collection.find_by_id(conversation.id) is None
        assert await _slots(user) == 0

        # The next upload starts over under the same collection id.
        [again] = await _attach(async_client, token, conversation, "b.txt")
        assert store.count(document_id=again["document_id"]) == again["chunk_count"]
        assert await Collection.find_by_id(conversation.id) is not None
    finally:
        await _cleanup(user, conversation)


async def test_racing_deletes_of_one_document_release_one_slot(async_client, store):
    user, token = await create_test_user_and_token()
    conversation = await _conversation(user)
    try:
        _, gone = await _attach(async_client, token, conversation, "a.txt", "b.txt")
        # Both requests pass the lookup before either removes the row.
        store.delete_delay = 0.2

        first, second = await asyncio.gather(
            _delete_doc(async_client, token, conversation.id, gone["document_id"]),
            _delete_doc(async_client, token, conversation.id, gone["document_id"]),
        )

        assert sorted([first.status_code, second.status_code]) == [200, 404]
        assert await _slots(user) == 1
    finally:
        await _cleanup(user, conversation)


async def test_qdrant_failure_keeps_the_document_for_a_retry(async_client, store):
    user, token = await create_test_user_and_token()
    conversation = await _conversation(user)
    try:
        [only] = await _attach(async_client, token, conversation, "a.txt")
        store.delete_error = RuntimeError("qdrant at 10.30.40.5 unavailable")

        failed = await _delete_doc(
            async_client, token, conversation.id, only["document_id"]
        )

        assert failed.status_code == 500
        assert failed.json()["detail"]["code"] == "conversation_document_cleanup_failed"
        assert "10.30.40.5" not in failed.text
        assert await Document.find_by_id(only["document_id"]) is not None
        assert await _slots(user) == 1

        store.delete_error = None
        retried = await _delete_doc(
            async_client, token, conversation.id, only["document_id"]
        )
        assert retried.status_code == 200
        assert store.count(document_id=only["document_id"]) == 0
        assert await _slots(user) == 0
    finally:
        await _cleanup(user, conversation)


async def test_delete_answers_404_for_a_document_outside_the_conversation(
    async_client, store
):
    user, token = await create_test_user_and_token()
    conversation = await _conversation(user)
    other = await _conversation(user)
    try:
        [mine] = await _attach(async_client, token, conversation, "a.txt")
        [theirs] = await _attach(async_client, token, other, "b.txt")

        for document_id in (theirs["document_id"], str(ObjectId()), "not-an-id"):
            resp = await _delete_doc(async_client, token, conversation.id, document_id)
            assert resp.status_code == 404, document_id

        assert store.count(document_id=theirs["document_id"]) == theirs["chunk_count"]
        assert store.count(document_id=mine["document_id"]) == mine["chunk_count"]
        assert await _slots(user) == 2
    finally:
        await _cleanup(user, other)
        await _cleanup(user, conversation)


async def test_delete_on_a_conversation_with_no_documents_answers_404(
    async_client, store
):
    user, token = await create_test_user_and_token()
    conversation = await _conversation(user)
    try:
        resp = await _delete_doc(async_client, token, conversation.id, str(ObjectId()))
        assert resp.status_code == 404
    finally:
        await _cleanup(user, conversation)


async def test_other_users_conversation_answers_404_on_list_and_delete(
    async_client, store
):
    owner, owner_token = await create_test_user_and_token()
    stranger, stranger_token = await create_test_user_and_token()
    conversation = await _conversation(owner)
    try:
        [only] = await _attach(async_client, owner_token, conversation, "a.txt")

        listed = await _list(async_client, stranger_token, conversation.id)
        deleted = await _delete_doc(
            async_client, stranger_token, conversation.id, only["document_id"]
        )

        assert listed.status_code == 404
        assert deleted.status_code == 404
        assert store.count(document_id=only["document_id"]) == only["chunk_count"]
        assert await _slots(owner) == 1
    finally:
        await _cleanup(owner, conversation)
        await cleanup_models([stranger])


async def test_flag_off_answers_404_on_list_and_delete(
    async_client, store, monkeypatch
):
    user, token = await create_test_user_and_token()
    conversation = await _conversation(user)
    try:
        [only] = await _attach(async_client, token, conversation, "a.txt")
        monkeypatch.setattr("src.config.FEATURE_CONVERSATION_DOCUMENTS", False)

        listed = await _list(async_client, token, conversation.id)
        deleted = await _delete_doc(
            async_client, token, conversation.id, only["document_id"]
        )

        assert listed.status_code == 404
        assert deleted.status_code == 404
        assert await Document.find_by_id(only["document_id"]) is not None
    finally:
        await _cleanup(user, conversation)


async def test_list_and_delete_read_on_the_primary(async_client, store, monkeypatch):
    """Right after an upload a secondary may not have the rows yet."""
    user, token = await create_test_user_and_token()
    conversation = await _conversation(user)
    try:
        [only] = await _attach(async_client, token, conversation, "a.txt")
        for model in (Conversation, Collection, Document):
            monkeypatch.setattr(
                model,
                "get_collection",
                classmethod(lambda cls: _LaggingSecondary(get_collection(cls.collection_name))),
            )

        listed = await _list(async_client, token, conversation.id)
        deleted = await _delete_doc(
            async_client, token, conversation.id, only["document_id"]
        )

        assert [d["name"] for d in listed.json()["documents"]] == ["a.txt"]
        assert deleted.status_code == 200, deleted.text
    finally:
        monkeypatch.undo()
        await _cleanup(user, conversation)


async def test_conversation_delete_takes_everything_back_to_the_prior_counter(
    async_client, store, monkeypatch
):
    """The ticket's acceptance: collection, rows, points gone, counter as before."""
    reject_pipeline_updates(monkeypatch)
    user, token = await create_test_user_and_token()
    conversation = await _conversation(user)
    try:
        # Documents in the user's own collections keep their slots.
        await _set_slots(user, 3)
        await _attach(async_client, token, conversation, "a.txt", "b.txt")
        assert await _slots(user) == 5
        assert store.count(collection_id=conversation.id) > 0

        deleted = await async_client.delete(
            f"/conversations/{conversation.id}",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert deleted.status_code == 200, deleted.text
        assert await Collection.find_by_id(conversation.id) is None
        assert await Document.count_documents({"collection_id": conversation.id}) == 0
        assert store.count(collection_id=conversation.id) == 0
        assert await _slots(user) == 3
    finally:
        await _cleanup(user, conversation)


async def test_conversation_delete_with_the_flag_off_still_frees_documents(
    async_client, store, monkeypatch
):
    user, token = await create_test_user_and_token()
    conversation = await _conversation(user)
    try:
        await _attach(async_client, token, conversation, "a.txt")
        monkeypatch.setattr("src.config.FEATURE_CONVERSATION_DOCUMENTS", False)

        deleted = await async_client.delete(
            f"/conversations/{conversation.id}",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert deleted.status_code == 200, deleted.text
        assert store.count(collection_id=conversation.id) == 0
        assert await _slots(user) == 0
    finally:
        await _cleanup(user, conversation)


async def test_retry_after_a_failed_collection_delete_leaves_the_counter_right(
    async_client, store, monkeypatch
):
    """Rows gone means a retry deletes none, so it could never release their slots."""
    user, token = await create_test_user_and_token()
    conversation = await _conversation(user)
    try:
        await _attach(async_client, token, conversation, "a.txt", "b.txt")
        real_delete = Collection.delete

        async def broken_delete(self):
            raise RuntimeError("mongo refused")

        monkeypatch.setattr(Collection, "delete", broken_delete)
        failed = await async_client.delete(
            f"/conversations/{conversation.id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert failed.status_code == 500
        assert failed.json()["detail"]["code"] == "conversation_document_cleanup_failed"

        monkeypatch.setattr(Collection, "delete", real_delete)
        retried = await async_client.delete(
            f"/conversations/{conversation.id}",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert retried.status_code == 200, retried.text
        assert await Collection.find_by_id(conversation.id) is None
        assert await _slots(user) == 0
    finally:
        monkeypatch.undo()
        await _cleanup(user, conversation)


async def test_a_conversation_attached_before_its_collection_shared_its_id(
    async_client, store, monkeypatch
):
    """Older conversations keep a collection whose _id is not the conversation id."""
    reject_pipeline_updates(monkeypatch)
    user, token = await create_test_user_and_token()
    conversation = await _conversation(user)
    await Message.get_collection().insert_one({"conversation_id": conversation.id})
    try:
        collection = await Collection.create(
            user_id=user.id, name="Attached documents", conversation_id=conversation.id
        )
        assert collection.id != conversation.id
        rows = [
            await Document.create(
                user_id=user.id,
                collection_id=collection.id,
                name=name,
                chunk_count=1,
            )
            for name in ("old-a.txt", "old-b.txt")
        ]
        store.points = [
            {"user_id": user.id, "collection_id": collection.id, "document_id": row.id}
            for row in rows
        ]
        await _set_slots(user, 2)

        listed = (await _list(async_client, token, conversation.id)).json()
        assert [d["name"] for d in listed["documents"]] == ["old-a.txt", "old-b.txt"]

        one = await _delete_doc(async_client, token, conversation.id, rows[0].id)
        assert one.status_code == 200, one.text
        assert store.count(document_id=rows[0].id) == 0
        assert store.count(document_id=rows[1].id) == 1
        assert await _slots(user) == 1
        # Removing a document leaves the conversation's messages alone.
        assert (
            await Message.count_documents({"conversation_id": conversation.id}) == 1
        )

        everything = await async_client.delete(
            f"/conversations/{conversation.id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert everything.status_code == 200, everything.text
        assert await Collection.find_by_id(collection.id) is None
        assert await Document.count_documents({"collection_id": collection.id}) == 0
        assert store.count(collection_id=collection.id) == 0
        assert await _slots(user) == 0
    finally:
        await Message.delete_many({"conversation_id": conversation.id})
        await _cleanup(user, conversation)
