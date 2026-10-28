"""POST /conversations/{id}/documents: documents attached to one conversation.

Parsing and chunking run for real; only the Qdrant boundary is replaced by an
in-memory store, so the number of points written can be compared with the
chunk_count the endpoint reports.
"""

import asyncio
import io

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
from src.database.models.user import User
from src.routers.message import apply_private_collections_to_request
from src.schemas.generation_request import GenerationRequest
from tests.utils.cleaner import cleanup_models
from tests.utils.utils import create_test_user_and_token


class FakeVectorStore:
    """Points by (user_id, collection_id), each carrying its document_id."""

    def __init__(self):
        self.points = []
        self.delay = 0.0

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
        if fake.delay:
            await asyncio.sleep(fake.delay)
        for doc in document_list:
            fake.points.append(
                {
                    "user_id": user_id,
                    "collection_id": collection_id,
                    "document_id": doc.metadata.get("document_id"),
                }
            )
        return [str(i) for i in range(len(document_list))]

    def delete_private_docs(self, *, user_id, collection_id, metadata=None, **_):
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
        assert await _slots(user) == 0
        assert await Document.count_documents({"user_id": user.id}) == 0
        assert store.count() == 0
    finally:
        await _cleanup(user, conversation)


async def test_other_users_conversation_is_forbidden(async_client, store):
    owner, _ = await create_test_user_and_token()
    other, other_token = await create_test_user_and_token()
    conversation = await _conversation(owner)
    try:
        resp = await async_client.post(
            f"/conversations/{conversation.id}/documents",
            headers={"Authorization": f"Bearer {other_token}"},
            files=[_txt("a.txt")],
        )
        assert resp.status_code == 403
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

        elsewhere = GenerationRequest(query="what does the file say?")
        await apply_private_collections_to_request(
            elsewhere, user.id, other_conversation.id
        )
        assert collection_id not in elsewhere.collection_ids

        monkeypatch.setattr("src.config.FEATURE_CONVERSATION_DOCUMENTS", False)
        flag_off = GenerationRequest(query="what does the file say?")
        await apply_private_collections_to_request(flag_off, user.id, conversation.id)
        assert collection_id not in flag_off.collection_ids
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
