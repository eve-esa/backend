"""Background document ingest through eve-esa/document-parser: 202 now, a job on parse-jobs,
the parser's result on parse-results, then chunks with page numbers, or a clean rollback."""

import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from bson import ObjectId

from src.database.models.collection import Collection
from src.database.models.document import Document
from src.database.models.document_ingest_job import DocumentIngestJob
from src.database.models.user import User
from src.services import document_ingest
from src.services.document_ingest import claim_next_job, handle_parse_result, process_job
from tests.utils.cleaner import cleanup_models
from tests.utils.fake_storage import FakeStorage
from tests.utils.utils import create_test_user_and_token

VSM = "src.core.vector_store_manager.VectorStoreManager"
PARSED = (Path(__file__).parent / "fixtures" / "docling_document.json").read_bytes()
# tests/fixtures/docling_document.json: 2 pages, one table.
# ATTACHED-DOCS-RETRIEVAL (2026-11-27): its short text pieces are merged (MIN_CHUNK_WORDS),
# 4 chunks before. Revert: PARSED_CHUNKS = 4 and the chunk cap below back to 2.
PARSED_CHUNKS = 2


class FakeWorld:
    def __init__(self):
        self.points = []  # chunk metadata
        self.fail_times = 0
        self.adds = 0
        self.sent = []  # parse jobs sent to SQS


@pytest.fixture
def world(monkeypatch):
    fake = FakeWorld()
    storage = FakeStorage()

    async def add_document_list(self, document_list, *, user_id, collection_id):
        fake.adds += 1
        # Written before failing: a real upload can die between batches.
        for doc in document_list:
            fake.points.append(doc.metadata)
        if fake.fail_times > 0:
            fake.fail_times -= 1
            raise RuntimeError("provider said something about the user's text")
        return []

    def delete_private_docs(self, *, user_id, collection_id, metadata=None, **_):
        target = (metadata or {}).get("metadata.document_id")
        fake.points = [p for p in fake.points if p.get("document_id") != target]

    async def send_jobs(bodies):
        fake.sent.extend(json.loads(b) for b in bodies)

    monkeypatch.setattr(f"{VSM}.ensure_private_collection", lambda self: None)
    monkeypatch.setattr(f"{VSM}.add_document_list", add_document_list)
    monkeypatch.setattr(f"{VSM}.delete_private_docs", delete_private_docs)
    monkeypatch.setattr("src.services.document_ingest.storage_service", storage)
    monkeypatch.setattr("src.services.document_ingest.parser_queue.send_jobs", send_jobs)
    monkeypatch.setattr("src.config.FEATURE_BACKGROUND_INGEST", True)
    monkeypatch.setattr("src.config.FEATURE_CONVERSATION_DOCUMENTS", True)
    fake.storage = storage
    return fake


@pytest.fixture(autouse=True)
async def _empty_queue():
    await DocumentIngestJob.delete_many({})
    yield
    await DocumentIngestJob.delete_many({})


async def _slots(user) -> int:
    row = await User.get_collection().find_one({"_id": ObjectId(user.id)})
    return int(row.get("private_document_count") or 0)


async def _setup(async_client):
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    created = await async_client.post("/collections", headers=headers, json={"name": "c"})
    return user, headers, created.json()["id"]


async def _upload(async_client, headers, coll, name="a.pdf", body=b"%PDF-1.4 a paper"):
    return await async_client.post(
        f"/collections/{coll}/documents",
        headers=headers,
        files={"files": (name, io.BytesIO(body), "application/pdf")},
        data={"metadata_names": name},
    )


async def _send(world) -> dict:
    """The worker claims the job and hands it to the parser; returns the one job message."""
    await process_job(await claim_next_job())
    [message] = world.sent
    return message


async def _parser_answers(world, message, status="ready", error_code=None):
    """What document-parser does: write document.json next to the original, report on parse-results."""
    folder = message["s3_key"].rsplit("/", 1)[0] + "/"
    result = {"version": 1, "document_id": message["document_id"], "status": status}
    if status == "ready":
        await world.storage.put_object(folder + "document.json", PARSED, "application/json")
        result.update(
            engine="blablador",
            pages=2,
            seconds=12.3,
            outputs={"json": folder + "document.json", "pages": folder + "pages.json"},
        )
    else:
        result.update(error_code=error_code, error="details for the logs")
    await handle_parse_result(result)


async def _job(async_client, headers, job_id) -> dict:
    return (await async_client.get(f"/document-ingest-jobs/{job_id}", headers=headers)).json()


async def _cleanup(user, coll):
    await Document.delete_many({"collection_id": coll})
    await Collection.delete_many({"_id": ObjectId(coll)})
    await cleanup_models([user])


async def test_upload_answers_202_and_nothing_is_embedded_yet(async_client, world):
    user, headers, coll = await _setup(async_client)
    try:
        resp = await _upload(async_client, headers, coll)

        assert resp.status_code == 202, resp.text
        body = resp.json()
        assert body["status"] == "queued"
        assert world.adds == 0 and world.sent == []
        [row] = await Document.find_all(filter_dict={"collection_id": coll})
        assert row.status == "queued"
        assert list(world.storage.objects) == [row.storage_key]
        assert row.storage_key == f"users/{user.id}/documents/{row.id}/original.pdf"
        assert await _slots(user) == 1
        assert (await _job(async_client, headers, body["job_id"]))["status"] == "queued"
    finally:
        await _cleanup(user, coll)


async def test_the_worker_sends_one_job_per_document_in_the_contract(async_client, world):
    user, headers, coll = await _setup(async_client)
    try:
        body = (await _upload(async_client, headers, coll)).json()
        message = await _send(world)

        [row] = await Document.find_all(filter_dict={"collection_id": coll})
        assert message["document_id"] == row.id and message["user_id"] == user.id
        assert message["s3_key"] == row.storage_key
        assert message["content_type"] == "application/pdf"
        assert (await _job(async_client, headers, body["job_id"]))["status"] == "parsing"
        assert world.adds == 0
    finally:
        await _cleanup(user, coll)


async def test_a_ready_result_is_chunked_with_page_numbers_and_embedded(async_client, world):
    user, headers, coll = await _setup(async_client)
    try:
        body = (await _upload(async_client, headers, coll)).json()
        await _parser_answers(world, await _send(world))

        assert (await _job(async_client, headers, body["job_id"]))["status"] == "done"
        [row] = await Document.find_all(filter_dict={"collection_id": coll})
        assert row.status == "ready" and row.pages == 2 and row.parser == "blablador"
        assert row.chunk_count == len(world.points) == PARSED_CHUNKS
        assert sorted({p["page"] for p in world.points}) == [1, 2]
        assert [p["element_type"] for p in world.points].count("table") == 1  # kept whole
        assert await _slots(user) == 1
        assert await claim_next_job() is None
    finally:
        await _cleanup(user, coll)


async def test_a_failed_result_rolls_everything_back(async_client, world):
    user, headers, coll = await _setup(async_client)
    try:
        body = (await _upload(async_client, headers, coll)).json()
        await _parser_answers(world, await _send(world), status="failed", error_code="ocr_rate_limited")

        job = await _job(async_client, headers, body["job_id"])
        assert job["status"] == "failed" and job["error_code"] == "document_parser_unavailable"
        assert await Document.count_documents({"collection_id": coll}) == 0
        assert await _slots(user) == 0
        assert world.storage.objects == {} and world.points == []
    finally:
        await _cleanup(user, coll)


async def test_an_unreadable_document_fails_as_unreadable(async_client, world):
    user, headers, coll = await _setup(async_client)
    try:
        body = (await _upload(async_client, headers, coll)).json()
        await _parser_answers(world, await _send(world), status="failed", error_code="ocr_unreadable")

        assert (await _job(async_client, headers, body["job_id"]))["error_code"] == "document_unreadable"
        assert await _slots(user) == 0
    finally:
        await _cleanup(user, coll)


async def test_a_result_received_twice_is_embedded_once(async_client, world):
    user, headers, coll = await _setup(async_client)
    try:
        await _upload(async_client, headers, coll)
        message = await _send(world)
        await _parser_answers(world, message)
        await _parser_answers(world, message)

        assert world.adds == 1 and len(world.points) == PARSED_CHUNKS
    finally:
        await _cleanup(user, coll)


async def test_an_embedding_failure_is_retried_without_duplicate_points(async_client, world, caplog):
    caplog.set_level("WARNING", logger=document_ingest.__name__)
    user, headers, coll = await _setup(async_client)
    try:
        await _upload(async_client, headers, coll)
        message = await _send(world)
        world.fail_times = 1
        with pytest.raises(RuntimeError):  # the result stays on the queue for another try
            await _parser_answers(world, message)
        await _parser_answers(world, message)

        [row] = await Document.find_all(filter_dict={"collection_id": coll})
        assert row.status == "ready"
        assert len(world.points) == row.chunk_count == PARSED_CHUNKS
        assert not any("user's text" in r.getMessage() for r in caplog.records)
    finally:
        await _cleanup(user, coll)


async def test_no_result_before_the_deadline_fails_as_parser_unavailable(async_client, world):
    user, headers, coll = await _setup(async_client)
    try:
        body = (await _upload(async_client, headers, coll)).json()
        await _send(world)
        await DocumentIngestJob.get_collection().update_one(
            {"_id": ObjectId(body["job_id"])},
            {"$set": {"lease_expires_at": datetime.now(timezone.utc) - timedelta(seconds=1)}},
        )
        await process_job(await claim_next_job())

        job = await _job(async_client, headers, body["job_id"])
        assert job["status"] == "failed" and job["error_code"] == "document_parser_unavailable"
        assert await _slots(user) == 0 and world.storage.objects == {}
    finally:
        await _cleanup(user, coll)


async def test_a_dead_workers_job_is_claimed_again_after_its_lease(async_client, world):
    user, headers, coll = await _setup(async_client)
    try:
        await _upload(async_client, headers, coll)
        job = await claim_next_job()
        assert await claim_next_job() is None  # leased

        await DocumentIngestJob.get_collection().update_one(
            {"_id": ObjectId(job.id)},
            {"$set": {"lease_expires_at": datetime.now(timezone.utc) - timedelta(seconds=1)}},
        )
        again = await claim_next_job()

        assert again.id == job.id and again.attempts == 2
    finally:
        await _cleanup(user, coll)


async def test_a_document_deleted_while_queued_is_never_sent(async_client, world):
    user, headers, coll = await _setup(async_client)
    try:
        await _upload(async_client, headers, coll)
        await Document.delete_many({"collection_id": coll})

        await process_job(await claim_next_job())

        assert world.sent == [] and world.adds == 0
        assert world.storage.objects == {}
    finally:
        await _cleanup(user, coll)


async def test_conversation_upload_uses_the_task_layout_and_keeps_its_chunk_cap(
    async_client, world, monkeypatch
):
    monkeypatch.setattr("src.routers.document.MAX_CONVERSATION_DOCUMENT_CHUNKS", 1)  # ATTACHED-DOCS-RETRIEVAL: was 2
    user, token = await create_test_user_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    conversation = (
        await async_client.post("/conversations", headers=headers, json={"name": "c"})
    ).json()["id"]
    try:
        resp = await async_client.post(
            f"/conversations/{conversation}/documents",
            headers=headers,
            files={"files": ("paper.pdf", io.BytesIO(b"%PDF-1.4 a paper"), "application/pdf")},
        )
        assert resp.status_code == 202, resp.text
        message = await _send(world)
        document_id = message["document_id"]
        assert message["s3_key"] == (
            f"users/{user.id}/conversations/{conversation}/uploads/{document_id}/original.pdf"
        )
        assert message["conversation_id"] == conversation

        await _parser_answers(world, message)  # 2 chunks over a cap of 1

        job = await _job(async_client, headers, resp.json()["job_id"])
        assert job["error_code"] == "document_too_many_chunks"
        assert world.adds == 0 and await _slots(user) == 0
        # The failed first upload leaves no empty collection for chat to search.
        assert await Collection.count_documents({"conversation_id": conversation}) == 0
    finally:
        await async_client.delete(f"/conversations/{conversation}", headers=headers)
        await cleanup_models([user])


async def test_another_users_job_answers_404(async_client, world):
    user, headers, coll = await _setup(async_client)
    other, other_token = await create_test_user_and_token()
    try:
        body = (await _upload(async_client, headers, coll)).json()
        resp = await async_client.get(
            f"/document-ingest-jobs/{body['job_id']}",
            headers={"Authorization": f"Bearer {other_token}"},
        )
        assert resp.status_code == 404
    finally:
        await _cleanup(user, coll)
        await cleanup_models([other])


async def test_flag_off_keeps_the_synchronous_route(async_client, world, monkeypatch):
    monkeypatch.setattr("src.config.FEATURE_BACKGROUND_INGEST", False)
    user, headers, coll = await _setup(async_client)
    try:
        resp = await _upload(
            async_client, headers, coll, name="a.txt", body=b"EVE reads attached text."
        )
        assert resp.status_code == 200, resp.text
        assert "job_id" not in resp.json()
        assert world.adds == 1 and world.sent == []
        assert await DocumentIngestJob.count_documents({}) == 0
    finally:
        await _cleanup(user, coll)
