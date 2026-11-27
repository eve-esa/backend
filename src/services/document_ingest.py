"""Parse and embed uploads outside the request (FEATURE_BACKGROUND_INGEST).

A request stores each original in object storage, writes one job and answers 202; the
document is ``queued``. A worker claims the job from Mongo with a lease and sends one message
per document to eve-esa/document-parser through SQS (parse-jobs), and the job waits in
``parsing``. The parser has every page read by the OCR model (Blablador alias-ocr) and writes
document.json, document.md and pages.json next to the original. A result consumer
(parse-results) chunks each document.json by section, tables whole, every chunk with its page
numbers, embeds it, and marks the document ``ready``; a failed result marks it failed. The job
closes once every file is ready or failed. A job with no final result after
PARSE_TIMEOUT_SECONDS fails as document_parser_unavailable.

Originals live under the task's layout:
``users/<user>/conversations/<conversation>/uploads/<document>/original.<ext>`` for a
conversation's documents, ``users/<user>/documents/<document>/`` for a private collection's.

A worker that dies mid job loses its lease and the job runs again; every attempt first removes
the points of the one before, so a retry cannot duplicate them. Slots and document rows stay
reserved while a job is pending, and go back only when it fails for good, so
``private_document_count`` always matches the document rows.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import traceback
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import List, Optional

import anyio
from bson import ObjectId
from fastapi import UploadFile
from langchain_core.documents import Document as ChunkDocument
from pymongo import ReturnDocument

from src import config
from src.core.vector_store_manager import VectorStoreManager
from src.database.models.collection import Collection
from src.database.models.document import Document
from src.database.models.document_ingest_job import (
    JOB_DONE,
    JOB_FAILED,
    JOB_PARSING,
    JOB_QUEUED,
    JOB_RUNNING,
    PARSE_DONE,
    PARSE_FAILED,
    PARSE_PROCESSING,
    PARSE_QUEUED,
    DocumentIngestJob,
    IngestJobFile,
)
from src.services import document_parser_queue as parser_queue
from src.services.conversation_documents import delete_conversation_collection_if_empty
from src.services.docling_chunks import chunks_from_docling, page_count
from src.services.private_document_limit import release_private_document_slots
from src.services.storage import storage_service

logger = logging.getLogger(__name__)

DOCUMENT_QUEUED = "queued"
DOCUMENT_READY = "ready"

MAX_ATTEMPTS = 3
# The lease outlives the job timeout: a job is never re-claimed while it can run.
JOB_TIMEOUT_S = config.INGEST_JOB_TIMEOUT_SECONDS
LEASE_S = JOB_TIMEOUT_S + 60
RETRY_BACKOFF_S = 15
QDRANT_TIMEOUT_S = 30.0
ERROR_INTERNAL = "document_ingest_failed"
ERROR_UNREADABLE = "document_unreadable"
ERROR_TOO_MANY_CHUNKS = "document_too_many_chunks"
ERROR_PARSER_UNAVAILABLE = "document_parser_unavailable"
ERROR_TOO_MANY_PAGES = "document_too_many_pages"

# document-parser error codes -> ours (stable, safe to show).
_PARSER_ERRORS = {
    "unsupported_type": ERROR_UNREADABLE,
    "file_missing": ERROR_UNREADABLE,
    "corrupt_file": ERROR_UNREADABLE,
    "encrypted_file": ERROR_UNREADABLE,
    "file_too_large": ERROR_UNREADABLE,
    "ocr_unreadable": ERROR_UNREADABLE,
    "too_many_pages": ERROR_TOO_MANY_PAGES,
    # The OCR service (Blablador alias-ocr) was rate limited or down: worth trying again later.
    "ocr_rate_limited": ERROR_PARSER_UNAVAILABLE,
    "ocr_unavailable": ERROR_PARSER_UNAVAILABLE,
}
# When every file of a job failed, the job reports the most specific reason.
_ERROR_PRIORITY = [
    ERROR_TOO_MANY_CHUNKS,
    ERROR_TOO_MANY_PAGES,
    ERROR_UNREADABLE,
    ERROR_PARSER_UNAVAILABLE,
    ERROR_INTERNAL,
]



def _now() -> datetime:
    return datetime.now(timezone.utc)


def document_prefix(user_id: str, document_id: str, conversation_id: Optional[str] = None) -> str:
    """Where a document's original and the parser's files live in object storage."""
    if conversation_id:
        return f"users/{user_id}/conversations/{conversation_id}/uploads/{document_id}/"
    return f"users/{user_id}/documents/{document_id}/"


def _storage_key(
    user_id: str, document_id: str, filename: str, conversation_id: Optional[str] = None
) -> str:
    extension = os.path.splitext(filename or "")[1].lower() or ".bin"
    return f"{document_prefix(user_id, document_id, conversation_id)}original{extension}"


async def _delete_prefix(prefix: str, document_id: str) -> None:
    try:
        for record in await storage_service.list_keys(prefix, 50):
            await _delete_object(record["key"])
    except Exception:
        logger.warning("document_ingest.objects_delete_failed document_id=%s", document_id)


async def delete_document_objects(
    user_id: str, document_id: str, conversation_id: Optional[str] = None
) -> None:
    """Best effort: remove a document's original and the parser's files."""
    await _delete_prefix(document_prefix(user_id, document_id, conversation_id), document_id)


async def enqueue_upload(
    *,
    user_id: str,
    collection_id: str,
    files: List[UploadFile],
    documents: List[Document],
    embeddings_model: str,
    chunk_size: int,
    chunk_overlap: int,
    source_urls: Optional[List[str]] = None,
    max_chunks: Optional[int] = None,
) -> DocumentIngestJob:
    """Store the files and write the job. The caller already holds the slots."""
    job_files: List[IngestJobFile] = []
    collection = await Collection.find_by_id_on_primary(collection_id)
    conversation_id = collection.conversation_id if collection is not None else None
    try:
        for index, (file, document) in enumerate(zip(files, documents)):
            key = _storage_key(user_id, document.id, file.filename or document.name, conversation_id)
            await file.seek(0)
            body = await file.read()
            await storage_service.put_object(
                key, body, file.content_type or "application/octet-stream"
            )
            job_files.append(
                IngestJobFile(
                    document_id=document.id,
                    filename=file.filename or document.name,
                    storage_key=key,
                    source_url=(source_urls[index] if source_urls else "") or "",
                )
            )
            await Document.get_collection().update_one(
                {"_id": ObjectId(document.id)}, {"$set": {"storage_key": key}}
            )
        return await DocumentIngestJob.create(
            user_id=user_id,
            collection_id=collection_id,
            files=job_files,
            embeddings_model=embeddings_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_chunks=max_chunks,
            available_at=_now(),
        )
    except Exception:
        for stored in job_files:
            await _delete_object(stored.storage_key)
        raise


async def claim_next_job() -> Optional[DocumentIngestJob]:
    """Take the oldest job that is due, or one whose worker lost its lease."""
    now = _now()
    doc = await DocumentIngestJob.get_primary_collection().find_one_and_update(
        {
            "$or": [
                {"status": JOB_QUEUED, "available_at": {"$lte": now}},
                {"status": JOB_RUNNING, "lease_expires_at": {"$lt": now}},
                # Sent to the parser, no final result before its deadline.
                {"status": JOB_PARSING, "lease_expires_at": {"$lt": now}},
            ]
        },
        {
            "$set": {
                "status": JOB_RUNNING,
                "lease_expires_at": now + timedelta(seconds=LEASE_S),
            },
            "$inc": {"attempts": 1},
        },
        sort=[("available_at", 1)],
        return_document=ReturnDocument.AFTER,
    )
    return DocumentIngestJob.from_dict(doc) if doc else None


async def _delete_object(key: str) -> None:
    try:
        await storage_service.delete_object(key)
    except Exception:
        logger.warning("document_ingest.object_delete_failed key=%s", key, exc_info=True)


async def _delete_points(job: DocumentIngestJob, document_ids: List[str]) -> None:
    store = VectorStoreManager(qdrant_timeout_s=QDRANT_TIMEOUT_S)
    for document_id in document_ids:
        await anyio.to_thread.run_sync(
            partial(
                store.delete_private_docs,
                user_id=job.user_id,
                collection_id=job.collection_id,
                metadata={"metadata.document_id": document_id},
            )
        )


async def _existing_document_ids(job: DocumentIngestJob) -> List[str]:
    """Rows still present: the user may have deleted them while the job waited."""
    wanted = [ObjectId(file.document_id) for file in job.files]
    rows = await Document.get_primary_collection().find(
        {"_id": {"$in": wanted}}, {"_id": 1}
    ).to_list(length=None)
    return [str(row["_id"]) for row in rows]


async def _finish(job: DocumentIngestJob, status: str, **fields) -> None:
    """Close the job. The originals stay: they belong to the documents now."""
    await DocumentIngestJob.get_collection().update_one(
        {"_id": ObjectId(job.id)},
        {"$set": {"status": status, "finished_at": _now(), "lease_expires_at": None, **fields}},
    )


async def _drop_documents(job: DocumentIngestJob, document_ids: List[str]) -> None:
    """Remove documents that will never be ready, with their points and slots."""
    if not document_ids:
        return
    await _delete_points(job, document_ids)
    deleted = await Document.delete_many(
        {"_id": {"$in": [ObjectId(document_id) for document_id in document_ids]}}
    )
    await release_private_document_slots(job.user_id, deleted)
    folders = {f.document_id: f.storage_key.rsplit("/", 1)[0] + "/" for f in job.files}
    for document_id in document_ids:
        if document_id in folders:
            await _delete_prefix(folders[document_id], document_id)


async def _fail(job: DocumentIngestJob, code: str) -> None:
    await _drop_documents(job, await _existing_document_ids(job))
    for file in job.files:
        await _delete_object(file.storage_key)
    await _finish(job, JOB_FAILED, error_code=code)
    collection = await Collection.find_by_id_on_primary(job.collection_id)
    if collection is not None and collection.conversation_id:
        await delete_conversation_collection_if_empty(collection)


async def _ingest_via_queue(job: DocumentIngestJob, document_ids: List[str]) -> None:
    """Hand the files to document-parser. The job then waits in parsing for the results."""
    if job.parse_state:
        # Claimed again: it was sent before and no final result came back in time.
        logger.warning("document_ingest.parser_timeout job_id=%s", job.id)
        await _fail(job, ERROR_PARSER_UNAVAILABLE)
        return
    files = [file for file in job.files if file.document_id in document_ids]
    # Sent before the job is marked: a failed send is retried like any other attempt.
    await parser_queue.send_jobs(
        [
            parser_queue.job_body(
                document_id=file.document_id,
                key=file.storage_key,
                filename=file.filename,
                user_id=job.user_id,
                collection_id=job.collection_id,
            )
            for file in files
        ]
    )
    await DocumentIngestJob.get_collection().update_one(
        {"_id": ObjectId(job.id)},
        {
            "$set": {
                "status": JOB_PARSING,
                "lease_expires_at": _now() + timedelta(seconds=config.PARSE_TIMEOUT_SECONDS),
                "parse_state": {file.document_id: PARSE_QUEUED for file in files},
                "parse_errors": {},
            }
        },
    )
    logger.info("document_ingest.sent_to_parser job_id=%s files=%s", job.id, len(files))


async def _ingest(job: DocumentIngestJob, document_ids: List[str]) -> None:
    await _ingest_via_queue(job, document_ids)


async def process_job(job: DocumentIngestJob) -> None:
    """Run one claimed job; a transient failure goes back in the queue."""
    try:
        document_ids = await _existing_document_ids(job)
        # Deleted while the job waited: their stored files go with them.
        for file in job.files:
            if file.document_id not in document_ids:
                await _delete_object(file.storage_key)
        if not document_ids:
            await _finish(job, JOB_DONE)
            return
        # A previous attempt may have written some batches.
        if job.attempts > 1:
            await _delete_points(job, document_ids)
        await _ingest(job, document_ids)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        # The class only: provider and Mongo errors can quote the user's text.
        logger.warning(
            "document_ingest.attempt_failed job_id=%s attempt=%s error=%s",
            job.id,
            job.attempts,
            type(exc).__name__,
        )
        if job.attempts >= MAX_ATTEMPTS:
            logger.error("document_ingest.failed job_id=%s", job.id)
            try:
                await _fail(job, ERROR_INTERNAL)
            except Exception as cleanup:
                logger.error(
                    "document_ingest.cleanup_failed job_id=%s error=%s",
                    job.id,
                    type(cleanup).__name__,
                )
            return
        await DocumentIngestJob.get_collection().update_one(
            {"_id": ObjectId(job.id)},
            {
                "$set": {
                    "status": JOB_QUEUED,
                    "lease_expires_at": None,
                    "available_at": _now() + timedelta(seconds=RETRY_BACKOFF_S * job.attempts),
                }
            },
        )


# --- Results from document-parser (parse-results) -----------------------------------------


def _parsed_key_allowed(file: IngestJobFile, key: Optional[str]) -> bool:
    """The result may only point at the document's own folder."""
    folder = file.storage_key.rsplit("/", 1)[0] + "/"
    return bool(key) and key.startswith(folder) and ".." not in key.split("/")


def _chunk_parsed(file: IngestJobFile, parsed: dict, engine: str, chunk_size: int) -> List[ChunkDocument]:
    extension = os.path.splitext(file.filename)[1].lower().lstrip(".")
    uploaded = _now().isoformat()
    documents = []
    for chunk in chunks_from_docling(parsed, max_chars=max(chunk_size, 1500)):
        heading = file.filename + (f" > {chunk.section}" if chunk.section else "")
        documents.append(
            ChunkDocument(
                page_content=f"{heading}\n{chunk.text}",
                metadata={
                    "source": file.source_url,
                    "source_name": file.filename,
                    "filename": file.filename,
                    "file_type": extension,
                    "upload_time": uploaded,
                    "document_id": file.document_id,
                    "page": chunk.page,
                    "pages": chunk.pages,
                    "section": chunk.section,
                    "element_type": chunk.element_type,
                    "parser": engine,
                },
            )
        )
    return documents


async def _mark(job: DocumentIngestJob, document_id: str, state: str, **fields) -> bool:
    """Move one file on, unless it is already final (a result can arrive twice)."""
    result = await DocumentIngestJob.get_collection().update_one(
        {
            "_id": ObjectId(job.id),
            "status": JOB_PARSING,
            f"parse_state.{document_id}": {"$nin": [PARSE_DONE, PARSE_FAILED]},
        },
        {"$set": {f"parse_state.{document_id}": state, **fields}},
    )
    return result.modified_count == 1


async def _embed_parsed(job: DocumentIngestJob, file: IngestJobFile, result: dict) -> None:
    document_id = file.document_id
    outputs = result.get("outputs") or {}
    if not await Document.get_primary_collection().find_one({"_id": ObjectId(document_id)}, {"_id": 1}):
        await _mark(job, document_id, PARSE_FAILED)  # deleted while parsing
        return
    key = outputs.get("json")
    if not _parsed_key_allowed(file, key):
        logger.warning("document_ingest.parsed_key_rejected document_id=%s", document_id)
        await _mark(job, document_id, PARSE_FAILED, **{f"parse_errors.{document_id}": ERROR_INTERNAL})
        return
    stored = await storage_service.get_object(key)
    parsed = json.loads(b"".join([c async for c in storage_service.stream_body(stored["Body"])]))
    engine = result.get("engine") or "document-parser"
    documents = _chunk_parsed(file, parsed, engine, job.chunk_size)
    if not documents:
        await _mark(job, document_id, PARSE_FAILED, **{f"parse_errors.{document_id}": ERROR_UNREADABLE})
        return
    already = sum(int(n) for d, n in job.chunk_counts.items() if d != document_id)
    if job.max_chunks and already + len(documents) > job.max_chunks:
        await _mark(job, document_id, PARSE_FAILED, **{f"parse_errors.{document_id}": ERROR_TOO_MANY_CHUNKS})
        return

    store = VectorStoreManager(embeddings_model=job.embeddings_model, qdrant_timeout_s=QDRANT_TIMEOUT_S)
    # The collection first: on a fresh Qdrant the delete below answers 404 and the result would
    # fail on every retry.
    await anyio.to_thread.run_sync(store.ensure_private_collection)
    # A result handled twice must not leave two copies of the points.
    await _delete_points(job, [document_id])
    with anyio.fail_after(JOB_TIMEOUT_S):
        await store.add_document_list(documents, user_id=job.user_id, collection_id=job.collection_id)
    pages = result.get("pages") or page_count(parsed)
    await Document.get_collection().update_one(
        {"_id": ObjectId(document_id)},
        {"$set": {"status": DOCUMENT_READY, "chunk_count": len(documents), "pages": pages, "parser": engine}},
    )
    await _mark(job, document_id, PARSE_DONE, **{f"chunk_counts.{document_id}": len(documents)})
    logger.info(
        "document_ingest.parsed document_id=%s pages=%s chunks=%s seconds=%s engine=%s flagged=%s",
        document_id, pages, len(documents), result.get("seconds"), engine,
        len(result.get("flagged_pages") or []),
    )


async def _close_parsed_job(job_id: str) -> None:
    """Finish the job once no file is still with the parser."""
    row = await DocumentIngestJob.get_primary_collection().find_one({"_id": ObjectId(job_id)})
    if row is None or row.get("status") != JOB_PARSING:
        return
    job = DocumentIngestJob.from_dict(row)
    existing = await _existing_document_ids(job)
    states = job.parse_state
    if any(states.get(d) in (PARSE_QUEUED, PARSE_PROCESSING) for d in existing):
        return
    # One closer only: the first consumer to take the job out of parsing.
    taken = await DocumentIngestJob.get_collection().update_one(
        {"_id": ObjectId(job_id), "status": JOB_PARSING},
        {"$set": {"status": JOB_RUNNING, "lease_expires_at": _now() + timedelta(seconds=LEASE_S)}},
    )
    if taken.modified_count != 1:
        return
    done = [d for d in existing if states.get(d) == PARSE_DONE]
    if not done:
        errors = set(job.parse_errors.values())
        await _fail(job, next((code for code in _ERROR_PRIORITY if code in errors), ERROR_UNREADABLE))
        return
    await _drop_documents(job, [d for d in existing if d not in done])
    gone = [d for d in job.chunk_counts if d not in existing]
    if gone:
        await _delete_points(job, gone)
    await _finish(job, JOB_DONE, chunk_counts={d: job.chunk_counts[d] for d in done if d in job.chunk_counts})


async def handle_parse_result(result: dict) -> None:
    """One message from parse-results. Raises to leave it on the queue for another try."""
    document_id = result["document_id"]
    row = await DocumentIngestJob.get_primary_collection().find_one(
        {"status": JOB_PARSING, "files.document_id": document_id}
    )
    if row is None:
        return  # finished, timed out or deleted: nothing waits for this result
    job = DocumentIngestJob.from_dict(row)
    file = next(f for f in job.files if f.document_id == document_id)
    status = result["status"]
    if status == "processing":
        await DocumentIngestJob.get_collection().update_one(
            {"_id": ObjectId(job.id), f"parse_state.{document_id}": PARSE_QUEUED},
            {"$set": {f"parse_state.{document_id}": PARSE_PROCESSING}},
        )
        return
    if job.parse_state.get(document_id) in (PARSE_DONE, PARSE_FAILED):
        return
    if status == "failed":
        code = _PARSER_ERRORS.get(result.get("error_code") or "", ERROR_INTERNAL)
        logger.warning(
            "document_ingest.parser_failed document_id=%s code=%s", document_id, result.get("error_code")
        )
        await _mark(job, document_id, PARSE_FAILED, **{f"parse_errors.{document_id}": code})
    else:
        await _embed_parsed(job, file, result)
    await _close_parsed_job(job.id)


async def _result_loop() -> None:
    while True:
        try:
            messages = await parser_queue.receive_results(wait_seconds=10)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("document_ingest.results_receive_failed error=%s", type(exc).__name__)
            await asyncio.sleep(5)
            continue
        for receipt, body in messages:
            result = parser_queue.parse_result(body)
            try:
                if result is not None:
                    await handle_parse_result(result)
                else:
                    logger.warning("document_ingest.result_ignored")
                await parser_queue.delete_result(receipt)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Left on the queue: it comes back after the visibility timeout. The class and
                # the line it was raised on, never the message: it can quote the user's text.
                frames = traceback.extract_tb(exc.__traceback__)
                where = f"{os.path.basename(frames[-1].filename)}:{frames[-1].lineno}" if frames else "?"
                logger.warning(
                    "document_ingest.result_failed document_id=%s error=%s at=%s",
                    (result or {}).get("document_id"), type(exc).__name__, where,
                )


async def _worker_loop(poll_s: float) -> None:
    while True:
        try:
            job = await claim_next_job()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("document_ingest.claim_failed error=%s", type(exc).__name__)
            job = None
        if job is None:
            # Jitter: workers across tasks should not poll Mongo in step.
            await asyncio.sleep(poll_s * random.uniform(0.5, 1.5))
            continue
        await process_job(job)


def start_workers(concurrency: Optional[int] = None) -> List[asyncio.Task]:
    """Start the claim loops of this process; none when the feature is off."""
    count = config.INGEST_WORKER_CONCURRENCY if concurrency is None else concurrency
    if not config.FEATURE_BACKGROUND_INGEST or count <= 0:
        return []
    tasks = [
        asyncio.create_task(_worker_loop(config.INGEST_POLL_SECONDS))
        for _ in range(count)
    ]
    tasks += [
        asyncio.create_task(_result_loop()) for _ in range(max(1, config.PARSE_RESULT_CONSUMERS))
    ]
    return tasks


async def stop_workers(tasks: List[asyncio.Task]) -> None:
    """Stop the loops. A running job keeps its lease and is retried after it expires."""
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
