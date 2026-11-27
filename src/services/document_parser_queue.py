"""SQS between the backend and eve-esa/document-parser (FEATURE_BACKGROUND_INGEST).

The backend sends one job per file to parse-jobs and reads processing / done / failed results
from parse-results. Message formats are the parser's (document-parser,
src/document_parser/messages.py); the result handling is in src/services/document_ingest.py.
SQS on AWS through the task role, ElasticMQ in compose.
"""

from __future__ import annotations

import json
import mimetypes
import logging
from functools import lru_cache
from typing import List, Optional, Tuple

import anyio
import boto3

from src import config

logger = logging.getLogger(__name__)

JOB_VERSION = 1


@lru_cache(maxsize=1)
def _client():
    kwargs = {"region_name": config.SQS_REGION}
    if config.SQS_ENDPOINT_URL:
        kwargs["endpoint_url"] = config.SQS_ENDPOINT_URL
    if config.SQS_ACCESS_KEY_ID and config.SQS_SECRET_ACCESS_KEY:
        kwargs["aws_access_key_id"] = config.SQS_ACCESS_KEY_ID
        kwargs["aws_secret_access_key"] = config.SQS_SECRET_ACCESS_KEY
    return boto3.client("sqs", **kwargs)


@lru_cache(maxsize=4)
def _queue_url(name_or_url: str) -> str:
    if name_or_url.startswith(("http://", "https://")):
        return name_or_url
    return _client().get_queue_url(QueueName=name_or_url)["QueueUrl"]


def _conversation_id(key: str) -> Optional[str]:
    """users/<user>/conversations/<conversation>/uploads/<document>/original.<ext>"""
    parts = key.split("/")
    if len(parts) > 3 and parts[0] == "users" and parts[2] == "conversations":
        return parts[3]
    return None


def job_body(*, document_id: str, key: str, filename: str, user_id: str, collection_id: str) -> str:
    """The parse job, in the contract of the document-parser task: document_id, user_id,
    conversation_id, s3_key, content_type (plus filename and bucket)."""
    return json.dumps(
        {
            "version": JOB_VERSION,
            "document_id": document_id,
            "user_id": user_id,
            "conversation_id": _conversation_id(key),
            "s3_key": key,
            "content_type": mimetypes.guess_type(filename)[0] or "application/octet-stream",
            "filename": filename,
            "bucket": config.S3_BUCKET_NAME or None,
            "collection_id": collection_id,
        }
    )


def _send_sync(bodies: List[str]) -> None:
    url = _queue_url(config.PARSE_JOBS_QUEUE)
    for body in bodies:
        _client().send_message(QueueUrl=url, MessageBody=body)


async def send_jobs(bodies: List[str]) -> None:
    await anyio.to_thread.run_sync(_send_sync, bodies)


def _receive_sync(wait_seconds: int) -> List[Tuple[str, str]]:
    url = _queue_url(config.PARSE_RESULTS_QUEUE)
    resp = _client().receive_message(
        QueueUrl=url, MaxNumberOfMessages=10, WaitTimeSeconds=wait_seconds
    )
    return [(m["ReceiptHandle"], m["Body"]) for m in resp.get("Messages", [])]


async def receive_results(wait_seconds: int = 10) -> List[Tuple[str, str]]:
    """(receipt handle, body) pairs; empty after wait_seconds without a message."""
    return await anyio.to_thread.run_sync(_receive_sync, wait_seconds, abandon_on_cancel=True)


async def delete_result(receipt: str) -> None:
    url = _queue_url(config.PARSE_RESULTS_QUEUE)
    await anyio.to_thread.run_sync(
        lambda: _client().delete_message(QueueUrl=url, ReceiptHandle=receipt)
    )


def parse_result(body: str) -> Optional[dict]:
    """The result as a dict, or None when it is not one this backend understands."""
    try:
        data = json.loads(body)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("document_id"), str):
        return None
    if data.get("version", JOB_VERSION) != JOB_VERSION:
        return None
    status = data.get("status")
    if status == "done":  # parsers before the alias-ocr contract
        data["status"] = status = "ready"
    if status not in ("processing", "ready", "failed"):
        return None
    return data
