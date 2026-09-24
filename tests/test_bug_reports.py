"""POST /bug-reports and GET /bug-reports/{id}/screenshot.

Storage is the in-memory fake except in the MinIO round trip, which runs only
where the stack's MinIO answers. The log event tests cover both halves: a
plain WARNING record with telemetry off, and the OTLP record exported from
inside the server span with telemetry on (in-memory exporters, same handler
class and ASGI middleware as production).
"""

import io
import json
import logging
import os
import socket
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import pytest
from bson import ObjectId
from httpx import ASGITransport, AsyncClient

import server
from src import observability
from src.database.models.bug_report import BugReport
from src.observability.redaction import RedactionFilter
from src.services import bug_reports as bug_report_service
from src.services.storage import StorageService
from src.utils.redaction import REDACTED, REDACTED_EMAIL
from tests.utils.cleaner import cleanup_models
from tests.utils.fake_storage import FakeStorage
from tests.utils.utils import create_test_user_and_token

JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0ZXN0LXVzZXIifQ.c2lnbmF0dXJlLXNlY3JldA"
EVE_KEY = "eve_" + "b2" * 32
EMAIL = "someone.private@example.org"
SECRETS = (JWT, EVE_KEY, EMAIL)

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64 + b"PNG-SCREENSHOT-MARKER"
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 64 + b"JPEG-SCREENSHOT-MARKER"
GIF_BYTES = b"GIF89a" + b"\x00" * 64

CONTEXT_FIELDS = {
    "session_id",
    "replay_url",
    "trace_id",
    "conversation_id",
    "message_id",
    "app_version",
    "app_commit",
    "environment",
    "user_agent",
    "viewport",
    "path",
    "console_errors",
    "privacy_mode",
}


def _context(**overrides) -> dict:
    ctx = {
        "session_id": "5f1c2a9e0d7b4c3a8e6f1b2d9c0a7e4f",
        "replay_url": "http://localhost:8081/sessions?sid=5f1c2a9e0d7b4c3a8e6f1b2d9c0a7e4f",
        "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
        "conversation_id": "66f1a2b3c4d5e6f7a8b9c0d1",
        "message_id": "66f1a2b3c4d5e6f7a8b9c0d2",
        "app_version": "v0.1.1",
        "app_commit": "8ea24e5",
        "environment": "local",
        "user_agent": "Mozilla/5.0 (X11; Linux x86_64) Firefox/131.0",
        "viewport": {"width": 1440, "height": 900},
        "path": "/chat/66f1a2b3c4d5e6f7a8b9c0d1",
        "console_errors": ["TypeError: x is undefined"],
        "privacy_mode": "mask",
    }
    ctx.update(overrides)
    return ctx


async def _post(client, token=None, *, description="It broke", context=None, raw_context=None, screenshot=None):
    """Always multipart, like the browser, even without a screenshot."""
    parts = {"description": (None, description)}
    if raw_context is not None:
        parts["context"] = (None, raw_context)
    elif context is not False:
        parts["context"] = (None, json.dumps(context if context is not None else _context()))
    if screenshot is not None:
        parts["screenshot"] = screenshot
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return await client.post("/bug-reports", headers=headers, files=parts)


@pytest.fixture
def fake_storage(monkeypatch) -> FakeStorage:
    fake = FakeStorage()
    monkeypatch.setattr(bug_report_service, "storage_service", fake)
    return fake


@pytest.fixture
async def author():
    user, token = await create_test_user_and_token()
    try:
        yield user, token
    finally:
        await BugReport.delete_many({"user_id": user.id})
        await cleanup_models([user])


# C1: contract, storage, redaction


async def test_create_stores_every_field_redacted(async_client, fake_storage, author):
    user, token = author
    description = (
        f"Chat froze. My header was Authorization: Bearer {JWT}, "
        f"my key {EVE_KEY}, write me at {EMAIL}"
    )
    ctx = _context(
        path=f"/chat?api_key={EVE_KEY}",
        console_errors=[f"401 for Bearer {JWT}", f"user {EMAIL} not found"],
        user_agent=f"agent {EVE_KEY}",
    )
    resp = await _post(async_client, token, description=description, context=ctx)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert set(body) == {"id", "created_at", "screenshot"}
    assert body["screenshot"] is False
    assert ObjectId.is_valid(body["id"])

    doc = await BugReport.get_collection().find_one({"_id": ObjectId(body["id"])})
    assert doc is not None
    assert doc["user_id"] == user.id
    assert set(doc["context"]) == CONTEXT_FIELDS
    stored = json.dumps(doc, default=str)
    for secret in SECRETS:
        assert secret not in stored
    assert f"Bearer {REDACTED}" in doc["description"]
    assert REDACTED_EMAIL in doc["description"]
    assert doc["description"].startswith("Chat froze.")
    assert doc["context"]["path"] == f"/chat?api_key={REDACTED}"
    assert doc["context"]["console_errors"] == [f"401 for Bearer {REDACTED}", f"user {REDACTED_EMAIL} not found"]
    # Untouched when there is nothing to redact.
    assert doc["context"]["session_id"] == ctx["session_id"]
    assert doc["context"]["replay_url"] == ctx["replay_url"]
    assert doc["context"]["viewport"] == {"width": 1440, "height": 900}
    assert doc["context"]["privacy_mode"] == "mask"
    assert doc["screenshot"] is None


async def test_empty_context_stores_every_field_as_null(async_client, fake_storage, author):
    _, token = author
    resp = await _post(async_client, token, context={"unknown_key": "dropped"})
    assert resp.status_code == 201, resp.text
    doc = await BugReport.get_collection().find_one({"_id": ObjectId(resp.json()["id"])})
    assert set(doc["context"]) == CONTEXT_FIELDS
    assert doc["context"]["console_errors"] == []
    assert all(v is None for k, v in doc["context"].items() if k != "console_errors")


# C2: rate limit


async def test_sixth_report_in_an_hour_is_rate_limited(async_client, fake_storage, author):
    user, token = author
    for i in range(5):
        resp = await _post(async_client, token, description=f"report {i}")
        assert resp.status_code == 201, resp.text
    resp = await _post(async_client, token, description="report 6")
    assert resp.status_code == 429
    assert resp.json()["detail"]["code"] == "bug_report_rate_limited"
    assert resp.headers["Retry-After"] == "3600"
    assert await BugReport.count_documents({"user_id": user.id}) == 5

    # Another user has a budget of their own.
    other, other_token = await create_test_user_and_token()
    try:
        resp = await _post(async_client, other_token)
        assert resp.status_code == 201
    finally:
        await BugReport.delete_many({"user_id": other.id})
        await cleanup_models([other])


async def test_reports_older_than_an_hour_do_not_count(async_client, fake_storage, author):
    user, token = author
    old = datetime.now(timezone.utc) - timedelta(hours=1, minutes=5)
    await BugReport.get_collection().insert_many(
        [{"user_id": user.id, "description": "old", "context": {}, "timestamp": old} for _ in range(5)]
    )
    resp = await _post(async_client, token)
    assert resp.status_code == 201, resp.text


async def test_post_insert_recount_refuses_a_report_that_slipped_past_the_count(
    async_client, fake_storage, author, monkeypatch
):
    """A request that passed the count while others were inserting is caught
    by the recount after its own insert: its row is deleted, nothing is
    uploaded, and the answer is the same 429."""
    user, token = author
    now = datetime.now(timezone.utc)
    await BugReport.get_collection().insert_many(
        [{"user_id": user.id, "description": "raced", "context": {}, "timestamp": now} for _ in range(5)]
    )

    async def count_saw_nothing(user_id, now=None):
        return None

    monkeypatch.setattr(bug_report_service, "enforce_rate_limit", count_saw_nothing)
    resp = await _post(async_client, token, screenshot=("s.png", io.BytesIO(PNG_BYTES), "image/png"))
    assert resp.status_code == 429
    assert resp.json()["detail"]["code"] == "bug_report_rate_limited"
    assert await BugReport.count_documents({"user_id": user.id}) == 5
    assert fake_storage.objects == {}


async def test_rate_limit_is_a_mongo_window_count_not_redis():
    import inspect

    assert "count_documents" in inspect.getsource(bug_report_service.enforce_rate_limit)
    assert "find(_window_filter" in inspect.getsource(bug_report_service._is_among_first_in_window)
    # No Redis client or helper imported into the module.
    assert not [name for name in vars(bug_report_service) if "redis" in name.lower()]


# C3: screenshot


async def test_png_screenshot_is_stored_and_served_inline_to_its_author(async_client, fake_storage, author):
    user, token = author
    resp = await _post(async_client, token, screenshot=("shot.png", io.BytesIO(PNG_BYTES), "image/png"))
    assert resp.status_code == 201, resp.text
    report_id = resp.json()["id"]
    assert resp.json()["screenshot"] is True

    key = f"bug-reports/{user.id}/{report_id}.png"
    assert list(fake_storage.objects) == [key]
    assert fake_storage.objects[key]["content_type"] == "image/png"
    doc = await BugReport.get_collection().find_one({"_id": ObjectId(report_id)})
    assert doc["screenshot"] == {"key": key, "content_type": "image/png", "size_bytes": len(PNG_BYTES)}

    got = await async_client.get(f"/bug-reports/{report_id}/screenshot", headers={"Authorization": f"Bearer {token}"})
    assert got.status_code == 200
    assert got.content == PNG_BYTES
    assert got.headers["content-type"] == "image/png"
    assert got.headers["content-disposition"].startswith("inline;")
    assert got.headers["x-content-type-options"] == "nosniff"
    assert "private" in got.headers["cache-control"]

    other, other_token = await create_test_user_and_token()
    try:
        denied = await async_client.get(
            f"/bug-reports/{report_id}/screenshot", headers={"Authorization": f"Bearer {other_token}"}
        )
        assert denied.status_code == 404
    finally:
        await cleanup_models([other])

    anonymous = await async_client.get(f"/bug-reports/{report_id}/screenshot")
    assert anonymous.status_code == 401


async def test_jpeg_screenshot_is_accepted_whatever_it_declares(async_client, fake_storage, author):
    user, token = author
    resp = await _post(
        async_client, token, screenshot=("shot.bin", io.BytesIO(JPEG_BYTES), "application/octet-stream")
    )
    assert resp.status_code == 201, resp.text
    key = f"bug-reports/{user.id}/{resp.json()['id']}.jpeg"
    assert fake_storage.objects[key]["content_type"] == "image/jpeg"


@pytest.mark.parametrize(
    "name,data,declared",
    [
        ("anim.gif", GIF_BYTES, "image/gif"),
        ("evil.png", b"<svg onload=alert(1)>", "image/png"),
        ("notes.txt", b"plain text", "text/plain"),
        ("doc.pdf", b"%PDF-1.4\n", "application/pdf"),
    ],
    ids=["gif", "svg-as-png", "text", "pdf"],
)
async def test_non_png_jpeg_screenshot_is_rejected(async_client, fake_storage, author, name, data, declared):
    user, token = author
    resp = await _post(async_client, token, screenshot=(name, io.BytesIO(data), declared))
    assert resp.status_code == 415
    assert fake_storage.objects == {}
    assert await BugReport.count_documents({"user_id": user.id}) == 0


async def test_screenshot_above_one_megabyte_is_rejected(async_client, fake_storage, author):
    user, token = author
    too_big = b"\x89PNG\r\n\x1a\n" + b"\x00" * (1024 * 1024)
    resp = await _post(async_client, token, screenshot=("big.png", io.BytesIO(too_big), "image/png"))
    assert resp.status_code == 413
    assert fake_storage.objects == {}
    assert await BugReport.count_documents({"user_id": user.id}) == 0

    exactly = b"\x89PNG\r\n\x1a\n" + b"\x00" * (1024 * 1024 - 8)
    resp = await _post(async_client, token, screenshot=("max.png", io.BytesIO(exactly), "image/png"))
    assert resp.status_code == 201


async def test_screenshot_route_404s(async_client, fake_storage, author):
    user, token = author
    headers = {"Authorization": f"Bearer {token}"}
    assert (await async_client.get("/bug-reports/not-an-id/screenshot", headers=headers)).status_code == 404
    assert (await async_client.get(f"/bug-reports/{ObjectId()}/screenshot", headers=headers)).status_code == 404
    resp = await _post(async_client, token)
    no_shot = await async_client.get(f"/bug-reports/{resp.json()['id']}/screenshot", headers=headers)
    assert no_shot.status_code == 404
    # The document outlived its bytes.
    resp = await _post(async_client, token, screenshot=("s.png", io.BytesIO(PNG_BYTES), "image/png"))
    fake_storage.objects.clear()
    gone = await async_client.get(f"/bug-reports/{resp.json()['id']}/screenshot", headers=headers)
    assert gone.status_code == 404


async def test_storage_failure_keeps_the_report_without_screenshot(async_client, fake_storage, author, monkeypatch):
    user, token = author

    async def broken_put(key, body, content_type):
        raise RuntimeError("bucket down")

    monkeypatch.setattr(fake_storage, "put_object", broken_put)
    resp = await _post(async_client, token, screenshot=("s.png", io.BytesIO(PNG_BYTES), "image/png"))
    assert resp.status_code == 201
    assert resp.json()["screenshot"] is False
    doc = await BugReport.get_collection().find_one({"_id": ObjectId(resp.json()["id"])})
    assert doc["screenshot"] is None


async def test_description_and_screenshot_never_reach_a_log(async_client, fake_storage, author, caplog):
    _, token = author
    caplog.set_level(logging.DEBUG)
    description = "UNIQUE-DESCRIPTION-TEXT-7731 the chat froze"
    resp = await _post(
        async_client, token, description=description, screenshot=("s.png", io.BytesIO(PNG_BYTES), "image/png")
    )
    assert resp.status_code == 201
    dumped = "\n".join(
        f"{r.getMessage()} {json.dumps({k: str(v) for k, v in vars(r).items()})}" for r in caplog.records
    )
    assert "UNIQUE-DESCRIPTION-TEXT-7731" not in dumped
    assert "PNG-SCREENSHOT-MARKER" not in dumped
    assert "bug_report.created" in dumped


def _minio_reachable() -> bool:
    endpoint = os.getenv("S3_ENDPOINT_URL", "").strip()
    if not endpoint:
        return False
    parsed = urlparse(endpoint)
    try:
        with socket.create_connection((parsed.hostname, parsed.port or 80), timeout=1):
            return True
    except OSError:
        return False


@pytest.mark.skipif(not _minio_reachable(), reason="needs the stack's MinIO (S3_ENDPOINT_URL)")
async def test_minio_round_trip_through_the_storage_service(async_client, author):
    """No fake: the real StorageService writes to and reads from the bucket."""
    user, token = author
    storage = StorageService()
    resp = await _post(async_client, token, screenshot=("s.png", io.BytesIO(PNG_BYTES), "image/png"))
    assert resp.status_code == 201, resp.text
    report_id = resp.json()["id"]
    key = f"bug-reports/{user.id}/{report_id}.png"
    try:
        stored = await storage.get_object(key)
        assert stored["Body"].read() == PNG_BYTES
        assert stored["ContentType"] == "image/png"
        got = await async_client.get(
            f"/bug-reports/{report_id}/screenshot", headers={"Authorization": f"Bearer {token}"}
        )
        assert got.status_code == 200 and got.content == PNG_BYTES
    finally:
        await storage.delete_object(key)


# C4: the log event


async def test_created_event_is_a_plain_warning_with_telemetry_off(async_client, fake_storage, author, caplog):
    user, token = author
    assert not observability.is_enabled()
    caplog.set_level(logging.INFO)
    ctx = _context()
    resp = await _post(async_client, token, description="SECRET-DESCRIPTION-991", context=ctx)
    assert resp.status_code == 201
    (record,) = [r for r in caplog.records if r.name == "eve.bug_report"]
    assert record.levelno == logging.WARNING
    assert record.getMessage().startswith("bug_report.created ")
    attrs = vars(record)
    assert attrs["eve.bug_report.id"] == resp.json()["id"]
    assert attrs["rum.sessionId"] == ctx["session_id"]
    assert attrs["eve.replay_url"] == ctx["replay_url"]
    assert attrs["gen_ai.conversation.id"] == ctx["conversation_id"]
    assert attrs["eve.message_id"] == ctx["message_id"]
    assert attrs["user.id"] == user.id
    assert attrs["deployment.environment.name"] == observability.deployment_environment()
    assert attrs["event.name"] == "bug_report.created"
    assert "SECRET-DESCRIPTION-991" not in json.dumps({k: str(v) for k, v in attrs.items()})
    doc = await BugReport.get_collection().find_one({"_id": ObjectId(resp.json()["id"])})
    assert doc["request_trace_id"] is None


async def test_created_event_is_exported_inside_the_request_span(fake_storage, author):
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter, SimpleLogRecordProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.trace import SpanKind

    user, token = author
    span_exporter = InMemorySpanExporter()
    tracer_provider = observability.build_tracer_provider(span_exporter, batch=False)
    log_exporter = InMemoryLogRecordExporter()
    logger_provider = LoggerProvider(resource=tracer_provider.resource)
    logger_provider.add_log_record_processor(SimpleLogRecordProcessor(log_exporter))
    # The production OTLP handler class, with the production redaction filter.
    handler = observability._otlp_log_handler_class()(logger_provider=logger_provider)
    handler.addFilter(RedactionFilter())
    event_logger = logging.getLogger("eve.bug_report")
    event_logger.addHandler(handler)

    outer = server.create_app()
    app = observability.wrap_asgi(
        outer, fastapi_app=observability._find_fastapi_app(outer), tracer_provider=tracer_provider
    )
    ctx = _context()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await _post(client, token, description="SECRET-DESCRIPTION-552", context=ctx)
    finally:
        event_logger.removeHandler(handler)
        logger_provider.shutdown()
        tracer_provider.shutdown()
    assert resp.status_code == 201, resp.text
    report_id = resp.json()["id"]

    (server_span,) = [s for s in span_exporter.get_finished_spans() if s.kind == SpanKind.SERVER]
    assert server_span.name == "POST /bug-reports"
    assert server_span.attributes["eve.bug_report.id"] == report_id
    assert server_span.attributes["user.id"] == user.id

    logs = [getattr(item, "log_record", item) for item in log_exporter.get_finished_logs()]
    (event,) = [r for r in logs if str(r.body).startswith("bug_report.created")]
    assert event.severity_text == "WARN"
    assert event.trace_id == server_span.context.trace_id
    assert event.span_id == server_span.context.span_id
    attrs = dict(event.attributes)
    assert attrs["eve.bug_report.id"] == report_id
    assert attrs["rum.sessionId"] == ctx["session_id"]
    assert attrs["eve.replay_url"] == ctx["replay_url"]
    assert attrs["gen_ai.conversation.id"] == ctx["conversation_id"]
    assert attrs["eve.message_id"] == ctx["message_id"]
    assert attrs["user.id"] == user.id
    assert attrs["deployment.environment.name"] == observability.deployment_environment()
    exported = f"{event.body} {attrs}"
    assert "SECRET-DESCRIPTION-552" not in exported

    doc = await BugReport.get_collection().find_one({"_id": ObjectId(report_id)})
    assert doc["request_trace_id"] == format(server_span.context.trace_id, "032x")


# C5: auth and validation


async def test_unauthenticated_is_401(async_client, fake_storage):
    resp = await _post(async_client, None)
    assert resp.status_code == 401


@pytest.mark.parametrize("description", ["", "   \n\t", "x" * 4001], ids=["empty", "blank", "4001-chars"])
async def test_bad_description_is_422(async_client, fake_storage, author, description):
    user, token = author
    resp = await _post(async_client, token, description=description)
    assert resp.status_code == 422, resp.text
    assert await BugReport.count_documents({"user_id": user.id}) == 0


async def test_description_of_exactly_4000_chars_is_accepted(async_client, fake_storage, author):
    _, token = author
    resp = await _post(async_client, token, description="x" * 4000)
    assert resp.status_code == 201


@pytest.mark.parametrize(
    "raw_context",
    [
        "not json",
        "[1, 2]",
        '{"viewport": {"width": "wide"}}',
        '{"console_errors": "not a list"}',
        json.dumps({"console_errors": ["e"] * 51}),
        json.dumps({"session_id": "s" * 129}),
    ],
    ids=["not-json", "array", "viewport-type", "console-errors-type", "51-console-errors", "long-session-id"],
)
async def test_invalid_context_is_422(async_client, fake_storage, author, raw_context):
    user, token = author
    resp = await _post(async_client, token, raw_context=raw_context)
    assert resp.status_code == 422, resp.text
    assert any("context" in err["loc"] for err in resp.json()["detail"])
    assert await BugReport.count_documents({"user_id": user.id}) == 0


async def test_missing_context_is_422(async_client, fake_storage, author):
    _, token = author
    resp = await _post(async_client, token, context=False)
    assert resp.status_code == 422
