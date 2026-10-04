"""GET /internal/bug-reports/{id}: the observability sink reads any report.

What matters: an unset secret closes the route, a wrong or absent secret is
403, an unknown or malformed id is 404, and a good read returns the author
view plus ``trace_id``, ``conversation_id`` and ``environment``, logged at
INFO. The author route is unchanged, which test_bug_reports.py covers.

``INTERNAL_API_SECRET`` is monkeypatched on the router module, because config
is read once at import.
"""

import json
import logging

import pytest
from bson import ObjectId

from src.database.models.bug_report import BugReport
from src.database.models.conversation import Conversation
from src.database.models.message import Message
from src.routers import internal_bug_reports
from tests.utils.cleaner import cleanup_models
from tests.utils.utils import create_test_user_and_token

SECRET = "internal-secret-for-tests"
TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
NEW_FIELDS = {"trace_id", "conversation_id", "environment"}


def _url(report_id) -> str:
    return f"/internal/bug-reports/{report_id}"


@pytest.fixture(autouse=True)
def _configured_secret(monkeypatch):
    monkeypatch.setattr(internal_bug_reports, "INTERNAL_API_SECRET", SECRET)


@pytest.fixture
async def report(async_client):
    """A report filed through the public route, with a real conversation."""
    user, token = await create_test_user_and_token()
    conv = Conversation(user_id=user.id, name="Internal read conversation")
    await conv.save()
    await Message(conversation_id=conv.id, input="hi", output="hello").save()
    context = {"trace_id": TRACE_ID, "conversation_id": conv.id, "environment": "local"}
    resp = await async_client.post(
        "/bug-reports",
        headers={"Authorization": f"Bearer {token}"},
        files={"description": (None, "It broke"), "context": (None, json.dumps(context))},
    )
    assert resp.status_code == 201, resp.text
    try:
        yield resp.json()["id"], user, token, conv
    finally:
        await BugReport.delete_many({"user_id": user.id})
        await Message.delete_many({"conversation_id": conv.id})
        await conv.delete()
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_read_returns_the_author_view_plus_ticket_fields(async_client, report, caplog, monkeypatch):
    report_id, user, token, conv = report
    monkeypatch.setattr(internal_bug_reports, "deployment_environment", lambda: "dev")

    with caplog.at_level(logging.INFO, logger=internal_bug_reports.__name__):
        got = await async_client.get(_url(report_id), headers={"X-Internal-Secret": SECRET})
    assert got.status_code == 200, got.text
    body = got.json()

    author = await async_client.get(f"/bug-reports/{report_id}", headers={"Authorization": f"Bearer {token}"})
    assert author.status_code == 200
    author_body = author.json()
    # Same fields and values as the author route, plus exactly the three additions.
    assert set(body) == set(author_body) | NEW_FIELDS
    assert not NEW_FIELDS & set(author_body)
    assert {k: body[k] for k in author_body} == author_body
    assert body["user_id"] == user.id
    assert body["trace_id"] == TRACE_ID
    assert body["conversation_id"] == conv.id
    assert body["environment"] == "dev"
    assert body["conversation"]["messages"][0]["output"] == "hello"

    lines = [r for r in caplog.records if r.name == internal_bug_reports.__name__]
    assert [(r.levelno, r.getMessage()) for r in lines] == [
        (logging.INFO, f"bug_report.read_internal id={report_id}")
    ]


@pytest.mark.asyncio
async def test_report_without_context_ids_reads_null_ticket_fields(async_client):
    user, _ = await create_test_user_and_token()
    doc = BugReport(user_id=user.id, description="bare", context={})
    await doc.save()
    try:
        got = await async_client.get(_url(doc.id), headers={"X-Internal-Secret": SECRET})
        assert got.status_code == 200, got.text
        body = got.json()
        assert body["trace_id"] is None
        assert body["conversation_id"] is None
        assert body["conversation"] is None
        assert body["environment"]
    finally:
        await BugReport.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
@pytest.mark.parametrize("headers", [{}, {"X-Internal-Secret": "wrong"}, {"X-Internal-Secret": ""}], ids=["absent", "wrong", "empty"])
async def test_bad_or_absent_secret_is_403(async_client, report, headers):
    report_id, *_ = report
    got = await async_client.get(_url(report_id), headers=headers)
    assert got.status_code == 403
    assert "description" not in got.text


@pytest.mark.asyncio
async def test_a_user_bearer_is_not_the_secret(async_client, report):
    """The author's own token opens the author route, never this one."""
    report_id, _, token, _ = report
    got = await async_client.get(_url(report_id), headers={"Authorization": f"Bearer {token}"})
    assert got.status_code == 403


@pytest.mark.asyncio
async def test_unset_secret_closes_the_route(async_client, report, monkeypatch):
    report_id, *_ = report
    monkeypatch.setattr(internal_bug_reports, "INTERNAL_API_SECRET", None)
    got = await async_client.get(_url(report_id), headers={"X-Internal-Secret": SECRET})
    assert got.status_code == 503


@pytest.mark.asyncio
@pytest.mark.parametrize("report_id", [str(ObjectId()), "not-an-id"], ids=["unknown", "malformed"])
async def test_unknown_or_malformed_id_is_404(async_client, report_id, caplog):
    with caplog.at_level(logging.INFO, logger=internal_bug_reports.__name__):
        got = await async_client.get(_url(report_id), headers={"X-Internal-Secret": SECRET})
    assert got.status_code == 404
    assert not [r for r in caplog.records if r.name == internal_bug_reports.__name__]


@pytest.mark.asyncio
async def test_secret_is_checked_before_the_lookup(async_client):
    """No secret, no oracle: a missing id answers 403, not 404."""
    got = await async_client.get(_url(ObjectId()), headers={"X-Internal-Secret": "wrong"})
    assert got.status_code == 403
