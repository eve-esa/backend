"""Tests for src.commands.send_cohort_invite, the launch mail in cohorts."""

import logging
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from motor.motor_asyncio import AsyncIOMotorCollection

from src.commands import send_cohort_invite as command
from src.commands.send_cohort_invite import (
    COLLECTION,
    CohortInviteError,
    mask_email,
    plan_cohort,
    read_emails_file,
    send_cohort_invite,
)
from src.database.mongo import AsyncMongoDBManager, async_mongo_manager, get_collection
from src.services.account_notifications import (
    COHORT_INVITE_SUBJECT,
    render_cohort_invite,
)
from tests.conftest import _resolve_test_mongo_uri


class FakeMailer:
    """Stands in for send_mail: records recipients, can fail chosen addresses."""

    def __init__(self, fail_for: set[str] | None = None):
        self.sent: list[dict] = []
        self.fail_for = fail_for or set()

    async def __call__(self, to: str, subject: str, html: str, text: str) -> None:
        if to in self.fail_for:
            raise RuntimeError("fake transport failure")
        self.sent.append({"to": to, "subject": subject, "html": html, "text": text})


@pytest.fixture
def fake_mailer(monkeypatch) -> FakeMailer:
    fake = FakeMailer()
    monkeypatch.setattr(command, "send_mail", fake)
    monkeypatch.setattr("src.services.mailer.MAIL_TRANSPORT", "ses")
    return fake


@pytest.fixture
def pauses(monkeypatch) -> list[float]:
    calls: list[float] = []

    async def _record(seconds: float) -> None:
        calls.append(seconds)

    monkeypatch.setattr(command, "_pause", _record)
    return calls


@pytest.fixture
async def world():
    """Unique cohort and address prefix; removes every row the test created."""
    tag = uuid.uuid4().hex[:10]
    state = {"tag": tag, "cohort": f"test-{tag}", "user_ids": []}
    yield state
    users = get_collection("users")
    ids = state["user_ids"]
    await users.delete_many({"_id": {"$in": ids}})
    await get_collection("external_identities").delete_many(
        {"user_id": {"$in": [str(i) for i in ids]}}
    )
    await get_collection(COLLECTION).delete_many({"cohort": {"$regex": f"^test-{tag}"}})


async def _user(world, name: str | None, **fields) -> tuple[str, str | None]:
    """Insert a raw users row; returns (user_id, email)."""
    email = f"{name}-{world['tag']}@example.org" if name else fields.pop("email", None)
    doc = {"timestamp": datetime.now(timezone.utc), **fields}
    if email is not None:
        doc["email"] = email
    result = await get_collection("users").insert_one(doc)
    world["user_ids"].append(result.inserted_id)
    return str(result.inserted_id), email


async def _sent_rows(cohort: str) -> list[dict]:
    return [doc async for doc in get_collection(COLLECTION).find({"cohort": cohort})]


# ---------------------------------------------------------------- pure units


@pytest.mark.no_db
@pytest.mark.parametrize(
    "address, masked",
    [
        ("alice@example.org", "a***@example.org"),
        ("Bob.Smith+tag@sub.example.org", "B***@sub.example.org"),
        ("@example.org", "***@example.org"),
        ("not-an-address", "***"),
        ("", "***"),
    ],
)
def test_mask_email(address, masked):
    assert mask_email(address) == masked


@pytest.mark.no_db
def test_read_emails_file_skips_comments_blanks_and_repeats(tmp_path):
    path = tmp_path / "list.txt"
    path.write_text("# vip list\nAlice@Example.org\n\n  bob@example.org \nalice@example.org\n")
    assert read_emails_file(path) == ["alice@example.org", "bob@example.org"]


@pytest.mark.no_db
def test_render_uses_url_and_escapes_the_text_block():
    subject, html, text = render_cohort_invite(
        "carol@example.org",
        sign_in_url="https://staging.eve-chat.chat",
        text_block="First line\nwrapped.\n\nSecond <b>para</b> & more.",
    )
    assert subject == COHORT_INVITE_SUBJECT
    assert 'href="https://staging.eve-chat.chat"' in html
    assert "<p style=\"margin:0 0 20px 0;\">First line wrapped.</p>" in html
    assert "Second &lt;b&gt;para&lt;/b&gt; &amp; more." in html
    assert "Second <b>para</b> & more." in text
    assert "Sign in to EVE: https://staging.eve-chat.chat" in text


@pytest.mark.no_db
def test_render_default_text_is_english_and_neutral():
    subject, html, text = render_cohort_invite("dave@example.org")
    assert subject == "The new EVE is ready for you"
    assert "Forgot password" in text
    assert "dave@example.org" in text


@pytest.mark.no_db
def test_render_uses_the_base_layout(monkeypatch):
    from src.services import account_notifications

    monkeypatch.setattr(account_notifications, "FRONTEND_URL", "https://eve.example.org")
    _, html, text = render_cohort_invite(
        "erin@example.org", sign_in_url="https://eve-chat.philab.esa.int"
    )
    assert "background-color:#003247" in html
    assert '<img src="https://eve.example.org/branding/eve-logo.png"' in html
    assert 'href="https://eve-chat.philab.esa.int"' in html
    assert 'European Space Agency. <a href="https://eve.example.org"' in html
    for gone in ("website-files.com", "box-shadow", "&#127757;"):
        assert gone not in html
    assert not any(ord(char) > 0x1F000 for char in html + text)
    footer = "You are receiving this message because"
    assert footer in html
    assert footer in text


@pytest.mark.no_db
def test_cli_rejects_emails_file_with_slicing():
    with pytest.raises(SystemExit):
        command._parse_args(["--cohort", "x", "--emails-file", "a.txt", "--limit", "5"])


@pytest.mark.no_db
def test_cli_requires_cohort():
    with pytest.raises(SystemExit):
        command._parse_args(["--limit", "5"])


# ------------------------------------------------------------- with Mongo


@pytest.mark.asyncio
async def test_skip_rules(world):
    _, ok_none = await _user(world, "none")
    _, ok_approved = await _user(world, "approved", approval_status="approved")
    await _user(world, "pending", approval_status="pending")
    await _user(world, None, email="")
    await _user(world, None)  # no email field at all
    migrated_id, migrated_email = await _user(world, "migrated")
    await get_collection("external_identities").insert_one(
        {"user_id": migrated_id, "issuer": "https://test", "subject": f"cohort-{world['tag']}"}
    )
    dup_upper = f"NONE-{world['tag']}@example.org"
    await _user(world, None, email=dup_upper)

    count_before = len(world["user_ids"])
    plan = await plan_cohort(
        world["cohort"],
        emails=[ok_none, ok_approved, f"pending-{world['tag']}@example.org", migrated_email, dup_upper.lower()],
    )
    assert [r.email for r in plan.recipients] == [ok_none, ok_approved]
    assert plan.skipped["not_approved"] == 1
    assert plan.skipped["migrated"] == 1
    assert plan.skipped["duplicate_email"] == 0  # the upper-case row is not matched by $in
    assert count_before == len(world["user_ids"])

    # Selected by _id slice, the address-less rows and the duplicate are counted too.
    total = await get_collection("users").count_documents({})
    offset = total - len(world["user_ids"])
    plan = await plan_cohort(world["cohort"], offset=offset, limit=len(world["user_ids"]))
    assert plan.selected == 7
    assert [r.email for r in plan.recipients] == [ok_none, ok_approved]
    assert plan.skipped["no_email"] == 2
    assert plan.skipped["not_approved"] == 1
    assert plan.skipped["migrated"] == 1
    assert plan.skipped["duplicate_email"] == 1

    plan = await plan_cohort(
        world["cohort"], offset=offset, limit=len(world["user_ids"]), include_migrated=True
    )
    assert migrated_email in [r.email for r in plan.recipients]


@pytest.mark.asyncio
async def test_offset_and_limit_follow_id_order(world):
    emails = [(await _user(world, f"slice{i}"))[1] for i in range(5)]
    total = await get_collection("users").count_documents({})
    base = total - 5
    plan = await plan_cohort(world["cohort"], offset=base + 1, limit=2)
    assert [r.email for r in plan.recipients] == emails[1:3]


@pytest.mark.asyncio
async def test_emails_file_counts_unknown_addresses(world):
    _, known = await _user(world, "known")
    plan = await plan_cohort(
        world["cohort"], emails=[known, f"ghost-{world['tag']}@example.org"]
    )
    assert plan.selected == 1
    assert plan.not_found == 1


@pytest.mark.asyncio
async def test_dry_run_sends_and_writes_nothing(world, fake_mailer, pauses, capsys):
    emails = [(await _user(world, f"dry{i}"))[1] for i in range(7)]

    summary = await send_cohort_invite(world["cohort"], emails=emails, batch_size=2)

    assert summary["recipients"] == 7
    assert summary["sent"] == 0
    assert fake_mailer.sent == []
    assert pauses == []
    assert await _sent_rows(world["cohort"]) == []
    out = capsys.readouterr().out
    assert "Recipients: 7 in 4 batch(es)" in out
    assert out.count("***@example.org") == 5
    for address in emails:
        assert address not in out


@pytest.mark.asyncio
async def test_apply_sends_in_batches_and_is_idempotent(world, fake_mailer, pauses, caplog):
    emails = [(await _user(world, f"send{i}"))[1] for i in range(5)]

    with caplog.at_level(logging.INFO, logger=command.__name__):
        first = await send_cohort_invite(
            world["cohort"], apply=True, emails=emails, batch_size=2, sleep_seconds=7
        )

    assert first["sent"] == 5
    assert sorted(m["to"] for m in fake_mailer.sent) == sorted(emails)
    assert pauses == [7, 7]  # three batches, a pause between each pair
    rows = await _sent_rows(world["cohort"])
    assert len(rows) == 5 and all(row["sent_at"] for row in rows)
    batch_lines = [r.getMessage() for r in caplog.records if "cohort_invite_batch" in r.getMessage()]
    assert len(batch_lines) == 3
    for line in batch_lines:
        assert world["cohort"] in line
        assert "@" not in line

    second = await send_cohort_invite(world["cohort"], apply=True, emails=emails, sleep_seconds=0)
    assert second["sent"] == 0
    assert second["skipped_already_sent"] == 5
    assert len(fake_mailer.sent) == 5

    # Idempotency is per cohort: a new cohort label mails them again.
    other = await send_cohort_invite(
        f"{world['cohort']}-b", apply=True, emails=emails[:1], sleep_seconds=0
    )
    assert other["sent"] == 1


@pytest.mark.asyncio
async def test_failed_send_releases_the_claim(world, fake_mailer, pauses):
    _, good = await _user(world, "good")
    _, bad = await _user(world, "bad")
    fake_mailer.fail_for = {bad}

    first = await send_cohort_invite(world["cohort"], apply=True, emails=[good, bad], sleep_seconds=0)
    assert (first["sent"], first["failed"]) == (1, 1)
    assert len(await _sent_rows(world["cohort"])) == 1

    fake_mailer.fail_for = set()
    second = await send_cohort_invite(world["cohort"], apply=True, emails=[good, bad], sleep_seconds=0)
    assert (second["sent"], second["skipped_already_sent"]) == (1, 1)
    assert [m["to"] for m in fake_mailer.sent] == [good, bad]


@pytest.mark.asyncio
async def test_claim_without_sent_at_is_in_doubt(world, fake_mailer, pauses):
    user_id, email = await _user(world, "doubt")
    await get_collection(COLLECTION).insert_one(
        {"user_id": user_id, "cohort": world["cohort"], "claimed_at": datetime.now(timezone.utc), "sent_at": None}
    )
    summary = await send_cohort_invite(world["cohort"], apply=True, emails=[email], sleep_seconds=0)
    assert summary["skipped_in_doubt"] == 1
    assert fake_mailer.sent == []


@pytest.mark.asyncio
async def test_apply_refuses_when_mail_is_off(world, fake_mailer, monkeypatch):
    _, email = await _user(world, "off")
    monkeypatch.setattr("src.services.mailer.MAIL_TRANSPORT", "off")
    with pytest.raises(CohortInviteError):
        await send_cohort_invite(world["cohort"], apply=True, emails=[email])
    assert fake_mailer.sent == []
    assert await _sent_rows(world["cohort"]) == []


# ------------------------------------------- connecting with the reader credential


class _IndexSpy:
    """Records create_index calls on every Motor collection, then delegates."""

    def __init__(self, real):
        self.real = real
        self.calls: list[tuple[str, object]] = []

    def install(self, monkeypatch) -> "_IndexSpy":
        spy = self

        async def create_index(collection, keys, **kwargs):
            spy.calls.append((collection.name, keys))
            return await spy.real(collection, keys, **kwargs)

        monkeypatch.setattr(AsyncIOMotorCollection, "create_index", create_index)
        return self


@pytest.fixture
def fresh_connection(monkeypatch):
    """Make the command open its own connection, as it does from the CLI.

    The suite's fixture has already connected and the command reuses that
    connection, so ``connect`` would never run. Calling the returned function
    (after the test rows are inserted) clears ``database`` and points the
    default URI at the test database, so the command takes the real path.
    """
    opened: list = []
    real_connect = async_mongo_manager.connect

    async def connect(connection_string=None, **kwargs):
        database = await real_connect(connection_string, **kwargs)
        opened.append(async_mongo_manager.client)
        return database

    def detach() -> list:
        monkeypatch.setattr("src.database.mongo.get_mongodb_uri", _resolve_test_mongo_uri)
        monkeypatch.setattr(async_mongo_manager, "connect", connect)
        monkeypatch.setattr(async_mongo_manager, "client", async_mongo_manager.client)
        monkeypatch.setattr(async_mongo_manager, "database", None)
        return opened

    yield detach
    monkeypatch.undo()
    for client in opened:
        client.close()


@pytest.mark.asyncio
async def test_dry_run_connects_without_creating_indexes(world, fresh_connection, monkeypatch):
    _, email = await _user(world, "reader")
    opened = fresh_connection()
    spy = _IndexSpy(AsyncIOMotorCollection.create_index).install(monkeypatch)

    summary = await send_cohort_invite(world["cohort"], emails=[email])

    assert len(opened) == 1
    assert summary["recipients"] == 1
    assert spy.calls == []


@pytest.mark.asyncio
async def test_apply_still_ensures_indexes(world, fake_mailer, pauses, fresh_connection, monkeypatch):
    _, email = await _user(world, "writer")
    opened = fresh_connection()
    spy = _IndexSpy(AsyncIOMotorCollection.create_index).install(monkeypatch)

    summary = await send_cohort_invite(world["cohort"], apply=True, emails=[email], sleep_seconds=0)

    assert len(opened) == 1
    assert summary["sent"] == 1
    touched = {name for name, _ in spy.calls}
    assert {"documents", COLLECTION} <= touched


@pytest.mark.no_db
@pytest.mark.asyncio
@pytest.mark.parametrize("ensure, expected", [(True, 2), (False, 0)])
async def test_connect_ensure_indexes_flag(monkeypatch, ensure, expected):
    collection = MagicMock()
    collection.create_index = AsyncMock()
    client = MagicMock()
    client.admin.command = AsyncMock(return_value={"ok": 1})
    client.get_database.return_value = MagicMock(__getitem__=MagicMock(return_value=collection))
    monkeypatch.setattr("src.database.mongo.AsyncIOMotorClient", lambda _uri: client)

    manager = AsyncMongoDBManager()
    if ensure:
        await manager.connect("mongodb://unused")
    else:
        await manager.connect("mongodb://unused", ensure_indexes=False)

    assert collection.create_index.await_count == expected
