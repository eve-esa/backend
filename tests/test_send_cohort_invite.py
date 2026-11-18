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
    MAIL_SPECS,
    mail_template_store,
    render_cohort_invite,
)
from src.services.mail_templates import BUNDLED_DIR

COHORT_INVITE_SUBJECT = MAIL_SPECS["cohort_invite"].subject
from tests.conftest import _resolve_test_mongo_uri


class FakeMailer:
    """Stands in for send_mail: records recipients, can fail chosen addresses."""

    def __init__(self, fail_for: set[str] | None = None):
        self.sent: list[dict] = []
        self.fallback_on_uncertain: list[bool] = []
        self.fail_for = fail_for or set()

    async def __call__(
        self, to: str, subject: str, html: str, text: str, *, fallback_on_uncertain: bool = True
    ) -> None:
        self.fallback_on_uncertain.append(fallback_on_uncertain)
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
def test_read_emails_file_skips_comments_and_blanks_and_keeps_repeats(tmp_path):
    # Repeats stay: plan_cohort normalises once and counts them as duplicate_email.
    path = tmp_path / "list.txt"
    path.write_text("# vip list\nAlice@Example.org\n\n  bob@example.org \nalice@example.org\n")
    assert read_emails_file(path) == ["Alice@Example.org", "bob@example.org", "alice@example.org"]


@pytest.mark.no_db
def test_read_emails_file_drops_a_byte_order_mark(tmp_path):
    """Spreadsheet exports start with U+FEFF, which str.strip() keeps."""
    path = tmp_path / "list.txt"
    path.write_bytes("\ufeffalice@example.org\r\nbob@example.org\r\n".encode("utf-8"))
    assert read_emails_file(path) == ["alice@example.org", "bob@example.org"]


@pytest.mark.no_db
@pytest.mark.parametrize(
    "value",
    [
        "a@b.org # vip",
        "Alice <x@y.org>",
        "a@b.org,Alice",
        "@",
        "x@",
        "@example.org",
        "x@localhost",
        "x@example.",
        "x@@example.org",
        "x@a@example.org",
        "x y@example.org",
        "x\u200b@example.org",
        "\ufeffx@example.org",
        "#x@example.org",
    ],
)
def test_a_value_that_is_not_one_bare_address_is_invalid(value):
    assert not command.is_bare_address(value)


@pytest.mark.no_db
@pytest.mark.parametrize("value", ["a@b.org", "first.last+tag@mail.example.co.uk", "o'neil@example.org"])
def test_a_bare_address_is_valid(value):
    assert command.is_bare_address(value)


@pytest.mark.no_db
def test_render_text_block_overrides_the_body_and_is_escaped():
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
    for structured in ("How to sign in", "HOW TO SIGN IN", "pre-registered"):
        assert structured not in html
        assert structured not in text
    for kept in ("Kind regards,", "You are receiving this message because"):
        assert kept in html
        assert kept in text


@pytest.mark.no_db
def test_render_default_is_the_structured_launch_mail():
    url = "https://eve-chat.chat"
    subject, html, text = render_cohort_invite("dave@example.org", sign_in_url=url)
    assert subject == "EVE is now open, come try it"
    assert "EVE is open to everyone" in html
    assert "EVE is open to everyone" in text
    # The button and the link in the sign-in section both point at the URL.
    assert html.count(f'href="{url}"') >= 2
    assert f'<a href="{url}" style="color:#003247;">{url}</a>' in html
    assert 'href="mailto:eve@picampus-school.com"' in html
    for section in ("How to sign in", "What EVE does", "What comes next", "Your feedback"):
        assert section in html
        assert section.upper() in text
    assert f"sign in at {url} with the e-mail address" in text
    assert "eve@picampus-school.com" in text
    assert "Forgot password" in text
    assert "<" not in text
    assert "dave@example.org" in text
    for gone in ("EVE, the Earth Virtual Expert,", "\u2014", "\u00b7"):
        assert gone not in html
        assert gone not in text


@pytest.mark.no_db
def test_mail_templates_carry_no_em_dash_or_middle_dot():
    for path in sorted(BUNDLED_DIR.iterdir()):
        content = path.read_text(encoding="utf-8")
        assert "\u2014" not in content, path.name
        assert "\u00b7" not in content, path.name


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
    # The list names ok_none twice (once through the upper-case row's address): one
    # recipient, the repeat counted. The upper-case users row itself is not matched.
    assert plan.skipped["duplicate_email"] == 1
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
async def test_apply_refreshes_the_templates_once_and_logs_their_source(
    world, fake_mailer, pauses, monkeypatch, caplog, capsys
):
    """A one-off task has no lifespan: it must fetch the pack before the first send."""
    emails = [(await _user(world, f"tpl{i}"))[1] for i in range(3)]
    refreshes = []

    async def _refresh():
        refreshes.append(len(fake_mailer.sent))
        return False

    monkeypatch.setattr(mail_template_store, "refresh", _refresh)
    with caplog.at_level(logging.INFO, logger=command.__name__):
        summary = await send_cohort_invite(
            world["cohort"], apply=True, emails=emails, sleep_seconds=0
        )

    assert summary["sent"] == 3
    assert refreshes == [0]
    assert "Mail templates: bundled" in capsys.readouterr().out
    sent_lines = [r.getMessage() for r in caplog.records if "cohort_invite_sent" in r.getMessage()]
    assert len(sent_lines) == 3
    for line in sent_lines:
        assert f"cohort={world['cohort']}" in line
        assert "template_source=bundled pack_version=none" in line
        assert "@" not in line


@pytest.mark.asyncio
async def test_dry_run_makes_no_s3_call_and_prints_no_template_line(
    world, fake_mailer, pauses, monkeypatch, capsys
):
    """The dry run output is the same as before the pack existed."""
    emails = [(await _user(world, f"dry{i}"))[1] for i in range(2)]
    refreshes = []

    async def _refresh():
        refreshes.append(True)
        return False

    monkeypatch.setattr(mail_template_store, "refresh", _refresh)
    summary = await send_cohort_invite(world["cohort"], emails=emails, sleep_seconds=0)

    assert summary["recipients"] == 2
    assert refreshes == []
    assert "Mail templates" not in capsys.readouterr().out
    assert fake_mailer.sent == []


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
@pytest.mark.parametrize("ensure, expected", [(True, 3), (False, 0)])
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


# ------------------------------------- pre-registered addresses without an account


SIGNUP_SUBJECT = MAIL_SPECS["cohort_invite_signup"].subject


class Killed(BaseException):
    """A process kill between the send and the stamp: no except clause catches it."""


# Operators Amazon DocumentDB 5.0 accepts in a partialFilterExpression, from
# https://docs.aws.amazon.com/documentdb/latest/devguide/partial-index.html
# ("Supported operators", read 2026-11-18). $and only at the top level; $all,
# $mod, $or, $xor, $not, $nor and array values are refused.
DOCUMENTDB_PARTIAL_FILTER_OPERATORS = {"$eq", "$exists", "$and", "$gt", "$gte", "$lt", "$lte"}


def _operators(expression, *, top_level=True) -> list[tuple[str, bool]]:
    """Every $operator in a filter expression, with whether it sits at the top."""
    found: list[tuple[str, bool]] = []
    if isinstance(expression, dict):
        for key, value in expression.items():
            if key.startswith("$"):
                found.append((key, top_level))
            found += _operators(value, top_level=False)
    elif isinstance(expression, list):
        for item in expression:
            found += _operators(item, top_level=False)
    return found


@pytest.mark.no_db
def test_claim_index_filters_use_only_documentdb_operators():
    specs = command.CLAIM_INDEXES
    assert {spec["name"] for spec in specs} == {
        "cohort_invites_cohort_user",
        "cohort_invites_cohort_email",
    }
    for spec in specs:
        assert spec["unique"] is True
        operators = _operators(spec["partialFilterExpression"])
        assert operators, spec["name"]
        for operator, top_level in operators:
            assert operator in DOCUMENTDB_PARTIAL_FILTER_OPERATORS, (spec["name"], operator)
            assert operator != "$and" or top_level, spec["name"]
        for value in spec["partialFilterExpression"].values():
            assert not isinstance(value, list), spec["name"]


@pytest.mark.no_db
def test_cli_takes_a_sign_up_url():
    args = command._parse_args(["--cohort", "x", "--sign-up-url", "https://eve.example.org"])
    assert args.sign_up_url == "https://eve.example.org"


@pytest.mark.asyncio
async def test_listed_address_without_account_gets_the_create_account_mail(
    world, fake_mailer, pauses, caplog
):
    address = f"prereg-{world['tag']}@example.org"
    with caplog.at_level(logging.INFO, logger=command.__name__):
        summary = await send_cohort_invite(
            world["cohort"],
            apply=True,
            emails=[address],
            sleep_seconds=0,
            sign_up_url="https://eve.example.org",
        )

    assert summary["sent"] == 1
    assert summary["sent_cohort_invite_signup"] == 1
    assert summary["sent_cohort_invite"] == 0
    assert [m["to"] for m in fake_mailer.sent] == [address]
    assert fake_mailer.sent[0]["subject"] == SIGNUP_SUBJECT
    assert 'href="https://eve.example.org"' in fake_mailer.sent[0]["html"]
    rows = await _sent_rows(world["cohort"])
    assert len(rows) == 1
    assert rows[0]["email"] == address
    assert "user_id" not in rows[0]  # absent, not null: the user index skips it
    assert rows[0]["sent_at"]
    lines = [r.getMessage() for r in caplog.records if "cohort_invite_sent" in r.getMessage()]
    assert len(lines) == 1 and "mail=cohort_invite_signup" in lines[0]
    assert "@" not in lines[0]


@pytest.mark.asyncio
async def test_address_listed_twice_in_any_case_gets_one_mail(world, fake_mailer, pauses):
    ghost = f"Twice-{world['tag']}@Example.org"
    _, known = await _user(world, "known")
    emails = [ghost, ghost.lower(), f"  {ghost.upper()} ", known.upper(), known]

    summary = await send_cohort_invite(world["cohort"], apply=True, emails=emails, sleep_seconds=0)

    assert sorted(m["to"] for m in fake_mailer.sent) == sorted([ghost.lower(), known])
    assert summary["sent"] == 2
    assert summary["skipped_duplicate_email"] == 3
    assert len(await _sent_rows(world["cohort"])) == 2


@pytest.mark.asyncio
async def test_two_rows_for_one_address_never_both_get_mail(world, fake_mailer, pauses):
    # Two users rows with one address: the plan drops the second row as a
    # duplicate before any claim is written.
    _, email = await _user(world, "twin")
    await _user(world, None, email=email)
    summary = await send_cohort_invite(world["cohort"], apply=True, emails=[email], sleep_seconds=0)
    assert [m["to"] for m in fake_mailer.sent] == [email]
    assert summary["skipped_duplicate_email"] == 1


@pytest.mark.asyncio
async def test_account_already_mailed_in_the_cohort_is_not_mailed_again(world, fake_mailer, pauses):
    """On both lists: the first send claimed (user_id, cohort) with no email field."""
    user_id, email = await _user(world, "legacy")
    await get_collection(COLLECTION).insert_one(
        {
            "user_id": user_id,
            "cohort": world["cohort"],
            "claimed_at": datetime.now(timezone.utc),
            "sent_at": datetime.now(timezone.utc),
        }
    )
    summary = await send_cohort_invite(world["cohort"], apply=True, emails=[email], sleep_seconds=0)
    assert summary["skipped_already_sent"] == 1
    assert fake_mailer.sent == []


@pytest.mark.asyncio
async def test_create_account_mail_then_account_is_not_mailed_again(world, fake_mailer, pauses):
    address = f"later-{world['tag']}@example.org"
    first = await send_cohort_invite(world["cohort"], apply=True, emails=[address], sleep_seconds=0)
    assert first["sent_cohort_invite_signup"] == 1

    # The person creates an account; the next run of the cohort finds the row.
    await _user(world, None, email=address)
    second = await send_cohort_invite(world["cohort"], apply=True, emails=[address], sleep_seconds=0)
    assert second["sent"] == 0
    assert second["skipped_already_sent"] == 1
    assert len(fake_mailer.sent) == 1


@pytest.mark.asyncio
async def test_account_created_after_the_plan_gets_the_account_mail_only(
    world, fake_mailer, pauses, monkeypatch, capsys
):
    address = f"between-{world['tag']}@example.org"
    dry = await send_cohort_invite(world["cohort"], emails=[address])
    assert dry["recipients_cohort_invite_signup"] == 1
    capsys.readouterr()

    real_plan = command.plan_cohort
    created: list[str] = []

    async def plan_then_sign_up(*args, **kwargs):
        plan = await real_plan(*args, **kwargs)
        created.append((await _user(world, None, email=address))[0])
        return plan

    monkeypatch.setattr(command, "plan_cohort", plan_then_sign_up)
    summary = await send_cohort_invite(world["cohort"], apply=True, emails=[address], sleep_seconds=0)

    assert summary["sent_cohort_invite"] == 1
    assert summary["sent_cohort_invite_signup"] == 0
    assert [m["subject"] for m in fake_mailer.sent] == [COHORT_INVITE_SUBJECT]
    rows = await _sent_rows(world["cohort"])
    assert len(rows) == 1
    assert rows[0]["user_id"] == created[0]
    assert rows[0]["email"] == address


@pytest.mark.asyncio
async def test_rerun_after_a_crash_sends_nothing_twice(world, fake_mailer, pauses, monkeypatch):
    _, account = await _user(world, "crash-account")
    signup = f"crash-signup-{world['tag']}@example.org"
    sent: list[str] = []

    async def send_then_die(to, subject, html, text, **_):
        sent.append(to)
        if to == signup:
            raise Killed()  # the mail left, the stamp never happened

    monkeypatch.setattr(command, "send_mail", send_then_die)
    with pytest.raises(Killed):
        await send_cohort_invite(
            world["cohort"], apply=True, emails=[account, signup], sleep_seconds=0
        )
    rows = {row["email"]: row for row in await _sent_rows(world["cohort"])}
    assert rows[account]["sent_at"] and rows[signup]["sent_at"] is None

    rerun = await send_cohort_invite(
        world["cohort"], apply=True, emails=[account, signup], sleep_seconds=0
    )
    assert rerun["sent"] == 0
    assert rerun["skipped_already_sent"] == 1
    assert rerun["skipped_in_doubt"] == 1
    assert sent == [account, signup]


@pytest.mark.asyncio
async def test_failed_smtp_send_logs_reason_and_code_never_the_address(
    world, fake_mailer, pauses, monkeypatch, caplog
):
    from src.services.mailer import MailDeliveryFailed

    _, account = await _user(world, "refused")
    signup = f"refused-signup-{world['tag']}@example.org"

    async def refuse(to, subject, html, text, **_):
        if to == account:
            raise MailDeliveryFailed(
                f"SMTP send failed for {to}", reason="SMTPDataError", code=550
            )
        raise RuntimeError(f"relay said no to {to}")

    monkeypatch.setattr(command, "send_mail", refuse)
    with caplog.at_level(logging.INFO, logger=command.__name__):
        summary = await send_cohort_invite(
            world["cohort"], apply=True, emails=[account, signup], sleep_seconds=0
        )

    assert summary["failed"] == 2
    assert summary["failed_in_doubt"] == 0
    assert await _sent_rows(world["cohort"]) == []  # released for the next run
    failed = [r for r in caplog.records if "cohort_invite_failed" in r.getMessage()]
    assert len(failed) == 2
    messages = [r.getMessage() for r in failed]
    assert all("claim=released" in m for m in messages)
    assert any("reason=SMTPDataError code=550" in m for m in messages)
    assert any("reason=RuntimeError" in m for m in messages)
    for record, message in zip(failed, messages):
        assert f"cohort={world['cohort']}" in message
        assert "claim_id=" in message
        assert "@" not in message
        assert record.exc_info is None
    assert account not in caplog.text and signup not in caplog.text


@pytest.mark.asyncio
async def test_dry_run_prints_one_count_per_mail_type(world, fake_mailer, pauses, capsys):
    accounts = [(await _user(world, f"type{i}"))[1] for i in range(2)]
    signups = [f"type-signup{i}-{world['tag']}@example.org" for i in range(3)]

    summary = await send_cohort_invite(world["cohort"], emails=accounts + signups, batch_size=2)

    assert summary["recipients"] == 5
    assert summary["recipients_cohort_invite"] == 2
    assert summary["recipients_cohort_invite_signup"] == 3
    out = capsys.readouterr().out
    assert "Launch mail to an account (cohort_invite): 2" in out
    assert "Create-account mail (cohort_invite_signup): 3" in out
    assert "Recipients: 5 in 3 batch(es)" in out
    for address in accounts + signups:
        assert address not in out
    assert fake_mailer.sent == []
    assert await _sent_rows(world["cohort"]) == []


@pytest.mark.asyncio
async def test_claim_indexes_replace_the_legacy_index_and_keep_old_claims(world):
    collection = get_collection(f"cohort_invites_idx_{world['tag']}")
    try:
        await collection.create_index(
            [("user_id", 1), ("cohort", 1)], name="cohort_invites_user_cohort", unique=True
        )
        await collection.insert_one({"user_id": "u1", "cohort": "c", "sent_at": None})

        await command.ensure_claim_indexes(collection)
        await command.ensure_claim_indexes(collection)  # a second run is a no-op

        info = await collection.index_information()
        assert "cohort_invites_user_cohort" not in info
        assert info["cohort_invites_cohort_user"]["partialFilterExpression"] == {
            "user_id": {"$exists": True}
        }
        assert info["cohort_invites_cohort_email"]["partialFilterExpression"] == {
            "email": {"$exists": True}
        }

        from pymongo.errors import DuplicateKeyError

        # The old claim keeps its user_id uniqueness.
        with pytest.raises(DuplicateKeyError):
            await collection.insert_one({"user_id": "u1", "cohort": "c", "email": "a@x.org"})
        # Two claims without a user id do not collide on a missing field.
        await collection.insert_one({"cohort": "c", "email": "b@x.org"})
        await collection.insert_one({"cohort": "c", "email": "c@x.org"})
        # A user claim and an email claim for one address collide.
        with pytest.raises(DuplicateKeyError):
            await collection.insert_one({"user_id": "u2", "cohort": "c", "email": "b@x.org"})
        await collection.insert_one({"cohort": "other", "email": "b@x.org"})
    finally:
        await collection.drop()


# ------------------------------------------------------------- review fixes


def _unclear(to: str):
    from src.services.mailer import MailDeliveryFailed

    return MailDeliveryFailed(
        f"SMTP send result unknown for {to}", reason="SMTPServerDisconnected", uncertain=True
    )


@pytest.mark.asyncio
async def test_twin_row_of_an_account_with_an_old_claim_gets_no_mail(world, fake_mailer, pauses):
    """The first row holds a first-send claim (user_id, no email): the twin is covered."""
    first_id, email = await _user(world, "twin-legacy")
    await _user(world, None, email=email)
    await get_collection(COLLECTION).insert_one(
        {
            "user_id": first_id,
            "cohort": world["cohort"],
            "claimed_at": datetime.now(timezone.utc),
            "sent_at": datetime.now(timezone.utc),
        }
    )

    summary = await send_cohort_invite(world["cohort"], apply=True, emails=[email], sleep_seconds=0)

    assert fake_mailer.sent == []
    assert summary["sent"] == 0
    assert summary["skipped_already_sent"] == 1
    assert summary["skipped_duplicate_email"] == 1
    assert len(await _sent_rows(world["cohort"])) == 1


@pytest.mark.asyncio
async def test_every_send_forbids_a_second_copy_through_the_fallback(world, fake_mailer, pauses):
    """An unclear SMTP send must stay in doubt, never be resent through SES."""
    _, account = await _user(world, "nofallback")
    signup = f"nofallback-signup-{world['tag']}@example.org"

    await send_cohort_invite(world["cohort"], apply=True, emails=[account, signup], sleep_seconds=0)

    assert [m["to"] for m in fake_mailer.sent] == [account, signup]
    assert fake_mailer.fallback_on_uncertain == [False, False]


@pytest.mark.asyncio
async def test_unclear_send_keeps_the_claim_and_a_rerun_sends_nothing(
    world, fake_mailer, pauses, monkeypatch, caplog
):
    """A disconnect after the message was on its way: the ECS restart must not resend."""
    _, account = await _user(world, "unclear")
    signup = f"unclear-signup-{world['tag']}@example.org"
    attempts: list[str] = []

    async def drop_the_line(to, subject, html, text, **_):
        attempts.append(to)
        raise _unclear(to)

    monkeypatch.setattr(command, "send_mail", drop_the_line)
    with caplog.at_level(logging.INFO, logger=command.__name__):
        first = await send_cohort_invite(
            world["cohort"], apply=True, emails=[account, signup], sleep_seconds=0
        )

    assert first["failed"] == 2  # the exit code stays 1
    assert first["failed_in_doubt"] == 2
    rows = await _sent_rows(world["cohort"])
    assert len(rows) == 2 and all(row["sent_at"] is None for row in rows)
    lines = [r.getMessage() for r in caplog.records if "cohort_invite_failed" in r.getMessage()]
    assert len(lines) == 2
    for line in lines:
        assert "claim=in_doubt" in line
        assert "reason=SMTPServerDisconnected" in line
        assert "@" not in line
    batch = [r.getMessage() for r in caplog.records if "cohort_invite_batch" in r.getMessage()]
    assert "failed=2 in_doubt=2" in batch[0]

    rerun = await send_cohort_invite(
        world["cohort"], apply=True, emails=[account, signup], sleep_seconds=0
    )
    assert rerun["skipped_in_doubt"] == 2
    assert attempts == [account, signup]


@pytest.mark.asyncio
async def test_definite_refusal_releases_the_claim_and_a_rerun_retries(
    world, fake_mailer, pauses, monkeypatch
):
    from src.services.mailer import MailDeliveryFailed

    _, account = await _user(world, "refused-once")
    calls: list[str] = []

    async def refuse_once(to, subject, html, text, **_):
        calls.append(to)
        if len(calls) == 1:
            raise MailDeliveryFailed(
                "SMTP send failed", reason="SMTPRecipientsRefused", code=550, status="5.1.1"
            )

    monkeypatch.setattr(command, "send_mail", refuse_once)
    first = await send_cohort_invite(world["cohort"], apply=True, emails=[account], sleep_seconds=0)
    assert (first["failed"], first["failed_in_doubt"]) == (1, 0)
    assert await _sent_rows(world["cohort"]) == []

    second = await send_cohort_invite(world["cohort"], apply=True, emails=[account], sleep_seconds=0)
    assert second["sent"] == 1
    assert calls == [account, account]


@pytest.mark.asyncio
async def test_failure_line_carries_the_enhanced_status(world, fake_mailer, pauses, monkeypatch, caplog):
    from src.services.mailer import MailDeliveryFailed

    _, account = await _user(world, "quota")

    async def daily_limit(to, subject, html, text, **_):
        raise MailDeliveryFailed("SMTP send failed", reason="SMTPDataError", code=550, status="5.4.5")

    monkeypatch.setattr(command, "send_mail", daily_limit)
    with caplog.at_level(logging.INFO, logger=command.__name__):
        await send_cohort_invite(world["cohort"], apply=True, emails=[account], sleep_seconds=0)
    [line] = [r.getMessage() for r in caplog.records if "cohort_invite_failed" in r.getMessage()]
    assert "claim=released reason=SMTPDataError code=550 status=5.4.5" in line


@pytest.mark.asyncio
async def test_users_row_stored_in_mixed_case_is_found(world, fake_mailer, pauses):
    user_id, _ = await _user(world, None, email=f" Mixed-{world['tag']}@Example.ORG ")
    listed = f"mixed-{world['tag']}@example.org"

    plan = await plan_cohort(world["cohort"], emails=[listed.upper()])
    assert [(r.user_id, r.mail) for r in plan.recipients] == [(user_id, "cohort_invite")]
    assert plan.not_found == 0

    summary = await send_cohort_invite(world["cohort"], apply=True, emails=[listed], sleep_seconds=0)
    assert summary["sent_cohort_invite"] == 1
    assert summary["sent_cohort_invite_signup"] == 0
    [row] = await _sent_rows(world["cohort"])
    assert row["user_id"] == user_id and row["email"] == listed


@pytest.mark.asyncio
async def test_mixed_case_account_created_after_the_plan_gets_the_account_mail(
    world, fake_mailer, pauses, monkeypatch
):
    listed = f"late-mixed-{world['tag']}@example.org"
    real_plan = command.plan_cohort
    created: list[str] = []

    async def plan_then_sign_up(*args, **kwargs):
        plan = await real_plan(*args, **kwargs)
        created.append((await _user(world, None, email=listed.title()))[0])
        return plan

    monkeypatch.setattr(command, "plan_cohort", plan_then_sign_up)
    summary = await send_cohort_invite(world["cohort"], apply=True, emails=[listed], sleep_seconds=0)
    assert summary["sent_cohort_invite"] == 1
    [row] = await _sent_rows(world["cohort"])
    assert row["user_id"] == created[0]


@pytest.mark.asyncio
async def test_address_regex_is_escaped(world, fake_mailer, pauses):
    """A dot in the listed address must not match any character in a stored one."""
    await _user(world, None, email=f"dotXcom-{world['tag']}@exampleXorg")
    plan = await plan_cohort(world["cohort"], emails=[f"dot.com-{world['tag']}@example.org"])
    assert plan.not_found == 1
    assert [r.mail for r in plan.recipients] == ["cohort_invite_signup"]


@pytest.mark.asyncio
async def test_invalid_list_values_are_counted_and_never_mailed(world, fake_mailer, pauses):
    bad = ["a@b.org # vip", "Alice <x@y.org>", "a@b.org,Alice", "@", "x@"]
    good = f"valid-{world['tag']}@example.org"
    summary = await send_cohort_invite(world["cohort"], apply=True, emails=bad + [good], sleep_seconds=0)
    assert summary["skipped_invalid_email"] == 5
    assert [m["to"] for m in fake_mailer.sent] == [good]


@pytest.mark.asyncio
async def test_file_path_counts_repeats_and_ignores_the_byte_order_mark(
    world, fake_mailer, pauses, tmp_path, capsys
):
    _, known = await _user(world, "bom")
    ghost = f"bom-ghost-{world['tag']}@example.org"
    path = tmp_path / "list.txt"
    path.write_bytes(f"﻿{known}\r\n{known.upper()}\r\n{ghost}\r\n# note\r\n{ghost}\r\n".encode("utf-8"))
    # What main() does with --emails-file (asyncio.run cannot nest in this loop).
    await send_cohort_invite(world["cohort"], emails=read_emails_file(path))
    out = capsys.readouterr().out
    assert "Skipped duplicate_email: 2" in out
    assert "Launch mail to an account (cohort_invite): 1" in out
    assert "Create-account mail (cohort_invite_signup): 1" in out
    assert "Skipped invalid_email" not in out


@pytest.mark.asyncio
async def test_a_render_error_leaves_no_claim(world, fake_mailer, pauses, monkeypatch):
    signup = f"render-{world['tag']}@example.org"

    def broken(*args, **kwargs):
        raise ValueError("template context mismatch")

    monkeypatch.setattr(command, "build_cohort_invite_signup", broken)
    with pytest.raises(ValueError):
        await send_cohort_invite(world["cohort"], apply=True, emails=[signup], sleep_seconds=0)
    assert await _sent_rows(world["cohort"]) == []
    assert fake_mailer.sent == []


@pytest.mark.asyncio
async def test_max_sends_caps_the_run_and_the_dry_run_says_so(world, fake_mailer, pauses, capsys):
    emails = [(await _user(world, f"cap{i}"))[1] for i in range(5)]

    dry = await send_cohort_invite(world["cohort"], emails=emails, max_sends=2, batch_size=2)
    assert (dry["recipients"], dry["to_send"]) == (5, 2)
    assert "Max sends: 2, this run sends at most 2" in capsys.readouterr().out

    first = await send_cohort_invite(
        world["cohort"], apply=True, emails=emails, max_sends=2, batch_size=1, sleep_seconds=3
    )
    assert first["sent"] == 2
    assert first["left_by_max_sends"] == 3
    assert pauses == [3]  # no pause after the cap is reached
    assert "Stopped at --max-sends 2: 3 recipient(s) left" in capsys.readouterr().out

    second = await send_cohort_invite(world["cohort"], apply=True, emails=emails, sleep_seconds=0)
    assert (second["sent"], second["skipped_already_sent"]) == (3, 2)
    assert sorted(m["to"] for m in fake_mailer.sent) == sorted(emails)


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_max_sends_must_be_positive():
    with pytest.raises(CohortInviteError):
        await send_cohort_invite("x", max_sends=0)


@pytest.mark.no_db
def test_cli_takes_max_sends():
    args = command._parse_args(["--cohort", "x", "--emails-file", "a.txt", "--max-sends", "300"])
    assert args.max_sends == 300


@pytest.mark.asyncio
async def test_account_gone_or_readdressed_after_the_plan_is_not_mailed(
    world, fake_mailer, pauses, monkeypatch
):
    gone_id, gone = await _user(world, "gone")
    moved_id, moved = await _user(world, "moved")
    real_plan = command.plan_cohort

    async def plan_then_change(*args, **kwargs):
        plan = await real_plan(*args, **kwargs)
        await get_collection("users").delete_one({"_id": command._row_key(gone_id)})
        await get_collection("users").update_one(
            {"_id": command._row_key(moved_id)}, {"$set": {"email": f"new-{world['tag']}@example.org"}}
        )
        return plan

    monkeypatch.setattr(command, "plan_cohort", plan_then_change)
    summary = await send_cohort_invite(world["cohort"], apply=True, emails=[gone, moved], sleep_seconds=0)
    assert summary["changed_since_plan"] == 2
    assert summary["sent"] == 0 and summary["failed"] == 0
    assert fake_mailer.sent == []
    assert await _sent_rows(world["cohort"]) == []


class _FakeIndexCollection:
    def __init__(self, drop_error=None):
        self.drop_error = drop_error
        self.created: list[str] = []

    async def create_index(self, keys, **kwargs):
        self.created.append(kwargs["name"])

    async def index_information(self):
        return {command.LEGACY_CLAIM_INDEX: {}}

    async def drop_index(self, name):
        if self.drop_error:
            raise self.drop_error


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_legacy_index_dropped_by_another_run_is_not_an_error():
    from pymongo.errors import OperationFailure

    collection = _FakeIndexCollection(OperationFailure("index not found with name", code=27))
    await command.ensure_claim_indexes(collection)
    assert collection.created == ["cohort_invites_cohort_user", "cohort_invites_cohort_email"]


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_other_drop_failures_still_raise():
    from pymongo.errors import OperationFailure

    collection = _FakeIndexCollection(OperationFailure("not authorized", code=13))
    with pytest.raises(OperationFailure):
        await command.ensure_claim_indexes(collection)


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_claim_queries_carry_the_partial_index_filter(monkeypatch):
    seen: list[dict] = []

    class _Cursor:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    class _Collection:
        def find(self, query):
            seen.append(query)
            return _Cursor()

    monkeypatch.setattr(command, "get_collection", lambda name: _Collection())
    await command._claims_by("c", "user_id", ["u1"])
    await command._claims_by("c", "email", ["a@b.org"])
    for key, query in zip(("user_id", "email"), seen):
        # The index's own filter, verbatim, inside a top-level $and.
        assert {key: {"$exists": True}} in query["$and"]
        assert {"cohort": "c", key: {"$in": ["u1"] if key == "user_id" else ["a@b.org"]}} in query["$and"]
