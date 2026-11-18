"""The mail template pack: loaded from S3, validated whole, bundled copy as fallback.

A mail is the one output of this service a person reads outside the app, and a
broken pack must never stop it from leaving. These tests pin the three promises
of ``src/services/mail_templates.py``: a blank bucket changes nothing, a bad pack
is rejected whole and the last good one stays, and a send never waits for S3.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import time
from pathlib import Path

import boto3
import pytest
import yaml
from botocore.exceptions import ClientError, ReadTimeoutError
from botocore.response import StreamingBody
from botocore.stub import Stubber

from src.commands import render_mail_pack
from src.services import account_notifications, mail_templates
from src.services.account_notifications import (
    KIND_APPROVED,
    KIND_PENDING,
    MAIL_SPECS,
    mail_template_store,
    notify_account_pending,
    render_account_mail,
    render_cohort_invite,
)
from src.services.mail_templates import (
    BUNDLED_DIR,
    MANIFEST_KEY,
    MailTemplateStore,
    PackRejected,
    load_pack,
)

pytestmark = pytest.mark.no_db

FRONTEND = "https://eve.example.org"
EMAIL = "person@example.com"
BUCKET = "eve-test-mail-templates"
GOLDEN = Path(__file__).parent / "golden" / "mail"

BASE = (
    "<div>{{ logo_url }}|{% block title %}{% endblock %}|"
    "{% block body %}{% endblock %}|{{ app_url }}</div>"
)
PENDING_HTML = (
    '{% extends "base.html" %}{% block title %}Pack hold{% endblock %}'
    "{% block body %}PACK pending for {{ email }}{% endblock %}"
)
PENDING_TXT = "PACK pending for {{ email }} at {{ app_url }}"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_pack(
    files: dict[str, str] | None = None,
    mails: dict | None = None,
    *,
    version: str = "1.0.0",
    contract: int = 1,
) -> tuple[bytes, dict[str, bytes]]:
    """Return ``(manifest bytes, {key: bytes})`` for a pack."""
    files = files if files is not None else {
        "base.html": BASE,
        "account_pending.html": PENDING_HTML,
        "account_pending.txt": PENDING_TXT,
    }
    mails = mails if mails is not None else {
        "account_pending": {
            "subject": "Pack subject",
            "html": "account_pending.html",
            "text": "account_pending.txt",
        }
    }
    blobs = {name: body.encode("utf-8") for name, body in files.items()}
    manifest = {
        "contract": contract,
        "version": version,
        "files": {name: _sha(data) for name, data in blobs.items()},
        "mails": mails,
    }
    return yaml.safe_dump(manifest, sort_keys=False).encode("utf-8"), blobs


class FakeS3:
    """Enough of the S3 client for the loader: conditional manifest GET, plain GETs."""

    def __init__(self, manifest: bytes | None, files: dict[str, bytes], etag: str = '"e1"'):
        self.objects = dict(files)
        if manifest is not None:
            self.objects[MANIFEST_KEY] = manifest
        self.etag = etag
        self.calls: list[dict] = []
        self.delay = 0.0
        self.error: Exception | None = None

    def get_object(self, **kwargs):
        self.calls.append(kwargs)
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        key = kwargs["Key"]
        if key == MANIFEST_KEY and kwargs.get("IfNoneMatch") == self.etag:
            raise ClientError(
                {"Error": {"Code": "304", "Message": "Not Modified"},
                 "ResponseMetadata": {"HTTPStatusCode": 304}},
                "GetObject",
            )
        if key not in self.objects:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "missing"},
                 "ResponseMetadata": {"HTTPStatusCode": 404}},
                "GetObject",
            )
        data = self.objects[key]
        return {
            "Body": StreamingBody(io.BytesIO(data), len(data)),
            "ETag": self.etag if key == MANIFEST_KEY else '"f"',
        }


def _client_error(code: str, status: int) -> ClientError:
    return ClientError(
        {"Error": {"Code": code, "Message": code}, "ResponseMetadata": {"HTTPStatusCode": status}},
        "GetObject",
    )


def store_with(client, bucket: str = BUCKET) -> MailTemplateStore:
    return MailTemplateStore(MAIL_SPECS, bucket=bucket, client_factory=lambda: client)


@pytest.fixture(autouse=True)
def _frontend_url(monkeypatch):
    monkeypatch.setattr(account_notifications, "FRONTEND_URL", FRONTEND)


@pytest.fixture
def default_store(monkeypatch):
    """The process-wide store, isolated per test and restored after."""
    monkeypatch.setattr(mail_template_store, "_bucket", None)
    monkeypatch.setattr(mail_template_store, "_client_factory", None)
    monkeypatch.setattr(mail_template_store, "_client", None)
    monkeypatch.setattr(mail_template_store, "_pack", None)
    monkeypatch.setattr(mail_template_store, "_etag", None)
    monkeypatch.setattr(mail_template_store, "_task", None)
    monkeypatch.setattr(mail_template_store, "_last_failure_log", None)
    return mail_template_store


def _log_text(caplog) -> str:
    return "\n".join(record.getMessage() for record in caplog.records)


# ------------------------------------------------------------ 1. blank bucket


GOLDEN_CASES = {
    "account_pending": lambda: render_account_mail(KIND_PENDING, EMAIL),
    "account_approved": lambda: render_account_mail(KIND_APPROVED, EMAIL),
    "account_approved_after_hold": lambda: render_account_mail(KIND_APPROVED, EMAIL, after_hold=True),
    "cohort_invite": lambda: render_cohort_invite(EMAIL, sign_in_url="https://eve-chat.example.org"),
    "cohort_invite_paragraphs": lambda: render_cohort_invite(
        EMAIL,
        sign_in_url="https://eve-chat.example.org",
        text_block="First line\nwrapped.\n\nSecond <b>para</b> & more.",
        subject="Custom subject",
    ),
}


@pytest.mark.parametrize("name", sorted(GOLDEN_CASES))
def test_blank_bucket_renders_todays_output_byte_for_byte(name, default_store, monkeypatch):
    monkeypatch.setattr(mail_templates.config, "MAIL_TEMPLATES_BUCKET", "")

    def _no_client(*args, **kwargs):
        raise AssertionError("no S3 client may be built while the bucket is blank")

    monkeypatch.setattr(mail_templates.boto3, "client", _no_client)

    subject, html, text = GOLDEN_CASES[name]()

    assert subject == (GOLDEN / f"{name}.subject").read_text(encoding="utf-8")
    assert html == (GOLDEN / f"{name}.html").read_text(encoding="utf-8")
    assert text == (GOLDEN / f"{name}.txt").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_blank_bucket_starts_no_task_and_builds_no_client(default_store, monkeypatch, caplog):
    monkeypatch.setattr(mail_templates.config, "MAIL_TEMPLATES_BUCKET", "")
    monkeypatch.setattr(
        mail_templates.boto3, "client", lambda *a, **k: pytest.fail("client built")
    )
    with caplog.at_level(logging.INFO, logger=mail_templates.logger.name):
        await default_store.start()
        render_account_mail(KIND_PENDING, EMAIL)
        await default_store.stop()
    assert default_store._task is None
    # A blank bucket is today's behaviour, not a degradation: no signal.
    assert "mail_template" not in _log_text(caplog)


# ------------------------------------------------------------ 2. valid pack


@pytest.mark.asyncio
async def test_valid_pack_is_used_for_its_mails(default_store, monkeypatch, caplog):
    manifest, files = make_pack()
    client = boto3.client(
        "s3", region_name="eu-west-1", aws_access_key_id="x", aws_secret_access_key="x"
    )
    stub = Stubber(client)
    stub.add_response(
        "get_object",
        {"Body": StreamingBody(io.BytesIO(manifest), len(manifest)), "ETag": '"e1"'},
        {"Bucket": BUCKET, "Key": MANIFEST_KEY},
    )
    for key in sorted(files):
        stub.add_response(
            "get_object",
            {"Body": StreamingBody(io.BytesIO(files[key]), len(files[key])), "ETag": '"f"'},
            {"Bucket": BUCKET, "Key": key},
        )
    monkeypatch.setattr(default_store, "_bucket", BUCKET)
    monkeypatch.setattr(default_store, "_client_factory", lambda: client)

    with stub, caplog.at_level(logging.INFO):
        assert await default_store.refresh() is True
        stub.assert_no_pending_responses()

    sent = {}

    async def _capture(**kwargs):
        sent.update(kwargs)

    monkeypatch.setattr(account_notifications, "send_mail", _capture)
    with caplog.at_level(logging.INFO):
        await notify_account_pending("64b7f2c1a2b3c4d5e6f70123", EMAIL)

    assert sent["subject"] == "Pack subject"
    assert sent["html"] == f"<div>{FRONTEND}/branding/eve-logo.png|Pack hold|PACK pending for {EMAIL}|{FRONTEND}</div>"
    assert sent["text"] == f"PACK pending for {EMAIL} at {FRONTEND}"
    logged = _log_text(caplog)
    assert "mail_templates_loaded version=1.0.0 etag=" in logged
    assert (
        "account_mail_sent kind=pending user_id=64b7f2c1a2b3c4d5e6f70123 "
        "template_source=pack pack_version=1.0.0" in logged
    )


# ------------------------------------------------------------ 3. 304


@pytest.mark.asyncio
async def test_unchanged_manifest_fetches_no_file_and_keeps_the_environment():
    manifest, files = make_pack()
    s3 = FakeS3(manifest, files)
    store = store_with(s3)
    assert await store.refresh() is True
    pack = store._pack

    client = boto3.client(
        "s3", region_name="eu-west-1", aws_access_key_id="x", aws_secret_access_key="x"
    )
    stub = Stubber(client)
    stub.add_client_error(
        "get_object",
        service_error_code="304",
        http_status_code=304,
        expected_params={"Bucket": BUCKET, "Key": MANIFEST_KEY, "IfNoneMatch": '"e1"'},
    )
    store._client = client
    with stub:
        assert await store.refresh() is True
        stub.assert_no_pending_responses()
    assert store._pack is pack
    assert store._pack.env_html is pack.env_html
    assert store._pack.env_text is pack.env_text


# ------------------------------------------------------------ 4. swap mid render


@pytest.mark.asyncio
async def test_a_render_started_before_a_swap_finishes_on_the_old_pack():
    manifest_a, files_a = make_pack(version="1.0.0")
    s3 = FakeS3(manifest_a, files_a, etag='"a"')
    store = store_with(s3)
    await store.refresh()
    old = store._pack

    files_b = {
        "base.html": BASE,
        "account_pending.html": PENDING_HTML.replace("PACK", "NEW"),
        "account_pending.txt": PENDING_TXT.replace("PACK", "NEW"),
    }
    manifest_b, blobs_b = make_pack(files_b, version="2.0.0")
    original = old.env_html.get_template
    swapped = []

    def _get_template_then_swap(name, *args, **kwargs):
        template = original(name, *args, **kwargs)
        if not swapped:
            s3.objects.update(blobs_b)
            s3.objects[MANIFEST_KEY] = manifest_b
            s3.etag = '"b"'
            store._refresh_sync()
            swapped.append(True)
        return template

    old.env_html.get_template = _get_template_then_swap
    rendered = store.render(
        "account_pending", {"email": EMAIL, "app_url": FRONTEND, "logo_url": "L"}
    )
    assert store._pack.version == "2.0.0"
    assert "PACK pending" in rendered.html
    assert "PACK pending" in rendered.text
    assert rendered.version == "1.0.0"


# ------------------------------------------------------------ 5. rejections


def _bad_packs():
    good_manifest, good_files = make_pack()

    def tamper(manifest_change=None, files_change=None, raw_manifest=None):
        manifest = yaml.safe_load(good_manifest)
        if manifest_change:
            manifest_change(manifest)
        files = dict(good_files)
        if files_change:
            files_change(files)
        body = raw_manifest if raw_manifest is not None else yaml.safe_dump(manifest).encode()
        return body, files

    def rehash(files, name, body):
        files[name] = body.encode()

    def with_template(name, body):
        files = {"base.html": BASE, "account_pending.html": PENDING_HTML,
                 "account_pending.txt": PENDING_TXT, name: body}
        return make_pack(files)

    return {
        "malformed_yaml": (tamper(raw_manifest=b"contract: [1\n"), "yaml"),
        "wrong_contract": (tamper(lambda m: m.update(contract=2)), "contract"),
        "schema": (tamper(lambda m: m.pop("version")), "schema"),
        "hash_mismatch": (tamper(files_change=lambda f: rehash(f, "account_pending.txt", "changed")), "file"),
        "missing_file": (tamper(files_change=lambda f: f.pop("account_pending.txt")), "file"),
        "syntax_error": (with_template("account_pending.txt", "{% if %}"), "template"),
        "extends_outside": (with_template("account_pending.html", '{% extends "other.html" %}'), "template"),
        "undeclared_variable": (with_template("account_pending.txt", "{{ password }}"), "template"),
        "oversize": (tamper(raw_manifest=b"#" * (mail_templates.MAX_MANIFEST_BYTES + 1)), "size"),
        # Caught by the trial render: it would fail on every send.
        "render_error": (with_template("account_pending.txt", "{{ email.no_such_attribute }}"), "render"),
        "sandbox_escape": (with_template("account_pending.txt", "{{ ''.__class__.__mro__ }}"), "render"),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("case", sorted(_bad_packs()))
async def test_a_bad_pack_is_rejected_whole_and_the_last_good_one_stays(case, caplog):
    manifest, files = make_pack()
    s3 = FakeS3(manifest, files, etag='"good"')
    store = store_with(s3)
    await store.refresh()
    good = store._pack

    (bad_manifest, bad_files), reason = _bad_packs()[case]
    s3.objects = dict(bad_files)
    s3.objects[MANIFEST_KEY] = bad_manifest
    s3.etag = '"bad"'
    with caplog.at_level(logging.WARNING, logger=mail_templates.logger.name):
        await store.refresh()

    assert store._pack is good
    rejected = [r.getMessage() for r in caplog.records if "mail_templates_rejected" in r.getMessage()]
    assert len(rejected) == 1, rejected
    assert f"reason={reason}" in rejected[0]


# ------------------------------------------------------------ 6. render errors


# Fails only for the address the tests send to: the trial render at load uses
# another address, so the pack loads and the failure happens at send time.
DATA_DEPENDENT_ERROR = '{% if email == "person@example.com" %}{{ email.nope }}{% endif %}ok'


@pytest.mark.parametrize(
    "body",
    [
        DATA_DEPENDENT_ERROR,
        '{% if email == "person@example.com" %}{{ email.__class__.__mro__ }}{% endif %}ok',
    ],
)
def test_a_render_error_falls_back_to_the_bundled_copy(body, caplog):
    files = {"base.html": BASE, "account_pending.html": PENDING_HTML, "account_pending.txt": body}
    manifest, blobs = make_pack(files)
    store = store_with(FakeS3(manifest, blobs))
    store._refresh_sync()
    assert store._pack is not None

    with caplog.at_level(logging.WARNING, logger=mail_templates.logger.name):
        rendered = store.render(
            "account_pending", {"email": EMAIL, "app_url": FRONTEND, "logo_url": "L"}
        )

    assert rendered.source == "bundled"
    assert "registered and on hold" in rendered.text
    assert rendered.subject == MAIL_SPECS["account_pending"].subject
    fallbacks = [r.getMessage() for r in caplog.records if "mail_template_fallback" in r.getMessage()]
    assert len(fallbacks) == 1
    assert "name=account_pending" in fallbacks[0]
    assert "pack_version=1.0.0" in fallbacks[0]
    assert any(cls in fallbacks[0] for cls in ("reason=UndefinedError", "reason=SecurityError"))


@pytest.mark.asyncio
async def test_a_render_error_still_sends_the_mail(default_store, monkeypatch):
    files = {"base.html": BASE, "account_pending.html": PENDING_HTML,
             "account_pending.txt": DATA_DEPENDENT_ERROR}
    manifest, blobs = make_pack(files)
    monkeypatch.setattr(default_store, "_bucket", BUCKET)
    monkeypatch.setattr(default_store, "_client_factory", lambda: FakeS3(manifest, blobs))
    await default_store.refresh()
    sent = []

    async def _capture(**kwargs):
        sent.append(kwargs)

    monkeypatch.setattr(account_notifications, "send_mail", _capture)
    await notify_account_pending("64b7f2c1a2b3c4d5e6f70123", EMAIL)
    assert len(sent) == 1
    assert "registered and on hold" in sent[0]["text"]


# ------------------------------------------------------------ 7. autoescape


def test_pack_html_escapes_the_address_and_the_text_keeps_it_literal():
    manifest, blobs = make_pack()
    store = store_with(FakeS3(manifest, blobs))
    store._refresh_sync()
    address = 'a"><script>@example.com'
    rendered = store.render(
        "account_pending", {"email": address, "app_url": FRONTEND, "logo_url": "L"}
    )
    assert rendered.source == "pack"
    assert "<script>" not in rendered.html
    assert "&lt;script&gt;" in rendered.html
    assert address in rendered.text


# ------------------------------------------------------------ 8. startup failures


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        _client_error("NoSuchBucket", 404),
        _client_error("AccessDenied", 403),
        ReadTimeoutError(endpoint_url="https://s3.example"),
    ],
    ids=["no_such_bucket", "access_denied", "read_timeout"],
)
async def test_startup_failure_returns_and_sends_use_bundled(error, caplog):
    s3 = FakeS3(None, {})
    s3.error = error
    store = store_with(s3)
    with caplog.at_level(logging.WARNING, logger=mail_templates.logger.name):
        await store.start(first_wait_s=5.0)
        rendered = store.render(
            "account_pending", {"email": EMAIL, "app_url": FRONTEND, "logo_url": "L"}
        )
        await store.stop()
    assert rendered.source == "bundled"
    logged = _log_text(caplog)
    assert "mail_templates_refresh_failed reason=" in logged
    assert "mail_template_fallback name=account_pending reason=no_pack" in logged


@pytest.mark.asyncio
async def test_a_slow_bucket_never_holds_the_lifespan_past_its_cap():
    manifest, blobs = make_pack()
    s3 = FakeS3(manifest, blobs)
    s3.delay = 1.0
    store = store_with(s3)
    started = time.monotonic()
    await store.start(first_wait_s=0.2)
    elapsed = time.monotonic() - started
    rendered = store.render(
        "account_pending", {"email": EMAIL, "app_url": FRONTEND, "logo_url": "L"}
    )
    await store.stop()
    assert elapsed < 0.9
    assert rendered.source == "bundled"


@pytest.mark.asyncio
async def test_refresh_failures_are_logged_once_then_rate_limited(caplog):
    s3 = FakeS3(None, {})
    s3.error = _client_error("AccessDenied", 403)
    store = store_with(s3)
    with caplog.at_level(logging.WARNING, logger=mail_templates.logger.name):
        for _ in range(3):
            await store.refresh()
    failures = [r for r in caplog.records if "mail_templates_refresh_failed" in r.getMessage()]
    assert len(failures) == 1
    assert "reason=AccessDenied" in failures[0].getMessage()


# ------------------------------------------------------------ 9. send path


def test_no_s3_call_happens_on_the_render_path():
    manifest, blobs = make_pack()
    s3 = FakeS3(manifest, blobs)
    store = store_with(s3)
    store._refresh_sync()
    calls_after_load = len(s3.calls)
    for name, ctx in (
        ("account_pending", {"email": EMAIL, "app_url": FRONTEND, "logo_url": "L"}),
        ("account_approved", {"email": EMAIL, "app_url": FRONTEND, "logo_url": "L", "after_hold": False}),
    ):
        store.render(name, ctx)
    assert len(s3.calls) == calls_after_load


# ------------------------------------------------------------ 10. no address in logs


@pytest.mark.asyncio
async def test_no_log_line_carries_the_recipient_address(default_store, monkeypatch, caplog):
    files = {"base.html": BASE, "account_pending.html": PENDING_HTML,
             "account_pending.txt": DATA_DEPENDENT_ERROR}
    manifest, blobs = make_pack(files)
    s3 = FakeS3(manifest, blobs)
    monkeypatch.setattr(default_store, "_bucket", BUCKET)
    monkeypatch.setattr(default_store, "_client_factory", lambda: s3)

    async def _swallow(**kwargs):
        return None

    monkeypatch.setattr(account_notifications, "send_mail", _swallow)
    with caplog.at_level(logging.DEBUG):
        await default_store.refresh()
        await notify_account_pending("64b7f2c1a2b3c4d5e6f70123", EMAIL)
        await account_notifications.notify_account_approved("64b7f2c1a2b3c4d5e6f70123", EMAIL)
        s3.etag = '"x"'
        s3.objects["account_pending.txt"] = b"tampered"
        await default_store.refresh()
    assert caplog.records
    assert EMAIL not in _log_text(caplog)


# ------------------------------------------------------------ 11. absent mail


def test_a_mail_the_pack_does_not_carry_renders_from_bundled(caplog):
    manifest, blobs = make_pack()
    store = store_with(FakeS3(manifest, blobs))
    store._refresh_sync()
    with caplog.at_level(logging.WARNING, logger=mail_templates.logger.name):
        rendered = store.render(
            "account_approved",
            {"email": EMAIL, "app_url": FRONTEND, "logo_url": "L", "after_hold": False},
        )
    assert rendered.source == "bundled"
    assert rendered.subject == "Your EVE account is ready"
    assert "Welcome to EVE!" in rendered.html
    assert "mail_template_fallback name=account_approved reason=absent pack_version=1.0.0" in _log_text(caplog)


# ------------------------------------------------------------ 12. render command


def _write_pack(directory: Path, manifest: bytes, files: dict[str, bytes], fixtures: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / MANIFEST_KEY).write_bytes(manifest)
    for name, data in files.items():
        (directory / name).write_bytes(data)
    (directory / "fixtures.yaml").write_text(yaml.safe_dump(fixtures), encoding="utf-8")


FIXTURES = {
    "variants": [
        {"name": "pending", "mail": "account_pending",
         "context": {"email": EMAIL, "app_url": FRONTEND, "logo_url": "L"}},
        {"name": "pending-other", "mail": "account_pending",
         "context": {"email": "other@example.com", "app_url": FRONTEND, "logo_url": "L"}},
    ]
}


def test_render_command_writes_every_fixture_variant(tmp_path):
    manifest, files = make_pack()
    _write_pack(tmp_path / "pack", manifest, files, FIXTURES)
    code = render_mail_pack.main(["--pack", str(tmp_path / "pack"), "--out", str(tmp_path / "out")])
    assert code == 0
    for variant in ("pending", "pending-other"):
        assert (tmp_path / "out" / f"{variant}.html").read_text().startswith("<div>L|Pack hold|")
        assert "PACK pending" in (tmp_path / "out" / f"{variant}.txt").read_text()


@pytest.mark.parametrize(
    "breakage",
    ["bad_hash", "unknown_mail_fixture", "missing_variable", "render_error"],
)
def test_render_command_fails_on_a_bad_pack(tmp_path, breakage):
    manifest, files = make_pack()
    fixtures = {"variants": [dict(v) for v in FIXTURES["variants"]]}
    if breakage == "bad_hash":
        files["account_pending.txt"] = b"changed"
    elif breakage == "unknown_mail_fixture":
        fixtures["variants"].append({"name": "x", "mail": "nope", "context": {}})
    elif breakage == "missing_variable":
        fixtures["variants"][0] = {"name": "pending", "mail": "account_pending",
                                   "context": {"email": EMAIL, "app_url": FRONTEND}}
    elif breakage == "render_error":
        manifest, files = make_pack({"base.html": BASE, "account_pending.html": PENDING_HTML,
                                     "account_pending.txt": "{{ email.nope }}"})
    _write_pack(tmp_path / "pack", manifest, files, fixtures)
    code = render_mail_pack.main(["--pack", str(tmp_path / "pack"), "--out", str(tmp_path / "out")])
    assert code != 0


# ------------------------------------------------------------ contract


def test_load_pack_ignores_a_mail_this_backend_does_not_know():
    files = {"base.html": BASE, "account_pending.html": PENDING_HTML,
             "account_pending.txt": PENDING_TXT, "future.txt": "{{ anything }}"}
    mails = {
        "account_pending": {"subject": "S", "html": "account_pending.html", "text": "account_pending.txt"},
        "future_mail": {"subject": "F", "html": "account_pending.html", "text": "future.txt"},
    }
    manifest, blobs = make_pack(files, mails)
    pack = load_pack(manifest, blobs.__getitem__, MAIL_SPECS, etag='"e"')
    assert set(pack.mails) == {"account_pending"}


@pytest.mark.parametrize(
    "subject",
    ["Hi\nBcc: x", "Hi\rBcc: x", "A\x00B", "A\x0bB", "A\x85B", "A\u2028B", "A\u2029B",
     "A\u200bB", "A\ue000B", "A\tB", "   "],
    ids=["lf", "cr", "nul", "vtab", "nel", "line_sep", "para_sep", "zero_width", "private_use",
         "tab", "blank"],
)
def test_subject_with_a_control_or_separator_character_is_rejected(subject):
    mails = {"account_pending": {"subject": subject, "html": "account_pending.html",
                                 "text": "account_pending.txt"}}
    manifest, blobs = make_pack(mails=mails)
    with pytest.raises(PackRejected) as excinfo:
        load_pack(manifest, blobs.__getitem__, MAIL_SPECS, etag='"e"')
    assert excinfo.value.reason == "schema"


def test_signup_invite_bundled_default_renders():
    rendered = account_notifications.build_cohort_invite_signup(
        EMAIL, sign_up_url="https://eve-chat.example.org/signup"
    )
    assert rendered.subject == MAIL_SPECS["cohort_invite_signup"].subject
    assert 'href="https://eve-chat.example.org/signup"' in rendered.html
    assert "https://eve-chat.example.org/signup" in rendered.text
    assert EMAIL in rendered.html and EMAIL in rendered.text
    assert "background-color:#003247" in rendered.html
    assert "<" not in rendered.text
    # Neutral default copy: campaign wording belongs to the published pack. The
    # shared frame (base.html) is covered by its own tests.
    for suffix in ("html", "txt"):
        source = (BUNDLED_DIR / f"cohort_invite_signup.{suffix}").read_text(encoding="utf-8")
        for gone in ("pilot", "Phi-lab", "philab", "ESA", "\u2014", "\u00b7"):
            assert gone not in source, (suffix, gone)


def test_every_bundled_template_passes_its_own_validation():
    """The bundled copy, published as a pack, must load: it is the seed of the first pack."""
    files: dict[str, bytes] = {}
    for path in sorted(BUNDLED_DIR.iterdir()):
        files[path.name] = path.read_bytes()
    mails = {
        name: {"subject": spec.subject, "html": spec.html, "text": spec.text}
        for name, spec in MAIL_SPECS.items()
    }
    manifest = yaml.safe_dump(
        {"contract": 1, "version": "0.1.0",
         "files": {n: _sha(d) for n, d in files.items()}, "mails": mails}
    ).encode()
    pack = load_pack(manifest, files.__getitem__, MAIL_SPECS, etag='"e"')
    assert set(pack.mails) == set(MAIL_SPECS)


# ------------------------------------------------------------ operator gallery


GALLERY = {
    "04-on-hold": ("account_pending", {}),
    "05-welcome": ("account_approved", {"after_hold": False}),
    "06-welcome-after-hold": ("account_approved", {"after_hold": True}),
    "07-launch": ("cohort_invite", {"sign_in_url": FRONTEND, "paragraphs": []}),
    "08-launch-signup": ("cohort_invite_signup", {"sign_up_url": f"{FRONTEND}/signup"}),
}


def _bundled_pack_dir(directory: Path, *, stale_hashes: bool = False) -> None:
    """The bundled templates laid out as a pack, with the gallery fixtures."""
    directory.mkdir(parents=True, exist_ok=True)
    lines = ["# seeded from the bundled templates", "contract: 1", "version: 0.1.0", "files:"]
    for path in sorted(BUNDLED_DIR.iterdir()):
        (directory / path.name).write_bytes(path.read_bytes())
        digest = "0" * 64 if stale_hashes else _sha(path.read_bytes())
        lines.append(f"  {path.name}: {digest}  # keep this comment")
    lines.append("mails:")
    for name, spec in MAIL_SPECS.items():
        lines += [f"  {name}:", f"    subject: {spec.subject}",
                  f"    html: {spec.html}", f"    text: {spec.text}"]
    (directory / MANIFEST_KEY).write_text("\n".join(lines) + "\n", encoding="utf-8")
    base = {"email": EMAIL, "app_url": FRONTEND, "logo_url": f"{FRONTEND}/branding/eve-logo.png"}
    fixtures = {"variants": [
        {"name": name, "mail": mail, "context": {**base, **extra}}
        for name, (mail, extra) in GALLERY.items()
    ]}
    (directory / "fixtures.yaml").write_text(yaml.safe_dump(fixtures), encoding="utf-8")


def test_render_command_names_files_after_the_gallery_variants(tmp_path, capsys):
    _bundled_pack_dir(tmp_path / "pack")
    code = render_mail_pack.main(["--pack", str(tmp_path / "pack"), "--out", str(tmp_path / "out")])
    assert code == 0
    written = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert written == sorted(f"{stem}.{ext}" for stem in GALLERY for ext in ("html", "txt"))
    # The pack is today's bundled copy, so the gallery equals the golden output.
    assert (tmp_path / "out" / "04-on-hold.html").read_text() == (GOLDEN / "account_pending.html").read_text()
    assert (tmp_path / "out" / "06-welcome-after-hold.txt").read_text() == (
        GOLDEN / "account_approved_after_hold.txt"
    ).read_text()
    out = capsys.readouterr().out
    assert "08-launch-signup: Create your EVE account" in out


def test_write_hashes_fixes_every_hash_and_keeps_the_rest(tmp_path):
    pack = tmp_path / "pack"
    _bundled_pack_dir(pack, stale_hashes=True)
    before = (pack / MANIFEST_KEY).read_text().splitlines()

    assert render_mail_pack.main(["--pack", str(pack), "--write-hashes"]) == 0

    after = (pack / MANIFEST_KEY).read_text().splitlines()
    assert len(after) == len(before)
    for old, new in zip(before, after):
        if "0" * 64 in old:
            name = old.split(":")[0].strip()
            assert new == old.replace("0" * 64, _sha((pack / name).read_bytes()))
        else:
            assert new == old
    # Hashes now match, so the pack loads and renders.
    assert render_mail_pack.main(["--pack", str(pack), "--out", str(tmp_path / "out")]) == 0
    # A second run changes nothing.
    assert render_mail_pack.main(["--pack", str(pack), "--write-hashes"]) == 0
    assert (pack / MANIFEST_KEY).read_text().splitlines() == after


def test_write_hashes_fails_on_a_listed_file_that_is_missing(tmp_path):
    pack = tmp_path / "pack"
    _bundled_pack_dir(pack)
    (pack / "cohort_invite_signup.txt").unlink()
    assert render_mail_pack.main(["--pack", str(pack), "--write-hashes"]) == 1


def test_render_command_needs_no_database_or_network(tmp_path):
    """The operator script runs it with dummy settings only: no Mongo, no Redis, no keys."""
    import os
    import subprocess
    import sys

    _bundled_pack_dir(tmp_path / "pack")
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path),
        "QDRANT_URL": "http://unused.invalid",
        "QDRANT_API_KEY": "unused",
        "AUTH_ISSUER": "https://unused.invalid",
        "AUTH_CLIENT_ID": "unused",
    }
    result = subprocess.run(
        [sys.executable, "-m", "src.commands.render_mail_pack",
         "--pack", str(tmp_path / "pack"), "--out", str(tmp_path / "out")],
        cwd=Path(__file__).resolve().parent.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert (tmp_path / "out" / "07-launch.html").exists()


# ------------------------------------------------------------ 13. hostile packs


def _pending_pack_with(html: str = PENDING_HTML, txt: str = PENDING_TXT, **extra_files: str):
    files = {"base.html": BASE, "account_pending.html": html, "account_pending.txt": txt}
    files.update(extra_files)
    return make_pack(files)


# Each body would stall a worker if it ever rendered (CPU or memory). They run
# in a subprocess with a hard timeout, so a regression fails the test instead
# of freezing the suite: 9**9**9 holds the GIL until it finishes.
HOSTILE_BODIES = {
    "pow": "{{ 9**9**9 }}",
    "nested_range": "{% for a in range(100000) %}{% for b in range(100000) %}.{% endfor %}{% endfor %}",
    "string_mul": '{{ ("x"*50000000)|length }}',
    "mul_variable": "{{ email * 100000000 }}",
    "mod_format": '{{ "%0999999999d" % 1 }}',
    "set_doubling": "{% set a = email ~ email %}" + "{% set a = a ~ a %}" * 40 + "{{ a|length }}",
    "with_doubling": "{% with a = email ~ email %}{% with b = a ~ a %}{{ b }}{% endwith %}{% endwith %}",
    "method_call": "{{ email.center(999999999) }}",
    "growing_filter": '{{ email|center(999999999) }}',
    "global_call": "{{ lipsum(1) }}",
    "macro_recursion": "{% macro f() %}{{ f() }}{% endmacro %}{{ f() }}",
    "nested_loop_over_variable": "{% for a in email %}{% for b in email %}.{% endfor %}{% endfor %}",
    "loop_over_literal": '{% for c in "abcdef" %}{{ c }}{% endfor %}',
    "include_in_loop": '{% for c in email %}{% include "base.html" %}{% endfor %}',
    "append": "{% set l = [] %}{{ l.append(1) }}",
    "autoescape_off": "{% autoescape false %}{{ email }}{% endautoescape %}",
    "safe_filter": "{{ email|safe }}",
}

_HOSTILE_SCRIPT = r"""
import hashlib, json, sys, time, yaml
from src.services.account_notifications import MAIL_SPECS
from src.services.mail_templates import PackRejected, load_pack

BASE = sys.argv[1]
results = {}
for case, body in json.loads(sys.argv[2]).items():
    files = {"base.html": BASE.encode(), "account_pending.html": body.encode(),
             "account_pending.txt": b"plain"}
    manifest = yaml.safe_dump({
        "contract": 1, "version": "1.0.0",
        "files": {n: hashlib.sha256(d).hexdigest() for n, d in files.items()},
        "mails": {"account_pending": {"subject": "S", "html": "account_pending.html",
                                      "text": "account_pending.txt"}},
    }).encode()
    started = time.monotonic()
    try:
        load_pack(manifest, files.__getitem__, MAIL_SPECS)
        outcome = "loaded"
    except PackRejected as exc:
        outcome = exc.reason
    results[case] = [outcome, time.monotonic() - started]
print(json.dumps(results))
"""


def test_a_hostile_pack_is_rejected_at_load_in_well_under_a_second(tmp_path):
    import json
    import os
    import subprocess
    import sys

    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path),
        "QDRANT_URL": "http://unused.invalid",
        "QDRANT_API_KEY": "unused",
        "AUTH_ISSUER": "https://unused.invalid",
        "AUTH_CLIENT_ID": "unused",
    }
    result = subprocess.run(
        [sys.executable, "-c", _HOSTILE_SCRIPT, BASE, json.dumps(HOSTILE_BODIES)],
        cwd=Path(__file__).resolve().parent.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    results = json.loads(result.stdout.strip().splitlines()[-1])
    assert set(results) == set(HOSTILE_BODIES)
    for case, (outcome, seconds) in results.items():
        assert outcome == "template", (case, outcome)
        assert seconds < 1.0, (case, seconds)


def test_a_reference_cycle_is_rejected():
    manifest, blobs = _pending_pack_with(
        html='{% include "loop_a.html" %}',
        **{"loop_a.html": '{% include "loop_b.html" %}', "loop_b.html": '{% include "loop_a.html" %}'},
    )
    with pytest.raises(PackRejected) as excinfo:
        load_pack(manifest, blobs.__getitem__, MAIL_SPECS)
    assert excinfo.value.reason == "template"


def test_the_trial_render_rejects_a_mail_above_the_size_cap():
    big = "x" * (mail_templates.MAX_RENDERED_BYTES + 1)
    manifest, blobs = _pending_pack_with(txt=big)
    with pytest.raises(PackRejected) as excinfo:
        load_pack(manifest, blobs.__getitem__, MAIL_SPECS)
    assert excinfo.value.reason == "size"
    assert excinfo.value.detail == "account_pending text"


def test_a_real_send_above_the_size_cap_falls_back_to_bundled(caplog):
    # Passes the trial render (two sample paragraphs), not a send with ten.
    block = "y" * 50_000
    html = "{% for p in paragraphs %}" + block + "{{ p }}{% endfor %}{{ email }}{{ sign_in_url }}"
    files = {"cohort_invite.html": html, "cohort_invite.txt": "{{ email }} {{ sign_in_url }}"}
    mails = {"cohort_invite": {"subject": "S", "html": "cohort_invite.html", "text": "cohort_invite.txt"}}
    manifest, blobs = make_pack(files, mails)
    store = store_with(FakeS3(manifest, blobs))
    store._refresh_sync()
    assert store.carries("cohort_invite")
    context = {"email": EMAIL, "app_url": FRONTEND, "logo_url": "L",
               "sign_in_url": FRONTEND, "paragraphs": ["p"] * 10}
    with caplog.at_level(logging.WARNING, logger=mail_templates.logger.name):
        rendered = store.render("cohort_invite", context)
    assert rendered.source == "bundled"
    assert "mail_template_fallback name=cohort_invite reason=size" in _log_text(caplog)


def test_a_partial_listed_only_by_an_unknown_mail_is_held_to_the_base_variables():
    manifest, blobs = make_pack(
        {
            "base.html": BASE,
            "account_pending.html": '{% include "q.html" %}',
            "account_pending.txt": PENDING_TXT,
            "q.html": "{{ secret_var }}",
            "q.txt": "x",
        },
        {
            "account_pending": {"subject": "S", "html": "account_pending.html", "text": "account_pending.txt"},
            "future_mail": {"subject": "F", "html": "q.html", "text": "q.txt"},
        },
    )
    with pytest.raises(PackRejected) as excinfo:
        load_pack(manifest, blobs.__getitem__, MAIL_SPECS)
    assert excinfo.value.reason == "template"
    assert "q.html" in excinfo.value.detail


def test_autoescape_follows_the_role_of_the_part_not_the_file_name():
    files = {"part.htm": "<b>{{ email }}</b>", "part.html": "{{ email }} {{ app_url }}"}
    mails = {"account_pending": {"subject": "S", "html": "part.htm", "text": "part.html"}}
    manifest, blobs = make_pack(files, mails)
    store = store_with(FakeS3(manifest, blobs))
    store._refresh_sync()
    address = 'a"><script>x</script>&@example.org'
    rendered = store.render("account_pending", {"email": address, "app_url": FRONTEND, "logo_url": "L"})
    assert rendered.source == "pack"
    assert "<script>" not in rendered.html
    assert "&lt;script&gt;" in rendered.html
    assert rendered.text.startswith(address)


@pytest.mark.parametrize(
    "manifest_change",
    [
        lambda m: m["mails"].update({"account_pending\nFAKE mail_templates_loaded version=9.9.9":
                                     m["mails"]["account_pending"]}),
        lambda m: m.update({"extra\nFAKE mail_templates_loaded version=9.9.9": 1}),
        lambda m: m["files"].update({"x.txt\nFAKE": "0" * 64}),
    ],
    ids=["mail_name", "top_level_key", "file_name"],
)
@pytest.mark.asyncio
async def test_a_forged_name_never_reaches_a_log_line(manifest_change, caplog):
    manifest, files = make_pack()
    raw = yaml.safe_load(manifest)
    manifest_change(raw)
    s3 = FakeS3(yaml.safe_dump(raw).encode(), files)
    store = store_with(s3)
    with caplog.at_level(logging.DEBUG, logger=mail_templates.logger.name):
        await store.refresh()
    assert store._pack is None
    lines = [r.getMessage() for r in caplog.records]
    assert any("mail_templates_rejected reason=schema" in line for line in lines)
    for line in lines:
        assert "\n" not in line
        assert "FAKE" not in line


@pytest.mark.asyncio
async def test_a_publish_in_progress_is_retried_and_loads_once_complete(caplog):
    manifest_v1, files_v1 = make_pack(version="1.0.0")
    s3 = FakeS3(manifest_v1, files_v1, etag='"v1"')
    store = store_with(s3)
    await store.refresh()

    # The publish wrote the new manifest; one listed file is still the old copy.
    files_v2 = {
        "base.html": BASE,
        "account_pending.html": PENDING_HTML.replace("PACK", "V2"),
        "account_pending.txt": PENDING_TXT.replace("PACK", "V2"),
    }
    manifest_v2, blobs_v2 = make_pack(files_v2, version="2.0.0")
    s3.objects[MANIFEST_KEY] = manifest_v2
    s3.objects["account_pending.html"] = blobs_v2["account_pending.html"]
    s3.etag = '"v2"'
    with caplog.at_level(logging.WARNING, logger=mail_templates.logger.name):
        await store.refresh()
        assert store.pack_version == "1.0.0"
        # The last file lands; the manifest (and its ETag) did not change.
        s3.objects["account_pending.txt"] = blobs_v2["account_pending.txt"]
        s3.calls.clear()
        await store.refresh()

    # The rejected manifest was not pinned: the retry fetched it in full.
    assert s3.calls[0].get("IfNoneMatch") != '"v2"'
    assert len(s3.calls) > 1
    assert store.pack_version == "2.0.0"
    rejected = [r.getMessage() for r in caplog.records if "mail_templates_rejected" in r.getMessage()]
    assert len(rejected) == 1 and "reason=file" in rejected[0]


@pytest.mark.asyncio
async def test_a_final_rejection_is_not_fetched_again():
    manifest, files = make_pack()
    s3 = FakeS3(manifest, files, etag='"good"')
    store = store_with(s3)
    await store.refresh()
    (bad_manifest, bad_files), _ = _bad_packs()["undeclared_variable"]
    s3.objects = dict(bad_files)
    s3.objects[MANIFEST_KEY] = bad_manifest
    s3.etag = '"bad"'
    await store.refresh()
    s3.calls.clear()
    await store.refresh()
    assert [call.get("IfNoneMatch") for call in s3.calls] == ['"bad"']


@pytest.mark.asyncio
async def test_a_subject_the_message_build_refuses_sends_the_bundled_mail(default_store, monkeypatch, caplog):
    import dataclasses

    from src.services import mailer

    manifest, blobs = make_pack()
    pack = load_pack(manifest, blobs.__getitem__, MAIL_SPECS, etag='"e"')
    # Built past the loader on purpose: the guard at send time is the last line.
    broken = dataclasses.replace(
        pack,
        mails={"account_pending": dataclasses.replace(pack.mails["account_pending"], subject="Hi\x0bthere")},
    )
    monkeypatch.setattr(default_store, "_bucket", BUCKET)
    monkeypatch.setattr(default_store, "_pack", broken)
    sent = []

    async def _build_like_smtp(**kwargs):
        mailer._build_message(sender="EVE <eve@example.org>", **kwargs).as_string()
        sent.append(kwargs)

    monkeypatch.setattr(account_notifications, "send_mail", _build_like_smtp)
    with caplog.at_level(logging.WARNING, logger=mail_templates.logger.name):
        await notify_account_pending("64b7f2c1a2b3c4d5e6f70123", EMAIL)
    assert len(sent) == 1
    assert sent[0]["subject"] == MAIL_SPECS["account_pending"].subject
    assert "mail_template_fallback name=account_pending reason=subject pack_version=1.0.0" in _log_text(caplog)


@pytest.mark.asyncio
async def test_the_cohort_command_refuses_a_subject_with_a_control_character():
    from src.commands.send_cohort_invite import CohortInviteError, send_cohort_invite

    for subject in ("A\x0bB", "A\x00B", "A B", "A\x85B"):
        with pytest.raises(CohortInviteError):
            await send_cohort_invite("c", subject=subject)


@pytest.mark.asyncio
async def test_the_s3_client_is_built_once_on_the_event_loop_thread():
    import threading

    manifest, blobs = make_pack()
    built_on = []

    def _factory():
        built_on.append(threading.current_thread())
        return FakeS3(manifest, blobs)

    store = MailTemplateStore(MAIL_SPECS, bucket=BUCKET, client_factory=_factory)
    await store.refresh()
    await store.refresh()
    assert built_on == [threading.main_thread()]


def test_the_default_client_uses_a_session_of_its_own(monkeypatch):
    sessions = []

    class _Session:
        def __init__(self):
            sessions.append(self)

        def client(self, service, **kwargs):
            return ("client", service)

    def _default_session_client(*args, **kwargs):
        raise AssertionError("the default boto3 session must not build the client")

    monkeypatch.setattr(mail_templates.boto3, "client", _default_session_client)
    monkeypatch.setattr(mail_templates.boto3.session, "Session", _Session)
    assert mail_templates._s3_client() == ("client", "s3")
    assert len(sessions) == 1


# ------------------------------------------------------------ 14. render command hardening


def test_a_variant_name_with_a_trailing_newline_is_refused(tmp_path):
    manifest, files = make_pack()
    fixtures = {"variants": [dict(FIXTURES["variants"][0], name="pending\n")]}
    _write_pack(tmp_path / "pack", manifest, files, fixtures)
    code = render_mail_pack.main(["--pack", str(tmp_path / "pack"), "--out", str(tmp_path / "out")])
    assert code == 1
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "entry",
    ['  "account_pending.txt": 0000', "  account_pending.txt: 0000#note", "  account_pending.txt: &a 0000"],
    ids=["quoted_key", "glued_comment", "anchor"],
)
def test_write_hashes_fails_on_a_files_line_it_cannot_read(tmp_path, entry):
    pack = tmp_path / "pack"
    _bundled_pack_dir(pack)
    path = pack / MANIFEST_KEY
    lines = path.read_text().splitlines()
    index = next(i for i, line in enumerate(lines) if line.strip().startswith("cohort_invite.txt:"))
    lines.insert(index, entry)
    original = "\n".join(lines) + "\n"
    path.write_text(original)
    assert render_mail_pack.main(["--pack", str(pack), "--write-hashes"]) == 1
    assert path.read_text() == original


def test_write_hashes_fails_when_the_parsed_files_disagree_with_the_edit(tmp_path):
    # A second files key as a flow mapping: the line edit fixes the block it
    # reads, YAML keeps the last key, so the parsed files stay stale.
    pack = tmp_path / "pack"
    _bundled_pack_dir(pack)
    path = pack / MANIFEST_KEY
    text = path.read_text() + "files: {base.html: " + "0" * 64 + "}\n"
    path.write_text(text)
    assert render_mail_pack.main(["--pack", str(pack), "--write-hashes"]) == 1
    assert path.read_text() == text


def test_write_hashes_keeps_crlf_line_endings(tmp_path):
    pack = tmp_path / "pack"
    _bundled_pack_dir(pack, stale_hashes=True)
    path = pack / MANIFEST_KEY
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    assert render_mail_pack.main(["--pack", str(pack), "--write-hashes"]) == 0
    data = path.read_bytes()
    assert b"0" * 64 not in data
    assert data.count(b"\r\n") == data.count(b"\n")
    assert yaml.safe_load(data)["files"]["base.html"] == _sha((pack / "base.html").read_bytes())


def test_write_hashes_replaces_the_manifest_atomically(tmp_path, monkeypatch):
    pack = tmp_path / "pack"
    _bundled_pack_dir(pack, stale_hashes=True)
    path = pack / MANIFEST_KEY
    original = path.read_bytes()
    replaced = []
    real_replace = render_mail_pack.os.replace

    def _replace(src, dst):
        replaced.append((Path(src).parent, Path(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(render_mail_pack.os, "replace", _replace)
    monkeypatch.setattr(Path, "write_text", lambda *a, **k: (_ for _ in ()).throw(AssertionError("in place")))
    monkeypatch.setattr(Path, "write_bytes", lambda *a, **k: (_ for _ in ()).throw(AssertionError("in place")))
    assert render_mail_pack.main(["--pack", str(pack), "--write-hashes"]) == 0
    assert replaced == [(pack, path)]
    assert path.read_bytes() != original
    assert sorted(p.name for p in pack.iterdir() if p.name.endswith(".tmp")) == []


@pytest.mark.parametrize("target", ["listed_file", "manifest"])
@pytest.mark.parametrize("mode", ["write_hashes", "render"])
def test_a_symlink_in_the_pack_fails_the_command(tmp_path, target, mode):
    pack = tmp_path / "pack"
    _bundled_pack_dir(pack)
    outside = tmp_path / "outside"
    outside.mkdir()
    name = "account_pending.txt" if target == "listed_file" else MANIFEST_KEY
    (outside / name).write_bytes((pack / name).read_bytes())
    (pack / name).unlink()
    (pack / name).symlink_to(outside / name)
    args = ["--write-hashes"] if mode == "write_hashes" else ["--out", str(tmp_path / "out")]
    assert render_mail_pack.main(["--pack", str(pack), *args]) == 1
    assert not (tmp_path / "out").exists()


def test_render_command_refuses_an_out_directory_that_is_not_empty(tmp_path):
    _bundled_pack_dir(tmp_path / "pack")
    out = tmp_path / "out"
    out.mkdir()
    (out / "03-old-variant.html").write_text("stale")
    code = render_mail_pack.main(["--pack", str(tmp_path / "pack"), "--out", str(out)])
    assert code == 1
    assert sorted(p.name for p in out.iterdir()) == ["03-old-variant.html"]
