"""What the two account emails actually say, and what they link to.

Templates are the one place in this feature where a mistake is invisible until a
person receives it: a broken logo URL, a link to the wrong environment, or an
address rendered as raw HTML all pass every other test in the suite. So these
assert on the rendered output rather than on the render being called.
"""

import pytest

from src.services import account_notifications
from src.services.account_notifications import (
    KIND_APPROVED,
    KIND_PENDING,
    notify_account_approved,
    notify_account_pending,
    render_account_mail,
)

pytestmark = pytest.mark.no_db

FRONTEND = "https://eve.example.org"


@pytest.fixture(autouse=True)
def _frontend_url(monkeypatch):
    monkeypatch.setattr(account_notifications, "FRONTEND_URL", FRONTEND)


def test_pending_mail_says_the_account_is_queued():
    subject, html, text = render_account_mail(KIND_PENDING, "person@example.com")

    assert subject == "Your EVE account is waiting for approval"
    assert "Thanks for joining EVE" in html
    assert "person@example.com" in html
    assert "in the queue for approval" in html
    assert f"{FRONTEND}/branding/eve-logo.png" in html
    assert FRONTEND in html
    assert "EVE - Earth Virtual Expert - European Space Agency." in html

    assert "person@example.com" in text
    assert "in the queue for approval" in text
    assert FRONTEND in text
    assert "<" not in text


def test_approved_mail_carries_the_sign_in_button():
    subject, html, text = render_account_mail(KIND_APPROVED, "person@example.com")

    assert subject == "Your EVE account is ready"
    assert "Your account has been enabled" in html
    assert "person@example.com" in html
    assert "Sign in to EVE" in html
    assert f'href="{FRONTEND}"' in html
    assert "background-color:#003247" in html
    assert f"{FRONTEND}/branding/eve-logo.png" in html

    assert "Sign in to EVE" in text
    assert FRONTEND in text


@pytest.mark.parametrize("kind", [KIND_PENDING, KIND_APPROVED])
def test_the_address_is_escaped_in_the_html(kind):
    """An address is untrusted input: it reaches the template from the provider."""
    _, html, text = render_account_mail(kind, 'a"><script>@example.com')

    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    # The text part must stay literal, or the reader sees HTML entities.
    assert "<script>" in text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "notify,kind,subject",
    [
        (notify_account_pending, "pending", "Your EVE account is waiting for approval"),
        (notify_account_approved, "approved", "Your EVE account is ready"),
    ],
)
async def test_notifying_hands_the_rendered_message_to_the_mailer(
    monkeypatch, caplog, notify, kind, subject
):
    sent = {}

    async def _capture(**kwargs):
        sent.update(kwargs)

    monkeypatch.setattr(account_notifications, "send_mail", _capture)

    with caplog.at_level("INFO", logger=account_notifications.logger.name):
        await notify("64b7f2c1a2b3c4d5e6f70123", "person@example.com")

    assert sent["to"] == "person@example.com"
    assert sent["subject"] == subject
    assert "person@example.com" in sent["html"]
    assert "person@example.com" in sent["text"]

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert f"account_mail_sent kind={kind} user_id=64b7f2c1a2b3c4d5e6f70123" in logged
    # The id is the operator's handle; the address is personal data.
    assert "person@example.com" not in logged


@pytest.mark.asyncio
async def test_a_failing_transport_is_logged_and_reraised(monkeypatch, caplog):
    async def _fail(**kwargs):
        raise RuntimeError("relay refused the message")

    monkeypatch.setattr(account_notifications, "send_mail", _fail)

    with caplog.at_level("ERROR", logger=account_notifications.logger.name):
        with pytest.raises(RuntimeError):
            await notify_account_pending("64b7f2c1a2b3c4d5e6f70123", "person@example.com")

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "account_mail_failed kind=pending user_id=64b7f2c1a2b3c4d5e6f70123" in logged
