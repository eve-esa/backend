"""The one way out for mail, and the three shapes it can take.

What is worth pinning here is not that a library gets called. It is that "off"
really sends nothing, that the SMTP path builds a message a client can read,
that the SES path fills the fields SES actually validates, and that a host with
a transport but no sender fails with a sentence somebody can act on instead of
mailing from a domain nobody owns.

Config is read once at import, so the knobs are monkeypatched on
``src.services.mailer`` rather than in the environment.
"""

from unittest.mock import MagicMock

import pytest

from src.services import mailer
from src.services.mailer import MailNotConfigured, send_mail

pytestmark = pytest.mark.no_db


@pytest.fixture(autouse=True)
def _sane_defaults(monkeypatch):
    monkeypatch.setattr(mailer, "EMAIL_FROM_ADDRESS", "no-reply@eve.test")
    monkeypatch.setattr(mailer, "EMAIL_FROM_NAME", "EVE")
    monkeypatch.setattr(mailer, "SES_CONFIGURATION_SET", "")
    mailer._reset_ses_client_for_tests()
    yield
    mailer._reset_ses_client_for_tests()


@pytest.mark.asyncio
async def test_off_sends_nothing_and_says_so(monkeypatch, caplog):
    monkeypatch.setattr(mailer, "MAIL_TRANSPORT", "off")

    def _boom(*args, **kwargs):
        raise AssertionError("no transport may run when MAIL_TRANSPORT is off")

    monkeypatch.setattr(mailer.smtplib, "SMTP", _boom)
    monkeypatch.setattr(mailer, "_get_ses_client", _boom)

    with caplog.at_level("INFO", logger=mailer.logger.name):
        await send_mail(
            to="person@example.com", subject="Hello", html="<p>Hi</p>", text="Hi"
        )

    message = "\n".join(record.getMessage() for record in caplog.records)
    assert "mail suppressed" in message
    # Exact token, not a substring of some URL: the log names the domain only.
    assert "recipient_domain=example.com" in message
    # The subject is safe to log, the address is not.
    assert "person@example.com" not in message


@pytest.mark.asyncio
async def test_smtp_builds_a_multipart_message(monkeypatch):
    monkeypatch.setattr(mailer, "MAIL_TRANSPORT", "smtp")
    monkeypatch.setattr(mailer, "SMTP_HOST", "mailcatcher")
    monkeypatch.setattr(mailer, "SMTP_PORT", 1025)
    monkeypatch.setattr(mailer, "SMTP_USERNAME", "")
    monkeypatch.setattr(mailer, "SMTP_PASSWORD", "")

    smtp = MagicMock()
    server = smtp.return_value.__enter__.return_value
    monkeypatch.setattr(mailer.smtplib, "SMTP", smtp)

    await send_mail(
        to="person@example.com", subject="Hello", html="<p>Hi</p>", text="Hi"
    )

    host, port = smtp.call_args.args
    assert (host, port) == ("mailcatcher", 1025)
    # No credentials, no STARTTLS: the local catcher offers no TLS at all.
    server.starttls.assert_not_called()
    server.login.assert_not_called()

    sent = server.send_message.call_args.args[0]
    assert sent["Subject"] == "Hello"
    assert sent["To"] == "person@example.com"
    assert sent["From"] == "EVE <no-reply@eve.test>"
    parts = sent.get_payload()
    assert [part.get_content_type() for part in parts] == [
        "text/plain",
        "text/html",
    ]
    # HTML last, because a reader renders the final part it understands.
    assert "<p>Hi</p>" in parts[1].get_payload(decode=True).decode()


@pytest.mark.asyncio
async def test_smtp_upgrades_and_authenticates_when_a_username_is_set(monkeypatch):
    monkeypatch.setattr(mailer, "MAIL_TRANSPORT", "smtp")
    monkeypatch.setattr(mailer, "SMTP_HOST", "relay.example.net")
    monkeypatch.setattr(mailer, "SMTP_PORT", 587)
    monkeypatch.setattr(mailer, "SMTP_USERNAME", "mailer")
    monkeypatch.setattr(mailer, "SMTP_PASSWORD", "secret")

    smtp = MagicMock()
    server = smtp.return_value.__enter__.return_value
    monkeypatch.setattr(mailer.smtplib, "SMTP", smtp)

    await send_mail(to="person@example.com", subject="Hi", html="<p>Hi</p>", text="Hi")

    server.starttls.assert_called_once()
    server.login.assert_called_once_with("mailer", "secret")


@pytest.mark.asyncio
async def test_ses_sends_simple_content_with_both_bodies(monkeypatch):
    monkeypatch.setattr(mailer, "MAIL_TRANSPORT", "ses")
    monkeypatch.setattr(mailer, "SES_CONFIGURATION_SET", "eve-x-mail-events")

    client = MagicMock()
    monkeypatch.setattr(mailer, "_get_ses_client", lambda: client)

    await send_mail(
        to="person@example.com",
        subject="Your EVE account is ready",
        html="<p>Ready</p>",
        text="Ready",
    )

    request = client.send_email.call_args.kwargs
    assert request["FromEmailAddress"] == "EVE <no-reply@eve.test>"
    assert request["Destination"] == {"ToAddresses": ["person@example.com"]}
    assert request["ConfigurationSetName"] == "eve-x-mail-events"
    simple = request["Content"]["Simple"]
    assert simple["Subject"]["Data"] == "Your EVE account is ready"
    assert simple["Body"]["Text"]["Data"] == "Ready"
    assert simple["Body"]["Html"]["Data"] == "<p>Ready</p>"


@pytest.mark.asyncio
async def test_ses_omits_the_configuration_set_when_unset(monkeypatch):
    monkeypatch.setattr(mailer, "MAIL_TRANSPORT", "ses")
    monkeypatch.setattr(mailer, "SES_CONFIGURATION_SET", "")

    client = MagicMock()
    monkeypatch.setattr(mailer, "_get_ses_client", lambda: client)

    await send_mail(to="person@example.com", subject="Hi", html="<p>Hi</p>", text="Hi")

    assert "ConfigurationSetName" not in client.send_email.call_args.kwargs


@pytest.mark.asyncio
async def test_a_bare_address_keeps_its_own_display_name(monkeypatch):
    """The deployed value already reads "Name <addr>"; do not double the name."""
    monkeypatch.setattr(mailer, "MAIL_TRANSPORT", "ses")
    monkeypatch.setattr(
        mailer, "EMAIL_FROM_ADDRESS", "EVE by ESA Phi Lab <no-reply@eve.test>"
    )

    client = MagicMock()
    monkeypatch.setattr(mailer, "_get_ses_client", lambda: client)

    await send_mail(to="person@example.com", subject="Hi", html="<p>Hi</p>", text="Hi")

    assert (
        client.send_email.call_args.kwargs["FromEmailAddress"]
        == "EVE by ESA Phi Lab <no-reply@eve.test>"
    )


@pytest.mark.asyncio
async def test_a_blank_sender_fails_loudly(monkeypatch):
    """Silence is only acceptable when the transport is off."""
    monkeypatch.setattr(mailer, "MAIL_TRANSPORT", "smtp")
    monkeypatch.setattr(mailer, "EMAIL_FROM_ADDRESS", "")

    def _boom(*args, **kwargs):
        raise AssertionError("nothing may be sent without a sender")

    monkeypatch.setattr(mailer.smtplib, "SMTP", _boom)

    with pytest.raises(MailNotConfigured) as excinfo:
        await send_mail(to="person@example.com", subject="Hi", html="<p>Hi</p>", text="Hi")

    assert "EMAIL_FROM_ADDRESS" in str(excinfo.value)


@pytest.mark.asyncio
async def test_an_unknown_transport_is_refused(monkeypatch):
    monkeypatch.setattr(mailer, "MAIL_TRANSPORT", "carrier-pigeon")

    with pytest.raises(MailNotConfigured) as excinfo:
        await send_mail(to="person@example.com", subject="Hi", html="<p>Hi</p>", text="Hi")

    assert "carrier-pigeon" in str(excinfo.value)
