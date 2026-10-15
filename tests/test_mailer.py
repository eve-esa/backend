"""The one way out for mail, and the three shapes it can take.

What is worth pinning here is not that a library gets called. It is that "off"
really sends nothing, that the SMTP path builds a message a client can read,
that the SES path fills the fields SES actually validates, and that a host with
a transport but no sender fails with a sentence somebody can act on instead of
mailing from a domain nobody owns.

Config is read once at import, so the knobs are monkeypatched on
``src.services.mailer`` rather than in the environment.
"""

import email.utils
import smtplib
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from src.services import mailer
from src.services.mailer import MailDeliveryFailed, MailNotConfigured, send_mail

pytestmark = pytest.mark.no_db


@pytest.fixture(autouse=True)
def _sane_defaults(monkeypatch):
    monkeypatch.setattr(mailer, "EMAIL_FROM_ADDRESS", "no-reply@eve.test")
    monkeypatch.setattr(mailer, "EMAIL_FROM_NAME", "EVE")
    monkeypatch.setattr(mailer, "SES_CONFIGURATION_SET", "")
    monkeypatch.setattr(mailer, "SMTP_FROM_ADDRESS", "")
    monkeypatch.setattr(mailer, "MAIL_CANARY_TAG", "")
    monkeypatch.setattr(mailer, "MAIL_FALLBACK_TRANSPORT", "")
    mailer._reset_ses_client_for_tests()
    mailer._reset_fallback_warning_for_tests()
    yield
    mailer._reset_ses_client_for_tests()
    mailer._reset_fallback_warning_for_tests()


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
    server = smtp.return_value
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
    server = smtp.return_value
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


CANARY_TAG = "+eve-mail-canary"
SMTP_SENDER = "EVE by ESA Phi Lab <eve@mail.example.org>"


@pytest.fixture
def both_transports(monkeypatch):
    """SMTP and SES both mocked, so a test can see which one carried the mail."""
    monkeypatch.setattr(mailer, "SMTP_HOST", "relay.example.net")
    monkeypatch.setattr(mailer, "SMTP_PORT", 587)
    monkeypatch.setattr(mailer, "SMTP_USERNAME", "mailer")
    monkeypatch.setattr(mailer, "SMTP_PASSWORD", "secret")
    smtp = MagicMock()
    monkeypatch.setattr(mailer.smtplib, "SMTP", smtp)
    ses = MagicMock()
    monkeypatch.setattr(mailer, "_get_ses_client", lambda: ses)
    return smtp, ses


def _smtp_message(smtp):
    return smtp.return_value.send_message.call_args.args[0]


@pytest.mark.asyncio
async def test_a_canary_recipient_goes_through_smtp_with_the_smtp_sender(
    monkeypatch, both_transports
):
    smtp, ses = both_transports
    monkeypatch.setattr(mailer, "MAIL_TRANSPORT", "ses")
    monkeypatch.setattr(mailer, "MAIL_CANARY_TAG", CANARY_TAG)
    monkeypatch.setattr(mailer, "SMTP_FROM_ADDRESS", SMTP_SENDER)

    await send_mail(
        to="person+eve-mail-canary-1@example.com",
        subject="Hi",
        html="<p>Hi</p>",
        text="Hi",
    )

    ses.send_email.assert_not_called()
    sent = _smtp_message(smtp)
    assert sent["From"] == SMTP_SENDER
    assert sent["To"] == "person+eve-mail-canary-1@example.com"


@pytest.mark.asyncio
async def test_the_canary_tag_matches_regardless_of_case(monkeypatch, both_transports):
    smtp, ses = both_transports
    monkeypatch.setattr(mailer, "MAIL_TRANSPORT", "ses")
    monkeypatch.setattr(mailer, "MAIL_CANARY_TAG", CANARY_TAG)

    await send_mail(
        to="Person+EVE-Mail-Canary@Example.com", subject="Hi", html="<p>Hi</p>", text="Hi"
    )

    ses.send_email.assert_not_called()
    smtp.return_value.send_message.assert_called_once()


@pytest.mark.asyncio
async def test_every_other_recipient_stays_on_ses(monkeypatch, both_transports):
    smtp, ses = both_transports
    monkeypatch.setattr(mailer, "MAIL_TRANSPORT", "ses")
    monkeypatch.setattr(mailer, "MAIL_CANARY_TAG", CANARY_TAG)
    monkeypatch.setattr(mailer, "SMTP_FROM_ADDRESS", SMTP_SENDER)

    await send_mail(to="person@example.com", subject="Hi", html="<p>Hi</p>", text="Hi")

    smtp.assert_not_called()
    # SES keeps its own sender: the SMTP one is for the SMTP path only.
    assert ses.send_email.call_args.kwargs["FromEmailAddress"] == "EVE <no-reply@eve.test>"


@pytest.mark.asyncio
async def test_without_a_tag_a_tagged_address_stays_on_ses(monkeypatch, both_transports):
    """An empty tag would match every address; it must match none."""
    smtp, ses = both_transports
    monkeypatch.setattr(mailer, "MAIL_TRANSPORT", "ses")
    monkeypatch.setattr(mailer, "MAIL_CANARY_TAG", "")

    await send_mail(
        to="person+eve-mail-canary@example.com", subject="Hi", html="<p>Hi</p>", text="Hi"
    )

    smtp.assert_not_called()
    ses.send_email.assert_called_once()


@pytest.mark.asyncio
async def test_the_tag_never_turns_off_into_a_send(monkeypatch, both_transports):
    smtp, ses = both_transports
    monkeypatch.setattr(mailer, "MAIL_TRANSPORT", "off")
    monkeypatch.setattr(mailer, "MAIL_CANARY_TAG", CANARY_TAG)

    await send_mail(
        to="person+eve-mail-canary@example.com", subject="Hi", html="<p>Hi</p>", text="Hi"
    )

    smtp.assert_not_called()
    ses.send_email.assert_not_called()


@pytest.mark.asyncio
async def test_smtp_transport_uses_the_smtp_sender_when_set(monkeypatch, both_transports):
    smtp, ses = both_transports
    monkeypatch.setattr(mailer, "MAIL_TRANSPORT", "smtp")
    monkeypatch.setattr(mailer, "SMTP_FROM_ADDRESS", SMTP_SENDER)

    await send_mail(to="person@example.com", subject="Hi", html="<p>Hi</p>", text="Hi")

    ses.send_email.assert_not_called()
    assert _smtp_message(smtp)["From"] == SMTP_SENDER


@pytest.mark.asyncio
async def test_smtp_message_carries_date_and_a_message_id_on_the_sender_domain(
    monkeypatch, both_transports
):
    smtp, _ = both_transports
    monkeypatch.setattr(mailer, "MAIL_TRANSPORT", "smtp")
    monkeypatch.setattr(mailer, "SMTP_FROM_ADDRESS", SMTP_SENDER)

    await send_mail(to="person@example.com", subject="Hi", html="<p>Hi</p>", text="Hi")

    sent = _smtp_message(smtp)
    assert sent["Date"]
    assert email.utils.parsedate_to_datetime(sent["Date"]).tzinfo is not None
    assert sent["Message-ID"].startswith("<")
    assert sent["Message-ID"].endswith("@mail.example.org>")
    assert sent["Reply-To"] is None


@pytest.mark.asyncio
async def test_one_info_line_names_the_transport_and_the_domain_only(
    monkeypatch, both_transports, caplog
):
    monkeypatch.setattr(mailer, "MAIL_TRANSPORT", "ses")
    monkeypatch.setattr(mailer, "MAIL_CANARY_TAG", CANARY_TAG)

    with caplog.at_level("INFO", logger=mailer.logger.name):
        await send_mail(
            to="person+eve-mail-canary@example.com",
            subject="Hi",
            html="<p>Hi</p>",
            text="Hi",
        )

    records = [r for r in caplog.records if r.name == mailer.logger.name]
    assert len(records) == 1
    line = records[0].getMessage()
    assert records[0].levelname == "INFO"
    assert "via smtp" in line
    assert "recipient_domain=example.com" in line
    assert "person" not in line
    assert "canary" not in line


@pytest.mark.asyncio
async def test_the_canary_path_upgrades_and_authenticates(monkeypatch, both_transports):
    smtp, _ = both_transports
    monkeypatch.setattr(mailer, "MAIL_TRANSPORT", "ses")
    monkeypatch.setattr(mailer, "MAIL_CANARY_TAG", CANARY_TAG)
    monkeypatch.setattr(mailer, "SMTP_FROM_ADDRESS", SMTP_SENDER)

    await send_mail(
        to="person+eve-mail-canary@example.com", subject="Hi", html="<p>Hi</p>", text="Hi"
    )

    server = smtp.return_value
    server.starttls.assert_called_once()
    server.login.assert_called_once_with("mailer", "secret")
    # Envelope recipient is the one address, never re-parsed from the header.
    assert server.send_message.call_args.kwargs["to_addrs"] == [
        "person+eve-mail-canary@example.com"
    ]


@pytest.mark.asyncio
async def test_a_relay_sender_without_credentials_fails_closed(
    monkeypatch, both_transports
):
    """The Gmail sender with no username would mean plain SMTP without TLS."""
    smtp, _ = both_transports
    monkeypatch.setattr(mailer, "MAIL_TRANSPORT", "smtp")
    monkeypatch.setattr(mailer, "SMTP_FROM_ADDRESS", SMTP_SENDER)
    monkeypatch.setattr(mailer, "SMTP_USERNAME", "")

    with pytest.raises(MailNotConfigured) as excinfo:
        await send_mail(
            to="person@example.com", subject="Hi", html="<p>Hi</p>", text="Hi"
        )

    assert "SMTP_USERNAME" in str(excinfo.value)
    smtp.assert_not_called()


@pytest.mark.asyncio
async def test_a_refused_recipient_never_reaches_the_error(monkeypatch, both_transports):
    """Callers log with exc_info, so the raised error must not carry the address."""
    smtp, _ = both_transports
    monkeypatch.setattr(mailer, "MAIL_TRANSPORT", "smtp")
    address = "person+eve-mail-canary@example.com"
    server = smtp.return_value
    server.send_message.side_effect = smtplib.SMTPRecipientsRefused(
        {address: (550, f"5.1.1 <{address}> does not exist".encode())}
    )

    with pytest.raises(MailDeliveryFailed) as excinfo:
        await send_mail(to=address, subject="Hi", html="<p>Hi</p>", text="Hi")

    assert "SMTPRecipientsRefused" in str(excinfo.value)
    assert "person" not in str(excinfo.value)
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__suppress_context__ is True


@pytest.mark.asyncio
async def test_an_unparseable_sender_fails_closed(monkeypatch, both_transports):
    """An unquoted comma splits the From in two and leaves no domain to sign with."""
    smtp, _ = both_transports
    monkeypatch.setattr(mailer, "MAIL_TRANSPORT", "smtp")
    monkeypatch.setattr(
        mailer, "SMTP_FROM_ADDRESS", "EVE by ESA, Phi Lab <eve@mail.example.org>"
    )

    with pytest.raises(MailNotConfigured):
        await send_mail(
            to="person@example.com", subject="Hi", html="<p>Hi</p>", text="Hi"
        )

    smtp.assert_not_called()


@pytest.mark.asyncio
async def test_the_tag_matches_the_local_part_only(monkeypatch, both_transports):
    smtp, ses = both_transports
    monkeypatch.setattr(mailer, "MAIL_TRANSPORT", "ses")
    monkeypatch.setattr(mailer, "MAIL_CANARY_TAG", "canary")

    await send_mail(
        to="person@canary.example.com", subject="Hi", html="<p>Hi</p>", text="Hi"
    )

    smtp.assert_not_called()
    ses.send_email.assert_called_once()


@pytest.mark.asyncio
async def test_smtp_with_a_tag_stays_smtp(monkeypatch, both_transports):
    smtp, ses = both_transports
    monkeypatch.setattr(mailer, "MAIL_TRANSPORT", "smtp")
    monkeypatch.setattr(mailer, "MAIL_CANARY_TAG", CANARY_TAG)

    await send_mail(to="person@example.com", subject="Hi", html="<p>Hi</p>", text="Hi")

    ses.send_email.assert_not_called()
    smtp.return_value.send_message.assert_called_once()


def _fallback_lines(caplog):
    return [
        r for r in caplog.records
        if r.name == mailer.logger.name and "mail.fallback" in r.getMessage()
    ]


def _message_rejected(address):
    """What SES answers for an unverified sender: the text names the address."""
    return ClientError(
        {
            "Error": {
                "Code": "MessageRejected",
                "Message": f"Email address is not verified: {address}",
            }
        },
        "SendEmail",
    )


@pytest.fixture
def smtp_to_ses(monkeypatch, both_transports):
    """MAIL_TRANSPORT=smtp with the SES fallback on and a relay sender."""
    monkeypatch.setattr(mailer, "MAIL_TRANSPORT", "smtp")
    monkeypatch.setattr(mailer, "MAIL_FALLBACK_TRANSPORT", "ses")
    monkeypatch.setattr(mailer, "SMTP_FROM_ADDRESS", SMTP_SENDER)
    monkeypatch.setattr(mailer, "SES_CONFIGURATION_SET", "eve-x-mail-events")
    return both_transports


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "step, failure, reason",
    [
        ("login", smtplib.SMTPAuthenticationError(535, b"5.7.8 Bad credentials"), "SMTPAuthenticationError code=535"),
        ("send_message", smtplib.SMTPDataError(550, b"5.4.5 Daily user sending limit exceeded"), "SMTPDataError code=550"),
        # smtplib turns a read timeout or a reset into SMTPServerDisconnected; a raw
        # OSError only comes out of the constructor (refused, DNS, connect timeout).
        ("connect", TimeoutError("timed out"), "TimeoutError"),
        ("connect", ConnectionRefusedError(111, "Connection refused"), "ConnectionRefusedError"),
    ],
)
async def test_an_smtp_failure_falls_back_to_ses_with_the_ses_sender(
    smtp_to_ses, caplog, step, failure, reason
):
    smtp, ses = smtp_to_ses
    if step == "connect":
        smtp.side_effect = failure
    else:
        getattr(smtp.return_value, step).side_effect = failure

    with caplog.at_level("INFO", logger=mailer.logger.name):
        await send_mail(
            to="person@example.com", subject="Hello", html="<p>Hi</p>", text="Hi"
        )

    request = ses.send_email.call_args.kwargs
    # The SES identity sends, never the Gmail mailbox it has no right to.
    assert request["FromEmailAddress"] == "EVE <no-reply@eve.test>"
    assert request["Destination"] == {"ToAddresses": ["person@example.com"]}
    assert request["ConfigurationSetName"] == "eve-x-mail-events"
    assert request["Content"]["Simple"]["Subject"]["Data"] == "Hello"
    assert request["Content"]["Simple"]["Body"]["Text"]["Data"] == "Hi"
    assert request["Content"]["Simple"]["Body"]["Html"]["Data"] == "<p>Hi</p>"

    [warning] = _fallback_lines(caplog)
    line = warning.getMessage()
    assert warning.levelname == "WARNING"
    assert line == (
        f"mail.fallback from=smtp to=ses reason={reason} recipient_domain=example.com"
    )
    assert "person" not in line


@pytest.mark.asyncio
async def test_a_disconnect_during_data_falls_back_at_least_once(smtp_to_ses):
    """The relay may have queued the message before the line dropped.

    Delivery is at-least-once by choice: a second copy of a code mail beats
    none, so an ambiguous end of DATA goes to SES.
    """
    smtp, ses = smtp_to_ses
    smtp.return_value.send_message.side_effect = smtplib.SMTPServerDisconnected(
        "Connection unexpectedly closed"
    )

    await send_mail(to="person@example.com", subject="Hi", html="<p>Hi</p>", text="Hi")

    ses.send_email.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        smtplib.SMTPResponseException(421, b"4.7.0 Try again later, closing connection"),
        smtplib.SMTPServerDisconnected("Connection unexpectedly closed"),
    ],
)
async def test_an_error_after_the_relay_accepted_the_message_is_not_a_failure(
    smtp_to_ses, caplog, failure
):
    """Once send_message returns, the mail is delivered: a bad QUIT must not resend."""
    smtp, ses = smtp_to_ses
    server = smtp.return_value
    server.quit.side_effect = failure

    with caplog.at_level("INFO", logger=mailer.logger.name):
        await send_mail(to="person@example.com", subject="Hi", html="<p>Hi</p>", text="Hi")

    server.send_message.assert_called_once()
    ses.send_email.assert_not_called()
    assert not _fallback_lines(caplog)
    records = [r for r in caplog.records if r.name == mailer.logger.name]
    assert all(r.levelname == "INFO" for r in records)
    assert records[-1].getMessage() == (
        "mail sent via smtp subject='Hi' recipient_domain=example.com"
    )
    # The socket is released even when QUIT failed.
    server.close.assert_called()


@pytest.mark.asyncio
async def test_a_failed_smtp_session_is_closed(smtp_to_ses):
    smtp, _ = smtp_to_ses
    server = smtp.return_value
    server.login.side_effect = smtplib.SMTPAuthenticationError(535, b"5.7.8 Bad credentials")

    await send_mail(to="person@example.com", subject="Hi", html="<p>Hi</p>", text="Hi")

    server.send_message.assert_not_called()
    server.close.assert_called()


@pytest.mark.asyncio
async def test_a_refused_relay_config_falls_back_too(monkeypatch, smtp_to_ses):
    """MailNotConfigured from the SMTP path is a reason to try SES, not to lose the mail."""
    smtp, ses = smtp_to_ses
    monkeypatch.setattr(mailer, "SMTP_USERNAME", "")

    await send_mail(to="person@example.com", subject="Hi", html="<p>Hi</p>", text="Hi")

    smtp.assert_not_called()
    ses.send_email.assert_called_once()


@pytest.mark.asyncio
async def test_a_misconfigured_ses_side_surfaces_during_a_fallback(
    monkeypatch, smtp_to_ses, caplog
):
    """The relay sender is set, the SES one is not: SMTP fails, SES cannot run."""
    smtp, ses = smtp_to_ses
    monkeypatch.setattr(mailer, "EMAIL_FROM_ADDRESS", "")
    smtp.return_value.login.side_effect = smtplib.SMTPAuthenticationError(
        535, b"5.7.8 Bad credentials"
    )

    with caplog.at_level("INFO", logger=mailer.logger.name):
        with pytest.raises(MailNotConfigured) as excinfo:
            await send_mail(
                to="person@example.com", subject="Hi", html="<p>Hi</p>", text="Hi"
            )

    assert "EMAIL_FROM_ADDRESS" in str(excinfo.value)
    assert len(_fallback_lines(caplog)) == 1
    ses.send_email.assert_not_called()


@pytest.mark.asyncio
async def test_a_failed_canary_falls_back_to_ses(monkeypatch, both_transports):
    """The chosen transport decides: a canary diverted to SMTP falls back like smtp."""
    smtp, ses = both_transports
    monkeypatch.setattr(mailer, "MAIL_TRANSPORT", "ses")
    monkeypatch.setattr(mailer, "MAIL_CANARY_TAG", CANARY_TAG)
    monkeypatch.setattr(mailer, "MAIL_FALLBACK_TRANSPORT", "ses")
    smtp.return_value.send_message.side_effect = smtplib.SMTPDataError(
        550, b"5.4.5 Daily user sending limit exceeded"
    )

    await send_mail(
        to="person+eve-mail-canary@example.com", subject="Hi", html="<p>Hi</p>", text="Hi"
    )

    ses.send_email.assert_called_once()


@pytest.mark.asyncio
async def test_when_ses_fails_too_the_ses_error_is_raised_without_the_address(
    smtp_to_ses, caplog
):
    smtp, ses = smtp_to_ses
    address = "person@example.com"
    smtp.return_value.login.side_effect = smtplib.SMTPAuthenticationError(
        534, b"5.7.9 Application-specific password required"
    )
    ses.send_email.side_effect = _message_rejected(address)

    with caplog.at_level("INFO", logger=mailer.logger.name):
        with pytest.raises(MailDeliveryFailed) as excinfo:
            await send_mail(to=address, subject="Hi", html="<p>Hi</p>", text="Hi")

    assert str(excinfo.value) == "SES send failed: ClientError MessageRejected"
    assert "person" not in str(excinfo.value)
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__suppress_context__ is True
    # The SMTP failure is in the log line, not in the raised error.
    assert "code=534" in _fallback_lines(caplog)[0].getMessage()
    assert not any("mail sent" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_a_plain_ses_error_never_reaches_the_caller_with_the_address(
    monkeypatch, both_transports
):
    """Callers log with exc_info, and MessageRejected names the address."""
    _, ses = both_transports
    monkeypatch.setattr(mailer, "MAIL_TRANSPORT", "ses")
    ses.send_email.side_effect = _message_rejected("person@example.com")

    with pytest.raises(MailDeliveryFailed) as excinfo:
        await send_mail(to="person@example.com", subject="Hi", html="<p>Hi</p>", text="Hi")

    assert str(excinfo.value) == "SES send failed: ClientError MessageRejected"
    assert excinfo.value.reason == "ClientError"
    assert excinfo.value.code == "MessageRejected"
    assert excinfo.value.__cause__ is None


@pytest.mark.asyncio
async def test_without_a_fallback_the_smtp_error_is_raised_as_today(
    both_transports, monkeypatch, caplog
):
    smtp, ses = both_transports
    monkeypatch.setattr(mailer, "MAIL_TRANSPORT", "smtp")
    smtp.return_value.login.side_effect = smtplib.SMTPAuthenticationError(
        535, b"5.7.8 Bad credentials"
    )

    with caplog.at_level("INFO", logger=mailer.logger.name):
        with pytest.raises(MailDeliveryFailed):
            await send_mail(
                to="person@example.com", subject="Hi", html="<p>Hi</p>", text="Hi"
            )

    ses.send_email.assert_not_called()
    assert not _fallback_lines(caplog)


@pytest.mark.asyncio
async def test_a_failed_ses_send_never_falls_back(monkeypatch, both_transports):
    smtp, ses = both_transports
    monkeypatch.setattr(mailer, "MAIL_TRANSPORT", "ses")
    monkeypatch.setattr(mailer, "MAIL_FALLBACK_TRANSPORT", "ses")
    ses.send_email.side_effect = RuntimeError("ses down")

    with pytest.raises(RuntimeError):
        await send_mail(to="person@example.com", subject="Hi", html="<p>Hi</p>", text="Hi")

    smtp.assert_not_called()
    ses.send_email.assert_called_once()


@pytest.mark.asyncio
async def test_off_with_a_fallback_still_sends_nothing(monkeypatch, both_transports):
    smtp, ses = both_transports
    monkeypatch.setattr(mailer, "MAIL_TRANSPORT", "off")
    monkeypatch.setattr(mailer, "MAIL_FALLBACK_TRANSPORT", "ses")

    await send_mail(to="person@example.com", subject="Hi", html="<p>Hi</p>", text="Hi")

    smtp.assert_not_called()
    ses.send_email.assert_not_called()


@pytest.mark.asyncio
async def test_an_unknown_fallback_logs_once_and_does_not_fall_back(
    monkeypatch, both_transports, caplog
):
    smtp, ses = both_transports
    monkeypatch.setattr(mailer, "MAIL_TRANSPORT", "smtp")
    monkeypatch.setattr(mailer, "MAIL_FALLBACK_TRANSPORT", "smtp")
    smtp.return_value.login.side_effect = smtplib.SMTPAuthenticationError(
        535, b"5.7.8 Bad credentials"
    )

    with caplog.at_level("INFO", logger=mailer.logger.name):
        for _ in range(2):
            with pytest.raises(MailDeliveryFailed):
                await send_mail(
                    to="person@example.com", subject="Hi", html="<p>Hi</p>", text="Hi"
                )

    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 1
    assert "MAIL_FALLBACK_TRANSPORT" in errors[0].getMessage()
    ses.send_email.assert_not_called()


@pytest.mark.asyncio
async def test_an_unknown_fallback_never_blocks_plain_ses_mail(
    monkeypatch, both_transports, caplog
):
    """The fallback only matters to SMTP: a typo in it must not stop every SES user."""
    _, ses = both_transports
    monkeypatch.setattr(mailer, "MAIL_TRANSPORT", "ses")
    monkeypatch.setattr(mailer, "MAIL_FALLBACK_TRANSPORT", "smtp")

    with caplog.at_level("INFO", logger=mailer.logger.name):
        await send_mail(to="person@example.com", subject="Hi", html="<p>Hi</p>", text="Hi")

    ses.send_email.assert_called_once()
    assert not [r for r in caplog.records if r.levelname == "ERROR"]


@pytest.mark.asyncio
async def test_a_fallback_send_logs_the_usual_info_line_via_ses(smtp_to_ses, caplog):
    smtp, _ = smtp_to_ses
    smtp.return_value.send_message.side_effect = smtplib.SMTPServerDisconnected("gone")

    with caplog.at_level("INFO", logger=mailer.logger.name):
        await send_mail(to="person@example.com", subject="Hi", html="<p>Hi</p>", text="Hi")

    infos = [r.getMessage() for r in caplog.records if r.levelname == "INFO"]
    assert infos == ["mail sent via ses subject='Hi' recipient_domain=example.com"]


@pytest.mark.asyncio
async def test_smtp_waits_at_most_ten_seconds_per_operation(monkeypatch, both_transports):
    smtp, _ = both_transports
    monkeypatch.setattr(mailer, "MAIL_TRANSPORT", "smtp")

    await send_mail(to="person@example.com", subject="Hi", html="<p>Hi</p>", text="Hi")

    assert smtp.call_args.kwargs["timeout"] == 10


def test_the_ses_client_has_short_deadlines_and_two_attempts(monkeypatch):
    client = MagicMock()
    factory = MagicMock(return_value=client)
    monkeypatch.setattr(mailer.boto3, "client", factory)

    assert mailer._get_ses_client() is client

    service = factory.call_args.args[0]
    config = factory.call_args.kwargs["config"]
    assert service == "sesv2"
    assert config.connect_timeout == 3
    assert config.read_timeout == 8
    assert config.retries == {"total_max_attempts": 2, "mode": "standard"}
