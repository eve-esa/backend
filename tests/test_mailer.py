"""The one way out for mail, and the three shapes it can take.

What is worth pinning here is not that a library gets called. It is that "off"
really sends nothing, that the SMTP path builds a message a client can read,
that the SES path fills the fields SES actually validates, and that a host with
a transport but no sender fails with a sentence somebody can act on instead of
mailing from a domain nobody owns.

Config is read once at import, so the knobs are monkeypatched on
``src.services.mailer`` rather than in the environment.
"""

import email
import email.utils
import smtplib
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from src.services import mailer
from src.services.mailer import MailDeliveryFailed, MailNotConfigured, send_mail

pytestmark = pytest.mark.no_db

OK = (250, b"2.0.0 OK")


def _relay() -> MagicMock:
    """A stand-in for ``smtplib.SMTP`` whose server accepts every command.

    ``mail``, ``rcpt`` and ``data`` return a reply tuple like the real ones,
    because the mailer reads their codes instead of trusting ``sendmail``.
    """
    smtp = MagicMock()
    server = smtp.return_value
    server.mail.return_value = OK
    server.rcpt.return_value = OK
    server.data.return_value = OK
    return smtp


def _smtp_message(smtp):
    """The message as the relay received it after DATA."""
    return email.message_from_bytes(smtp.return_value.data.call_args.args[0])


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

    smtp = _relay()
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

    sent = _smtp_message(smtp)
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

    smtp = _relay()
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
    smtp = _relay()
    monkeypatch.setattr(mailer.smtplib, "SMTP", smtp)
    ses = MagicMock()
    monkeypatch.setattr(mailer, "_get_ses_client", lambda: ses)
    return smtp, ses


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
    smtp.return_value.data.assert_called_once()


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
    assert server.mail.call_args.args[0] == "eve@mail.example.org"
    assert server.rcpt.call_args.args[0] == "person+eve-mail-canary@example.com"


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
    server.rcpt.return_value = (550, f"5.1.1 <{address}> does not exist".encode())

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
    smtp.return_value.data.assert_called_once()


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
        ("login", smtplib.SMTPAuthenticationError(535, b"5.7.8 Bad credentials"), "SMTPAuthenticationError code=535 status=5.7.8"),
        ("data", smtplib.SMTPDataError(550, b"5.4.5 Daily user sending limit exceeded"), "SMTPDataError code=550 status=5.4.5"),
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
    smtp.return_value.data.side_effect = smtplib.SMTPServerDisconnected(
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
    """Once the relay answered DATA, the mail is delivered: a bad QUIT must not resend."""
    smtp, ses = smtp_to_ses
    server = smtp.return_value
    server.quit.side_effect = failure

    with caplog.at_level("INFO", logger=mailer.logger.name):
        await send_mail(to="person@example.com", subject="Hi", html="<p>Hi</p>", text="Hi")

    server.data.assert_called_once()
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

    server.mail.assert_not_called()
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
    smtp.return_value.data.side_effect = smtplib.SMTPDataError(
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
    smtp.return_value.data.side_effect = smtplib.SMTPServerDisconnected("gone")

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


@pytest.mark.asyncio
async def test_smtp_greets_the_relay_with_the_smtp_sender_domain(monkeypatch, both_transports):
    smtp, _ = both_transports
    monkeypatch.setattr(mailer, "MAIL_TRANSPORT", "smtp")
    monkeypatch.setattr(mailer, "SMTP_FROM_ADDRESS", SMTP_SENDER)

    await send_mail(to="person@example.com", subject="Hi", html="<p>Hi</p>", text="Hi")

    assert smtp.call_args.kwargs["local_hostname"] == "mail.example.org"


@pytest.mark.asyncio
async def test_smtp_greets_with_the_fallback_sender_domain_when_no_smtp_sender(
    monkeypatch, both_transports
):
    smtp, _ = both_transports
    monkeypatch.setattr(mailer, "MAIL_TRANSPORT", "smtp")
    monkeypatch.setattr(mailer, "SMTP_USERNAME", "")

    await send_mail(to="person@example.com", subject="Hi", html="<p>Hi</p>", text="Hi")

    assert smtp.call_args.kwargs["local_hostname"] == "eve.test"


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


# ------------------------------------------- definite refusal or unclear result


@pytest.fixture
def smtp_only(monkeypatch, both_transports):
    """MAIL_TRANSPORT=smtp, no fallback: the caller sees the SMTP verdict."""
    monkeypatch.setattr(mailer, "MAIL_TRANSPORT", "smtp")
    return both_transports


async def _verdict(**kwargs) -> MailDeliveryFailed:
    with pytest.raises(MailDeliveryFailed) as excinfo:
        await send_mail(to="person@example.com", subject="Hi", html="<p>Hi</p>", text="Hi", **kwargs)
    return excinfo.value


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        smtplib.SMTPServerDisconnected("Connection unexpectedly closed"),
        TimeoutError("timed out"),
        ConnectionResetError(104, "Connection reset by peer"),
    ],
)
async def test_a_dropped_line_once_data_started_is_an_unclear_result(smtp_only, failure):
    """After DATA the relay may hold the whole message: the outcome is unknown."""
    smtp, _ = smtp_only
    smtp.return_value.data.side_effect = failure

    error = await _verdict()

    assert error.uncertain is True
    assert mailer.delivery_uncertain(error) is True
    assert error.reason == type(failure).__name__
    assert mailer.failure_reason(error) == type(failure).__name__
    assert error.__cause__ is None and error.__suppress_context__ is True


@pytest.mark.asyncio
@pytest.mark.parametrize("step", ["mail", "rcpt"])
@pytest.mark.parametrize(
    "failure",
    [
        smtplib.SMTPServerDisconnected("Connection unexpectedly closed"),
        smtplib.SMTPServerDisconnected("timed out"),
    ],
)
async def test_a_dropped_line_before_data_is_a_definite_failure(smtp_only, step, failure):
    """Nothing can be queued before DATA (RFC 5321 4.1.1.4): the mail did not leave."""
    smtp, _ = smtp_only
    getattr(smtp.return_value, step).side_effect = failure

    error = await _verdict()

    assert error.uncertain is False
    assert mailer.failure_reason(error) == "SMTPServerDisconnected"
    smtp.return_value.data.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "step, outcome, line",
    [
        # A reply code that is not 250 to MAIL FROM, RCPT TO or the final dot.
        ("mail", (421, b"4.7.0 Try again later"), "SMTPSenderRefused code=421 status=4.7.0"),
        (
            "rcpt",
            (550, b"5.1.1 <person@example.com> does not exist"),
            "SMTPRecipientsRefused code=550 status=5.1.1",
        ),
        ("data", (550, b"5.4.5 Daily user sending limit exceeded."), "SMTPDataError code=550 status=5.4.5"),
        ("data", (554, b"Transaction failed"), "SMTPDataError code=554"),
        # DATA itself refused: smtplib raises before sending the message.
        ("data", smtplib.SMTPDataError(451, b"4.3.0 Try later"), "SMTPDataError code=451 status=4.3.0"),
    ],
)
async def test_a_reply_from_the_relay_is_a_definite_refusal(smtp_only, step, outcome, line):
    smtp, _ = smtp_only
    command = getattr(smtp.return_value, step)
    if isinstance(outcome, Exception):
        command.side_effect = outcome
    else:
        command.return_value = outcome

    error = await _verdict()

    assert error.uncertain is False
    assert mailer.failure_reason(error) == line
    assert "person" not in str(error)
    assert error.__cause__ is None and error.__suppress_context__ is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "step, failure",
    [
        ("connect", smtplib.SMTPServerDisconnected("closed during the greeting")),
        ("starttls", smtplib.SMTPServerDisconnected("closed during STARTTLS")),
        ("login", smtplib.SMTPServerDisconnected("closed during AUTH")),
        ("login", smtplib.SMTPAuthenticationError(535, b"5.7.8 Bad credentials")),
    ],
)
async def test_a_failure_before_the_transaction_is_never_unclear(smtp_only, step, failure):
    smtp, _ = smtp_only
    if step == "connect":
        smtp.side_effect = failure
    else:
        getattr(smtp.return_value, step).side_effect = failure

    error = await _verdict()

    assert error.uncertain is False
    smtp.return_value.mail.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("step", ["mail", "data"])
async def test_a_code_bug_inside_the_transaction_is_raised_as_is(smtp_only, step):
    """Only a dropped line is a delivery outcome; a bug keeps its class and traceback."""
    smtp, _ = smtp_only
    bug = ValueError("not a delivery problem")
    getattr(smtp.return_value, step).side_effect = bug

    with pytest.raises(ValueError) as excinfo:
        await send_mail(to="person@example.com", subject="Hi", html="<p>Hi</p>", text="Hi")

    assert excinfo.value is bug
    assert excinfo.value.__traceback__ is not None
    smtp.return_value.close.assert_called()


@pytest.mark.asyncio
async def test_an_unclear_send_skips_ses_when_the_caller_forbids_a_second_copy(
    smtp_to_ses, caplog
):
    """The cohort mail is at most once: an unclear SMTP send must not go out again via SES."""
    smtp, ses = smtp_to_ses
    smtp.return_value.data.side_effect = smtplib.SMTPServerDisconnected("gone")

    with caplog.at_level("INFO", logger=mailer.logger.name):
        error = await _verdict(fallback_on_uncertain=False)

    assert error.uncertain is True
    assert mailer.failure_reason(error) == "SMTPServerDisconnected"
    ses.send_email.assert_not_called()
    lines = [r for r in caplog.records if r.name == mailer.logger.name]
    assert [(r.levelname, r.getMessage()) for r in lines] == [
        (
            "WARNING",
            "mail.no_fallback from=smtp to=ses reason=SMTPServerDisconnected "
            "uncertain=true recipient_domain=example.com",
        )
    ]


@pytest.mark.asyncio
async def test_a_definite_smtp_failure_still_falls_back_when_a_second_copy_is_forbidden(
    smtp_to_ses,
):
    """Nothing left through SMTP, so SES cannot make a duplicate."""
    smtp, ses = smtp_to_ses
    smtp.return_value.rcpt.side_effect = smtplib.SMTPServerDisconnected("gone")

    await send_mail(
        to="person@example.com", subject="Hi", html="<p>Hi</p>", text="Hi",
        fallback_on_uncertain=False,
    )

    ses.send_email.assert_called_once()


@pytest.mark.asyncio
async def test_an_unclear_send_falls_back_to_ses_by_default(smtp_to_ses):
    """Account mails keep at-least-once: a duplicate code mail beats none."""
    smtp, ses = smtp_to_ses
    smtp.return_value.data.side_effect = TimeoutError("timed out")

    await send_mail(to="person@example.com", subject="Hi", html="<p>Hi</p>", text="Hi")

    ses.send_email.assert_called_once()


@pytest.mark.asyncio
async def test_an_unclear_smtp_send_stays_unclear_when_the_ses_fallback_fails(smtp_to_ses):
    smtp, ses = smtp_to_ses
    smtp.return_value.data.side_effect = smtplib.SMTPServerDisconnected("gone")
    ses.send_email.side_effect = _message_rejected("person@example.com")

    error = await _verdict()

    assert error.uncertain is True
    assert error.code == "MessageRejected"
    assert "person" not in str(error)


@pytest.mark.asyncio
async def test_a_definite_smtp_refusal_keeps_the_ses_verdict_when_ses_fails(smtp_to_ses):
    smtp, ses = smtp_to_ses
    smtp.return_value.login.side_effect = smtplib.SMTPAuthenticationError(535, b"5.7.8 Bad")
    ses.send_email.side_effect = _message_rejected("person@example.com")

    error = await _verdict()

    assert error.uncertain is False
    assert str(error) == "SES send failed: ClientError MessageRejected"


@pytest.mark.parametrize(
    "error, line",
    [
        (MailDeliveryFailed("x", reason="SMTPDataError", code=550, status="5.4.5"), "SMTPDataError code=550 status=5.4.5"),
        (MailDeliveryFailed("x", reason="ClientError", code="Throttling"), "ClientError code=Throttling"),
        (MailDeliveryFailed("x", reason="SMTPServerDisconnected", uncertain=True), "SMTPServerDisconnected"),
        (RuntimeError("relay said no to person@example.com"), "RuntimeError"),
    ],
)
def test_failure_reason_is_public_and_never_carries_the_address(error, line):
    assert mailer.failure_reason(error) == line


# ------------------------------------------------- a real smtplib on a socket


class _ScriptedRelay:
    """A minimal SMTP relay on a local socket that can drop the line on cue.

    Exercises the real smtplib, so the classification is checked against what
    smtplib actually raises, not against what a mock was told to raise.
    ``drop_at`` is "mail", "rcpt" or "dot" (after the message, before the 250).
    """

    def __init__(self, drop_at: str | None = None):
        import socket
        import threading

        self.drop_at = drop_at
        self.received = b""
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(1)
        self.port = self._sock.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        conn, _ = self._sock.accept()
        reader = conn.makefile("rb")
        try:
            conn.sendall(b"220 relay.test ESMTP\r\n")
            while True:
                line = reader.readline()
                if not line:
                    return
                verb = line[:4].upper()
                if verb in (b"EHLO", b"HELO"):
                    conn.sendall(b"250 relay.test\r\n")
                elif verb == b"MAIL":
                    if self.drop_at == "mail":
                        return
                    conn.sendall(b"250 2.1.0 OK\r\n")
                elif verb == b"RCPT":
                    if self.drop_at == "rcpt":
                        return
                    conn.sendall(b"250 2.1.5 OK\r\n")
                elif verb == b"DATA":
                    conn.sendall(b"354 go ahead\r\n")
                    while not self.received.endswith(b"\r\n.\r\n"):
                        chunk = reader.readline()
                        if not chunk:
                            return
                        self.received += chunk
                    if self.drop_at == "dot":
                        return
                    conn.sendall(b"250 2.0.0 queued\r\n")
                elif verb == b"QUIT":
                    conn.sendall(b"221 bye\r\n")
                    return
                else:
                    conn.sendall(b"502 5.5.1 unknown\r\n")
        finally:
            reader.close()
            conn.close()
            self._sock.close()

    def join(self):
        self._thread.join(timeout=5)


@pytest.fixture
def real_relay(monkeypatch):
    monkeypatch.setattr(mailer, "MAIL_TRANSPORT", "smtp")
    monkeypatch.setattr(mailer, "SMTP_HOST", "127.0.0.1")
    monkeypatch.setattr(mailer, "SMTP_USERNAME", "")
    monkeypatch.setattr(mailer, "SMTP_PASSWORD", "")

    def _start(drop_at=None):
        relay = _ScriptedRelay(drop_at)
        monkeypatch.setattr(mailer, "SMTP_PORT", relay.port)
        return relay

    return _start


@pytest.mark.asyncio
async def test_a_real_relay_gets_the_message_with_crlf_line_ends(real_relay):
    """Many relays refuse bare LF in DATA; smtplib only fixes line ends of a str."""
    relay = real_relay()

    await send_mail(to="person@example.com", subject="Hi", html="<p>Hi</p>", text="Hi")
    relay.join()

    body = relay.received
    assert b"Subject: Hi\r\n" in body
    assert b"\n" not in body.replace(b"\r\n", b"")


@pytest.mark.asyncio
@pytest.mark.parametrize("drop_at, uncertain", [("mail", False), ("rcpt", False), ("dot", True)])
async def test_a_real_dropped_line_is_unclear_only_after_data(real_relay, drop_at, uncertain):
    relay = real_relay(drop_at)

    error = await _verdict()
    relay.join()

    assert error.uncertain is uncertain
    assert mailer.failure_reason(error) == "SMTPServerDisconnected"
