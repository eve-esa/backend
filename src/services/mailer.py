"""One way out for transactional mail, with the transport picked by config.

Callers hand over a rendered message and never learn how it left the process.
That is the whole point: local compose talks to Mailpit over plain SMTP, the
deployed environments hand the message to Amazon SES, and a host with no mail
configuration at all sends nothing. None of those three is a code change.

``MAIL_TRANSPORT`` defaults to "off" on purpose. A missing SES permission or a
forgotten SMTP host must not turn into a failed sign-in for somebody who only
wanted to log in, so an unconfigured host is silent rather than broken. What it
must not be is *quietly* silent: the suppressed path logs, and any transport
other than "off" refuses to send without a sender rather than mailing from a
domain nobody owns.

With ``MAIL_TRANSPORT=ses`` a non-empty ``MAIL_CANARY_TAG`` diverts the
recipients whose local part carries the tag to SMTP, so a new relay can be
proven on a few test mailboxes while every real user stays on SES. The choice
is made in one place, :func:`_choose_transport`.

``MAIL_FALLBACK_TRANSPORT=ses`` keeps a relay outage from losing mail: when
the chosen transport is SMTP and the send raises, the same message goes out
through SES with the SES sender, and one WARNING line says why.

Recipients are never logged in full. A subject and a domain are enough to see
that mail is flowing; the address belongs in the message, not in CloudWatch.
"""

from __future__ import annotations

import asyncio
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr, formatdate, make_msgid, parseaddr

import boto3

from src.config import (
    EMAIL_FROM_ADDRESS,
    EMAIL_FROM_NAME,
    MAIL_CANARY_TAG,
    MAIL_FALLBACK_TRANSPORT,
    MAIL_TRANSPORT,
    SES_CONFIGURATION_SET,
    SMTP_FROM_ADDRESS,
    SMTP_HOST,
    SMTP_PASSWORD,
    SMTP_PORT,
    SMTP_USERNAME,
)

logger = logging.getLogger(__name__)

# Long enough for a slow relay handshake, short enough that a black-holed host
# cannot pin a worker thread for minutes.
SMTP_TIMEOUT_SECONDS = 20.0

TRANSPORT_OFF = "off"
TRANSPORT_SMTP = "smtp"
TRANSPORT_SES = "ses"


class MailNotConfigured(RuntimeError):
    """The chosen transport cannot run with the configuration it was given."""


class MailDeliveryFailed(RuntimeError):
    """The relay refused the message. Names the error class, never the address.

    smtplib errors such as ``SMTPRecipientsRefused`` carry the recipient in their
    text, and callers log failures with ``exc_info``. The class name and the
    SMTP reply code survive as attributes, for the fallback log line.
    """

    def __init__(self, message: str, *, reason: str = "", smtp_code: int | None = None):
        super().__init__(message)
        self.reason = reason
        self.smtp_code = smtp_code


_ses_client = None


def _recipient_domain(address: str) -> str:
    """The part of an address that is safe to log."""
    _, _, domain = address.rpartition("@")
    return domain or "unknown"


def _smtp_code(exc: smtplib.SMTPException) -> int | None:
    """The reply code of a refused SMTP command, when the server sent one."""
    if isinstance(exc, smtplib.SMTPResponseException):
        return exc.smtp_code
    if isinstance(exc, smtplib.SMTPRecipientsRefused) and exc.recipients:
        code, _ = next(iter(exc.recipients.values()))
        return code
    return None


def _failure_reason(exc: BaseException) -> str:
    """``reason=<class>[ code=<n>]`` for a failed send, with no address in it."""
    if isinstance(exc, MailDeliveryFailed) and exc.reason:
        reason = exc.reason
        if exc.smtp_code is not None:
            reason += f" code={exc.smtp_code}"
        return reason
    return type(exc).__name__


def _choose_transport(transport: str, to: str) -> str:
    """The transport that carries a message to this recipient.

    Only "ses" is ever diverted, and only to "smtp": "off" stays silent for a
    tagged address too, and an empty tag matches nobody.
    """
    tag = (MAIL_CANARY_TAG or "").lower()
    if transport != TRANSPORT_SES or not tag:
        return transport
    local_part, _, _ = to.rpartition("@")
    return TRANSPORT_SMTP if tag in local_part.lower() else transport


def _sender(transport: str) -> str:
    """The From header, as a single RFC 5322 address.

    The SMTP path uses ``SMTP_FROM_ADDRESS`` when it is set, because a relay
    such as Google Workspace only sends as its own mailbox; blank falls back to
    ``EMAIL_FROM_ADDRESS`` like every other transport.

    ``EMAIL_FROM_ADDRESS`` is allowed to carry its own display name, because
    that is how the value is already written in the deployed environments
    ("EVE by ESA Phi Lab <no-reply@dev.eve-chat.chat>"). Only a bare address
    gets ``EMAIL_FROM_NAME`` attached, so a configured name is never doubled.
    A value that does not parse to one address with a domain is refused: an
    unquoted comma in the display name would split the From in two.
    """
    address = ""
    if transport == TRANSPORT_SMTP:
        address = (SMTP_FROM_ADDRESS or "").strip()
    address = address or (EMAIL_FROM_ADDRESS or "").strip()
    if not address:
        raise MailNotConfigured(
            "MAIL_TRANSPORT is set but EMAIL_FROM_ADDRESS is blank: set a sender "
            "address (optionally 'Display Name <user@domain>') or set "
            "MAIL_TRANSPORT=off"
        )
    if "<" not in address:
        name = (EMAIL_FROM_NAME or "").strip()
        address = formataddr((name, address)) if name else address
    _, parsed = parseaddr(address)
    if not parsed.rpartition("@")[2]:
        raise MailNotConfigured(
            f"The sender for MAIL_TRANSPORT={transport} does not parse to one "
            "address with a domain: quote a display name that contains a comma"
        )
    return address


def _build_message(
    *, sender: str, to: str, subject: str, html: str, text: str
) -> MIMEMultipart:
    """A multipart/alternative message: plain text first, HTML second.

    Order is not cosmetic. A reader picks the last part it can render, so the
    HTML has to come after the text or every modern client shows the fallback.

    Date and Message-ID are set here because SMTP, unlike SES, leaves them to
    the client, and receiving filters score a message without them as spam.
    The Message-ID carries the From domain so it aligns with the sender.
    """
    _, sender_address = parseaddr(sender)
    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = to
    message["Date"] = formatdate(localtime=False, usegmt=True)
    message["Message-ID"] = make_msgid(domain=sender_address.rpartition("@")[2] or None)
    message.attach(MIMEText(text, "plain", "utf-8"))
    message.attach(MIMEText(html, "html", "utf-8"))
    return message


def _send_smtp(*, sender: str, to: str, subject: str, html: str, text: str) -> None:
    """Blocking SMTP send, called from a worker thread.

    STARTTLS only when a username is configured. The local Mailpit container
    (compose service "mailcatcher") speaks plain SMTP on port 1025 and offers no
    TLS at all, so upgrading unconditionally would break the one environment
    this transport exists for. A relay sender (``SMTP_FROM_ADDRESS``) without a
    username is refused instead, so a deployed relay never gets mail in clear.
    """
    if SMTP_FROM_ADDRESS and not SMTP_USERNAME:
        raise MailNotConfigured(
            "SMTP_FROM_ADDRESS is set but SMTP_USERNAME is blank: a relay needs "
            "STARTTLS and a login, plain SMTP is only for the local catcher"
        )
    message = _build_message(
        sender=sender, to=to, subject=subject, html=html, text=text
    )
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=SMTP_TIMEOUT_SECONDS) as server:
            if SMTP_USERNAME:
                server.starttls()
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(message, to_addrs=[to])
    except smtplib.SMTPException as exc:
        raise MailDeliveryFailed(
            f"SMTP send failed: {type(exc).__name__}",
            reason=type(exc).__name__,
            smtp_code=_smtp_code(exc),
        ) from None


def _get_ses_client():
    """Lazily build and cache the SES v2 client.

    Built on first use rather than at import: creating a boto3 client resolves
    credentials and region, and a module that is imported by every host must not
    do that work on a host whose transport is "off". Region comes from the
    normal boto3 resolution chain, which is AWS_REGION on the ECS task.
    """
    global _ses_client
    if _ses_client is None:
        _ses_client = boto3.client("sesv2")
    return _ses_client


def _reset_ses_client_for_tests() -> None:
    global _ses_client
    _ses_client = None


def _send_ses(*, sender: str, to: str, subject: str, html: str, text: str) -> None:
    """Blocking SES v2 send, called from a worker thread.

    Simple content rather than Raw: SES builds the MIME itself, so there is no
    second place where the multipart ordering could be got wrong.
    """
    request = {
        "FromEmailAddress": sender,
        "Destination": {"ToAddresses": [to]},
        "Content": {
            "Simple": {
                "Subject": {"Data": subject, "Charset": "UTF-8"},
                "Body": {
                    "Text": {"Data": text, "Charset": "UTF-8"},
                    "Html": {"Data": html, "Charset": "UTF-8"},
                },
            }
        },
    }
    if SES_CONFIGURATION_SET:
        request["ConfigurationSetName"] = SES_CONFIGURATION_SET
    _get_ses_client().send_email(**request)


async def send_mail(to: str, subject: str, html: str, text: str) -> None:
    """Deliver one message, or explain why it could not be delivered.

    Raises :class:`MailNotConfigured` when the transport cannot run at all, and
    lets whatever the transport raises through otherwise. Callers decide what a
    failure means: the approval endpoint answers 502, the sign-in path logs and
    carries on.
    """
    transport = _choose_transport((MAIL_TRANSPORT or TRANSPORT_OFF).strip().lower(), to)

    if transport == TRANSPORT_OFF:
        logger.info(
            "mail suppressed (MAIL_TRANSPORT=off) subject=%r recipient_domain=%s",
            subject,
            _recipient_domain(to),
        )
        return

    fallback = (MAIL_FALLBACK_TRANSPORT or "").strip().lower()
    if fallback not in ("", TRANSPORT_SES):
        raise MailNotConfigured(
            f"Unknown MAIL_FALLBACK_TRANSPORT {fallback!r}: expected blank or "
            f"{TRANSPORT_SES!r}"
        )

    if transport == TRANSPORT_SMTP and fallback == TRANSPORT_SES:
        try:
            await asyncio.to_thread(
                _send_smtp,
                sender=_sender(TRANSPORT_SMTP),
                to=to,
                subject=subject,
                html=html,
                text=text,
            )
        except Exception as exc:
            # The SMTP error is only logged: the caller sees SES's verdict.
            logger.warning(
                "mail.fallback from=%s to=%s reason=%s recipient_domain=%s",
                TRANSPORT_SMTP,
                TRANSPORT_SES,
                _failure_reason(exc),
                _recipient_domain(to),
            )
            transport = TRANSPORT_SES
            await asyncio.to_thread(
                _send_ses,
                sender=_sender(TRANSPORT_SES),
                to=to,
                subject=subject,
                html=html,
                text=text,
            )
        _log_sent(transport, subject, to)
        return

    sender = _sender(transport)

    if transport == TRANSPORT_SMTP:
        await asyncio.to_thread(
            _send_smtp, sender=sender, to=to, subject=subject, html=html, text=text
        )
    elif transport == TRANSPORT_SES:
        await asyncio.to_thread(
            _send_ses, sender=sender, to=to, subject=subject, html=html, text=text
        )
    else:
        raise MailNotConfigured(
            f"Unknown MAIL_TRANSPORT {transport!r}: expected one of "
            f"{TRANSPORT_OFF!r}, {TRANSPORT_SMTP!r}, {TRANSPORT_SES!r}"
        )

    _log_sent(transport, subject, to)


def _log_sent(transport: str, subject: str, to: str) -> None:
    logger.info(
        "mail sent via %s subject=%r recipient_domain=%s",
        transport,
        subject,
        _recipient_domain(to),
    )
