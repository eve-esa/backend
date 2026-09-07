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

Recipients are never logged in full. A subject and a domain are enough to see
that mail is flowing; the address belongs in the message, not in CloudWatch.
"""

from __future__ import annotations

import asyncio
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr

import boto3

from src.config import (
    EMAIL_FROM_ADDRESS,
    EMAIL_FROM_NAME,
    MAIL_TRANSPORT,
    SES_CONFIGURATION_SET,
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


_ses_client = None


def _recipient_domain(address: str) -> str:
    """The part of an address that is safe to log."""
    _, _, domain = address.rpartition("@")
    return domain or "unknown"


def _sender() -> str:
    """The From header, as a single RFC 5322 address.

    ``EMAIL_FROM_ADDRESS`` is allowed to carry its own display name, because
    that is how the value is already written in the deployed environments
    ("EVE by ESA Phi Lab <no-reply@dev.eve-chat.chat>"). Only a bare address
    gets ``EMAIL_FROM_NAME`` attached, so a configured name is never doubled.
    """
    address = (EMAIL_FROM_ADDRESS or "").strip()
    if not address:
        raise MailNotConfigured(
            "MAIL_TRANSPORT is set but EMAIL_FROM_ADDRESS is blank: set a sender "
            "address (optionally 'Display Name <user@domain>') or set "
            "MAIL_TRANSPORT=off"
        )
    if "<" in address:
        return address
    name = (EMAIL_FROM_NAME or "").strip()
    return formataddr((name, address)) if name else address


def _build_message(
    *, sender: str, to: str, subject: str, html: str, text: str
) -> MIMEMultipart:
    """A multipart/alternative message: plain text first, HTML second.

    Order is not cosmetic. A reader picks the last part it can render, so the
    HTML has to come after the text or every modern client shows the fallback.
    """
    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = to
    message.attach(MIMEText(text, "plain", "utf-8"))
    message.attach(MIMEText(html, "html", "utf-8"))
    return message


def _send_smtp(*, sender: str, to: str, subject: str, html: str, text: str) -> None:
    """Blocking SMTP send, called from a worker thread.

    STARTTLS only when a username is configured. The local Mailpit container
    (compose service "mailcatcher") speaks plain SMTP on port 1025 and offers no
    TLS at all, so upgrading unconditionally would break the one environment
    this transport exists for.
    """
    message = _build_message(
        sender=sender, to=to, subject=subject, html=html, text=text
    )
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=SMTP_TIMEOUT_SECONDS) as server:
        if SMTP_USERNAME:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(message)


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
    transport = (MAIL_TRANSPORT or TRANSPORT_OFF).strip().lower()

    if transport == TRANSPORT_OFF:
        logger.info(
            "mail suppressed (MAIL_TRANSPORT=off) subject=%r recipient_domain=%s",
            subject,
            _recipient_domain(to),
        )
        return

    sender = _sender()

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

    logger.info(
        "mail sent via %s subject=%r recipient_domain=%s",
        transport,
        subject,
        _recipient_domain(to),
    )
