"""The mails the platform owes a person, rendered from one spec table.

An account that lands in the queue gets told so, an account that gets in,
straight away or after the wait, gets the welcome message, and the launch mails
go to accounts and pre-registered people in cohorts. Each is rendered here and
handed to ``src/services/mailer.py``, which decides how it leaves the process.

Copy lives in templates, not in this module: in the published pack when one is
configured, else in ``src/templates/mail/`` (``src/services/mail_templates.py``).
``MAIL_SPECS`` is the backend half of the pack contract: each mail's default
subject, bundled files and the exact variables it receives. A new mail is one
entry here plus its bundled templates, and ``docs/mail-templates.md``.

Log lines carry the user id and never the address. The id is what an operator
looks up in the back office; the address is personal data that would then live
in every log sink the platform ships to.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from src.config import FRONTEND_URL
from src.services.mail_templates import (
    BASE_VARIABLES,
    MailSpec,
    MailTemplateStore,
    RenderedMail,
)
from src.services.mailer import send_mail

logger = logging.getLogger(__name__)

KIND_PENDING = "pending"
KIND_APPROVED = "approved"

MAIL_ACCOUNT_PENDING = "account_pending"
MAIL_ACCOUNT_APPROVED = "account_approved"
MAIL_COHORT_INVITE = "cohort_invite"
MAIL_COHORT_INVITE_SIGNUP = "cohort_invite_signup"


# The sample values a pack is trial rendered with when it loads: every branch
# a real send can take (with and without the optional parts).
_SAMPLE_BASE = {
    "email": "trial.recipient@example.org",
    "app_url": "https://eve.example.org",
    "logo_url": "https://eve.example.org/branding/eve-logo.png",
}
_SAMPLE_PARAGRAPHS = ["A first sample paragraph.", "A second sample paragraph."]


def _spec(name: str, subject: str, *variants: dict) -> MailSpec:
    """One mail: its name, default subject, and its extra variables by sample.

    Each variant is the extra part of one trial context; the first one also
    names the extra variables the mail receives.
    """
    variants = variants or ({},)
    return MailSpec(
        name=name,
        subject=subject,
        html=f"{name}.html",
        text=f"{name}.txt",
        variables=BASE_VARIABLES | frozenset(variants[0]),
        samples=tuple({**_SAMPLE_BASE, **variant} for variant in variants),
    )


MAIL_SPECS: dict[str, MailSpec] = {
    spec.name: spec
    for spec in (
        _spec(MAIL_ACCOUNT_PENDING, "Your EVE account is on hold for now"),
        _spec(
            MAIL_ACCOUNT_APPROVED,
            "Your EVE account is ready",
            {"after_hold": False},
            {"after_hold": True},
        ),
        # The launch mail to legacy accounts (decision D16). ``paragraphs`` is
        # the optional text block from the command line, split on blank lines.
        _spec(
            MAIL_COHORT_INVITE,
            "EVE is now open, come try it",
            {"sign_in_url": "https://eve.example.org", "paragraphs": []},
            {"sign_in_url": "https://eve.example.org", "paragraphs": _SAMPLE_PARAGRAPHS},
        ),
        # The launch mail to a pre-registered person with no account yet.
        _spec(
            MAIL_COHORT_INVITE_SIGNUP,
            "Create your EVE account",
            {"sign_up_url": "https://eve.example.org/signup"},
        ),
    )
}

_KIND_TO_MAIL = {KIND_PENDING: MAIL_ACCOUNT_PENDING, KIND_APPROVED: MAIL_ACCOUNT_APPROVED}

# The process-wide store: server.py starts its refresh task, the cohort command
# refreshes it once before sending.
mail_template_store = MailTemplateStore(MAIL_SPECS)


def _base_context(email: str) -> dict:
    # The logo and the links derive from FRONTEND_URL, so a mail sent from dev
    # points at dev and one sent from prod points at prod.
    return {
        "email": email,
        "app_url": FRONTEND_URL,
        "logo_url": f"{FRONTEND_URL}/branding/eve-logo.png",
    }


def build_account_mail(kind: str, email: str, after_hold: bool = False) -> RenderedMail:
    """Render one of the two account messages, with its template source."""
    name = _KIND_TO_MAIL[kind]
    context = _base_context(email)
    if name == MAIL_ACCOUNT_APPROVED:
        # Only the welcome message thanks an account that waited in the queue.
        context["after_hold"] = after_hold
    return mail_template_store.render(name, context)


def render_account_mail(
    kind: str, email: str, after_hold: bool = False
) -> Tuple[str, str, str]:
    """Return ``(subject, html, text)`` for one of the two account messages."""
    return build_account_mail(kind, email, after_hold)[:3]


async def _notify(
    kind: str, user_id: str, email: str, after_hold: bool = False
) -> None:
    mail = build_account_mail(kind, email, after_hold)
    try:
        await send_mail(to=email, subject=mail.subject, html=mail.html, text=mail.text)
    except Exception:
        # Logged here because this is the layer that knows which message failed;
        # re-raised because only the caller knows whether that is fatal.
        logger.error(
            "account_mail_failed kind=%s user_id=%s", kind, user_id, exc_info=True
        )
        raise
    logger.info(
        "account_mail_sent kind=%s user_id=%s template_source=%s pack_version=%s",
        kind,
        user_id,
        mail.source,
        mail.version,
    )


async def notify_account_pending(user_id: str, email: str) -> None:
    """Tell a brand-new account that it is in the approval queue."""
    await _notify(KIND_PENDING, user_id, email)


async def notify_account_approved(
    user_id: str, email: str, *, after_hold: bool = False
) -> None:
    """Welcome an account that can sign in, whether it waited first or not."""
    await _notify(KIND_APPROVED, user_id, email, after_hold=after_hold)


def _paragraphs(text: str) -> list[str]:
    """Split a plain-text block on blank lines, joining wrapped lines."""
    blocks = [block.strip() for block in text.replace("\r\n", "\n").split("\n\n")]
    return [" ".join(line.strip() for line in block.splitlines()) for block in blocks if block]


def build_cohort_invite(
    email: str,
    *,
    sign_in_url: Optional[str] = None,
    text_block: Optional[str] = None,
    subject: Optional[str] = None,
) -> RenderedMail:
    """Render the launch mail of one account, with its template source.

    Without a text block the templates render the launch copy they own. A text
    block is plain text from a file the product owner approves; its paragraphs
    replace that copy and are escaped in the HTML part like every other value.
    ``subject`` wins over the pack and the default subject.
    """
    context = {
        **_base_context(email),
        "sign_in_url": (sign_in_url or FRONTEND_URL).strip(),
        "paragraphs": _paragraphs(text_block) if text_block else [],
    }
    return mail_template_store.render(MAIL_COHORT_INVITE, context, subject=subject)


def render_cohort_invite(
    email: str,
    *,
    sign_in_url: Optional[str] = None,
    text_block: Optional[str] = None,
    subject: Optional[str] = None,
) -> Tuple[str, str, str]:
    """Return ``(subject, html, text)`` for the launch mail of one account."""
    return build_cohort_invite(
        email, sign_in_url=sign_in_url, text_block=text_block, subject=subject
    )[:3]


def build_cohort_invite_signup(
    email: str, *, sign_up_url: Optional[str] = None, subject: Optional[str] = None
) -> RenderedMail:
    """Render the launch mail of a pre-registered person who has no account yet."""
    context = {
        **_base_context(email),
        "sign_up_url": (sign_up_url or FRONTEND_URL).strip(),
    }
    return mail_template_store.render(MAIL_COHORT_INVITE_SIGNUP, context, subject=subject)
