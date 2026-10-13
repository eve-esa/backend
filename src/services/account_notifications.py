"""The two messages the approval gate owes a person, and nothing else.

An account that lands in the queue gets told so, and an account that gets in,
straight away or after the wait, gets the welcome message. Both are rendered here and handed to
``src/services/mailer.py``, which decides how they leave the process.

Copy lives in ``src/templates/mail/`` rather than in this module so that
changing a sentence is not a Python change, and so the HTML and plain-text
versions sit next to each other and stay in step.

Log lines carry the user id and never the address. The id is what an operator
looks up in the back office; the address is personal data that would then live
in every log sink the platform ships to.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.config import FRONTEND_URL
from src.services.mailer import send_mail

logger = logging.getLogger(__name__)

KIND_PENDING = "pending"
KIND_APPROVED = "approved"

# Subject, HTML template, text template. One table so a new message is one entry
# and cannot be half-wired.
_MESSAGES = {
    KIND_PENDING: (
        "Your EVE account is on hold for now",
        "account_pending.html",
        "account_pending.txt",
    ),
    KIND_APPROVED: (
        "Your EVE account is ready",
        "account_approved.html",
        "account_approved.txt",
    ),
}

# Shipped inside the image: the Dockerfile does ``COPY src/ ./src/``, so anything
# under src/templates travels with the code and needs no package-data wiring.
_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates" / "mail"

# Autoescape by extension, not unconditionally: an address with an ampersand in
# it must be escaped in the HTML part and must stay literal in the text part.
_environment = Environment(
    loader=FileSystemLoader(str(_TEMPLATES_DIR)),
    autoescape=select_autoescape(
        enabled_extensions=("html", "xml"), default_for_string=True
    ),
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_account_mail(
    kind: str, email: str, after_hold: bool = False
) -> Tuple[str, str, str]:
    """Return ``(subject, html, text)`` for one of the two account messages.

    The logo and the link both derive from ``FRONTEND_URL``, so a message sent
    from dev points at dev and one sent from prod points at prod, with no
    per-environment template. ``after_hold`` only matters to the welcome
    message, which thanks an account that waited in the queue.
    """
    subject, html_template, text_template = _MESSAGES[kind]
    context = {
        "email": email,
        "app_url": FRONTEND_URL,
        "logo_url": f"{FRONTEND_URL}/branding/eve-logo.png",
        "after_hold": after_hold,
    }
    html = _environment.get_template(html_template).render(**context)
    text = _environment.get_template(text_template).render(**context)
    return subject, html, text


async def _notify(
    kind: str, user_id: str, email: str, after_hold: bool = False
) -> None:
    subject, html, text = render_account_mail(kind, email, after_hold)
    try:
        await send_mail(to=email, subject=subject, html=html, text=text)
    except Exception:
        # Logged here because this is the layer that knows which message failed;
        # re-raised because only the caller knows whether that is fatal.
        logger.error(
            "account_mail_failed kind=%s user_id=%s", kind, user_id, exc_info=True
        )
        raise
    logger.info("account_mail_sent kind=%s user_id=%s", kind, user_id)


async def notify_account_pending(user_id: str, email: str) -> None:
    """Tell a brand-new account that it is in the approval queue."""
    await _notify(KIND_PENDING, user_id, email)


async def notify_account_approved(
    user_id: str, email: str, *, after_hold: bool = False
) -> None:
    """Welcome an account that can sign in, whether it waited first or not."""
    await _notify(KIND_APPROVED, user_id, email, after_hold=after_hold)


# The launch mail to legacy accounts (decision D16, sent in cohorts by
# ``src/commands/send_cohort_invite.py``). The subject and the text block are
# approved by product and arrive from the command line, so these defaults are
# deliberately neutral: they say what is true for every legacy account and
# nothing more.
COHORT_INVITE_SUBJECT = "The new EVE is ready for you"
COHORT_INVITE_TEXT = (
    "EVE, the Earth Virtual Expert by ESA Phi-lab, has a new version, and your "
    "account comes with it.\n"
    "\n"
    "Sign in with the same e-mail address and password you used before. If you "
    "do not remember your password, choose \"Forgot password\" on the sign-in "
    "page and follow the steps."
)


def _paragraphs(text: str) -> list[str]:
    """Split a plain-text block on blank lines, joining wrapped lines."""
    blocks = [block.strip() for block in text.replace("\r\n", "\n").split("\n\n")]
    return [" ".join(line.strip() for line in block.splitlines()) for block in blocks if block]


def render_cohort_invite(
    email: str,
    *,
    sign_in_url: str | None = None,
    text_block: str | None = None,
    subject: str | None = None,
) -> Tuple[str, str, str]:
    """Return ``(subject, html, text)`` for the launch mail of one account.

    The text block is plain text from a file the product owner approves; it is
    escaped in the HTML part like every other value, so a stray ``<`` cannot
    change the layout.
    """
    context = {
        "email": email,
        "app_url": FRONTEND_URL,
        "logo_url": f"{FRONTEND_URL}/branding/eve-logo.png",
        "sign_in_url": (sign_in_url or FRONTEND_URL).strip(),
        "paragraphs": _paragraphs(text_block or COHORT_INVITE_TEXT),
    }
    html = _environment.get_template("cohort_invite.html").render(**context)
    text = _environment.get_template("cohort_invite.txt").render(**context)
    return (subject or COHORT_INVITE_SUBJECT).strip(), html, text
