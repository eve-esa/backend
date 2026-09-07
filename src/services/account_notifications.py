"""The two messages the approval gate owes a person, and nothing else.

An account that lands in the queue gets told so, and an account somebody
approves gets told that too. Both are rendered here and handed to
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


def render_account_mail(kind: str, email: str) -> Tuple[str, str, str]:
    """Return ``(subject, html, text)`` for one of the two account messages.

    The logo and the link both derive from ``FRONTEND_URL``, so a message sent
    from dev points at dev and one sent from prod points at prod, with no
    per-environment template.
    """
    subject, html_template, text_template = _MESSAGES[kind]
    context = {
        "email": email,
        "app_url": FRONTEND_URL,
        "logo_url": f"{FRONTEND_URL}/branding/eve-logo.png",
    }
    html = _environment.get_template(html_template).render(**context)
    text = _environment.get_template(text_template).render(**context)
    return subject, html, text


async def _notify(kind: str, user_id: str, email: str) -> None:
    subject, html, text = render_account_mail(kind, email)
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


async def notify_account_approved(user_id: str, email: str) -> None:
    """Tell an account that somebody approved it and it can sign in."""
    await _notify(KIND_APPROVED, user_id, email)
