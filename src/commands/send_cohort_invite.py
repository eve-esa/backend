"""Send the launch mail to legacy accounts, one cohort at a time (decision D16).

The accounts that existed before the public opening get one mail telling them
the new EVE is live; everybody else registers on their own. The mail goes out in
cohorts so that SES reputation, support load and sign-in traffic grow in steps.

    python -m src.commands.send_cohort_invite --cohort launch-1 --limit 50
    python -m src.commands.send_cohort_invite --cohort launch-1 --limit 50 --apply
    python -m src.commands.send_cohort_invite --cohort vip --emails-file vip.txt --apply

Dry run by default: it reads, prints the counts and the first five recipients
masked, and writes nothing. A dry run works with the reader credential: it
connects without creating indexes. ``--apply`` needs the writer.

Selection. ``--limit`` and ``--offset`` slice ``users`` sorted by ``_id``, before
any skip rule, so a slice is the same set of rows on every run and consecutive
cohorts never overlap (``--offset 0 --limit 100``, then ``--offset 100 --limit
100``). ``--emails-file`` selects the rows whose address is listed instead, one
address per line, blank lines and ``#`` comments ignored.

Skipped, each counted on its own line:
  no_email        the row has no address
  not_approved    ``approval_status`` is set and is not "approved" (None or an
                  absent field means the row was never gated, so it is eligible)
  migrated        the user already signed in to the new version (a row in
                  ``external_identities``), unless ``--include-migrated``
  duplicate_email a second row with the same address in this run
  already_sent    ``cohort_invites`` holds a sent row for (user_id, cohort)
  in_doubt        a claim without ``sent_at``: a previous run died between the
                  claim and the send, so whether the mail left is unknown

Idempotency. ``cohort_invites`` holds one document per (``user_id``,
``cohort``), unique. The row is inserted as a claim before the send and gets
``sent_at`` after it: two runs of the same cohort cannot both mail a person, and
a re-run skips everybody already served. A failed send removes its claim, so the
next run retries it. A claim left by a killed process is reported as in_doubt
and never resent automatically; to resend, delete it:
``db.cohort_invites.deleteMany({cohort: "<name>", sent_at: null})``.

Logs carry counts and the cohort, never an address.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from pymongo.errors import DuplicateKeyError

from src.config import configure_logging
from src.database.mongo import async_mongo_manager, get_collection
from src.services import mailer
from src.services.account_notifications import render_cohort_invite
from src.services.mailer import send_mail

configure_logging()
logger = logging.getLogger(__name__)

COLLECTION = "cohort_invites"
DEFAULT_BATCH_SIZE = 50
DEFAULT_SLEEP_SECONDS = 60.0
PREVIEW_COUNT = 5

SKIP_REASONS = (
    "no_email",
    "not_approved",
    "migrated",
    "duplicate_email",
    "already_sent",
    "in_doubt",
)


class CohortInviteError(RuntimeError):
    """The command refuses to run with the arguments or configuration given."""


@dataclass
class Recipient:
    user_id: str
    email: str


@dataclass
class CohortPlan:
    cohort: str
    selected: int = 0
    not_found: int = 0
    recipients: list[Recipient] = field(default_factory=list)
    skipped: Counter = field(default_factory=Counter)


def mask_email(address: str) -> str:
    """``alice@example.org`` becomes ``a***@example.org``."""
    local, at, domain = (address or "").strip().rpartition("@")
    if not at:
        return "***"
    return f"{local[:1]}***@{domain}"


def read_emails_file(path: str | Path) -> list[str]:
    """Addresses from a file, lowercased, in order, without repeats."""
    seen: dict[str, None] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            seen.setdefault(value.lower(), None)
    return list(seen)


def _normalize_cohort(cohort: str) -> str:
    value = (cohort or "").strip()
    if not value:
        raise CohortInviteError("--cohort must be a non-empty label")
    return value


async def _select_users(
    *, limit: Optional[int], offset: int, emails: Optional[list[str]]
) -> list[dict]:
    projection = {"email": 1, "approval_status": 1}
    users = get_collection("users")
    if emails is not None:
        cursor = users.find({"email": {"$in": emails}}, projection).sort("_id", 1)
    else:
        cursor = users.find({}, projection).sort("_id", 1).skip(offset)
        if limit is not None:
            cursor = cursor.limit(limit)
    return [doc async for doc in cursor]


async def _user_ids_with(collection: str, query: dict, ids: list[str]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not ids:
        return rows
    cursor = get_collection(collection).find({**query, "user_id": {"$in": ids}})
    async for doc in cursor:
        rows.setdefault(doc["user_id"], doc)
    return rows


async def plan_cohort(
    cohort: str,
    *,
    limit: Optional[int] = None,
    offset: int = 0,
    emails: Optional[list[str]] = None,
    include_migrated: bool = False,
) -> CohortPlan:
    """Select the rows for a cohort and apply the skip rules. Reads only."""
    cohort = _normalize_cohort(cohort)
    docs = await _select_users(limit=limit, offset=offset, emails=emails)
    plan = CohortPlan(cohort=cohort, selected=len(docs))

    if emails is not None:
        found = {str(doc.get("email") or "").strip().lower() for doc in docs}
        plan.not_found = sum(1 for address in emails if address not in found)

    ids = [str(doc["_id"]) for doc in docs]
    invites = await _user_ids_with(COLLECTION, {"cohort": cohort}, ids)
    migrated = await _user_ids_with("external_identities", {}, ids)

    seen_emails: set[str] = set()
    for doc in docs:
        user_id = str(doc["_id"])
        email = str(doc.get("email") or "").strip()
        status = doc.get("approval_status")
        if not email:
            reason = "no_email"
        elif status is not None and status != "approved":
            reason = "not_approved"
        elif user_id in migrated and not include_migrated:
            reason = "migrated"
        elif email.lower() in seen_emails:
            reason = "duplicate_email"
        elif user_id in invites:
            reason = "already_sent" if invites[user_id].get("sent_at") else "in_doubt"
        else:
            reason = None

        if reason:
            plan.skipped[reason] += 1
            continue
        seen_emails.add(email.lower())
        plan.recipients.append(Recipient(user_id=user_id, email=email))
    return plan


def _batches(items: list[Recipient], size: int) -> Iterable[list[Recipient]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


async def _pause(seconds: float) -> None:
    """Indirection so tests can count the pauses without waiting."""
    await asyncio.sleep(seconds)


async def _claim(collection, user_id: str, cohort: str) -> bool:
    try:
        await collection.insert_one(
            {
                "user_id": user_id,
                "cohort": cohort,
                "claimed_at": datetime.now(timezone.utc),
                "sent_at": None,
            }
        )
    except DuplicateKeyError:
        return False
    return True


async def _send_one(
    collection,
    recipient: Recipient,
    cohort: str,
    *,
    sign_in_url: Optional[str],
    text_block: Optional[str],
    subject: Optional[str],
) -> str:
    """Claim, send, stamp. Returns "sent", "failed" or "claimed_elsewhere"."""
    if not await _claim(collection, recipient.user_id, cohort):
        return "claimed_elsewhere"
    mail_subject, html, text = render_cohort_invite(
        recipient.email, sign_in_url=sign_in_url, text_block=text_block, subject=subject
    )
    try:
        await send_mail(to=recipient.email, subject=mail_subject, html=html, text=text)
    except Exception:
        # Release the claim so the next run retries this person.
        await collection.delete_one(
            {"user_id": recipient.user_id, "cohort": cohort, "sent_at": None}
        )
        logger.error(
            "cohort_invite_failed cohort=%s user_id=%s",
            cohort,
            recipient.user_id,
            exc_info=True,
        )
        return "failed"
    await collection.update_one(
        {"user_id": recipient.user_id, "cohort": cohort},
        {"$set": {"sent_at": datetime.now(timezone.utc)}},
    )
    return "sent"


def _print_plan(plan: CohortPlan, *, batch_size: int, sleep_seconds: float) -> None:
    print(f"Cohort: {plan.cohort}")
    print(f"Selected rows: {plan.selected}")
    if plan.not_found:
        print(f"Listed addresses with no user row: {plan.not_found}")
    for reason in SKIP_REASONS:
        if plan.skipped[reason]:
            print(f"Skipped {reason}: {plan.skipped[reason]}")
    batches = -(-len(plan.recipients) // batch_size) if plan.recipients else 0
    print(
        f"Recipients: {len(plan.recipients)} in {batches} batch(es) of up to "
        f"{batch_size}, {sleep_seconds:g} s between batches"
    )
    for recipient in plan.recipients[:PREVIEW_COUNT]:
        print(f"  {mask_email(recipient.email)}")


async def send_cohort_invite(
    cohort: str,
    *,
    apply: bool = False,
    limit: Optional[int] = None,
    offset: int = 0,
    emails: Optional[list[str]] = None,
    include_migrated: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    sign_in_url: Optional[str] = None,
    text_block: Optional[str] = None,
    subject: Optional[str] = None,
) -> dict[str, int]:
    """Plan the cohort, print it, and with ``apply`` send it. Returns a summary."""
    if batch_size < 1:
        raise CohortInviteError("--batch-size must be at least 1")
    if sleep_seconds < 0 or offset < 0 or (limit is not None and limit < 0):
        raise CohortInviteError("--sleep, --offset and --limit must not be negative")
    if apply:
        transport = (mailer.MAIL_TRANSPORT or mailer.TRANSPORT_OFF).strip().lower()
        if transport == mailer.TRANSPORT_OFF:
            # send_mail drops the message and returns normally when the
            # transport is off: every row would be stamped sent and nobody mailed.
            raise CohortInviteError(
                "MAIL_TRANSPORT is off: --apply would record mails that never leave"
            )

    if async_mongo_manager.database is None:
        # A dry run must not create indexes: the reader credential may not.
        await async_mongo_manager.connect(ensure_indexes=apply)

    plan = await plan_cohort(
        cohort,
        limit=limit,
        offset=offset,
        emails=emails,
        include_migrated=include_migrated,
    )
    _print_plan(plan, batch_size=batch_size, sleep_seconds=sleep_seconds)

    summary = {
        "selected": plan.selected,
        "recipients": len(plan.recipients),
        "sent": 0,
        "failed": 0,
        "claimed_elsewhere": 0,
        **{f"skipped_{reason}": plan.skipped[reason] for reason in SKIP_REASONS},
    }
    if not apply:
        print("Dry run: nothing sent, nothing written. Add --apply to send.")
        return summary

    collection = get_collection(COLLECTION)
    await collection.create_index(
        [("user_id", 1), ("cohort", 1)], name="cohort_invites_user_cohort", unique=True
    )

    batches = list(_batches(plan.recipients, batch_size))
    for number, batch in enumerate(batches, start=1):
        outcome: Counter = Counter()
        for recipient in batch:
            outcome[
                await _send_one(
                    collection,
                    recipient,
                    plan.cohort,
                    sign_in_url=sign_in_url,
                    text_block=text_block,
                    subject=subject,
                )
            ] += 1
        for key in ("sent", "failed", "claimed_elsewhere"):
            summary[key] += outcome[key]
        logger.info(
            "cohort_invite_batch cohort=%s batch=%d/%d size=%d sent=%d failed=%d",
            plan.cohort,
            number,
            len(batches),
            len(batch),
            outcome["sent"],
            outcome["failed"],
        )
        if number < len(batches) and sleep_seconds:
            await _pause(sleep_seconds)

    print(
        f"Sent: {summary['sent']}  Failed: {summary['failed']}  "
        f"Claimed by another run: {summary['claimed_elsewhere']}"
    )
    return summary


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.commands.send_cohort_invite",
        description="Send the launch mail to legacy accounts in a named cohort.",
    )
    parser.add_argument("--cohort", required=True, help="Free label, the idempotency key with the user id.")
    parser.add_argument("--apply", action="store_true", help="Send. Without it the command only reports.")
    parser.add_argument("--limit", type=int, help="Rows of users, sorted by _id, after --offset.")
    parser.add_argument("--offset", type=int, default=0, help="Rows of users to skip, sorted by _id.")
    parser.add_argument("--emails-file", help="One address per line, instead of --limit and --offset.")
    parser.add_argument("--include-migrated", action="store_true", help="Also mail users who already signed in to the new version.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Mails per batch (default 50).")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SECONDS, help="Seconds between batches (default 60).")
    parser.add_argument("--text-file", help="Plain-text block for the mail body, paragraphs split by blank lines.")
    parser.add_argument("--subject", help="Mail subject; a neutral default otherwise.")
    parser.add_argument("--sign-in-url", help="Button target; FRONTEND_URL otherwise.")
    args = parser.parse_args(argv)
    if args.emails_file and (args.limit is not None or args.offset):
        parser.error("--emails-file cannot be combined with --limit or --offset")
    return args


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    emails = read_emails_file(args.emails_file) if args.emails_file else None
    text_block = Path(args.text_file).read_text(encoding="utf-8") if args.text_file else None
    try:
        summary = asyncio.run(
            send_cohort_invite(
                args.cohort,
                apply=args.apply,
                limit=args.limit,
                offset=args.offset,
                emails=emails,
                include_migrated=args.include_migrated,
                batch_size=args.batch_size,
                sleep_seconds=args.sleep,
                sign_in_url=args.sign_in_url,
                text_block=text_block,
                subject=args.subject,
            )
        )
    except CohortInviteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
