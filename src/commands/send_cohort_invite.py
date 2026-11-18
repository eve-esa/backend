"""Send the launch mail in cohorts: to accounts, and to listed addresses without one.

The accounts that existed before the public opening get one mail telling them
the new EVE is live (``cohort_invite``). A listed address with no account, such
as a person who pre-registered, gets the create-account mail instead
(``cohort_invite_signup``). The mail goes out in cohorts so that sender
reputation, support load and sign-in traffic grow in steps.

    python -m src.commands.send_cohort_invite --cohort launch-1 --limit 50
    python -m src.commands.send_cohort_invite --cohort launch-1 --limit 50 --apply
    python -m src.commands.send_cohort_invite --cohort vip --emails-file vip.txt --apply

Dry run by default: it reads, prints the counts (one per mail type) and the
first five recipients masked, and writes nothing. A dry run works with the reader credential: it
connects without creating indexes. ``--apply`` needs the writer.

Selection. ``--limit`` and ``--offset`` slice ``users`` sorted by ``_id``, before
any skip rule, so a slice is the same set of rows on every run and consecutive
cohorts never overlap (``--offset 0 --limit 100``, then ``--offset 100 --limit
100``). ``--emails-file`` selects by address instead, one per line, blank lines
and ``#`` comment lines ignored, a UTF-8 byte order mark dropped. Every listed
address is normalised (stripped, lowercased) once, then looked up: exactly, and
when that misses once more ignoring case, so a users row stored in mixed case
is still found. A listed address with a users row gets the account mail under
the rules below, one without a row gets the create-account mail, whose button
opens ``--sign-up-url`` (``FRONTEND_URL`` otherwise: the app has no sign-up
route of its own, and the sign-in page it sends people to offers "Create an
account").

Right before each claim the users row is looked up again. Someone who created
an account after the plan gets the account mail. An account whose row is gone,
or whose address changed, since the plan is not mailed and is counted as
changed since the plan.

Skipped, each counted on its own line:
  no_email        the row has no address
  not_approved    ``approval_status`` is set and is not "approved" (None or an
                  absent field means the row was never gated, so it is eligible)
  migrated        the user already signed in to the new version (a row in
                  ``external_identities``), unless ``--include-migrated``
  duplicate_email a second row, or a repeated listed address, in this run
  invalid_email   a listed value that is not one bare address (one ``@``, a
                  local part, a domain with a dot; no space, comma, angle
                  bracket or ``#``)
  already_sent    ``cohort_invites`` holds a sent row for the cohort with this
                  user id or this address
  in_doubt        a claim without ``sent_at``: a previous run died between the
                  claim and the send, or its send ended with no answer from the
                  relay, so whether the mail left is unknown

Idempotency. ``cohort_invites`` holds one claim per person and cohort. Every
new claim stores the normalised ``email``; an account claim also stores
``user_id``, a claim for an address without an account has no ``user_id`` field
at all. Two unique partial indexes guard it: (``cohort``, ``user_id``) where
``user_id`` exists, and (``cohort``, ``email``) where ``email`` exists. Claims
written before the second mail existed carry ``user_id`` and no ``email``, and
stay covered by the first; a row skipped for such a claim also covers its
address for the rest of the run. The mail is rendered, then the claim is
inserted, then the mail is sent and the claim gets ``sent_at``: two runs of the
same cohort cannot both mail a person, an address on both lists is mailed once,
and a re-run skips everybody already served. A send refused before anything
left (connection, login, a refused sender or recipient, any definite reply
from the relay) removes its claim, so the next run retries it. A send that
ended with no answer once the message was on its way (a dropped line or a
timeout during the transaction) keeps its claim without ``sent_at``. Such a
claim, like one left by a killed process, is reported as in_doubt and never
resent automatically, not even by the ECS restart that follows a failed exit;
to resend, delete it:
``db.cohort_invites.deleteMany({cohort: "<name>", sent_at: null})``.

Quota. ``--max-sends N`` stops the run after N send attempts: the Workspace
mailbox allows 2,000 messages a day, shared by staging, prod and the sign-up
codes. The cap counts within one process: an ECS restart after a failed exit
starts again from zero, so stop the task on repeated failures.

Templates. With ``--apply`` and ``MAIL_TEMPLATES_BUCKET`` set, the command
fetches the published pack once before the first send (a one-off task has no
lifespan to refresh it) and prints which copy the mails will use; a dry run
makes no S3 call; each sent mail logs
``template_source`` and ``pack_version``. ``--subject`` still wins over the pack.

Logs carry counts, the cohort, the claim id and the user id, never an address.
A failed send logs whether its claim was released or kept in doubt, the error
class, the SMTP reply code or AWS error code, and the SMTP enhanced status code
(``status=5.4.5`` is the Gmail daily limit).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from bson import ObjectId
from pymongo.errors import DuplicateKeyError, OperationFailure

from src.config import configure_logging
from src.database.mongo import async_mongo_manager, get_collection
from src.services import mailer
from src.services.account_notifications import (
    MAIL_COHORT_INVITE,
    MAIL_COHORT_INVITE_SIGNUP,
    build_cohort_invite,
    build_cohort_invite_signup,
    mail_template_store,
)
from src.services.identity import normalize_email
from src.services.mail_templates import subject_problem
from src.services.mailer import delivery_uncertain, failure_reason, send_mail

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
    "invalid_email",
    "already_sent",
    "in_doubt",
)
CLAIM_SKIPS = ("already_sent", "in_doubt")
MAIL_TYPES = (MAIL_COHORT_INVITE, MAIL_COHORT_INVITE_SIGNUP)
MAIL_TYPE_LABELS = {
    MAIL_COHORT_INVITE: "Launch mail to an account",
    MAIL_COHORT_INVITE_SIGNUP: "Create-account mail",
}

# The claim indexes. Only $exists in the filters: Amazon DocumentDB 5.0 partial
# indexes accept $eq, $exists, a top-level $and and the range operators, not
# $type. The key order (cohort first) differs from the index they replace, so
# both can exist while the old one is dropped and uniqueness never lapses.
LEGACY_CLAIM_INDEX = "cohort_invites_user_cohort"
CLAIM_INDEXES = (
    {
        "keys": [("cohort", 1), ("user_id", 1)],
        "name": "cohort_invites_cohort_user",
        "unique": True,
        "partialFilterExpression": {"user_id": {"$exists": True}},
    },
    {
        "keys": [("cohort", 1), ("email", 1)],
        "name": "cohort_invites_cohort_email",
        "unique": True,
        "partialFilterExpression": {"email": {"$exists": True}},
    },
)
# MongoDB and DocumentDB answer a drop of a missing index with IndexNotFound.
INDEX_NOT_FOUND = 27

# Characters that never belong in a bare addr-spec as we accept it: a display
# name, a list separator, a comment, or a quoted local part.
_NOT_IN_ADDRESS = set('<>,;#"()[]\\')


class CohortInviteError(RuntimeError):
    """The command refuses to run with the arguments or configuration given."""


@dataclass
class Recipient:
    """One mail to send. No ``user_id`` means no account: the create-account mail."""

    user_id: Optional[str]
    email: str

    @property
    def mail(self) -> str:
        return MAIL_COHORT_INVITE if self.user_id else MAIL_COHORT_INVITE_SIGNUP


@dataclass
class CohortPlan:
    cohort: str
    selected: int = 0
    not_found: int = 0
    recipients: list[Recipient] = field(default_factory=list)
    skipped: Counter = field(default_factory=Counter)

    @property
    def mail_counts(self) -> Counter:
        return Counter(recipient.mail for recipient in self.recipients)


def mask_email(address: str) -> str:
    """``alice@example.org`` becomes ``a***@example.org``."""
    local, at, domain = (address or "").strip().rpartition("@")
    if not at:
        return "***"
    return f"{local[:1]}***@{domain}"


def is_bare_address(value: str) -> bool:
    """One addr-spec and nothing else: ``local@domain.tld``.

    No whitespace, control or invisible character, display name, comma or
    comment; exactly one ``@``; a non-empty local part; a domain of at least
    two non-empty labels.
    """
    if not value or any(
        ch.isspace() or not ch.isprintable() or ch in _NOT_IN_ADDRESS for ch in value
    ):
        return False
    local, at, domain = value.partition("@")
    if not at or not local or "@" in domain:
        return False
    labels = domain.split(".")
    return len(labels) >= 2 and all(labels)


def read_emails_file(path: str | Path) -> list[str]:
    """The listed values of a file, in order: no blank line, no ``#`` comment line.

    A byte order mark (spreadsheet exports) and surrounding whitespace are
    dropped. Normalising, validating and counting repeats is left to
    :func:`plan_cohort`, so the dry run reports every repeat.
    """
    values = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        value = line.replace("﻿", "").strip()
        if value and not value.startswith("#"):
            values.append(value)
    return values


def _normalize_cohort(cohort: str) -> str:
    value = (cohort or "").strip()
    if not value:
        raise CohortInviteError("--cohort must be a non-empty label")
    return value


def _address_ignoring_case(address: str) -> dict:
    """An anchored, escaped, case-insensitive match of one stored address."""
    return {"$regex": f"^\\s*{re.escape(address)}\\s*$", "$options": "i"}


async def _select_users(
    *, limit: Optional[int], offset: int, emails: Optional[list[str]]
) -> list[dict]:
    projection = {"email": 1, "approval_status": 1}
    users = get_collection("users")
    if emails is None:
        cursor = users.find({}, projection).sort("_id", 1).skip(offset)
        if limit is not None:
            cursor = cursor.limit(limit)
        return [doc async for doc in cursor]

    docs = [
        doc async for doc in users.find({"email": {"$in": emails}}, projection).sort("_id", 1)
    ]
    found = {normalize_email(doc.get("email")) for doc in docs}
    for address in emails:
        if address in found:
            continue
        # A row stored in mixed case or with spaces misses the exact lookup.
        cursor = users.find({"email": _address_ignoring_case(address)}, projection)
        docs.extend([doc async for doc in cursor])
    docs.sort(key=lambda doc: str(doc["_id"]))
    return docs


async def _find_user(address: str) -> Optional[dict]:
    """The users row of an address: exact first, then ignoring case."""
    users = get_collection("users")
    row = await users.find_one({"email": address}, {"email": 1})
    if row is None:
        row = await users.find_one(
            {"email": _address_ignoring_case(address)}, {"email": 1}, sort=[("_id", 1)]
        )
    return row


async def _rows_by(collection: str, query: dict, key: str, values: list[str]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not values:
        return rows
    cursor = get_collection(collection).find({**query, key: {"$in": values}})
    async for doc in cursor:
        rows.setdefault(doc[key], doc)
    return rows


async def _claims_by(cohort: str, key: str, values: list[str]) -> dict[str, dict]:
    """Claims of a cohort by ``user_id`` or ``email``.

    The query carries the partial filter of the key's index: DocumentDB uses a
    partial index only for a query whose filter contains the index's filter.
    """
    rows: dict[str, dict] = {}
    if not values:
        return rows
    query = {"$and": [{key: {"$exists": True}}, {"cohort": cohort, key: {"$in": values}}]}
    async for doc in get_collection(COLLECTION).find(query):
        rows.setdefault(doc[key], doc)
    return rows


def _claim_state(claim: dict) -> str:
    return "already_sent" if claim.get("sent_at") else "in_doubt"


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
    plan = CohortPlan(cohort=cohort)

    listed: Optional[list[str]] = None
    if emails is not None:
        listed = []
        seen_listed: set[str] = set()
        for raw in emails:
            address = normalize_email(raw)
            if address is None:
                plan.skipped["no_email"] += 1
            elif not is_bare_address(address):
                plan.skipped["invalid_email"] += 1
            elif address in seen_listed:
                plan.skipped["duplicate_email"] += 1
            else:
                seen_listed.add(address)
                listed.append(address)

    docs = await _select_users(limit=limit, offset=offset, emails=listed)
    plan.selected = len(docs)
    found = {normalize_email(doc.get("email")) for doc in docs}
    without_account = [a for a in (listed or []) if a not in found]
    plan.not_found = len(without_account)

    ids = [str(doc["_id"]) for doc in docs]
    addresses = sorted({a for a in found if a} | set(without_account))
    user_claims = await _claims_by(cohort, "user_id", ids)
    email_claims = await _claims_by(cohort, "email", addresses)
    migrated = await _rows_by("external_identities", {}, "user_id", ids)

    seen_emails: set[str] = set()
    for doc in docs:
        user_id = str(doc["_id"])
        email = str(doc.get("email") or "").strip()
        address = normalize_email(email)
        status = doc.get("approval_status")
        claim = user_claims.get(user_id) or email_claims.get(address)
        if not email:
            reason = "no_email"
        elif status is not None and status != "approved":
            reason = "not_approved"
        elif user_id in migrated and not include_migrated:
            reason = "migrated"
        elif address in seen_emails:
            reason = "duplicate_email"
        elif claim is not None:
            reason = _claim_state(claim)
        else:
            reason = None

        if reason in CLAIM_SKIPS:
            # The claim covers the address, whichever row it was written for:
            # an older claim has a user id and no email, so a second row with
            # the same address would not collide with it on any index.
            seen_emails.add(address)
        if reason:
            plan.skipped[reason] += 1
            continue
        seen_emails.add(address)
        plan.recipients.append(Recipient(user_id=user_id, email=email))

    for address in without_account:
        if address in email_claims:
            plan.skipped[_claim_state(email_claims[address])] += 1
        else:
            plan.recipients.append(Recipient(user_id=None, email=address))
    return plan


def _batches(items: list[Recipient], size: int) -> Iterable[list[Recipient]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


async def _pause(seconds: float) -> None:
    """Indirection so tests can count the pauses without waiting."""
    await asyncio.sleep(seconds)


async def ensure_claim_indexes(collection) -> None:
    """Create the two partial claim indexes, then drop the one they replace.

    The new indexes are built first, so a claim is never without a unique
    guard. Existing claims carry ``user_id`` and no ``email``: the user index
    covers them and the email index skips them, so neither build needs a
    backfill. The old index (``user_id``, ``cohort``) is unique on every
    document, so two claims without a user id would collide on the missing
    field: it has to go. Another run may drop it between the check and the
    drop; that is the outcome wanted, not an error.
    """
    for spec in CLAIM_INDEXES:
        await collection.create_index(
            spec["keys"],
            name=spec["name"],
            unique=spec["unique"],
            partialFilterExpression=spec["partialFilterExpression"],
        )
    if LEGACY_CLAIM_INDEX in await collection.index_information():
        try:
            await collection.drop_index(LEGACY_CLAIM_INDEX)
        except OperationFailure as exc:
            if exc.code != INDEX_NOT_FOUND:
                raise


def _row_key(user_id: str):
    return ObjectId(user_id) if ObjectId.is_valid(user_id) else user_id


async def _resolve(recipient: Recipient) -> Optional[Recipient]:
    """Look up the users row again right before the claim.

    An account recipient is mailed only if its row still exists with the same
    address; ``None`` otherwise. An address planned without an account gets
    the account mail if a row appeared since the plan. Approval and migration
    rules are not applied to such a row: the account is minutes old and the
    person asked for it.
    """
    if recipient.user_id:
        row = await get_collection("users").find_one(
            {"_id": _row_key(recipient.user_id)}, {"email": 1}
        )
        if row is None or normalize_email(row.get("email")) != normalize_email(recipient.email):
            return None
        return recipient
    row = await _find_user(recipient.email)
    if row is None:
        return recipient
    return Recipient(user_id=str(row["_id"]), email=recipient.email)


async def _send_one(
    collection,
    recipient: Recipient,
    cohort: str,
    *,
    sign_in_url: Optional[str],
    sign_up_url: Optional[str],
    text_block: Optional[str],
    subject: Optional[str],
) -> tuple[str, str]:
    """Resolve, render, claim, send, stamp.

    Returns the outcome and the mail type. Outcomes: "sent"; "failed" (claim
    released, the next run retries); "in_doubt" (no answer from the relay,
    claim kept); "claimed_elsewhere"; "changed" (the account is gone or has
    another address since the plan, nothing written).
    """
    planned = recipient
    recipient = await _resolve(recipient)
    if recipient is None:
        return "changed", planned.mail

    # Render before the claim: a template error must not leave an in_doubt
    # claim for someone who was never mailed.
    if recipient.user_id:
        mail = build_cohort_invite(
            recipient.email, sign_in_url=sign_in_url, text_block=text_block, subject=subject
        )
    else:
        mail = build_cohort_invite_signup(recipient.email, sign_up_url=sign_up_url)

    claim = {
        "cohort": cohort,
        "email": normalize_email(recipient.email),
        "claimed_at": datetime.now(timezone.utc),
        "sent_at": None,
    }
    if recipient.user_id:
        # Absent, never null, for an address without an account: the user
        # index is partial on the field existing, and null exists.
        claim["user_id"] = recipient.user_id
    try:
        claim_id = (await collection.insert_one(claim)).inserted_id
    except DuplicateKeyError:
        return "claimed_elsewhere", recipient.mail

    try:
        # At most once: an unclear SMTP send stays in doubt instead of going
        # out a second time through the SES fallback.
        await send_mail(
            to=recipient.email,
            subject=mail.subject,
            html=mail.html,
            text=mail.text,
            fallback_on_uncertain=False,
        )
    except Exception as exc:
        if delivery_uncertain(exc):
            # The relay may have the message: keep the claim, so no rerun
            # (the ECS restart included) mails this person again.
            state, outcome = "in_doubt", "in_doubt"
        else:
            # Nothing left: release the claim so the next run retries.
            await collection.delete_one({"_id": claim_id, "sent_at": None})
            state, outcome = "released", "failed"
        # No exc_info: relay errors and their tracebacks can carry the address.
        logger.error(
            "cohort_invite_failed cohort=%s claim_id=%s mail=%s claim=%s reason=%s",
            cohort,
            claim_id,
            recipient.mail,
            state,
            failure_reason(exc),
        )
        return outcome, recipient.mail
    await collection.update_one(
        {"_id": claim_id}, {"$set": {"sent_at": datetime.now(timezone.utc)}}
    )
    logger.info(
        "cohort_invite_sent cohort=%s claim_id=%s mail=%s user_id=%s "
        "template_source=%s pack_version=%s",
        cohort,
        claim_id,
        recipient.mail,
        recipient.user_id or "none",
        mail.source,
        mail.version,
    )
    return "sent", recipient.mail


def _print_plan(
    plan: CohortPlan, *, batch_size: int, sleep_seconds: float, max_sends: Optional[int]
) -> None:
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
    counts = plan.mail_counts
    for name in MAIL_TYPES:
        print(f"  {MAIL_TYPE_LABELS[name]} ({name}): {counts[name]}")
    if max_sends is not None:
        print(
            f"Max sends: {max_sends}, this run sends at most "
            f"{min(max_sends, len(plan.recipients))}"
        )
    for recipient in plan.recipients[:PREVIEW_COUNT]:
        print(f"  {mask_email(recipient.email)} {recipient.mail}")


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
    max_sends: Optional[int] = None,
    sign_in_url: Optional[str] = None,
    sign_up_url: Optional[str] = None,
    text_block: Optional[str] = None,
    subject: Optional[str] = None,
) -> dict[str, int]:
    """Plan the cohort, print it, and with ``apply`` send it. Returns a summary."""
    if batch_size < 1:
        raise CohortInviteError("--batch-size must be at least 1")
    if max_sends is not None and max_sends < 1:
        raise CohortInviteError("--max-sends must be at least 1")
    if sleep_seconds < 0 or offset < 0 or (limit is not None and limit < 0):
        raise CohortInviteError("--sleep, --offset and --limit must not be negative")
    if subject is not None:
        problem = subject_problem(subject)
        if problem is not None:
            # The same rule as a pack subject: a control character or a line
            # separator breaks the header, and every mail of the run with it.
            raise CohortInviteError(f"--subject is not a valid mail subject: {problem}")
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
    _print_plan(plan, batch_size=batch_size, sleep_seconds=sleep_seconds, max_sends=max_sends)

    counts = plan.mail_counts
    to_send = len(plan.recipients) if max_sends is None else min(max_sends, len(plan.recipients))
    summary = {
        "selected": plan.selected,
        "not_found": plan.not_found,
        "recipients": len(plan.recipients),
        **{f"recipients_{name}": counts[name] for name in MAIL_TYPES},
        "to_send": to_send,
        "sent": 0,
        **{f"sent_{name}": 0 for name in MAIL_TYPES},
        "failed": 0,
        "failed_in_doubt": 0,
        "claimed_elsewhere": 0,
        "changed_since_plan": 0,
        "left_by_max_sends": 0,
        **{f"skipped_{reason}": plan.skipped[reason] for reason in SKIP_REASONS},
    }
    if not apply:
        print("Dry run: nothing sent, nothing written. Add --apply to send.")
        return summary

    # Before the first send, never in a dry run: the pack this run renders
    # with, or the bundled copy.
    if await mail_template_store.refresh():
        for name in MAIL_TYPES:
            source = (
                f"pack {mail_template_store.pack_version}"
                if mail_template_store.carries(name)
                else "bundled"
            )
            print(f"Mail templates for {name}: {source}")
    else:
        print("Mail templates: bundled")

    collection = get_collection(COLLECTION)
    await ensure_claim_indexes(collection)

    attempts = 0
    handled = 0
    batches = list(_batches(plan.recipients, batch_size))
    for number, batch in enumerate(batches, start=1):
        outcome: Counter = Counter()
        for recipient in batch:
            if max_sends is not None and attempts >= max_sends:
                break
            result, mail = await _send_one(
                collection,
                recipient,
                plan.cohort,
                sign_in_url=sign_in_url,
                sign_up_url=sign_up_url,
                text_block=text_block,
                subject=subject,
            )
            handled += 1
            outcome[result] += 1
            if result in ("sent", "failed", "in_doubt"):
                attempts += 1
            if result == "sent":
                summary[f"sent_{mail}"] += 1
        failed = outcome["failed"] + outcome["in_doubt"]
        summary["sent"] += outcome["sent"]
        summary["failed"] += failed
        summary["failed_in_doubt"] += outcome["in_doubt"]
        summary["claimed_elsewhere"] += outcome["claimed_elsewhere"]
        summary["changed_since_plan"] += outcome["changed"]
        logger.info(
            "cohort_invite_batch cohort=%s batch=%d/%d size=%d sent=%d failed=%d in_doubt=%d",
            plan.cohort,
            number,
            len(batches),
            len(batch),
            outcome["sent"],
            failed,
            outcome["in_doubt"],
        )
        if max_sends is not None and attempts >= max_sends:
            break
        if number < len(batches) and sleep_seconds:
            await _pause(sleep_seconds)

    summary["left_by_max_sends"] = len(plan.recipients) - handled
    print(
        f"Sent: {summary['sent']} ({MAIL_COHORT_INVITE}: {summary['sent_' + MAIL_COHORT_INVITE]}, "
        f"{MAIL_COHORT_INVITE_SIGNUP}: {summary['sent_' + MAIL_COHORT_INVITE_SIGNUP]})  "
        f"Failed: {summary['failed']} (in doubt, claim kept: {summary['failed_in_doubt']})  "
        f"Claimed by another run: {summary['claimed_elsewhere']}  "
        f"Changed since the plan: {summary['changed_since_plan']}"
    )
    if summary["left_by_max_sends"]:
        print(
            f"Stopped at --max-sends {max_sends}: {summary['left_by_max_sends']} "
            "recipient(s) left for the next run"
        )
    return summary


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.commands.send_cohort_invite",
        description=(
            "Send the launch mail in a named cohort: to accounts, and the "
            "create-account mail to listed addresses without one."
        ),
    )
    parser.add_argument("--cohort", required=True, help="Free label, the idempotency key with the user id or the address.")
    parser.add_argument("--apply", action="store_true", help="Send. Without it the command only reports.")
    parser.add_argument("--limit", type=int, help="Rows of users, sorted by _id, after --offset.")
    parser.add_argument("--offset", type=int, default=0, help="Rows of users to skip, sorted by _id.")
    parser.add_argument("--emails-file", help="One address per line, instead of --limit and --offset.")
    parser.add_argument("--include-migrated", action="store_true", help="Also mail users who already signed in to the new version.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Mails per batch (default 50).")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SECONDS, help="Seconds between batches (default 60).")
    parser.add_argument(
        "--max-sends",
        type=int,
        help="Stop after this many send attempts in this run (default no cap); the mailbox allows 2,000 a day.",
    )
    parser.add_argument("--text-file", help="Plain-text block for the mail body, paragraphs split by blank lines.")
    parser.add_argument("--subject", help="Subject of the account mail; a neutral default otherwise.")
    parser.add_argument("--sign-in-url", help="Button target of the account mail; FRONTEND_URL otherwise.")
    parser.add_argument(
        "--sign-up-url",
        help="Button target of the create-account mail; FRONTEND_URL otherwise (its sign-in page offers Create an account).",
    )
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
                max_sends=args.max_sends,
                sign_in_url=args.sign_in_url,
                sign_up_url=args.sign_up_url,
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
