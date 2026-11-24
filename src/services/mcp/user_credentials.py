"""A user's own credentials for a toolkit: parse the spec, check, encrypt, store, read.

This is the only module that ever holds the plaintext values. They are envelope
encrypted into the DocumentDB row with the ``byok`` key, bound to the purpose, the
user and the toolkit. One KMS key serves every user at a flat monthly cost, while
one Secrets Manager secret per user and toolkit would bill per secret and add an
API call per turn. The context makes a blob useless on any other row or for any
other use, so isolation does not depend on a separate secret per user.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional

import httpx
from pydantic import ValidationError

from src import config
from src.database.models.mcp_server import MCPServer, UserCredentialSpec
from src.database.models.mcp_server_credential import MCPServerCredential
from src.services.custom_model_cipher import CipherContextError, SecretCipher

logger = logging.getLogger(__name__)

PURPOSE = "mcp_server_credential"
MAX_VALUE_LENGTH = 1024
CHECK_TIMEOUT_S = 10
_HINT_MIN_LENGTH = 8
_HINT_PREFIX = "…"

_cipher = SecretCipher(context_keys=("purpose", "user_id", "server_id"))

# Server names already warned about, so a broken operator spec logs once per
# process instead of once per turn.
_warned_invalid_specs: set[str] = set()


class CredentialsError(Exception):
    """Base class. Messages never contain a submitted or stored value."""


class CredentialsInvalid(CredentialsError):
    """A submitted value fails the format rules, before any network call."""

    def __init__(self, field: Optional[str], message: str) -> None:
        super().__init__(f"{field}: {message}" if field else message)
        self.field = field
        self.message = message


class CredentialsRejected(CredentialsError):
    """The provider answered that these credentials are wrong."""


class CredentialsCheckUnavailable(CredentialsError):
    """The check could not tell: the provider failed, timed out or answered oddly."""


class CredentialsUnreadable(CredentialsError):
    """A stored row no longer decrypts or parses."""


def effective_spec(server: MCPServer | Mapping[str, Any]) -> Optional[UserCredentialSpec]:
    """The credential spec users may fill for ``server``, or None.

    None when the feature is off, the row belongs to a user (only operator rows
    may ask for credentials), the row declares nothing, or the declaration is
    malformed. A malformed one must not break the toolkit for everyone, so it is
    ignored with one warning.
    """
    if not config.FEATURE_TOOLKIT_CREDENTIALS:
        return None
    if isinstance(server, MCPServer):
        owner, raw, name = server.user_id, server.user_credentials, server.name
    else:
        owner, raw, name = server.get("user_id"), server.get("user_credentials"), server.get("name")
    if owner or raw is None:
        return None
    if isinstance(raw, UserCredentialSpec):
        return raw
    try:
        return UserCredentialSpec.model_validate(raw)
    except ValidationError as exc:
        key = str(name)
        if key not in _warned_invalid_specs:
            _warned_invalid_specs.add(key)
            # Error types only: the messages and inputs echo operator data.
            types = sorted({str(err.get("type")) for err in exc.errors()})
            logger.warning(
                "mcp.user_credentials.invalid_spec server=%s errors=%s", key, ",".join(types)
            )
        return None


def _is_printable_ascii(value: str) -> bool:
    # The values become HTTP header values: CR or LF would split the header and
    # anything outside printable ASCII is not portable across HTTP stacks.
    return all(0x20 <= ord(ch) <= 0x7E for ch in value)


def validate_values(spec: UserCredentialSpec, values: Any) -> dict[str, str]:
    """Exactly the declared fields, stripped and checked. Raises CredentialsInvalid."""
    if not isinstance(values, Mapping):
        raise CredentialsInvalid(None, "values must be an object")
    declared = [f.name for f in spec.fields]
    for key in values:
        if key not in declared:
            # The key is user input too: name it only when it looks like a field name.
            shown = key if isinstance(key, str) and key.isidentifier() and len(key) <= 32 else None
            raise CredentialsInvalid(shown, "is not a field of this toolkit")
    clean: dict[str, str] = {}
    for name in declared:
        if name not in values:
            raise CredentialsInvalid(name, "is required")
        value = values[name]
        if not isinstance(value, str):
            raise CredentialsInvalid(name, "must be a string")
        value = value.strip()
        if not value:
            raise CredentialsInvalid(name, "must not be empty")
        if len(value) > MAX_VALUE_LENGTH:
            raise CredentialsInvalid(name, f"must be at most {MAX_VALUE_LENGTH} characters")
        if not _is_printable_ascii(value):
            raise CredentialsInvalid(
                name, "may only contain printable ASCII characters, no line breaks or tabs"
            )
        clean[name] = value
    return clean


async def verify_values(spec: UserCredentialSpec, values: Mapping[str, str]) -> bool:
    """Run the declared check on validated values. True when a check ran and passed.

    Raises CredentialsRejected when the provider says no and
    CredentialsCheckUnavailable when the answer proves nothing either way.
    """
    check = spec.check
    if check is None:
        return False
    form = {
        "grant_type": "client_credentials",
        "client_id": values[check.client_id_field],
        "client_secret": values[check.client_secret_field],
    }
    try:
        # No redirects: a 3xx could carry the form to a host the operator never named.
        async with httpx.AsyncClient(timeout=CHECK_TIMEOUT_S, follow_redirects=False) as client:
            response = await client.post(
                check.token_url, data=form, headers={"Accept": "application/json"}
            )
    except httpx.HTTPError as exc:
        logger.warning("mcp.user_credentials.check_failed error=%s", type(exc).__name__)
        # from None: the httpx exception holds the request, and the request holds the form.
        raise CredentialsCheckUnavailable("the credential check could not reach the provider") from None

    status = response.status_code
    if status in (400, 401):
        logger.info("mcp.user_credentials.check_rejected status=%s", status)
        raise CredentialsRejected("the provider rejected these credentials")
    if status == 200:
        try:
            body = response.json()
        except ValueError:
            body = None
        token = body.get("access_token") if isinstance(body, dict) else None
        if isinstance(token, str) and token.strip():
            return True
    # Never log the body: a provider may echo what it received.
    logger.warning("mcp.user_credentials.check_unavailable status=%s", status)
    raise CredentialsCheckUnavailable("the credential check gave no usable answer")


def _context(user_id: str, server_id: str) -> dict[str, str]:
    return {"purpose": PURPOSE, "user_id": user_id, "server_id": server_id}


def _hint(value: str) -> str:
    return _HINT_PREFIX + value[-4:] if len(value) > _HINT_MIN_LENGTH else _HINT_PREFIX


async def save(
    *, user_id: str, server_id: str, spec: UserCredentialSpec, values: Any
) -> MCPServerCredential:
    """Validate, check, encrypt and upsert the user's values for one toolkit."""
    clean = validate_values(spec, values)
    checked = await verify_values(spec, clean)
    # KMS is a blocking boto3 call, keep it off the event loop.
    blob = await asyncio.to_thread(
        _cipher.encrypt, json.dumps(clean, sort_keys=True), _context(user_id, server_id)
    )
    hints = {f.name: _hint(clean[f.name]) for f in spec.fields if not f.secret}
    now = datetime.now(timezone.utc)
    key = {"user_id": user_id, "server_id": server_id}
    # Update operators only: DocumentDB rejects pipeline updates.
    await MCPServerCredential.get_collection().update_one(
        key,
        {
            "$set": {
                "encrypted_values": blob,
                "hints": hints,
                "verified_at": now if checked else None,
                "updated_at": now,
            },
            "$setOnInsert": {"created_at": now, "timestamp": now},
        },
        upsert=True,
    )
    # Primary: on a replica set a secondary may not have the write yet.
    row = await MCPServerCredential.get_primary_collection().find_one(key)
    return MCPServerCredential.from_dict(row)


async def delete(*, user_id: str, server_id: str) -> bool:
    """Remove the user's values for one toolkit. True when a row was removed."""
    result = await MCPServerCredential.get_collection().delete_one(
        {"user_id": user_id, "server_id": server_id}
    )
    return result.deleted_count > 0


async def status_for(
    *, user_id: str, server_ids: Iterable[str]
) -> dict[str, MCPServerCredential]:
    """The user's stored rows for these toolkits, keyed by server id, in one query."""
    ids = list(dict.fromkeys(server_ids))
    if not ids:
        return {}
    cursor = MCPServerCredential.get_collection().find(
        {"user_id": user_id, "server_id": {"$in": ids}}
    )
    rows = await cursor.to_list(length=len(ids))
    return {row["server_id"]: MCPServerCredential.from_dict(row) for row in rows}


async def headers_for(
    *, user_id: str, server_id: str, spec: UserCredentialSpec
) -> Optional[dict[str, str]]:
    """Upstream headers built from the user's stored values, or None.

    Decrypted per call on purpose: no plaintext cache outlives the request.
    """
    row = await MCPServerCredential.get_collection().find_one(
        {"user_id": user_id, "server_id": server_id}
    )
    if row is None:
        return None
    try:
        plaintext = await asyncio.to_thread(
            _cipher.decrypt, row.get("encrypted_values") or "", _context(user_id, server_id)
        )
        values = json.loads(plaintext)
        if not isinstance(values, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in values.items()
        ):
            raise ValueError("stored values are not a string map")
    except (CipherContextError, ValueError) as exc:
        logger.error(
            "mcp.user_credentials.unreadable user_id=%s server_id=%s error=%s",
            user_id,
            server_id,
            type(exc).__name__,
        )
        raise CredentialsUnreadable("stored credentials could not be read") from None
    if any(f.name not in values for f in spec.fields):
        # The operator added a field after the user saved: ask again rather than
        # call the toolkit with a partial set.
        logger.warning("mcp.user_credentials.stale server_id=%s", server_id)
        return None
    return {f.header: values[f.name] for f in spec.fields}
