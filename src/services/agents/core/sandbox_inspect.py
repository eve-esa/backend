"""Read-only views of a conversation's sandbox and of its saved files.

Nothing here creates a sandbox, takes the pool lock, or moves the idle
deadline. ``get_sandbox`` reads the pool map only. ``list_sandbox_files``
attaches to the live session, runs one ``find``, and detaches; it never
calls ``destroy`` or ``renew``.
"""

from __future__ import annotations

import asyncio
import logging
import shlex
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi.concurrency import run_in_threadpool

from src.schemas.sandbox import FileEntry, FileListing, SandboxRead
from src.services.agents.core.sandbox import (
    SandboxGone,
    attach_session,
    sandbox_provider,
)
from src.services.agents.core.sandbox_pool import get_store
from src.services.agents.core.workspace import _command
from src.services.storage import storage_service

logger = logging.getLogger(__name__)

# The session root holds these four trees; see docs/sandbox.md, "Workspace".
SANDBOX_ROOTS = ("workspace", "skills", "artifacts", "large_tool_results")


class SandboxNotFound(Exception):
    """The conversation has no sandbox record."""


class InvalidPath(ValueError):
    """A path or prefix that is absolute, empty where a file is needed, or climbs."""


def conversation_prefix(user_id: str, conversation_id: str) -> str:
    return f"users/{user_id}/conversations/{conversation_id}/"


def relative_path(raw: str, *, allow_prefix: bool) -> str:
    """Clean a client path. ``allow_prefix`` accepts ``""`` and a trailing ``/``."""
    value = (raw or "").replace("\\", "/").lstrip("/")
    if not value:
        if allow_prefix:
            return ""
        raise InvalidPath(raw)
    body = value[:-1] if allow_prefix and value.endswith("/") else value
    if not body or any(part in {"", ".", ".."} for part in body.split("/")):
        raise InvalidPath(raw)
    return value


def _epoch(value: Any) -> Optional[datetime]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return datetime.fromtimestamp(number, tz=timezone.utc)


async def get_sandbox(conversation_id: str) -> SandboxRead:
    if sandbox_provider() in ("", "none"):
        return SandboxRead(status="disabled")
    store = get_store()
    record = await store.get(conversation_id)
    if record is None:
        return SandboxRead(status="none")
    deadline = await store.deadline(conversation_id)
    try:
        inflight = max(int(record.get("inflight") or "0"), 0)
    except ValueError:
        inflight = 0
    created = _epoch(record.get("created_at"))
    expires = None
    try:
        lifetime = float(record.get("max_lifetime_s") or 0)
    except ValueError:
        lifetime = 0
    if created is not None and lifetime > 0:
        expires = _epoch(created.timestamp() + lifetime)
    return SandboxRead(
        status="busy" if inflight > 0 else "idle",
        provider=record.get("provider") or None,
        sandbox_id=record.get("sandbox_id") or None,
        inflight=inflight,
        created_at=created,
        expires_at=expires,
        idle_deadline=_epoch(deadline),
        skills_digest=record.get("skills_digest") or None,
    )


def _find_command(prefix: str, limit: int) -> str:
    """List files under the session root whose path starts with *prefix*.

    The raw shell may start in the session root or inside ``workspace/``, so
    it moves to the directory that holds ``workspace`` first, as
    ``workspace._FIND`` does.
    """
    directory = prefix.rsplit("/", 1)[0] if "/" in prefix else ""
    roots = [directory] if directory else list(SANDBOX_ROOTS)
    quoted = " ".join(shlex.quote(root) for root in roots)
    # awk's index() with an empty needle is 0, so no prefix means no filter.
    match = (
        f"| awk -v p={shlex.quote(prefix)} 'index($0, p) == 1' " if prefix else ""
    )
    return (
        "if [ ! -d workspace ] && [ -d ../workspace ]; then cd ..; fi; "
        f"for d in {quoted}; do "
        "[ -d \"$d\" ] && find -H \"$d\" -type f -printf '%p\\t%s\\t%T@\\n'; "
        "done 2>/dev/null "
        f"{match}"
        f"| head -n {int(limit) + 1}"
    )


def _parse_listing(output: str, limit: int) -> FileListing:
    items: list[FileEntry] = []
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        path, raw_size, mtime = parts
        try:
            size = int(raw_size)
        except ValueError:
            continue
        items.append(FileEntry(path=path, size=size, modified_at=_epoch(mtime.strip())))
    truncated = len(items) > limit
    items = sorted(items[:limit], key=lambda item: item.path)
    return FileListing(items=items, truncated=truncated)


def _list_live(record: dict[str, str], prefix: str, limit: int) -> FileListing:
    session = attach_session(record)
    try:
        output, ok = _command(session.backend, _find_command(prefix, limit))
    finally:
        try:
            session.detach()
        except Exception:
            logger.exception(
                "sandbox.inspect_detach_failed sandbox_id=%s",
                record.get("sandbox_id"),
            )
    if not ok:
        raise RuntimeError("find failed in the sandbox")
    return _parse_listing(output, limit)


async def list_sandbox_files(
    conversation_id: str, prefix: str, limit: int
) -> FileListing:
    """Files in the live sandbox, relative to the session root.

    Raises:
        SandboxNotFound: no record for this conversation.
        SandboxGone: the provider reports the sandbox ended.
    """
    record = await get_store().get(conversation_id)
    if record is None:
        raise SandboxNotFound(conversation_id)
    return await asyncio.to_thread(_list_live, record, prefix, limit)


async def list_conversation_files(
    user_id: str, conversation_id: str, prefix: str, limit: int
) -> FileListing:
    """Saved objects under the conversation prefix, relative to it."""
    base = conversation_prefix(user_id, conversation_id)
    records, truncated = await run_in_threadpool(
        storage_service.list_key_records, base + prefix, limit
    )
    items = []
    for record in records:
        key = record.get("key") or ""
        if not key.startswith(base):
            continue
        modified = record.get("last_modified")
        items.append(
            FileEntry(
                path=key[len(base) :],
                size=int(record.get("size") or 0),
                modified_at=modified if isinstance(modified, datetime) else None,
            )
        )
    items.sort(key=lambda item: item.path)
    return FileListing(items=items, truncated=truncated)


__all__ = [
    "InvalidPath",
    "SANDBOX_ROOTS",
    "SandboxGone",
    "SandboxNotFound",
    "conversation_prefix",
    "get_sandbox",
    "list_conversation_files",
    "list_sandbox_files",
    "relative_path",
]
