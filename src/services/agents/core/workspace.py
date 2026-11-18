"""Conversation copy of the sandbox working directory.

S3 prefix ``users/{user_id}/conversations/{conversation_id}/workspace/`` is
the working directory, minus ``skills/``, ``large_tool_results/``, and
``artifacts/``. Those trees have their own stores. Paths outside the
session working directory are not listed.

A new sandbox copies the prefix in before the turn. After a turn, a sync
waits until no process is running beyond the baseline taken at sandbox
start and two listings match, then writes the change. A kill saves whatever
is there and does not wait. The sidecar ``workspace_state`` remembers size,
mtime, and sha256 so an unchanged file is not downloaded again. It sits
beside the prefix, so restore does not copy it back into the sandbox.

AgentCore ``executeCommand`` blocks until the foreground command exits.
https://github.com/aws/bedrock-agentcore-sdk-python/blob/main/bedrock_agentcore/tools/code_interpreter_client.py
A command the model backgrounds can keep writing after that return, which
is why the save waits.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
import urllib.parse
from typing import Any, Awaitable, Callable, Optional

from fastapi.concurrency import run_in_threadpool

from src.services.agents.core.sandbox import (
    after_turn,
    sandbox_idle_ttl_s,
    sandbox_reuse_limit_s,
)
from src.services.agents.core.tool_results import (
    RESTORE_MAX_BYTES,
    RESTORE_MAX_OBJECTS,
)
from src.services.storage import ObjectNotFoundError, storage_service

logger = logging.getLogger(__name__)

# These names are sibling directories of workspace, not files inside it.
_EXCLUDED = ("skills", "large_tool_results", "artifacts")
_BACKOFF_S = (2, 5, 15, 30)
_SCAN_CAP = 500
_PS = "ps -eo pid=,args="
# List only the workspace directory. skills, large_tool_results, and
# artifacts are siblings of it, so they are not in this tree. The
# shell may already be inside workspace, so fall back to ``.``.
_FIND = (
    "if [ -d workspace ]; then root=workspace; else root=.; fi; "
    "find \"$root\" -type f -printf '%P\\t%s\\t%T@\\n'"
)
_PS_LINE = re.compile(r"^(\d+)\s*(.*)$")
_STATE_NAME = "workspace_state"
# What the AgentCore integration returns, with exit code 1, for a command
# sent to a session that has ended.
_SESSION_EXPIRED = "AgentCore session has expired"

Alive = Callable[[], Awaitable[bool]]


class _SandboxClosed(Exception):
    """The sandbox ended under the save. Nothing more is written."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


async def _check_alive(alive: Optional[Alive]) -> None:
    if alive is not None and not await alive():
        raise _SandboxClosed("sandbox_discarded")


def _raise_if_closed(output: str, ok: bool) -> None:
    if not ok and _SESSION_EXPIRED in output:
        raise _SandboxClosed("session_expired")


def workspace_prefix(user_id: str, conversation_id: str) -> str:
    return f"users/{user_id}/conversations/{conversation_id}/workspace/"


def workspace_state_key(user_id: str, conversation_id: str) -> str:
    return (
        f"users/{user_id}/conversations/{conversation_id}/{_STATE_NAME}"
    )


def _content_type(data: bytes, relative: str) -> str:
    """Stored type of a workspace file: the bytes decide, never the name alone."""
    from src.services.mcp.artifact_ingestion import detect_content_type

    return detect_content_type(data, relative.rsplit("/", 1)[-1])


def capture_process_baseline(backend: Any) -> str:
    """Pids already running, space-separated. Empty when ``ps`` cannot be read.

    An empty baseline disables the process check. Two matching listings are
    then enough, so a sandbox without ``ps`` still saves.
    """
    execute = getattr(backend, "_eve_raw_execute", None) or getattr(
        backend, "execute", None
    )
    if execute is None:
        return ""
    try:
        output, ok = _command(backend, _PS)
    except Exception:
        logger.warning("workspace.baseline_failed", exc_info=True)
        return ""
    if not ok:
        return ""
    pids = _pids(output, drop_probes=True)
    return " ".join(sorted(pids, key=int))


async def restore_workspace(
    sandbox_backend: Any, user_id: str, conversation_id: str
) -> None:
    """Copy this conversation's ``workspace`` objects into a new sandbox.

    A failure is logged and swallowed. The turn still runs.
    """
    try:
        await _restore(sandbox_backend, user_id, conversation_id)
    except Exception:
        logger.warning(
            "workspace.restore_failed conversation_id=%s",
            conversation_id,
            exc_info=True,
        )


async def save_workspace(
    sandbox_backend: Any,
    user_id: str,
    conversation_id: str,
    *,
    alive: Optional[Alive] = None,
) -> None:
    """Write the working directory to S3 once. A failure is logged.

    *alive* is asked before every write. False stops the save: the sandbox
    was discarded, and a conversation delete erases the prefix next.
    """
    try:
        await _save(sandbox_backend, user_id, conversation_id, alive=alive)
    except _SandboxClosed as closed:
        logger.info(
            "workspace.save_skipped conversation_id=%s reason=%s",
            conversation_id,
            closed.reason,
        )
    except Exception:
        logger.warning(
            "workspace.save_failed conversation_id=%s",
            conversation_id,
            exc_info=True,
        )


async def settle_workspace(
    sandbox_backend: Any,
    user_id: str,
    conversation_id: str,
    *,
    baseline: str = "",
    extend: Optional[Callable[[], Awaitable[bool]]] = None,
    max_wait_s: Optional[float] = None,
    alive: Optional[Alive] = None,
) -> None:
    """Save once the tree is quiet, or when *max_wait_s* runs out.

    Quiet means no pid outside *baseline* (probe commands do not count) and
    two listings with the same path, size, and mtime. The default wait is
    the sandbox idle TTL. A sandbox that is gone (*alive* false, or the
    session expired) ends the wait without a save, logged at INFO.
    """
    try:
        await _settle(
            sandbox_backend,
            user_id,
            conversation_id,
            baseline=baseline,
            extend=extend,
            max_wait_s=max_wait_s,
            alive=alive,
        )
    except _SandboxClosed as closed:
        logger.info(
            "workspace.sync_skipped conversation_id=%s reason=%s",
            conversation_id,
            closed.reason,
        )


async def _settle(
    sandbox_backend: Any,
    user_id: str,
    conversation_id: str,
    *,
    baseline: str,
    extend: Optional[Callable[[], Awaitable[bool]]],
    max_wait_s: Optional[float],
    alive: Optional[Alive],
) -> None:
    async def save() -> None:
        await save_workspace(
            sandbox_backend, user_id, conversation_id, alive=alive
        )

    started = time.monotonic()
    limit = float(sandbox_idle_ttl_s() if max_wait_s is None else max_wait_s)
    previous: Optional[tuple] = None
    step = 0
    while True:
        await _check_alive(alive)
        running = await _jobs_running(sandbox_backend, baseline)
        signature = await _listing_signature(sandbox_backend)
        # A sandbox that cannot list will not start listing later. Save once
        # (that save also sees the failure and does not delete) and stop.
        if not running and signature is None:
            await save()
            return
        if (
            not running
            and previous is not None
            and signature is not None
            and previous == signature
        ):
            await save()
            return
        previous = None if running or signature is None else signature
        if time.monotonic() - started >= limit:
            await save()
            return
        if extend is not None and not await extend():
            await save()
            return
        delay = _BACKOFF_S[min(step, len(_BACKOFF_S) - 1)]
        step += 1
        await asyncio.sleep(delay)


async def sync_workspace(
    session: Any, user_id: str, conversation_id: str
) -> None:
    """Settle, save, then detach. The answer does not wait for this.

    A conversation delete can discard the sandbox while this runs. The save
    then stops at the next step and logs at INFO, not ERROR.
    """
    try:
        with after_turn():
            await settle_workspace(
                session.backend,
                user_id,
                conversation_id,
                baseline=str(session.metadata.get("process_baseline") or ""),
                extend=lambda: _extend_deadline(session, conversation_id),
                alive=lambda: _sandbox_current(session, conversation_id),
            )
    except Exception:
        logger.warning(
            "workspace.sync_failed conversation_id=%s",
            conversation_id,
            exc_info=True,
        )
    finally:
        try:
            await asyncio.to_thread(session.detach)
        except Exception:
            logger.exception(
                "workspace.detach_failed conversation_id=%s", conversation_id
            )


async def _restore(
    sandbox_backend: Any, user_id: str, conversation_id: str
) -> None:
    if sandbox_backend is None or not user_id or not conversation_id:
        return
    upload = getattr(sandbox_backend, "upload_files", None)
    if upload is None:
        logger.warning("workspace.restore_skipped reason=no_upload")
        return
    prefix = workspace_prefix(user_id, conversation_id)
    records, truncated = await run_in_threadpool(
        storage_service.list_key_records, prefix, RESTORE_MAX_OBJECTS
    )
    if truncated:
        logger.info(
            "workspace.restore_truncated conversation_id=%s", conversation_id
        )
    files: list[tuple[str, bytes]] = []
    total = 0
    for record in records:
        size = int(record.get("size") or 0)
        if size > RESTORE_MAX_BYTES or total + size > RESTORE_MAX_BYTES:
            logger.info(
                "workspace.restore_skipped_size key=%s size=%s",
                record.get("key"),
                size,
            )
            continue
        key = record.get("key") or ""
        relative = key[len(prefix) :] if key.startswith(prefix) else ""
        if _rejected(relative):
            continue
        try:
            response = await storage_service.get_object(key)
            body = await run_in_threadpool(_read_body, response)
        except ObjectNotFoundError:
            continue
        except Exception:
            logger.warning(
                "workspace.restore_read_failed key=%s", key, exc_info=True
            )
            continue
        total += len(body)
        files.append((f"/workspace/{relative}", body))
    if not files:
        return
    _log_upload_errors(await run_in_threadpool(upload, files))


async def _save(
    sandbox_backend: Any,
    user_id: str,
    conversation_id: str,
    *,
    alive: Optional[Alive] = None,
) -> None:
    if sandbox_backend is None or not user_id or not conversation_id:
        return
    await _check_alive(alive)
    listed, truncated = await _list_files(sandbox_backend)
    if listed is None:
        return
    prefix = workspace_prefix(user_id, conversation_id)
    previous = await _load_state(user_id, conversation_id)
    kept: dict[str, dict[str, Any]] = {}
    total = 0
    for relative, size, mtime in listed:
        prior = previous.get(relative)
        if (
            prior is not None
            and int(prior.get("size") or -1) == size
            and str(prior.get("mtime") or "") == mtime
        ):
            kept[relative] = prior
            total += size
            continue
        if size > RESTORE_MAX_BYTES or total + size > RESTORE_MAX_BYTES:
            logger.info(
                "workspace.save_skipped_size path=%s size=%s", relative, size
            )
            if prior is not None:
                kept[relative] = prior
            continue
        data = await _download(sandbox_backend, f"workspace/{relative}")
        if data is None:
            if prior is not None:
                kept[relative] = prior
            continue
        digest = hashlib.sha256(data).hexdigest()
        if prior is None or prior.get("sha256") != digest:
            await _check_alive(alive)
            await storage_service.put_object(
                f"{prefix}{relative}", data, _content_type(data, relative)
            )
        kept[relative] = {"size": size, "mtime": mtime, "sha256": digest}
        total += size
    await _check_alive(alive)
    if not truncated:
        await _delete_missing(prefix, set(kept))
        state = kept
    else:
        logger.info(
            "workspace.save_truncated conversation_id=%s", conversation_id
        )
        state = dict(previous)
        state.update(kept)
    await storage_service.put_object(
        workspace_state_key(user_id, conversation_id),
        json.dumps({"files": state}).encode("utf-8"),
        "application/json",
    )


async def _delete_missing(prefix: str, kept: set[str]) -> None:
    records, truncated = await run_in_threadpool(
        storage_service.list_key_records, prefix, 1000
    )
    if truncated:
        logger.info("workspace.delete_skipped reason=truncated_list")
        return
    for record in records:
        key = record.get("key") or ""
        relative = key[len(prefix) :] if key.startswith(prefix) else ""
        if not relative or relative in kept or _rejected(relative):
            continue
        try:
            await storage_service.delete_object(key)
        except Exception:
            logger.warning(
                "workspace.delete_failed key=%s", key, exc_info=True
            )


async def _jobs_running(backend: Any, baseline: str) -> bool:
    known = {pid for pid in baseline.split() if pid}
    if not known:
        return False
    try:
        output, ok = await asyncio.to_thread(_command, backend, _PS)
    except Exception:
        logger.warning("workspace.ps_failed", exc_info=True)
        return False
    _raise_if_closed(output, ok)
    if not ok:
        return False
    return bool(_pids(output, drop_probes=True) - known)


async def _listing_signature(backend: Any) -> Optional[tuple]:
    listed, _truncated = await _list_files(backend)
    if listed is None:
        return None
    return tuple(listed)


async def _list_files(
    backend: Any,
) -> tuple[Optional[list[tuple[str, int, str]]], bool]:
    """``(rows, truncated)``. Rows is ``None`` when ``find`` failed.

    A failed listing must not be treated as an empty tree: the caller then
    skips deletes.
    """
    try:
        output, ok = await asyncio.to_thread(_command, backend, _FIND)
    except Exception:
        logger.warning("workspace.list_failed", exc_info=True)
        return None, True
    _raise_if_closed(output, ok)
    if not ok:
        logger.warning("workspace.list_failed")
        return None, True
    rows: list[tuple[str, int, str]] = []
    overflow = False
    for line in output.splitlines():
        parsed = _parse_find_line(line)
        if parsed is None:
            continue
        rows.append(parsed)
        if len(rows) > _SCAN_CAP:
            overflow = True
            break
    truncated = overflow or len(rows) > RESTORE_MAX_OBJECTS
    if len(rows) > RESTORE_MAX_OBJECTS:
        rows.sort(key=_mtime_key, reverse=True)
        rows = rows[:RESTORE_MAX_OBJECTS]
    rows.sort(key=lambda item: item[0])
    return rows, truncated


def _mtime_key(item: tuple[str, int, str]) -> float:
    try:
        return float(item[2])
    except ValueError:
        return 0.0


def _parse_find_line(line: str) -> Optional[tuple[str, int, str]]:
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 3:
        return None
    relative, raw_size, mtime = parts
    if _rejected(relative):
        return None
    try:
        size = int(raw_size)
    except ValueError:
        return None
    if size < 0:
        return None
    return relative, size, mtime.strip()


def _rejected(relative: str) -> bool:
    if not relative or relative.startswith("/") or relative.startswith("./"):
        return True
    parts = relative.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return True
    return parts[0] in _EXCLUDED


def _command(backend: Any, command: str) -> tuple[str, bool]:
    execute = getattr(backend, "_eve_raw_execute", None)
    if execute is None:
        execute = getattr(backend, "execute", None)
    if execute is None:
        return "", False
    result = execute(command)
    output = getattr(result, "output", None)
    if output is None and isinstance(result, dict):
        output = result.get("output", "")
    exit_code = getattr(result, "exit_code", None)
    if exit_code is None and isinstance(result, dict):
        exit_code = result.get("exit_code")
    ok = exit_code in (0, None)
    return str(output or ""), ok


def _pids(output: str, *, drop_probes: bool) -> set[str]:
    found: set[str] = set()
    for line in output.splitlines():
        match = _PS_LINE.match(line.strip())
        if match is None:
            continue
        pid, args = match.group(1), match.group(2)
        if drop_probes and _is_probe(args):
            continue
        found.add(pid)
    return found


def _is_probe(args: str) -> bool:
    return _PS in args or "-printf" in args


def _read_candidates(path: str) -> list[str]:
    """Paths that may hold a file named in markdown.

    A file written before ``workspace/`` existed is still at the session root.
    """
    name = path.rsplit("/", 1)[-1]
    ordered = [path]
    if path.startswith("workspace/"):
        ordered.append(path[len("workspace/") :])
    elif name:
        ordered.append(f"workspace/{name}")
    if name and name not in ordered:
        ordered.append(name)
    return [item for item in ordered if item]


async def _download_first(backend: Any, paths: list[str]) -> Optional[bytes]:
    raw = getattr(backend, "_eve_download_files", None)
    for path in paths:
        data = await _download(backend, path)
        if data:
            return data
        if raw is None or path.startswith("workspace/"):
            continue
        data = await _download_raw(raw, path)
        if data:
            return data
    return None


async def _download_raw(download: Any, path: str) -> Optional[bytes]:
    try:
        responses = await run_in_threadpool(download, [path])
    except Exception:
        logger.info("workspace.download_skipped path=%s error=read_failed", path)
        return None
    if not responses:
        return None
    response = responses[0]
    if isinstance(response, dict):
        error = response.get("error")
        data = response.get("content")
    else:
        error = getattr(response, "error", None)
        data = getattr(response, "content", None)
    if error or data is None:
        logger.info("workspace.download_skipped path=%s error=%s", path, error)
        return None
    if isinstance(data, str):
        return data.encode("utf-8")
    return bytes(data)


async def _download(backend: Any, relative: str) -> Optional[bytes]:
    responses = await _download_many(backend, [relative])
    if not responses:
        return None
    response = responses[0]
    if isinstance(response, dict):
        error = response.get("error")
        data = response.get("content")
    else:
        error = getattr(response, "error", None)
        data = getattr(response, "content", None)
    if error or data is None:
        logger.info(
            "workspace.download_skipped path=%s error=%s", relative, error
        )
        return None
    if isinstance(data, str):
        return data.encode("utf-8")
    return bytes(data)


async def _download_many(backend: Any, paths: list[str]) -> Any:
    download = getattr(backend, "adownload_files", None)
    if download is not None:
        return await download(paths)
    sync_download = getattr(backend, "download_files", None)
    if sync_download is None:
        return []
    return await run_in_threadpool(sync_download, paths)


async def _load_state(user_id: str, conversation_id: str) -> dict[str, dict]:
    try:
        response = await storage_service.get_object(
            workspace_state_key(user_id, conversation_id)
        )
        raw = await run_in_threadpool(_read_body, response)
    except ObjectNotFoundError:
        return {}
    except Exception:
        logger.warning(
            "workspace.state_read_failed conversation_id=%s",
            conversation_id,
            exc_info=True,
        )
        return {}
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(files, dict):
        return {}
    return {key: value for key, value in files.items() if isinstance(value, dict)}


async def _extend_deadline(session: Any, conversation_id: str) -> bool:
    """Push the idle deadline forward, not past the AgentCore reuse margin.

    A deadline that is already later is left alone, so a turn that started
    while this sync is still running keeps its longer window. False means
    the hard cap is too close and the caller should save now.
    """
    from src.services.agents.core.sandbox_pool import get_store

    store = get_store()
    now = time.time()
    when = now + float(sandbox_idle_ttl_s())
    record = await store.get(conversation_id)
    if record is None:
        return False
    created = record.get("created_at")
    lifetime = record.get("max_lifetime_s")
    if created and lifetime:
        try:
            limit = float(created) + float(lifetime) - sandbox_reuse_limit_s()
            when = min(when, limit)
        except ValueError:
            pass
    if when <= now:
        return False
    current = await store.deadline(conversation_id)
    if current is None or current < when:
        await store.set_deadline(conversation_id, when)
    if getattr(session, "can_renew", False):
        try:
            await asyncio.to_thread(session.renew, max(when - now, 1.0))
        except Exception:
            logger.warning(
                "workspace.renew_failed conversation_id=%s",
                conversation_id,
                exc_info=True,
            )
    return True


async def _sandbox_current(session: Any, conversation_id: str) -> bool:
    """False once the conversation's record is gone or names another sandbox.

    A conversation delete stops the sandbox, then drops the record. A store
    error counts as alive, so a store blip does not lose a save.
    """
    from src.services.agents.core.sandbox_pool import get_store

    try:
        record = await get_store().get(conversation_id)
    except Exception:
        logger.info(
            "workspace.alive_check_failed conversation_id=%s", conversation_id
        )
        return True
    if record is None:
        return False
    return record.get("sandbox_id") == getattr(session, "sandbox_id", None)


def _read_body(response: dict) -> bytes:
    body = response["Body"]
    try:
        return body.read()
    finally:
        close = getattr(body, "close", None)
        if close is not None:
            close()


def _upload_error(response: Any) -> Optional[str]:
    if isinstance(response, dict):
        return response.get("error")
    return getattr(response, "error", None)


def _log_upload_errors(responses: Any) -> None:
    for response in responses or []:
        error = _upload_error(response)
        if error:
            logger.warning("workspace.upload_failed error=%s", error)


_MD_TARGET = re.compile(r"(!?\[[^\]]*])\(([^)\s]+)\)")


def session_path_for_markdown(url: str) -> Optional[str]:
    """Session-relative path for a markdown URL the chat cannot load yet.

    A relative path, or ``/workspace/...``, is a file in the working directory.
    ``/skills/...`` and ``/large_tool_results/...`` are the sibling directories.
    A host is stripped first. ``.../workspace/ndvi.png`` is still that file.
    ``/artifacts/...`` is left to the artifact rewrite.
    """
    raw = urllib.parse.unquote(str(url or "").split("?", 1)[0].split("#", 1)[0]).strip()
    if not raw or raw.startswith("mailto:"):
        return None
    if raw.startswith(("http://", "https://")):
        raw = urllib.parse.urlparse(raw).path or ""
    if not raw or raw.startswith("/artifacts/"):
        return None
    if ".." in raw.split("/"):
        return None
    for marker, prefix in (
        ("/workspace/", "workspace/"),
        ("/skills/", "skills/"),
        ("/large_tool_results/", "large_tool_results/"),
    ):
        at = raw.find(marker)
        if at >= 0:
            relative = prefix + raw[at + len(marker) :]
            return _markdown_relative(relative)
    if raw.startswith("./"):
        raw = raw[2:]
    if raw.startswith("/"):
        return None
    return _markdown_relative("workspace/" + raw)


def _markdown_relative(relative: str) -> Optional[str]:
    if not relative or relative.endswith("/"):
        return None
    if any(part in {"", ".", ".."} for part in relative.split("/")):
        return None
    return relative


def workspace_file_url(conversation_id: str, relative: str) -> str:
    """URL the chat loads. The bytes stay in the workspace prefix."""
    return f"/conversations/{conversation_id}/workspace/{relative}"


async def publish_markdown_files(
    answer: str, edits: Optional[list[tuple[int, int, int]]] = None
) -> str:
    """Point a workspace markdown path at the saved workspace file.

    The chat loads ``/conversations/{id}/workspace/{path}``. The bytes are the
    workspace object, not a second artifact. A missing file is left as
    written, with a warning. Each rewritten link is appended to ``edits`` as
    ``(start, end, new length)``.
    """
    if not answer:
        return answer
    from src.services.mcp.artifact_context import get_artifact_context

    ctx = get_artifact_context()
    if (
        ctx is None
        or ctx.sandbox_backend is None
        or not ctx.user_id
        or not ctx.conversation_id
    ):
        return answer
    pieces: list[str] = []
    cursor = 0
    for match in _MD_TARGET.finditer(answer):
        pieces.append(answer[cursor : match.start()])
        path = session_path_for_markdown(match.group(2))
        relative = _workspace_file(path)
        if relative is None:
            pieces.append(match.group(0))
        else:
            piece = await _link_workspace_file(
                match.group(0),
                match.group(1),
                path or "",
                relative,
                ctx,
            )
            if edits is not None and piece != match.group(0):
                edits.append((match.start(), match.end(), len(piece)))
            pieces.append(piece)
        cursor = match.end()
    pieces.append(answer[cursor:])
    return "".join(pieces)


def _workspace_file(path: Optional[str]) -> Optional[str]:
    """The path inside the workspace folder, or ``None`` for another tree."""
    if not path or not path.startswith("workspace/"):
        return None
    return _markdown_relative(path[len("workspace/") :])


async def _link_workspace_file(
    original: str, label: str, session_path: str, relative: str, ctx: Any
) -> str:
    key = f"{workspace_prefix(ctx.user_id, ctx.conversation_id)}{relative}"
    data = await _download_first(ctx.sandbox_backend, _read_candidates(session_path))
    if data:
        await storage_service.put_object(key, data, _content_type(data, relative))
    elif not await _workspace_object_exists(key):
        logger.warning("workspace.markdown_unreadable url=%s", original)
        return original
    url = workspace_file_url(ctx.conversation_id, relative)
    logger.warning("workspace.markdown_repaired url=%s to=%s", original, url)
    return f"{label}({url})"


async def _workspace_object_exists(key: str) -> bool:
    try:
        await storage_service.get_object(key)
    except ObjectNotFoundError:
        return False
    except Exception:
        logger.warning("workspace.lookup_failed key=%s", key, exc_info=True)
        return False
    return True
