"""Conversation copy of an MCP artifact, and the sandbox file that mirrors it.

The artifact itself stays at ``users/{user_id}/artifacts/``. This module also
stores the same bytes at
``users/{user_id}/conversations/{conversation_id}/tool_artifacts/{file_path}``.
``file_path`` is ``{artifact_id}/{filename}``. The model hears only the sandbox
path ``/tool_artifacts/{file_path}``, which is that key with the conversation
prefix removed.

A new sandbox copies the prefix in before the turn. A file persisted while a
sandbox is already up is uploaded immediately. An upload failure is logged.
The MinIO object remains the copy a later sandbox restores.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from fastapi.concurrency import run_in_threadpool

from src.services.agents.core.tool_results import (
    RESTORE_MAX_BYTES,
    RESTORE_MAX_OBJECTS,
)
from src.services.storage import ObjectNotFoundError, storage_service

logger = logging.getLogger(__name__)

# No trailing slash. The shell rewrite matches this token, same as ``/skills``.
SANDBOX_PREFIX = "/tool_artifacts"


def file_path_for(artifact_id: str, filename: str) -> str:
    """``{artifact_id}/{basename}`` under the conversation ``tool_artifacts`` folder."""
    raw = (filename or "").replace("\\", "/").strip()
    name = os.path.basename(raw) or "file"
    if name in {".", ".."}:
        name = "file"
    return f"{artifact_id}/{name}"


def object_key(user_id: str, conversation_id: str, file_path: str) -> str:
    return (
        f"users/{user_id}/conversations/{conversation_id}/tool_artifacts/{file_path}"
    )


def sandbox_path(file_path: str) -> str:
    """Path the model reads. The conversation prefix is not part of it."""
    return f"{SANDBOX_PREFIX}/{file_path}"


def conversation_prefix(user_id: str, conversation_id: str) -> str:
    return f"users/{user_id}/conversations/{conversation_id}/tool_artifacts/"


async def upload_tool_artifacts(
    sandbox_backend: Any, path: str, data: bytes
) -> None:
    """Copy one file into the live sandbox. A missing uploader is a no-op."""
    if sandbox_backend is None:
        return
    payload = [(path, data)]
    upload = getattr(sandbox_backend, "aupload_files", None)
    try:
        if upload is not None:
            _log_upload_errors(await upload(payload))
            return
        sync_upload = getattr(sandbox_backend, "upload_files", None)
        if sync_upload is None:
            return
        _log_upload_errors(await run_in_threadpool(sync_upload, payload))
    except Exception:
        logger.warning("tool_artifacts.upload_failed path=%s", path, exc_info=True)


async def restore_tool_artifacts(
    sandbox_backend: Any, user_id: str, conversation_id: str
) -> None:
    """Copy this conversation's ``tool_artifacts`` objects into a new sandbox.

    A failure is logged and swallowed. The turn still runs.
    """
    try:
        await _restore(sandbox_backend, user_id, conversation_id)
    except Exception:
        logger.warning(
            "tool_artifacts.restore_failed conversation_id=%s",
            conversation_id,
            exc_info=True,
        )


async def _restore(
    sandbox_backend: Any, user_id: str, conversation_id: str
) -> None:
    if sandbox_backend is None or not user_id or not conversation_id:
        return
    upload = getattr(sandbox_backend, "upload_files", None)
    if upload is None:
        logger.warning("tool_artifacts.restore_skipped reason=no_upload")
        return
    prefix = conversation_prefix(user_id, conversation_id)
    records, truncated = await run_in_threadpool(
        storage_service.list_key_records, prefix, RESTORE_MAX_OBJECTS
    )
    if truncated:
        logger.info(
            "tool_artifacts.restore_truncated conversation_id=%s", conversation_id
        )
    files: list[tuple[str, bytes]] = []
    total = 0
    for record in records:
        size = int(record.get("size") or 0)
        if size > RESTORE_MAX_BYTES or total + size > RESTORE_MAX_BYTES:
            logger.info(
                "tool_artifacts.restore_skipped_size key=%s size=%s",
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
                "tool_artifacts.restore_read_failed key=%s", key, exc_info=True
            )
            continue
        total += len(body)
        files.append((sandbox_path(relative), body))
    if not files:
        return
    _log_upload_errors(await run_in_threadpool(upload, files))


def _rejected(relative: str) -> bool:
    if not relative or relative.startswith("/"):
        return True
    return any(part in {"", ".", ".."} for part in relative.split("/"))


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
            logger.warning("tool_artifacts.upload_failed error=%s", error)
