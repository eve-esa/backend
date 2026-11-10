"""Offloaded tool results in S3, mirrored into the sandbox.

deepagents writes a large tool result with ``backend.awrite`` and then shows
the model a stub. This backend stores that text in the bucket so a later turn
can still read it, and copies it into the sandbox so shell code can open the
same path. The model's ``read_file`` and ``grep`` go to S3. ``edit`` and
``delete`` refuse: the offloaded text is read-only for the model.

The composite strips ``/large_tool_results/`` before the call, so every path
here is ``/{tool_call_id}``.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Optional

from fastapi.concurrency import run_in_threadpool

from deepagents.backends.protocol import BackendProtocol

from src.services.storage import ObjectNotFoundError, storage_service

logger = logging.getLogger(__name__)

TOOL_RESULT_TAG = "kind=tool-result"
_CONTENT_TYPE = "text/plain; charset=utf-8"
_ROUTE = "/large_tool_results"

# grep downloads every object it scans. A conversation can accumulate more
# results than one search should read.
GREP_SCAN_LIMIT = 50
# Stop the download once this many bytes have been accepted. Same order of
# magnitude as the sandbox restore cap.
GREP_MAX_BYTES = 32 * 1024 * 1024
# A new sandbox is filled from S3. Stop before the copy dominates the turn.
RESTORE_MAX_OBJECTS = 50
RESTORE_MAX_BYTES = 32 * 1024 * 1024
_LIST_LIMIT = 1000

_OFFLOAD_STUB = re.compile(
    r"saved in the filesystem at this path:\s*(/large_tool_results/\S+)"
)
_READ_ONLY = "Error: offloaded tool results are read-only"


def tool_result_prefix(user_id: str, conversation_id: str) -> str:
    """Object-key prefix for one conversation's offloaded results."""
    return (
        f"users/{user_id}/conversations/{conversation_id}/large_tool_results/"
    )


def offloaded_result_path(content: Any) -> Optional[str]:
    """The ``/large_tool_results/...`` path named by an offload stub.

    ``None`` when ``content`` is not the stub deepagents writes. The check
    is the first line of that stub, so ordinary tool output is left alone.
    """
    if not isinstance(content, str) or not content.startswith("Tool result too large"):
        return None
    match = _OFFLOAD_STUB.search(content)
    return match.group(1) if match else None


def read_offloaded_tool_result(
    user_id: str, conversation_id: str, path: str
) -> Optional[str]:
    """The object at ``path``, or ``None`` when it is missing or unreadable."""
    name = _object_name(path.removeprefix(_ROUTE))
    if name is None:
        return None
    key = f"{tool_result_prefix(user_id, conversation_id)}{name}"
    try:
        response = storage_service.get_object_sync(key)
        return _decode(_read_body(response))
    except ObjectNotFoundError:
        return None
    except Exception:
        logger.warning(
            "tool_results.read_failed key=%s", key, exc_info=True
        )
        return None


def expand_offloaded_content(
    content: Any, user_id: Optional[str], conversation_id: Optional[str]
) -> Any:
    """Replace an offload stub with the stored text.

    A turn with no conversation, or an object that cannot be read, keeps
    the stub. Callers that parse the tool output then see what they saw
    before offload existed.
    """
    if not user_id or not conversation_id:
        return content
    path = offloaded_result_path(content)
    if path is None:
        return content
    full = read_offloaded_tool_result(user_id, conversation_id, path)
    return content if full is None else full


class S3ToolResultBackend(BackendProtocol):
    """``/large_tool_results/{tool_call_id}`` stored in the bucket.

    One instance per turn. Fetched text stays on the instance, so paging one
    file costs a single GET. ``sandbox_backend`` receives a copy at the full
    ``/large_tool_results/`` path when a write succeeds. A failed copy is
    logged: S3 is the copy that must succeed.
    """

    def __init__(
        self,
        user_id: str,
        conversation_id: str,
        sandbox_backend: Any = None,
    ) -> None:
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.sandbox_backend = sandbox_backend
        self.prefix = tool_result_prefix(user_id, conversation_id)
        self._cache: dict[str, str] = {}

    def ls(self, path: str):
        from deepagents.backends.protocol import LsResult

        try:
            records, truncated = storage_service.list_key_records(
                self.prefix, _LIST_LIMIT
            )
        except Exception as exc:
            logger.warning("tool_results.list_failed", exc_info=True)
            return LsResult(error=f"Error: {exc}")
        if truncated:
            logger.info("tool_results.list_truncated prefix=%s", self.prefix)
        return LsResult(entries=_ls_entries(records, self.prefix, path))

    async def als(self, path: str):
        from deepagents.backends.protocol import LsResult

        try:
            records, truncated = await run_in_threadpool(
                storage_service.list_key_records, self.prefix, _LIST_LIMIT
            )
        except Exception as exc:
            logger.warning("tool_results.list_failed", exc_info=True)
            return LsResult(error=f"Error: {exc}")
        if truncated:
            logger.info("tool_results.list_truncated prefix=%s", self.prefix)
        return LsResult(entries=_ls_entries(records, self.prefix, path))

    def read(self, file_path: str, offset: int = 0, limit: int = 2000):
        try:
            text = self._get_text(file_path)
        except Exception as exc:
            logger.warning(
                "tool_results.read_failed path=%s", file_path, exc_info=True
            )
            from deepagents.backends.protocol import ReadResult

            return ReadResult(error=f"Error: {exc}")
        return _slice(file_path, text, offset, limit)

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000):
        try:
            text = await self._aget_text(file_path)
        except Exception as exc:
            logger.warning(
                "tool_results.read_failed path=%s", file_path, exc_info=True
            )
            from deepagents.backends.protocol import ReadResult

            return ReadResult(error=f"Error: {exc}")
        return _slice(file_path, text, offset, limit)

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ):
        try:
            records, truncated = storage_service.list_key_records(
                self.prefix, GREP_SCAN_LIMIT
            )
            files, byte_capped = self._files(records)
        except Exception as exc:
            logger.warning("tool_results.grep_failed", exc_info=True)
            from deepagents.backends.protocol import GrepResult

            return GrepResult(error=f"Error: {exc}")
        return _grep(files, pattern, path, glob, max_count, truncated or byte_capped)

    async def agrep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ):
        try:
            records, truncated = await run_in_threadpool(
                storage_service.list_key_records, self.prefix, GREP_SCAN_LIMIT
            )
            files, byte_capped = await self._afiles(records)
        except Exception as exc:
            logger.warning("tool_results.grep_failed", exc_info=True)
            from deepagents.backends.protocol import GrepResult

            return GrepResult(error=f"Error: {exc}")
        return _grep(files, pattern, path, glob, max_count, truncated or byte_capped)

    def glob(self, pattern: str, path: str | None = None):
        try:
            records, truncated = storage_service.list_key_records(
                self.prefix, _LIST_LIMIT
            )
        except Exception as exc:
            logger.warning("tool_results.glob_failed", exc_info=True)
            from deepagents.backends.protocol import GlobResult

            return GlobResult(error=f"Error: {exc}")
        return _glob(records, self.prefix, pattern, path, truncated)

    async def aglob(self, pattern: str, path: str | None = None):
        try:
            records, truncated = await run_in_threadpool(
                storage_service.list_key_records, self.prefix, _LIST_LIMIT
            )
        except Exception as exc:
            logger.warning("tool_results.glob_failed", exc_info=True)
            from deepagents.backends.protocol import GlobResult

            return GlobResult(error=f"Error: {exc}")
        return _glob(records, self.prefix, pattern, path, truncated)

    def write(self, file_path: str, content: str):
        from deepagents.backends.protocol import WriteResult

        name = _object_name(file_path)
        if name is None:
            return WriteResult(error=f"Error: invalid tool result path '{file_path}'")
        key = f"{self.prefix}{name}"
        try:
            if _object_exists(key):
                return WriteResult(error=_exists_error(file_path))
            storage_service.put_object_sync(
                key, content.encode("utf-8"), _CONTENT_TYPE, TOOL_RESULT_TAG
            )
        except Exception as exc:
            logger.warning("tool_results.write_failed key=%s", key, exc_info=True)
            return WriteResult(error=f"Error: could not store tool result: {exc}")
        self._cache[_cache_key(file_path)] = content
        self._mirror(name, content)
        return WriteResult(path=_cache_key(file_path))

    async def awrite(self, file_path: str, content: str):
        from deepagents.backends.protocol import WriteResult

        name = _object_name(file_path)
        if name is None:
            return WriteResult(error=f"Error: invalid tool result path '{file_path}'")
        key = f"{self.prefix}{name}"
        try:
            if await _aobject_exists(key):
                return WriteResult(error=_exists_error(file_path))
            await storage_service.put_object(
                key, content.encode("utf-8"), _CONTENT_TYPE, TOOL_RESULT_TAG
            )
        except Exception as exc:
            logger.warning("tool_results.write_failed key=%s", key, exc_info=True)
            return WriteResult(error=f"Error: could not store tool result: {exc}")
        self._cache[_cache_key(file_path)] = content
        await self._amirror(name, content)
        return WriteResult(path=_cache_key(file_path))

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ):
        from deepagents.backends.protocol import EditResult

        return EditResult(error=_READ_ONLY)

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ):
        return self.edit(file_path, old_string, new_string, replace_all)

    def delete(self, file_path: str):
        from deepagents.backends.protocol import DeleteResult

        return DeleteResult(error=_READ_ONLY)

    async def adelete(self, file_path: str):
        return self.delete(file_path)

    def _get_text(self, file_path: str) -> Optional[str]:
        key_path = _cache_key(file_path)
        cached = self._cache.get(key_path)
        if cached is not None:
            return cached
        name = _object_name(file_path)
        if name is None:
            return None
        try:
            response = storage_service.get_object_sync(f"{self.prefix}{name}")
        except ObjectNotFoundError:
            return None
        text = _decode(_read_body(response))
        self._cache[key_path] = text
        return text

    async def _aget_text(self, file_path: str) -> Optional[str]:
        key_path = _cache_key(file_path)
        cached = self._cache.get(key_path)
        if cached is not None:
            return cached
        name = _object_name(file_path)
        if name is None:
            return None
        try:
            response = await storage_service.get_object(f"{self.prefix}{name}")
        except ObjectNotFoundError:
            return None
        text = _decode(await run_in_threadpool(_read_body, response))
        self._cache[key_path] = text
        return text

    def _files(self, records: list[dict]) -> tuple[dict[str, Any], bool]:
        from deepagents.backends.utils import create_file_data

        chosen, capped = _grep_records(records)
        files: dict[str, Any] = {}
        for record in chosen:
            path = _virtual_path(self.prefix, record["key"])
            text = self._get_text(path)
            if text is None:
                continue
            files[path] = _with_modified(create_file_data(text), record)
        return files, capped

    async def _afiles(self, records: list[dict]) -> tuple[dict[str, Any], bool]:
        from deepagents.backends.utils import create_file_data

        chosen, capped = _grep_records(records)
        files: dict[str, Any] = {}
        for record in chosen:
            path = _virtual_path(self.prefix, record["key"])
            text = await self._aget_text(path)
            if text is None:
                continue
            files[path] = _with_modified(create_file_data(text), record)
        return files, capped

    def _mirror(self, name: str, content: str) -> None:
        sandbox = self.sandbox_backend
        upload = getattr(sandbox, "upload_files", None) if sandbox is not None else None
        if upload is None:
            return
        payload = [(_sandbox_path(name), content.encode("utf-8"))]
        try:
            _log_upload_errors(upload(payload))
        except Exception:
            logger.warning(
                "tool_results.mirror_failed path=%s",
                _sandbox_path(name),
                exc_info=True,
            )

    async def _amirror(self, name: str, content: str) -> None:
        sandbox = self.sandbox_backend
        if sandbox is None:
            return
        payload = [(_sandbox_path(name), content.encode("utf-8"))]
        upload = getattr(sandbox, "aupload_files", None)
        try:
            if upload is not None:
                _log_upload_errors(await upload(payload))
                return
            sync_upload = getattr(sandbox, "upload_files", None)
            if sync_upload is None:
                return
            _log_upload_errors(await run_in_threadpool(sync_upload, payload))
        except Exception:
            logger.warning(
                "tool_results.mirror_failed path=%s",
                _sandbox_path(name),
                exc_info=True,
            )


async def restore_tool_results(
    sandbox_backend: Any, user_id: str, conversation_id: str
) -> None:
    """Copy this conversation's offloaded results into a new sandbox.

    A failure is logged and swallowed. The turn still runs: the model can
    read the same files from S3, and only the shell copy is missing.
    """
    try:
        await _restore(sandbox_backend, user_id, conversation_id)
    except Exception:
        logger.warning(
            "tool_results.restore_failed conversation_id=%s",
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
        logger.warning("tool_results.restore_skipped reason=no_upload")
        return
    prefix = tool_result_prefix(user_id, conversation_id)
    records, truncated = await run_in_threadpool(
        storage_service.list_key_records, prefix, RESTORE_MAX_OBJECTS
    )
    if truncated:
        logger.info(
            "tool_results.restore_truncated conversation_id=%s",
            conversation_id,
        )
    files: list[tuple[str, bytes]] = []
    total = 0
    for record in records:
        size = int(record.get("size") or 0)
        if size > RESTORE_MAX_BYTES or total + size > RESTORE_MAX_BYTES:
            logger.info(
                "tool_results.restore_skipped_size key=%s size=%s",
                record.get("key"),
                size,
            )
            continue
        key = record["key"]
        name = key[len(prefix) :] if key.startswith(prefix) else ""
        if _object_name(f"/{name}") is None:
            continue
        try:
            response = await storage_service.get_object(key)
            body = await run_in_threadpool(_read_body, response)
        except ObjectNotFoundError:
            continue
        except Exception:
            logger.warning(
                "tool_results.restore_read_failed key=%s", key, exc_info=True
            )
            continue
        total += len(body)
        files.append((_sandbox_path(name), body))
    if not files:
        return
    _log_upload_errors(await run_in_threadpool(upload, files))


def _object_name(file_path: str) -> Optional[str]:
    """The key suffix for a backend path, or ``None`` when it is not a file."""
    path = file_path if file_path.startswith("/") else f"/{file_path}"
    name = path[1:]
    if not name or name.endswith("/"):
        return None
    parts = name.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return None
    return name


def _cache_key(file_path: str) -> str:
    return file_path if file_path.startswith("/") else f"/{file_path}"


def _sandbox_path(name: str) -> str:
    return f"{_ROUTE}/{name}"


def _virtual_path(prefix: str, key: str) -> str:
    name = key[len(prefix) :] if key.startswith(prefix) else key
    return f"/{name}"


def _iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return ""


def _read_body(response: dict) -> bytes:
    body = response["Body"]
    try:
        data = body.read()
    finally:
        close = getattr(body, "close", None)
        if close is not None:
            close()
    return data


def _decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _with_modified(file_data: dict, record: dict) -> dict:
    modified = _iso(record.get("last_modified"))
    if modified:
        file_data["modified_at"] = modified
    return file_data


def _result_path(file_path: str) -> str:
    """Model-facing path. Composite read does not rewrite ``ReadResult.error``."""
    path = _cache_key(file_path)
    route = f"{_ROUTE}/"
    if path.startswith(route):
        return path
    return f"{route}{path.lstrip('/')}"


def _exists_error(file_path: str) -> str:
    return f"Error: tool result '{_result_path(file_path)}' already exists"


def _object_exists(key: str) -> bool:
    # HeadObject then PutObject. IfNoneMatch='*' overwrites on MinIO builds
    # that ignore the wildcard. A create between the two calls can still win.
    return storage_service.exists_sync(key)


async def _aobject_exists(key: str) -> bool:
    return await run_in_threadpool(_object_exists, key)


def _grep_records(records: list[dict]) -> tuple[list[dict], bool]:
    """Objects to GET. Stop before a download would pass ``GREP_MAX_BYTES``."""
    chosen: list[dict] = []
    total = 0
    for record in records:
        size = int(record.get("size") or 0)
        if total >= GREP_MAX_BYTES or total + size > GREP_MAX_BYTES:
            return chosen, True
        chosen.append(record)
        total += size
    return chosen, False


def _slice(file_path: str, text: Optional[str], offset: int, limit: int):
    from deepagents.backends.protocol import ReadResult
    from deepagents.backends.utils import create_file_data, slice_read_response

    if text is None:
        return ReadResult(error=f"File '{_result_path(file_path)}' not found")
    return slice_read_response(create_file_data(text), offset, limit)


def _ls_entries(records: list[dict], prefix: str, path: str) -> list[dict]:
    normalized = path if path.endswith("/") else path + "/"
    infos: list[dict] = []
    subdirs: set[str] = set()
    for record in records:
        key_path = _virtual_path(prefix, record["key"])
        if not key_path.startswith(normalized):
            continue
        relative = key_path[len(normalized) :]
        if "/" in relative:
            subdirs.add(normalized + relative.split("/")[0] + "/")
            continue
        infos.append(
            {
                "path": key_path,
                "is_dir": False,
                "size": int(record.get("size") or 0),
                "modified_at": _iso(record.get("last_modified")),
            }
        )
    infos.extend(
        {"path": subdir, "is_dir": True, "size": 0, "modified_at": ""}
        for subdir in sorted(subdirs)
    )
    infos.sort(key=lambda item: item.get("path", ""))
    return infos


def _grep(files, pattern, path, glob, max_count, listing_truncated):
    from deepagents.backends.utils import grep_matches_from_files

    result = grep_matches_from_files(
        files, pattern, path if path is not None else "/", glob, max_count=max_count
    )
    if listing_truncated and result.error is None:
        result.truncated = True
    return result


def _glob(records, prefix, pattern, path, listing_truncated):
    from deepagents.backends.protocol import FileInfo, GlobResult
    from deepagents.backends.utils import InvalidGlobPatternError, _glob_search_files

    files: dict[str, dict] = {}
    sizes: dict[str, int] = {}
    for record in records:
        found = _virtual_path(prefix, record["key"])
        files[found] = _with_modified(
            {"content": "", "encoding": "utf-8"}, record
        )
        sizes[found] = int(record.get("size") or 0)
    try:
        result = _glob_search_files(files, pattern, path)
    except InvalidGlobPatternError as exc:
        return GlobResult(error=str(exc))
    if result == "No files found":
        return GlobResult(matches=[], truncated=listing_truncated)
    infos: list[FileInfo] = []
    for found in result.split("\n"):
        file_data = files.get(found) or {}
        infos.append(
            {
                "path": found,
                "is_dir": False,
                "size": sizes.get(found, 0),
                "modified_at": file_data.get("modified_at", ""),
            }
        )
    return GlobResult(matches=infos, truncated=listing_truncated)


def _upload_error(response: Any) -> Optional[str]:
    if isinstance(response, dict):
        return response.get("error")
    return getattr(response, "error", None)


def _log_upload_errors(responses: Any) -> None:
    for response in responses or []:
        error = _upload_error(response)
        if error:
            logger.warning("tool_results.upload_failed error=%s", error)
