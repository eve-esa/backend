"""Sandbox providers for one agent turn.

Every setting this service owns starts with ``SANDBOX_``. ``SANDBOX_PROVIDER``
picks the row. Values are read when a turn starts, not at import time, so a
``.env`` loaded later still applies. Credentials a provider SDK already reads
stay on that SDK's own names.

``none`` and an unset provider start nothing. ``open_sandbox`` / ``close_sandbox``
are the turn with no conversation: the handle is destroyed when the turn ends,
including when the turn fails. A conversation reuses one sandbox; that lifetime
lives in ``sandbox_pool``.

AgentCore's ``sessionTimeoutSeconds`` is a hard cap. Activity does not extend
it, and nothing in the API renews it, so a reused session is replaced before
the cap can cut a turn off. The cap is 28,800 seconds.
https://docs.aws.amazon.com/bedrock-agentcore/latest/APIReference/API_StartCodeInterpreterSession.html

OpenSandbox ``renew`` sets the expiry to now plus the duration given, so the
idle timer and the in-turn window are the same call.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
import re
import shlex
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from src.config import AGENTIC_TURN_DEADLINE

logger = logging.getLogger(__name__)

_PROVIDER_ENV = "SANDBOX_PROVIDER"
_AGENTCORE_TTL_MAX = 28800
_REUSE_MARGIN_S = 60
_DEFAULT_AGENTCORE_ID = "aws.codeinterpreter.v1"
_DEFAULT_OPENSANDBOX_IMAGE = "python:3.12"
_SKILLS_PREFIX = "/skills/"
_CONVERSATION_HISTORY = "/conversation_history/"
_LARGE_TOOL_RESULTS = "/large_tool_results/"
# writeFiles stores these under the session cwd. The shell rewrite points a
# command at that directory. Longest prefix first, and none has a trailing
# slash, so ``/skills`` itself is covered. A raw replace of ``/skills`` would
# rewrite the suffix of ``/large_tool_results/skills_1``. ``/artifacts`` is
# an MCP file stored for this conversation.
_SHELL_PREFIXES = ("/large_tool_results", "/artifacts", "/skills")


# OpenSandbox logs every failed endpoint lookup at WARNING before it raises,
# including the 404 for a sandbox that expired or was removed. Reattach and
# kill expect that answer and log it themselves at INFO with ids only.
_OPENSANDBOX_SDK_LOGGER = "opensandbox.sync.adapters.sandboxes_adapter"
_OPENSANDBOX_GONE_LINE = re.compile(
    r"\bsandbox \S+ not found\b|\bHTTP 4(04|10)\b", re.IGNORECASE
)
_expect_gone: ContextVar[bool] = ContextVar("sandbox_expect_gone", default=False)


class _GoneLookupFilter(logging.Filter):
    """Drop the SDK's WARNING for a gone sandbox, only inside our attach.

    Outside ``_attach_opensandbox`` every line passes. Inside it, a timeout, a
    5xx or an auth error keeps its WARNING. The readiness 404 the SDK retries
    while a pod has no IP yet is not a gone sandbox and also keeps it.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not _expect_gone.get():
            return True
        message = record.getMessage()
        if not message.startswith("Failed to retrieve sandbox endpoint"):
            return True
        if "POD_IP_NOT_AVAILABLE" in message:
            return True
        return _OPENSANDBOX_GONE_LINE.search(message) is None


# The SDK transport logs each retry at WARNING, also while execd starts and
# the readiness ping meets a closed connection. Inside our create and connect
# calls (which run that ping) a retry is expected, and retries that run out
# raise and are logged once at WARNING by ``_start_opensandbox`` or
# ``_attach_opensandbox``. Everywhere else (command runs, file uploads inside
# a tool call) a retry keeps its WARNING, so an outage mid-turn still shows.
_OPENSANDBOX_RETRY_LOGGERS = (
    "opensandbox.transport._sync_retry",
    "opensandbox.transport._async_retry",
)
_quiet_retries: ContextVar[bool] = ContextVar("sandbox_quiet_retries", default=False)


class _RetryAsInfoFilter(logging.Filter):
    """Log the SDK's ``retrying ...`` lines at INFO inside create and connect."""

    def filter(self, record: logging.LogRecord) -> bool:
        if (
            _quiet_retries.get()
            and record.levelno == logging.WARNING
            and str(record.msg).startswith("retrying ")
        ):
            record.levelno = logging.INFO
            record.levelname = logging.getLevelName(logging.INFO)
        return True


# A user Stop cancels the tool call while the SDK thread still streams the
# command; the closed HTTP stream then fails the read (``Errno 9``) and the
# SDK logs ERROR "Failed to run command". ``asyncio.to_thread`` copies the
# context, so the thread holds the turn's Stop binding and sees an event set
# after it started. A Stop is not an error; without one the ERROR stays.
_OPENSANDBOX_COMMAND_LOGGERS = (
    "opensandbox.sync.adapters.command_adapter",
    "opensandbox.adapters.command_adapter",
)
_COMMAND_FAILED = "Failed to run command"


class _StoppedCommandFilter(logging.Filter):
    """Log a command failure caused by a user Stop at INFO, with the reason."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno < logging.WARNING or record.msg != _COMMAND_FAILED:
            return True
        from src.observability.context import stop_requested

        if not stop_requested():
            return True
        exc = record.exc_info[1] if record.exc_info else None
        record.msg = "%s: turn stopped by the user reason=stop error=%s"
        record.args = (_COMMAND_FAILED, type(exc).__name__ if exc else "none")
        record.exc_info = None
        record.exc_text = None
        record.levelno = logging.INFO
        record.levelname = logging.getLevelName(logging.INFO)
        return True


# The AgentCore integration logs ERROR "AgentCore session expired while ..."
# and returns exit code 1 when the session is gone. After a turn the save of
# the workspace can meet a session that a conversation delete just stopped,
# so there the line drops to INFO and the save logs the outcome itself: INFO
# when the sandbox was stopped on purpose, WARNING workspace.save_failed
# reason=session_expired when its record still names it. During a turn, and
# in the reaper snapshot, the ERROR stays.
_AGENTCORE_SDK_LOGGER = "langchain_agentcore_codeinterpreter.sandbox"
_AGENTCORE_EXPIRED = "AgentCore session expired while"
_after_turn: ContextVar[bool] = ContextVar("sandbox_after_turn", default=False)


class _ClosedAfterTurnFilter(logging.Filter):
    """Log an expired session at INFO, with the reason, inside the post-turn save."""

    def filter(self, record: logging.LogRecord) -> bool:
        if (
            not _after_turn.get()
            or record.levelno < logging.WARNING
            or not str(record.msg).startswith(_AGENTCORE_EXPIRED)
        ):
            return True
        record.msg = f"{record.msg} reason=session_closed_after_turn"
        record.levelno = logging.INFO
        record.levelname = logging.getLevelName(logging.INFO)
        return True


@contextlib.contextmanager
def after_turn() -> Iterator[None]:
    """Mark sandbox calls made here as post-turn work.

    ``asyncio.to_thread`` copies the context, so a command run in a thread
    inside this block is marked too.
    """
    token = _after_turn.set(True)
    try:
        yield
    finally:
        _after_turn.reset(token)


# Marks our filters on the SDK loggers. A module reload (local development)
# builds new ContextVars, so an older instance is replaced, not kept.
_FILTER_MARK = "_eve_sandbox_filter"


def _add_filter_once(name: str, filter_: logging.Filter) -> None:
    target = logging.getLogger(name)
    kind = type(filter_).__name__
    for old in list(target.filters):
        if getattr(old, _FILTER_MARK, None) == kind:
            target.removeFilter(old)
    setattr(filter_, _FILTER_MARK, kind)
    target.addFilter(filter_)


_add_filter_once(_OPENSANDBOX_SDK_LOGGER, _GoneLookupFilter())
for _name in _OPENSANDBOX_RETRY_LOGGERS:
    _add_filter_once(_name, _RetryAsInfoFilter())
for _name in _OPENSANDBOX_COMMAND_LOGGERS:
    _add_filter_once(_name, _StoppedCommandFilter())
_add_filter_once(_AGENTCORE_SDK_LOGGER, _ClosedAfterTurnFilter())


class SandboxError(RuntimeError):
    """A sandbox could not be started or prepared."""


class SandboxGone(SandboxError):
    """The provider reports that this sandbox no longer exists."""


class SandboxSession:
    """One provider session and the backend the graph compiles with.

    ``destroy`` stops the remote sandbox. ``detach`` drops local clients and
    leaves the remote sandbox running, which is what a conversation release
    does. ``close`` is ``destroy``: a turn with no conversation still ends the
    sandbox it started.
    """

    def __init__(
        self,
        backend: Any,
        *,
        provider: str,
        sandbox_id: str,
        destroy: Callable[[], None],
        detach: Callable[[], None],
        renew: Optional[Callable[[float], None]] = None,
        metadata: Optional[dict[str, str]] = None,
        skills_digest: str = "",
    ) -> None:
        self.backend = backend
        self.provider = provider
        self.sandbox_id = sandbox_id
        self.metadata = dict(metadata or {})
        self.skills_digest = skills_digest
        self._destroy = destroy
        self._detach = detach
        self._renew = renew
        self._closed = False
        self._workspace_task: Optional[asyncio.Task] = None

    @property
    def can_renew(self) -> bool:
        """True when the provider can move its own expiry."""
        return self._renew is not None

    def renew(self, seconds: float) -> None:
        """Set the provider expiry to now plus *seconds*. No-op when it cannot."""
        if self._renew is not None:
            self._renew(seconds)

    def detach(self) -> None:
        """Drop local clients. The remote sandbox keeps running."""
        if self._closed:
            return
        self._closed = True
        self._detach()

    def destroy(self) -> None:
        """Stop the remote sandbox. A second call does nothing."""
        if self._closed:
            return
        self._closed = True
        self._destroy()

    def close(self) -> None:
        """Destroy the remote sandbox. A second call does nothing."""
        self.destroy()

    def workspace_sync_active(self) -> bool:
        """True while the post-turn workspace save has not finished."""
        task = self._workspace_task
        return task is not None and not task.done()

    def start_workspace_sync(self, user_id: str, conversation_id: str) -> None:
        """Save the working directory after this turn. A second call does nothing.

        The task detaches the session when the save finishes. Release skips
        detach while the task is running, so a background job can still be read.
        """
        if not user_id or not conversation_id or self.workspace_sync_active():
            return
        from src.services.agents.core.workspace import sync_workspace

        self._workspace_task = asyncio.create_task(
            sync_workspace(self, user_id, conversation_id),
            name=f"workspace-sync-{conversation_id}",
        )


def _noop() -> None:
    return None


def _setting(name: str, default: int, *, upper: Optional[int] = None) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    if value < 1:
        value = default
    if upper is not None:
        value = min(value, upper)
    return value


def sandbox_provider() -> str:
    """``SANDBOX_PROVIDER`` as a lower-case name. Empty when unset."""
    return os.environ.get(_PROVIDER_ENV, "").strip().lower()


def sandbox_idle_ttl_s() -> int:
    """How long a conversation sandbox stays up after its last turn."""
    return _setting("SANDBOX_IDLE_TTL_S", 600)


def sandbox_max_lifetime_s() -> int:
    """AgentCore's hard session cap. Activity does not extend it."""
    return _setting("SANDBOX_MAX_LIFETIME_S", 3600, upper=_AGENTCORE_TTL_MAX)


def sandbox_reaper_interval_s() -> int:
    """How often a worker looks for sandboxes past their idle deadline."""
    return _setting("SANDBOX_REAPER_INTERVAL_S", 30)


def sandbox_busy_ttl_s() -> int:
    """How long the sandbox stays up when a turn starts.

    Idle window plus AGENTIC_TURN_DEADLINE. Default 600 + 480 = 1080. A turn
    that starts at 10:00 is covered until 10:18. If it ends at 10:02, release
    sets the deadline to 10:12.
    """
    return sandbox_idle_ttl_s() + AGENTIC_TURN_DEADLINE


def sandbox_reuse_limit_s() -> int:
    """AgentCore time that must remain before a session is reused.

    AGENTIC_TURN_DEADLINE plus 60s. Default 540. A session created at 10:00
    ends at 11:00. A turn at 10:50 (600s left) reuses it. A turn at 10:52
    (480s left) replaces it.
    """
    return AGENTIC_TURN_DEADLINE + _REUSE_MARGIN_S


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SandboxError(f"{name} is required")
    return value


def _agentcore_session_ended(exc: BaseException) -> bool:
    """True when AgentCore reports the session is already gone."""
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    error = response.get("Error", {})
    if not isinstance(error, dict):
        return False
    code = error.get("Code", "")
    message = error.get("Message", "")
    return (
        code == "ResourceNotFoundException"
        or (code == "ValidationException" and "is not active" in message)
        or (code == "ConflictException" and "already terminated" in message)
    )


def _agentcore_client(region: str, identifier: str, sandbox_id: str) -> Any:
    from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter
    from langchain_agentcore_codeinterpreter.sandbox import INTEGRATION_SOURCE

    interpreter = CodeInterpreter(
        region=region, integration_source=INTEGRATION_SOURCE
    )
    interpreter.identifier = identifier
    interpreter.session_id = sandbox_id
    return interpreter


def _stop_agentcore_interpreter(interpreter: Any) -> None:
    try:
        interpreter.stop()
    except Exception as exc:
        if not _agentcore_session_ended(exc):
            raise


def _agentcore_handle(
    sandbox: Any,
    *,
    sandbox_id: str,
    region: str,
    identifier: str,
    lifetime: int,
    created_at: Optional[str],
    destroy: Callable[[], None],
) -> SandboxSession:
    return SandboxSession(
        prepare_workspace_layout(_bind_skills_shell(sandbox)),
        provider="agentcore",
        sandbox_id=sandbox_id,
        destroy=destroy,
        detach=_noop,
        metadata={
            "region": region,
            "identifier": identifier,
            "created_at": created_at or str(int(time.time())),
            "max_lifetime_s": str(lifetime),
        },
    )


def _start_agentcore() -> SandboxSession:
    region = _require("SANDBOX_AGENTCORE_REGION")
    identifier = (
        os.environ.get("SANDBOX_AGENTCORE_IDENTIFIER", "").strip()
        or _DEFAULT_AGENTCORE_ID
    )
    lifetime = sandbox_max_lifetime_s()
    from langchain_agentcore_codeinterpreter import AgentCoreSandbox

    sandbox = AgentCoreSandbox.create(
        region=region,
        identifier=identifier,
        session_timeout_seconds=lifetime,
    )
    # writeFiles stores "/skills/..." under the session cwd, and the process
    # cannot create a real "/skills". Skill text still calls that absolute
    # path, so the shell rewrites it to the directory the upload used.
    # close() stops the session only when this sandbox started it.
    return _agentcore_handle(
        sandbox,
        sandbox_id=str(getattr(sandbox, "id", "") or ""),
        region=region,
        identifier=identifier,
        lifetime=lifetime,
        created_at=None,
        destroy=sandbox.close,
    )


def _attach_agentcore(record: dict[str, str]) -> SandboxSession:
    region = record["region"]
    identifier = record["identifier"]
    sandbox_id = record["sandbox_id"]
    lifetime = int(record.get("max_lifetime_s") or sandbox_max_lifetime_s())
    interpreter = _agentcore_client(region, identifier, sandbox_id)
    try:
        info = interpreter.get_session()
    except Exception as exc:
        if _agentcore_session_ended(exc):
            raise SandboxGone(f"agentcore session {sandbox_id} has ended") from exc
        raise
    status = str(info.get("status", "")).upper()
    if status == "TERMINATED":
        raise SandboxGone(f"agentcore session {sandbox_id} is terminated")
    if status != "READY":
        raise SandboxError(f"agentcore session status {status or 'unknown'}")
    from langchain_agentcore_codeinterpreter import AgentCoreSandbox

    # This wrapper did not start the session, so its close() would leave the
    # session running. destroy() stops it through the client instead.
    def destroy() -> None:
        _stop_agentcore_interpreter(interpreter)

    session = _agentcore_handle(
        AgentCoreSandbox(interpreter=interpreter),
        sandbox_id=sandbox_id,
        region=region,
        identifier=identifier,
        lifetime=lifetime,
        created_at=record.get("created_at"),
        destroy=destroy,
    )
    session.skills_digest = record.get("skills_digest", "")
    return session


def _kill_agentcore(record: dict[str, str]) -> None:
    interpreter = _agentcore_client(
        record["region"], record["identifier"], record["sandbox_id"]
    )
    _stop_agentcore_interpreter(interpreter)


def _opensandbox_connection() -> tuple[str, Any]:
    domain = _require("SANDBOX_OPENSANDBOX_DOMAIN")
    protocol = os.environ.get("SANDBOX_OPENSANDBOX_PROTOCOL", "").strip() or "http"
    if protocol not in ("http", "https"):
        raise SandboxError("SANDBOX_OPENSANDBOX_PROTOCOL must be http or https")
    api_key = os.environ.get("SANDBOX_OPENSANDBOX_API_KEY", "").strip() or None
    image = (
        os.environ.get("SANDBOX_OPENSANDBOX_IMAGE", "").strip()
        or _DEFAULT_OPENSANDBOX_IMAGE
    )
    from opensandbox.config import ConnectionConfigSync

    config = ConnectionConfigSync(
        domain=domain,
        protocol=protocol,
        api_key=api_key,
    )
    return image, config


def _opensandbox_session(remote: Any, *, skills_digest: str = "") -> SandboxSession:
    from datetime import timedelta

    from langchain_opensandbox import OpenSandboxBackend

    def renew(seconds: float) -> None:
        remote.renew(timedelta(seconds=seconds))

    # close() frees the local HTTP client. destroy() also kills the remote.
    return SandboxSession(
        prepare_workspace_layout(OpenSandboxBackend(sandbox=remote)),
        provider="opensandbox",
        sandbox_id=str(getattr(remote, "id", "") or ""),
        destroy=remote.destroy,
        detach=remote.close,
        renew=renew,
        skills_digest=skills_digest,
    )


def _start_opensandbox() -> SandboxSession:
    from datetime import timedelta

    # Config is checked before the SDK import. A missing domain or a bad
    # protocol must raise SandboxError even when the package is not installed.
    image, config = _opensandbox_connection()
    from opensandbox import SandboxSync
    quiet = _quiet_retries.set(True)
    try:
        remote = SandboxSync.create(
            image,
            connection_config=config,
            timeout=timedelta(seconds=sandbox_busy_ttl_s()),
        )
    except Exception as exc:
        logger.warning(
            "sandbox.create_failed provider=opensandbox error=%s",
            type(exc).__name__,
        )
        raise
    finally:
        _quiet_retries.reset(quiet)
    return _opensandbox_session(remote)


def _attach_opensandbox(record: dict[str, str]) -> SandboxSession:
    _image, config = _opensandbox_connection()
    from opensandbox import SandboxSync
    sandbox_id = record["sandbox_id"]
    # Set and reset in this thread: the SDK logs from the same call stack.
    token = _expect_gone.set(True)
    quiet = _quiet_retries.set(True)
    try:
        remote = SandboxSync.connect(sandbox_id, connection_config=config)
    except SandboxGone:
        raise
    except Exception as exc:
        if getattr(exc, "status_code", None) in (404, 410):
            raise SandboxGone(f"opensandbox {sandbox_id} is gone") from exc
        logger.warning(
            "sandbox.connect_failed provider=opensandbox sandbox_id=%s error=%s",
            sandbox_id,
            type(exc).__name__,
        )
        raise
    finally:
        _quiet_retries.reset(quiet)
        _expect_gone.reset(token)
    return _opensandbox_session(remote, skills_digest=record.get("skills_digest", ""))


def _kill_opensandbox(record: dict[str, str]) -> None:
    try:
        session = _attach_opensandbox(record)
    except SandboxGone:
        logger.info(
            "sandbox.already_gone provider=opensandbox sandbox_id=%s",
            record.get("sandbox_id"),
        )
        return
    session.destroy()


# New providers are new rows. Each starter reads only its own SANDBOX_* keys.
_PROVIDERS: dict[str, Callable[[], SandboxSession]] = {
    "agentcore": _start_agentcore,
    "opensandbox": _start_opensandbox,
}
_ATTACH: dict[str, Callable[[dict[str, str]], SandboxSession]] = {
    "agentcore": _attach_agentcore,
    "opensandbox": _attach_opensandbox,
}
_KILL: dict[str, Callable[[dict[str, str]], None]] = {
    "agentcore": _kill_agentcore,
    "opensandbox": _kill_opensandbox,
}


def attach_session(record: dict[str, str]) -> SandboxSession:
    """Reconnect to the sandbox *record* names. Raise when it is not usable."""
    provider = record.get("provider", "")
    attach = _ATTACH.get(provider)
    if attach is None:
        raise SandboxError(f"cannot attach sandbox provider {provider!r}")
    return attach(record)


def kill_record(record: dict[str, str]) -> None:
    """Stop the sandbox *record* names. Already gone is success."""
    provider = record.get("provider", "")
    kill = _KILL.get(provider)
    if kill is None:
        logger.warning("sandbox.kill_unknown_provider provider=%s", provider)
        return
    kill(record)


def _skills_directory() -> Optional[Path]:
    """Skills folder for this process.

    ``None`` when neither ``EVE_AGENT_SKILLS_DIR`` nor the deep graph's
    ``skills_dir`` is available, so the upload is skipped and the turn still
    runs.
    """
    from src.services.agents.core.skills import packaged_skills_dir

    return packaged_skills_dir()


def _skill_files(root: Path) -> list[tuple[str, bytes]]:
    if not root.is_dir():
        return []
    files: list[tuple[str, bytes]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        files.append((f"{_SKILLS_PREFIX}{relative}", path.read_bytes()))
    return files


def _resolved_skill_files(skills_root: Optional[Path]) -> list[tuple[str, bytes]]:
    root = _skills_directory() if skills_root is None else skills_root
    if root is None:
        return []
    return _skill_files(root)


def _skills_digest(files: list[tuple[str, bytes]]) -> str:
    """sha256 over the ``(sandbox path, bytes)`` list, in upload order."""
    digest = hashlib.sha256()
    for path, content in files:
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


def _upload_error(response: Any) -> Optional[str]:
    if isinstance(response, dict):
        return response.get("error")
    return getattr(response, "error", None)


def _upload_files(backend: Any, files: list[tuple[str, bytes]]) -> None:
    """Copy *files* to ``/skills/`` inside the sandbox. One ``upload_files`` call."""
    if not files:
        return
    responses = list(backend.upload_files(files))
    if len(responses) != len(files):
        raise SandboxError(
            f"skills upload returned {len(responses)} results for {len(files)} files"
        )
    failed = [
        f"{path}: {_upload_error(response)}"
        for (path, _content), response in zip(files, responses)
        if _upload_error(response)
    ]
    if failed:
        raise SandboxError("skills upload failed: " + "; ".join(failed))


def _upload_skills(backend: Any, skills_root: Optional[Path] = None) -> str:
    """Copy the skills tree to ``/skills/`` and return its digest.

    *skills_root* is the merged tree for this turn. ``None`` keeps the packaged
    folder from the deep graph. Any error closes nothing by itself; the caller
    closes the session so a turn never starts with half the skills.
    """
    files = _resolved_skill_files(skills_root)
    digest = _skills_digest(files)
    _upload_files(backend, files)
    return digest


def _clear_skills(backend: Any) -> None:
    """Remove ``/skills`` so a changed tree does not leave files behind.

    The AgentCore shell rewrite points ``/skills`` at the directory the upload
    used. A backend without ``execute`` cannot drop stale files, so the caller
    replaces the sandbox instead of guessing.
    """
    execute = getattr(backend, "execute", None)
    if execute is None:
        raise SandboxError("sandbox cannot clear /skills/")
    result = execute("rm -rf /skills")
    exit_code = getattr(result, "exit_code", None)
    if exit_code not in (0, None):
        raise SandboxError(f"failed to clear /skills/: exit {exit_code}")


def sync_skills(session: SandboxSession, skills_root: Optional[Path] = None) -> str:
    """Make *session* hold *skills_root*. Return the digest now in the sandbox.

    The same digest skips the upload. A different one clears ``/skills`` first,
    so a skill that left the tree does not stay in the sandbox.
    """
    files = _resolved_skill_files(skills_root)
    digest = _skills_digest(files)
    if digest == session.skills_digest:
        return digest
    _clear_skills(session.backend)
    _upload_files(session.backend, files)
    session.skills_digest = digest
    return digest


def _rewrite_skills_command(command: str, cwd: str) -> str:
    """Point shell paths at the directory ``writeFiles`` used.

    A cwd of ``/`` already stores the upload at the absolute path. A command
    that already names ``cwd`` plus one prefix is left alone for that prefix,
    so a second call does not nest it. Only a whole path token is replaced,
    longest prefix first.
    """
    if not cwd or cwd == "/":
        return command
    for prefix in _SHELL_PREFIXES:
        root = f"{cwd}{prefix}"
        if root in command:
            continue
        command = _replace_path_token(command, prefix, root)
    return command


def _replace_path_token(command: str, prefix: str, root: str) -> str:
    """Replace ``prefix`` only where it begins a path token.

    The character before the match is not a path character, and the token
    continues with ``/`` or ends. ``/skills`` inside
    ``/large_tool_results/skills_1`` is left alone.
    """
    pattern = re.compile(
        rf"(?<![\w./\-]){re.escape(prefix)}(?=/|(?![\w./\-]))"
    )
    return pattern.sub(lambda _match: root, command)


def _bind_skills_shell(backend: Any) -> Any:
    """Rewrite ``/skills``, ``/artifacts``, and ``/large_tool_results`` in ``execute``.

    The object stays an ``AgentCoreSandbox``. ``CompositeBackend`` checks that
    type before it forwards a shell command, so a wrapper class would make
    ``execute`` raise.
    """
    original = getattr(backend, "execute", None)
    if original is None:
        return backend
    backend._eve_raw_execute = original
    cwd_box: dict[str, Optional[str]] = {"cwd": None}

    def execute(command: str, *args: Any, **kwargs: Any):
        cwd = cwd_box["cwd"]
        if cwd is None:
            probe = original("pwd")
            text = (getattr(probe, "output", "") or "").strip()
            cwd = text.splitlines()[-1].rstrip("/") if text else ""
            cwd_box["cwd"] = cwd
        return original(_rewrite_skills_command(command, cwd or ""), *args, **kwargs)

    backend.execute = execute
    return backend


# Sibling directories under the session root. The shell starts in workspace.
# The others stay beside it, not inside it.
_SIBLING_ROOTS = (
    "/skills",
    "/large_tool_results",
    "/artifacts",
    "/workspace",
    "/conversation_history",
)


def place_in_workspace(path: str, session_cwd: str = "") -> str:
    """Point a model path at ``workspace/`` unless it names a sibling directory.

    A path the provider already resolved under the session root is unchanged.
    ``..`` is unchanged so a caller can refuse it instead of escaping.
    """
    raw = str(path or "").strip()
    if not raw:
        return raw
    root = session_cwd.rstrip("/")
    if root and (raw == root or raw.startswith(root + "/")):
        return raw
    if ".." in raw.split("/"):
        return raw
    virtual = raw if raw.startswith("/") else f"/{raw.lstrip('./')}"
    for prefix in _SIBLING_ROOTS:
        if virtual == prefix or virtual.startswith(prefix + "/"):
            return virtual
    return "/workspace/" + raw.lstrip("./")


def _wrap_leading_path(
    method: Any, session_cwd: str, index: int, keyword: str
) -> Any:
    def wrapped(*args: Any, **kwargs: Any):
        if isinstance(kwargs.get(keyword), str):
            kwargs[keyword] = place_in_workspace(kwargs[keyword], session_cwd)
        elif len(args) > index and isinstance(args[index], str):
            args = list(args)
            args[index] = place_in_workspace(args[index], session_cwd)
            args = tuple(args)
        return method(*args, **kwargs)

    return wrapped


# The model names the workspace three ways: ``x`` (relative to the shell,
# which starts there), ``workspace/x`` (as the file tools accept it), and
# ``/workspace/x`` (the virtual path the file tools report). The command text
# is never edited: the other spellings resolve on disk instead. A relative
# self-link ``workspace -> .`` inside the workspace makes ``workspace/x``,
# ``./workspace/x`` and ``cd workspace`` land in the workspace, also inside a
# script. Where the session root is not ``/`` (AgentCore), a link
# ``/workspace`` to the real directory is tried; a sandbox that does not let
# the process write ``/`` keeps ``/workspace`` unresolved in the shell.
SELF_LINK = "workspace"
_VIRTUAL_WORKSPACE = "/workspace"


def _self_link_shell(target: str) -> str:
    """Shell that creates the self-link once. A failure never stops the caller."""
    link = f"{target}/{SELF_LINK}"
    return f"{{ [ -e {link} ] || ln -s . {link}; }} 2>/dev/null"


def workspace_shell_command(command: str, workspace_dir: Optional[str]) -> str:
    """Run a model command in its own shell from the workspace directory.

    ``workspace_dir`` is the real directory where the file tools store a
    relative path; without it the shell enters ``workspace`` relative to
    where it starts. The command is one quoted argument to ``sh -c`` and is
    passed byte for byte, so its quotes, ``$``, comments and heredocs reach
    that shell unchanged.
    """
    target = shlex.quote(workspace_dir) if workspace_dir else SELF_LINK
    return (
        f"mkdir -p {target} && {_self_link_shell(target)}; "
        f"cd {target} && exec sh -c {shlex.quote(command)}"
    )


def _layout_shell(workspace_dir: Optional[str]) -> str:
    """Create the workspace, its self-link and, if allowed, ``/workspace``."""
    target = shlex.quote(workspace_dir) if workspace_dir else SELF_LINK
    shell = f"mkdir -p {target} && {_self_link_shell(target)}"
    if workspace_dir and workspace_dir != _VIRTUAL_WORKSPACE:
        alias = shlex.quote(_VIRTUAL_WORKSPACE)
        shell += f"; {{ [ -e {alias} ] || ln -s {target} {alias}; }} 2>/dev/null"
    return f"{shell}; true"


def _probe_cwd(backend: Any, raw_execute: Any) -> str:
    """Session cwd, read once more through the raw shell after a failed probe.

    Caches the value on the provider (``_cwd``), so its lazy ``pwd`` never
    runs through the workspace shell and caches ``<cwd>/workspace``.
    Returns ``""`` when the shell cannot tell.
    """
    try:
        result = raw_execute("pwd")
    except Exception:
        logger.warning("workspace.cwd_retry_failed", exc_info=True)
        return ""
    text = (getattr(result, "output", "") or "").strip()
    if getattr(result, "exit_code", 0) not in (0, None) or not text:
        return ""
    cwd = text.splitlines()[-1].rstrip("/") or "/"
    if hasattr(backend, "_cwd"):
        backend._cwd = cwd.rstrip("/")
    return cwd


def prepare_workspace_layout(backend: Any) -> Any:
    """Create ``workspace`` and start shell commands inside it.

    ``skills``, ``large_tool_results``, and ``artifacts`` stay siblings
    of that directory. File paths that do not name one of them are stored
    under ``workspace``. A second call does nothing.

    The session root is the provider cwd when it reports one (AgentCore) and
    ``/`` otherwise (OpenSandbox stores ``/workspace/x`` as an absolute path).
    Internal commands (workspace listing, process baseline) keep running from
    that root through ``_eve_raw_execute``.
    """
    if backend is None or getattr(backend, "_eve_workspace_ready", False):
        return backend
    session_cwd = ""
    root: Optional[str] = "/"
    execute = getattr(backend, "execute", None)
    get_cwd = getattr(backend, "_get_cwd", None)
    if get_cwd is not None:
        try:
            session_cwd = str(get_cwd() or "").rstrip("/")
            root = session_cwd or "/"
        except Exception:
            logger.warning("workspace.cwd_probe_failed", exc_info=True)
            raw = getattr(backend, "_eve_raw_execute", None) or execute
            probed = _probe_cwd(backend, raw) if raw is not None else ""
            session_cwd = probed.rstrip("/")
            root = probed or None
    workspace_dir = f"{root.rstrip('/')}/workspace" if root else None
    if execute is not None:
        if getattr(backend, "_eve_raw_execute", None) is None:
            if root:
                root_q = shlex.quote(root)

                def execute_from_root(command: str, *args: Any, **kwargs: Any):
                    return execute(f"cd {root_q} && {command}", *args, **kwargs)

                backend._eve_raw_execute = execute_from_root
            else:
                backend._eve_raw_execute = execute
        try:
            execute(_layout_shell(workspace_dir))
        except Exception:
            logger.warning("workspace.mkdir_failed", exc_info=True)

        def execute_in_workspace(command: str, *args: Any, **kwargs: Any):
            return execute(
                workspace_shell_command(command, workspace_dir),
                *args,
                **kwargs,
            )

        backend.execute = execute_in_workspace
    for name, index, keyword in (
        ("write", 0, "file_path"),
        ("read", 0, "file_path"),
        ("edit", 0, "file_path"),
        ("ls", 0, "path"),
        ("grep", 1, "path"),
        ("glob", 1, "path"),
    ):
        method = getattr(backend, name, None)
        if method is not None:
            setattr(
                backend,
                name,
                _wrap_leading_path(method, session_cwd, index, keyword),
            )
    upload = getattr(backend, "upload_files", None)
    if upload is not None:

        def upload_files(files: list, *args: Any, **kwargs: Any):
            mapped = []
            for item in files:
                path, data = item[0], item[1]
                mapped.append(
                    (place_in_workspace(path, session_cwd), data, *item[2:])
                )
            return upload(mapped, *args, **kwargs)

        backend.upload_files = upload_files
    download = getattr(backend, "download_files", None)
    if download is not None:
        backend._eve_download_files = download

        def download_files(paths: list, *args: Any, **kwargs: Any):
            return download(
                [place_in_workspace(path, session_cwd) for path in paths],
                *args,
                **kwargs,
            )

        backend.download_files = download_files
    backend._eve_workspace_ready = True
    return backend


def _open_sync(skills_root: Optional[Path] = None) -> SandboxSession:
    provider = sandbox_provider()
    start = _PROVIDERS.get(provider)
    if start is None:
        known = ", ".join(sorted(_PROVIDERS))
        raise SandboxError(f"unknown SANDBOX_PROVIDER {provider!r}; expected {known}")
    session = start()
    try:
        session.skills_digest = _upload_skills(session.backend, skills_root)
        from src.services.agents.core.workspace import capture_process_baseline

        session.metadata["process_baseline"] = capture_process_baseline(
            session.backend
        )
    except Exception:
        try:
            session.destroy()
        except Exception:
            logger.exception("Failed to close the sandbox after a skills upload error")
        raise
    return session


async def open_sandbox(skills_root: Optional[Path] = None) -> Optional[SandboxSession]:
    """Start a sandbox for a turn that has no conversation, and upload skills.

    ``None`` when ``SANDBOX_PROVIDER`` is unset or ``none``. *skills_root*
    ``None`` uploads the packaged skills folder. The provider calls block, so
    they run on a worker thread. The caller destroys this sandbox when the
    turn ends. A conversation uses ``sandbox_pool`` instead.
    """
    provider = sandbox_provider()
    if provider in ("", "none"):
        return None
    return await asyncio.to_thread(_open_sync, skills_root)


async def close_sandbox(session: Optional[SandboxSession]) -> None:
    """Destroy *session*. ``None`` and a second close are no-ops.

    A close failure is logged and swallowed so it cannot replace the turn's
    own result.
    """
    if session is None:
        return
    try:
        await asyncio.to_thread(session.close)
    except Exception:
        logger.exception("Failed to close the sandbox")


def backend_for_compile(sandbox_backend: Any, tool_results: Any = None) -> Any:
    """Composite whose default is the sandbox.

    ``/skills/`` is not routed: the turn uploaded that tree into the sandbox,
    and the file tools and the shell both read it there. ``/conversation_history/``
    stays in graph state. ``/large_tool_results/`` uses ``tool_results`` when
    the turn has a conversation. With no conversation that prefix stays on the
    sandbox. A second ``StateBackend`` would share the files channel, and the
    composite would strip the prefix, so ``/large_tool_results/{id}`` and
    ``/{id}`` would be the same key.

    This composite stores no permissions. The runner passes
    ``large_tool_result_permissions`` when the route exists. Those paths sit
    under the route, which is what an execute backend accepts. ``/skills/``
    is not a route here, so a skills rule is not added. Offload calls
    ``awrite`` and does not consult the deny, so a new id still writes.
    """
    from deepagents.backends import CompositeBackend, StateBackend

    routes: dict[str, Any] = {_CONVERSATION_HISTORY: StateBackend()}
    if tool_results is not None:
        routes[_LARGE_TOOL_RESULTS] = tool_results
    return CompositeBackend(default=sandbox_backend, routes=routes)


def large_tool_result_permissions() -> list:
    """Deny model writes on ``/large_tool_results/**``. Reads stay allowed.

    The pattern starts with the route prefix, so a composite whose default
    can execute accepts it.
    """
    from deepagents.middleware.filesystem import FilesystemPermission

    return [
        FilesystemPermission(
            operations=["write"],
            paths=["/large_tool_results/**"],
            mode="deny",
        )
    ]
