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
import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional

from src.config import AGENTIC_TIMEOUT

logger = logging.getLogger(__name__)

_PROVIDER_ENV = "SANDBOX_PROVIDER"
_AGENTCORE_TTL_MAX = 28800
_REUSE_MARGIN_S = 60
_DEFAULT_AGENTCORE_ID = "aws.codeinterpreter.v1"
_DEFAULT_OPENSANDBOX_IMAGE = "python:3.12"
_SKILLS_PREFIX = "/skills/"
_OFFLOAD_ROUTES = ("/large_tool_results/", "/conversation_history/")


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
    """Provider expiry while a turn is in flight: the idle window plus the turn cap."""
    return sandbox_idle_ttl_s() + AGENTIC_TIMEOUT


def sandbox_reuse_limit_s() -> int:
    """Minimum hard-TTL left before a session is worth reusing."""
    return AGENTIC_TIMEOUT + _REUSE_MARGIN_S


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
        _bind_skills_shell(sandbox),
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
        OpenSandboxBackend(sandbox=remote),
        provider="opensandbox",
        sandbox_id=str(getattr(remote, "id", "") or ""),
        destroy=remote.destroy,
        detach=remote.close,
        renew=renew,
        skills_digest=skills_digest,
    )


def _start_opensandbox() -> SandboxSession:
    from datetime import timedelta

    from opensandbox import SandboxSync

    image, config = _opensandbox_connection()
    remote = SandboxSync.create(
        image,
        connection_config=config,
        timeout=timedelta(seconds=sandbox_busy_ttl_s()),
    )
    return _opensandbox_session(remote)


def _attach_opensandbox(record: dict[str, str]) -> SandboxSession:
    from opensandbox import SandboxSync

    _image, config = _opensandbox_connection()
    sandbox_id = record["sandbox_id"]
    try:
        remote = SandboxSync.connect(sandbox_id, connection_config=config)
    except SandboxGone:
        raise
    except Exception as exc:
        if getattr(exc, "status_code", None) in (404, 410):
            raise SandboxGone(f"opensandbox {sandbox_id} is gone") from exc
        raise
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
    """Point a ``/skills`` shell path at the directory ``writeFiles`` used.

    A cwd of ``/`` already stores the upload at ``/skills``. A command that
    already names ``cwd/skills`` is left alone so a second call does not
    nest the prefix.
    """
    if not cwd or cwd == "/":
        return command
    root = f"{cwd}/skills"
    if root in command:
        return command
    return command.replace("/skills", root)


def _bind_skills_shell(backend: Any) -> Any:
    """Rewrite ``/skills`` in ``execute`` and leave every other method in place.

    The object stays an ``AgentCoreSandbox``. ``CompositeBackend`` checks that
    type before it forwards a shell command, so a wrapper class would make
    ``execute`` raise.
    """
    original = getattr(backend, "execute", None)
    if original is None:
        return backend
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


def _open_sync(skills_root: Optional[Path] = None) -> SandboxSession:
    provider = sandbox_provider()
    start = _PROVIDERS.get(provider)
    if start is None:
        known = ", ".join(sorted(_PROVIDERS))
        raise SandboxError(f"unknown SANDBOX_PROVIDER {provider!r}; expected {known}")
    session = start()
    try:
        session.skills_digest = _upload_skills(session.backend, skills_root)
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


def backend_for_compile(sandbox_backend: Any) -> Any:
    """Composite whose default is the sandbox.

    ``/skills/`` is not routed: the turn uploaded that tree into the sandbox,
    and the file tools and the shell both read it there. The offload prefixes
    stay in graph state. An agents build that still evicts large messages then
    keeps those paths after this sandbox is destroyed.
    """
    from deepagents.backends import CompositeBackend, StateBackend

    return CompositeBackend(
        default=sandbox_backend,
        routes={route: StateBackend() for route in _OFFLOAD_ROUTES},
    )
