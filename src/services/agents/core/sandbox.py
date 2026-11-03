"""Sandbox providers for one agent turn.

Every setting this service owns starts with ``SANDBOX_``. ``SANDBOX_PROVIDER``
picks the row. Values are read when a turn starts, not at import time, so a
``.env`` loaded later still applies. Credentials a provider SDK already reads
stay on that SDK's own names.

``none`` and an unset provider start nothing. The caller closes the handle
when the turn ends, including when the turn fails.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_PROVIDER_ENV = "SANDBOX_PROVIDER"
_AGENTCORE_TIMEOUT_S = 900
_DEFAULT_AGENTCORE_ID = "aws.codeinterpreter.v1"
_DEFAULT_OPENSANDBOX_IMAGE = "python:3.12"
_SKILLS_PREFIX = "/skills/"
_OFFLOAD_ROUTES = ("/large_tool_results/", "/conversation_history/")


class SandboxError(RuntimeError):
    """A sandbox could not be started or prepared."""


class SandboxSession:
    """One provider session and the backend the graph compiles with."""

    def __init__(self, backend: Any, close: Callable[[], None]) -> None:
        self.backend = backend
        self._close = close
        self._closed = False

    def close(self) -> None:
        """Release the remote session. A second call does nothing."""
        if self._closed:
            return
        self._closed = True
        self._close()


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SandboxError(f"{name} is required")
    return value


def _start_agentcore() -> SandboxSession:
    region = _require("SANDBOX_AGENTCORE_REGION")
    identifier = (
        os.environ.get("SANDBOX_AGENTCORE_IDENTIFIER", "").strip()
        or _DEFAULT_AGENTCORE_ID
    )
    from langchain_agentcore_codeinterpreter import AgentCoreSandbox

    sandbox = AgentCoreSandbox.create(
        region=region,
        identifier=identifier,
        session_timeout_seconds=_AGENTCORE_TIMEOUT_S,
    )
    # writeFiles stores "/skills/..." under the session cwd, and the process
    # cannot create a real "/skills". Skill text still calls that absolute
    # path, so the shell rewrites it to the directory the upload used.
    return SandboxSession(backend=_bind_skills_shell(sandbox), close=sandbox.close)


def _start_opensandbox() -> SandboxSession:
    domain = _require("SANDBOX_OPENSANDBOX_DOMAIN")
    protocol = os.environ.get("SANDBOX_OPENSANDBOX_PROTOCOL", "").strip() or "http"
    if protocol not in ("http", "https"):
        raise SandboxError("SANDBOX_OPENSANDBOX_PROTOCOL must be http or https")
    api_key = os.environ.get("SANDBOX_OPENSANDBOX_API_KEY", "").strip() or None
    image = (
        os.environ.get("SANDBOX_OPENSANDBOX_IMAGE", "").strip()
        or _DEFAULT_OPENSANDBOX_IMAGE
    )
    from langchain_opensandbox import OpenSandboxBackend
    from opensandbox import SandboxSync
    from opensandbox.config import ConnectionConfigSync

    remote = SandboxSync.create(
        image,
        connection_config=ConnectionConfigSync(
            domain=domain,
            protocol=protocol,
            api_key=api_key,
        ),
    )
    backend = OpenSandboxBackend(sandbox=remote)
    return SandboxSession(backend=backend, close=remote.destroy)


# New providers are new rows. Each starter reads only its own SANDBOX_* keys.
_PROVIDERS: dict[str, Callable[[], SandboxSession]] = {
    "agentcore": _start_agentcore,
    "opensandbox": _start_opensandbox,
}


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


def _upload_error(response: Any) -> Optional[str]:
    if isinstance(response, dict):
        return response.get("error")
    return getattr(response, "error", None)


def _upload_skills(backend: Any, skills_root: Optional[Path] = None) -> None:
    """Copy the host skills tree to ``/skills/`` inside the sandbox.

    *skills_root* is the merged tree for this turn. ``None`` keeps the packaged
    folder from the deep graph. One ``upload_files`` call. Any error closes
    nothing by itself; the caller closes the session so a turn never starts
    with half the skills.
    """
    root = _skills_directory() if skills_root is None else skills_root
    if root is None:
        return
    files = _skill_files(root)
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


def _open_sync(provider: str, skills_root: Optional[Path] = None) -> SandboxSession:
    start = _PROVIDERS.get(provider)
    if start is None:
        known = ", ".join(sorted(_PROVIDERS))
        raise SandboxError(f"unknown SANDBOX_PROVIDER {provider!r}; expected {known}")
    session = start()
    try:
        _upload_skills(session.backend, skills_root)
    except Exception:
        try:
            session.close()
        except Exception:
            logger.exception("Failed to close the sandbox after a skills upload error")
        raise
    return session


async def open_sandbox(skills_root: Optional[Path] = None) -> Optional[SandboxSession]:
    """Start this turn's sandbox and upload skills into it.

    ``None`` when ``SANDBOX_PROVIDER`` is unset or ``none``. *skills_root*
    ``None`` uploads the packaged skills folder. The provider calls block, so
    they run on a worker thread.
    """
    provider = os.environ.get(_PROVIDER_ENV, "").strip().lower()
    if provider in ("", "none"):
        return None
    return await asyncio.to_thread(_open_sync, provider, skills_root)


async def close_sandbox(session: Optional[SandboxSession]) -> None:
    """Release *session*. ``None`` and a second close are no-ops.

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
