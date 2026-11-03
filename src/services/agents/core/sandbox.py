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
    return SandboxSession(backend=sandbox, close=sandbox.close)


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
    """Skills folder from the deep graph.

    ``None`` when that graph cannot be imported (agents v0.5.5 has none), so
    the upload is skipped and the turn still runs.
    """
    try:
        from agents.graphs.deep.graph import skills_dir
    except ImportError:
        return None
    return Path(skills_dir())


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


def _upload_skills(backend: Any) -> None:
    """Copy the host skills tree to ``/skills/`` inside the sandbox.

    One ``upload_files`` call. Any error closes nothing by itself; the caller
    closes the session so a turn never starts with half the skills.
    """
    root = _skills_directory()
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


def _open_sync(provider: str) -> SandboxSession:
    start = _PROVIDERS.get(provider)
    if start is None:
        known = ", ".join(sorted(_PROVIDERS))
        raise SandboxError(f"unknown SANDBOX_PROVIDER {provider!r}; expected {known}")
    session = start()
    try:
        _upload_skills(session.backend)
    except Exception:
        try:
            session.close()
        except Exception:
            logger.exception("Failed to close the sandbox after a skills upload error")
        raise
    return session


async def open_sandbox() -> Optional[SandboxSession]:
    """Start this turn's sandbox and upload skills into it.

    ``None`` when ``SANDBOX_PROVIDER`` is unset or ``none``. The provider
    calls block, so they run on a worker thread.
    """
    provider = os.environ.get(_PROVIDER_ENV, "").strip().lower()
    if provider in ("", "none"):
        return None
    return await asyncio.to_thread(_open_sync, provider)


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
