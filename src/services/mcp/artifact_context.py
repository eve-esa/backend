"""Per-request context for MCP artifact ingestion.

``ArtifactInterceptor`` (see ``artifact_ingestion.py``) is constructed once per
``MultiServerMCPClient`` — which may be reused across requests/users — so it
must never read per-request state (who's asking, which conversation/message)
from its own constructor or instance attributes. Instead, each entry point
that runs the agentic pipeline stashes the current request's identity in this
module's ``contextvars.ContextVar`` before invoking the graph, and the
interceptor reads it back at tool-call time. Native to asyncio: each task gets
its own copy, so concurrent requests never see each other's context.
"""

import contextlib
import contextvars
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple


@dataclass
class ArtifactRequestContext:
    """Identity of the request currently running the agentic pipeline.

    ``collected_artifact_ids`` is mutated in place by the interceptor as it
    persists artifacts, so callers can read it back after the run completes
    (before the context is reset) to link them onto the Message.
    """

    user_id: str
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None
    collected_artifact_ids: List[str] = field(default_factory=list)


_artifact_context: contextvars.ContextVar[Optional[ArtifactRequestContext]] = (
    contextvars.ContextVar("artifact_context", default=None)
)


def set_artifact_context(
    user_id: str,
    conversation_id: Optional[str] = None,
    message_id: Optional[str] = None,
) -> Tuple[ArtifactRequestContext, contextvars.Token]:
    """Start an artifact context for the current task; returns (context, reset token).

    Callers MUST reset with the returned token (typically in a ``finally``
    block) once the agentic run completes, regardless of outcome.
    """
    ctx = ArtifactRequestContext(
        user_id=user_id, conversation_id=conversation_id, message_id=message_id
    )
    token = _artifact_context.set(ctx)
    return ctx, token


def get_artifact_context() -> Optional[ArtifactRequestContext]:
    """Return the current task's artifact context, or None if unset.

    None means "no context" — the interceptor treats this as passthrough
    (e.g. a tool call made outside the agentic pipeline, or in a test that
    doesn't set one up).
    """
    return _artifact_context.get()


def reset_artifact_context(token: contextvars.Token) -> None:
    """Reset the contextvar to its state before the matching ``set_artifact_context``."""
    _artifact_context.reset(token)


@contextlib.contextmanager
def artifact_context(
    user_id: str,
    conversation_id: Optional[str] = None,
    message_id: Optional[str] = None,
) -> Iterator[ArtifactRequestContext]:
    """Context manager form of set/reset, for callers that prefer ``with``."""
    ctx, token = set_artifact_context(
        user_id, conversation_id=conversation_id, message_id=message_id
    )
    try:
        yield ctx
    finally:
        reset_artifact_context(token)
