"""Per-request context for retrieval tool results.

``RetrievalContextInterceptor`` (see ``retrieval_ingestion.py``) is constructed
once per ``MultiServerMCPClient``, which may be reused across requests and
users, so it never keeps per-request state on itself. Each agentic entry point
opens a context before invoking the graph, the interceptor appends the full
documents it saw to it, and the runner reads them back once the run completes.
Same pattern as ``artifact_context.py``.
"""

import contextlib
import contextvars
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple


@dataclass
class RetrievalCall:
    """One retrieval tool call seen by the interceptor.

    ``tool_name`` is the graph facing name (``<server>_<tool>``, lowercased),
    so the runner can keep the same RAG tool gate it applies to ToolMessages.
    ``documents`` are the complete Document-shaped records (id, score,
    payload, ...) as ``extract_document_data`` builds them; ``latencies`` is the
    ``latencies`` object the /retrieve endpoint returned, if any.
    """

    tool_name: str
    documents: List[Dict[str, Any]] = field(default_factory=list)
    latencies: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalRequestContext:
    """Retrieval calls made during the current agentic run.

    The model only sees a reduced copy of each result; the UI and the
    persisted message get the full documents from ``calls``.
    """

    calls: List[RetrievalCall] = field(default_factory=list)

    @property
    def documents(self) -> List[Dict[str, Any]]:
        """Every document of every call, in call order."""
        return [doc for call in self.calls for doc in call.documents]

    def for_tools(self, tool_names: set) -> List[RetrievalCall]:
        """Calls made by one of ``tool_names`` (graph facing, lowercased)."""
        return [call for call in self.calls if call.tool_name in tool_names]


_retrieval_context: contextvars.ContextVar[Optional[RetrievalRequestContext]] = (
    contextvars.ContextVar("retrieval_context", default=None)
)


def set_retrieval_context() -> Tuple[RetrievalRequestContext, contextvars.Token]:
    """Start a retrieval context for the current task; returns (context, reset token).

    Callers MUST reset with the returned token in a ``finally`` block.
    """
    ctx = RetrievalRequestContext()
    token = _retrieval_context.set(ctx)
    return ctx, token


def get_retrieval_context() -> Optional[RetrievalRequestContext]:
    """Return the current task's retrieval context, or None if unset.

    None means the interceptor passes tool results through untouched: a tool
    call made outside the agentic pipeline, or a test that sets no context.
    """
    return _retrieval_context.get()


def reset_retrieval_context(token: contextvars.Token) -> None:
    """Reset the contextvar to its state before the matching ``set_retrieval_context``."""
    _retrieval_context.reset(token)


@contextlib.contextmanager
def retrieval_context() -> Iterator[RetrievalRequestContext]:
    """Context manager form of set/reset."""
    ctx, token = set_retrieval_context()
    try:
        yield ctx
    finally:
        reset_retrieval_context(token)
