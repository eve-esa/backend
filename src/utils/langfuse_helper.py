"""Langfuse observability helpers (SDK v4).

Langfuse SDK v4 is OpenTelemetry-based. The correct way to attach user_id /
session_id to a LangChain / LangGraph trace is:

  1. Wrap the invocation in ``start_as_current_observation(as_type='span')``
     to create a root OTel span (note: ``start_as_current_span`` does NOT exist
     in v4 — use ``start_as_current_observation``).
  2. Use ``propagate_attributes(user_id=..., session_id=..., tags=...)``
     so all nested observations (including those from CallbackHandler) inherit
     the trace-level attributes.  The span itself does NOT have
     ``update_trace()``; ``propagate_attributes`` is the only way.

Use ``langfuse_context(...)`` as a context manager around every LangGraph call.
Use ``get_callbacks()`` to obtain the ``[CallbackHandler()]`` for config['callbacks'].

All functions degrade gracefully when Langfuse is unavailable.
"""

import contextlib
import logging
import os
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

_langfuse_available = False
_langfuse_client = None  # singleton, used only for flush()

try:
    from langfuse import Langfuse as _Langfuse, get_client as _get_langfuse_client
    from langfuse import propagate_attributes as _propagate_attributes
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
    _langfuse_available = True
except Exception:
    _Langfuse = None  # type: ignore
    _get_langfuse_client = None  # type: ignore
    _propagate_attributes = None  # type: ignore
    LangfuseCallbackHandler = None  # type: ignore


def _ensure_langfuse_host() -> None:
    """Override LANGFUSE_BASE_URL with the Docker-internal URL when available.

    load_dotenv(override=True) in config.py replaces the docker-compose value
    LANGFUSE_BASE_URL=http://langfuse-web:3000 with http://localhost:3000.
    LANGFUSE_INTERNAL_URL is only set inside docker-compose so dotenv never
    touches it, giving us the correct in-network address.
    """
    internal = os.environ.get("LANGFUSE_INTERNAL_URL")
    if internal:
        os.environ["LANGFUSE_BASE_URL"] = internal


def is_langfuse_enabled() -> bool:
    """Return True if the SDK is installed and keys are configured."""
    if not _langfuse_available:
        return False
    from src.config import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
    return bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)


def flush() -> None:
    """Flush pending traces to Langfuse.

    In SDK v3, flushing is managed on the global singleton client.
    """
    global _langfuse_client
    if not is_langfuse_enabled():
        return
    try:
        _ensure_langfuse_host()
        if _langfuse_client is None:
            _langfuse_client = _Langfuse()
        _langfuse_client.flush()
    except Exception as exc:
        logger.debug("Langfuse flush error: %s", exc)


def get_callbacks() -> List[Any]:
    """Return ``[LangfuseCallbackHandler()]`` for config['callbacks'].

    In SDK v3 the handler takes no constructor args — trace attributes
    (user_id, session_id, tags) must be set via a ``langfuse_context`` block
    that wraps the LangGraph invocation.

    Returns an empty list when Langfuse is unavailable so callers never branch.
    """
    if not is_langfuse_enabled():
        return []
    try:
        _ensure_langfuse_host()
        return [LangfuseCallbackHandler()]
    except Exception as exc:
        logger.warning("Could not create Langfuse callback handler: %s", exc)
        return []


@contextlib.contextmanager
def langfuse_context(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    trace_name: str = "langchain-call",
) -> Generator[Any, None, None]:
    """Sync context manager that attaches Langfuse trace attributes.

    Wrap every LangGraph ``ainvoke`` / ``astream`` call with this so that
    user_id and session_id appear in the Langfuse UI.

    Works in async contexts (asyncio propagates contextvars to child tasks),
    so it is safe to use around ``async for`` loops inside async generators.

    If Langfuse is unavailable the block runs without tracing overhead.

    Langfuse SDK v4:
      - ``start_as_current_observation(as_type='span')`` creates the root span
        (note: ``start_as_current_span`` does not exist in v4)
      - ``propagate_attributes(user_id, session_id, tags)`` is the correct way
        to set trace-level attributes so they appear in the UI

    Example::

        with langfuse_context(user_id=uid, session_id=conv_id, trace_name="stream"):
            async for chunk, meta in graph.astream(..., config=config):
                ...
    """
    if not is_langfuse_enabled() or _get_langfuse_client is None:
        yield None
        return

    # Separate initialisation from execution so that exceptions raised *inside*
    # the with-block propagate normally.  Yielding after throw() violates the
    # @contextmanager protocol and causes RuntimeError("generator didn't stop
    # after throw()") which breaks the caller's async generator.
    try:
        _ensure_langfuse_host()
        lf = _get_langfuse_client()
    except Exception as exc:
        logger.warning("Langfuse context setup failed, tracing disabled: %s", exc)
        yield None
        return

    prop_kwargs: Dict[str, Any] = {}
    if user_id:
        prop_kwargs["user_id"] = user_id
    if session_id:
        prop_kwargs["session_id"] = session_id
    if tags:
        prop_kwargs["tags"] = tags
    if trace_name:
        prop_kwargs["trace_name"] = trace_name

    # Any exception raised inside the caller's with-block will propagate through
    # these context managers and out of this generator cleanly (no yield after throw).
    with lf.start_as_current_observation(name=trace_name, as_type="span") as span:
        if prop_kwargs and _propagate_attributes is not None:
            with _propagate_attributes(**prop_kwargs):
                yield span
        else:
            yield span
