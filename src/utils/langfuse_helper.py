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

``record_error_kind`` is an optional overlay on Mongo ``error_logs``. It
attaches the same ``kind`` taxonomy as a Langfuse EVENT (and ``kind:<name>``
tags) when ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY`` are set.
Without keys it is a no-op. Hosted can enable later with URL + keys.
See https://langfuse.com/docs/observability/sdk/instrumentation
"""

import contextlib
import logging
import os
from contextvars import ContextVar
from typing import Any, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)

_langfuse_available = False
_langfuse_client = None  # kept for tests that patch this name; flush uses get_client()

# Survives LangGraph hopping off the OTel current span, and leftover persist
# after ``langfuse_context`` exits. Keyed by Langfuse session_id (= conversation).
_trace_id_ctx: ContextVar[Optional[str]] = ContextVar("lf_trace_id", default=None)
_observation_id_ctx: ContextVar[Optional[str]] = ContextVar(
    "lf_observation_id", default=None
)
_root_span_ctx: ContextVar[Any] = ContextVar("lf_root_span", default=None)
_callback_handler_ctx: ContextVar[Any] = ContextVar("lf_cb_handler", default=None)
_session_id_ctx: ContextVar[Optional[str]] = ContextVar("lf_session_id", default=None)
_trace_by_session: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
# Latest LangGraph node spans, kept after CallbackHandler pops them on_chain_end.
_node_spans_by_session: Dict[str, Dict[str, List[Any]]] = {}

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


if LangfuseCallbackHandler is not None:

    class _NodeKindCallbackHandler(LangfuseCallbackHandler):
        """Remember node spans before Langfuse pops them on_chain_end/error."""

        def _attach_observation(self, run_id, observation):  # type: ignore[no-untyped-def]
            super()._attach_observation(run_id, observation)
            _remember_span(observation)

else:  # pragma: no cover
    _NodeKindCallbackHandler = None  # type: ignore


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

    Uses ``get_client()`` — the same singleton ``create_event`` writes to.
    A second ``Langfuse()`` instance would leave those events unexported.
    """
    if not is_langfuse_enabled() or _get_langfuse_client is None:
        return
    try:
        _ensure_langfuse_host()
        _get_langfuse_client().flush()
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
        handler = (
            _NodeKindCallbackHandler()
            if _NodeKindCallbackHandler is not None
            else LangfuseCallbackHandler()
        )
        _callback_handler_ctx.set(handler)
        return [handler]
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
        trace_id = getattr(span, "trace_id", None) or lf.get_current_trace_id()
        observation_id = getattr(span, "id", None) or lf.get_current_observation_id()
        token_t = _trace_id_ctx.set(trace_id)
        token_o = _observation_id_ctx.set(observation_id)
        token_s = _root_span_ctx.set(span)
        token_sess = _session_id_ctx.set(session_id)
        if session_id and trace_id:
            _trace_by_session[session_id] = (trace_id, observation_id)
            _node_spans_by_session[session_id] = {}
        try:
            if prop_kwargs and _propagate_attributes is not None:
                with _propagate_attributes(**prop_kwargs):
                    yield span
            else:
                yield span
        finally:
            with contextlib.suppress(Exception):
                lf.flush()
            _trace_id_ctx.reset(token_t)
            _observation_id_ctx.reset(token_o)
            _root_span_ctx.reset(token_s)
            _session_id_ctx.reset(token_sess)


_KIND_METADATA_KEYS = ("signal", "tool", "server", "attempt")


def _kind_event_metadata(
    kind: str,
    *,
    node: Optional[str] = None,
    graph: Optional[str] = None,
    source: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {"kind": kind}
    if node:
        metadata["node"] = node
    if graph:
        metadata["graph"] = graph
    if source:
        metadata["source"] = source
    if extra:
        for key in _KIND_METADATA_KEYS:
            value = extra.get(key)
            if value is not None:
                metadata[key] = value
    return metadata


def _event_kwargs(
    kind: str,
    *,
    node: Optional[str] = None,
    graph: Optional[str] = None,
    source: Optional[str] = None,
    description: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metadata = _kind_event_metadata(
        kind, node=node, graph=graph, source=source, extra=extra
    )
    tags = [f"kind:{kind}"]
    if node:
        tags.append(f"node:{node}")
    metadata["tags"] = tags
    kwargs: Dict[str, Any] = {
        "name": kind,
        "level": "WARNING" if kind == "retry" else "ERROR",
        "metadata": metadata,
    }
    status = (description or "").strip()[:500] or None
    if status:
        kwargs["status_message"] = status
    return kwargs


def _emit_as_otel_child(lf: Any, event_kwargs: Dict[str, Any], ctx: Dict[str, str]) -> bool:
    """Attach an event as a child span without Langfuse ``AS_ROOT``.

    ``Langfuse.create_event(trace_context=...)`` sets ``AS_ROOT`` so the UI
    lists timeout/retry as separate traces. Parent via OTel instead.
    https://langfuse.com/docs/observability/features/trace-ids-and-distributed-tracing
    """
    try:
        from opentelemetry import trace as otel_trace_api
    except Exception:
        return False
    trace_id = ctx.get("trace_id") or ""
    parent_span_id = ctx.get("parent_span_id") or ""
    try:
        int_trace_id = int(trace_id, 16)
        int_parent = int(parent_span_id, 16) if parent_span_id else 0
    except ValueError:
        return False
    if not int_parent:
        return False
    parent = otel_trace_api.NonRecordingSpan(
        otel_trace_api.SpanContext(
            trace_id=int_trace_id,
            span_id=int_parent,
            is_remote=False,
            trace_flags=otel_trace_api.TraceFlags(0x01),
        )
    )
    create_event = getattr(lf, "create_event", None)
    if not callable(create_event):
        return False
    with otel_trace_api.use_span(parent):
        create_event(**event_kwargs)
    return True


def _span_buckets() -> Dict[str, List[Any]]:
    session = _session_id_ctx.get()
    if not session:
        try:
            from src.utils.error_logger import get_conversation_context

            session = get_conversation_context()
        except Exception:
            session = None
    if not session:
        session = "_default"
    return _node_spans_by_session.setdefault(session, {})


def _remember_span(span: Any) -> None:
    name = _observation_name(span)
    if not name or name.startswith("__error_handler__"):
        return
    if name in ("LangGraph", "agentic_generation_stream", "agentic_generation"):
        return
    bucket = _span_buckets().setdefault(name, [])
    if not bucket or bucket[-1] is not span:
        bucket.append(span)


def _ingest_live_runs() -> None:
    handler = _callback_handler_ctx.get()
    runs = getattr(handler, "_runs", None) if handler is not None else None
    if not isinstance(runs, dict):
        return
    for span in runs.values():
        _remember_span(span)


def _observation_name(span: Any) -> Optional[str]:
    otel = getattr(span, "_otel_span", None)
    if otel is not None:
        name = getattr(otel, "name", None)
        if name:
            return str(name)
    name = getattr(span, "name", None)
    return str(name) if name else None


def _node_observation_span(node: Optional[str], *, kind: Optional[str] = None) -> Any:
    """Span to parent a kind event on.

    Retry is emitted at the *start* of attempt N, after the new AGENT span
    exists — that looks one-off (retry sits on the new attempt). Parent it
    to the previous ``node`` span (the attempt that actually failed).

    Timeout/fallback run from ``error_handler`` after CallbackHandler has
    already popped the node span from ``_runs``. Use the cached last span.
    """
    _ingest_live_runs()
    if not node:
        return None
    bucket = _span_buckets().get(node) or []
    if not bucket:
        return None
    if kind == "retry" and len(bucket) >= 2:
        span = bucket[-2]
    else:
        span = bucket[-1]
    if callable(getattr(span, "create_event", None)):
        return span
    return None


def record_error_kind(
    kind: Optional[str],
    *,
    node: Optional[str] = None,
    graph: Optional[str] = None,
    source: Optional[str] = None,
    description: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Record ``kind`` as a child EVENT of the LangGraph **node** span.

    CallbackHandler observations are not the OTel current span, so we look up
    the latest ``_runs`` entry named ``node`` (``agent``, ``agent_fallback``,
    …) and call its ``create_event``. Never pass ``trace_context`` (that sets
    ``AS_ROOT`` and lists timeout as its own trace). Drop the event rather
    than opening a new root. Never raises.
    """
    if not kind or not is_langfuse_enabled() or _get_langfuse_client is None:
        return
    try:
        _ensure_langfuse_host()
        lf = _get_langfuse_client()
        kwargs = _event_kwargs(
            kind,
            node=node,
            graph=graph,
            source=source,
            description=description,
            extra=extra,
        )
        node_span = _node_observation_span(node, kind=kind)
        if node_span is not None:
            node_span.create_event(**kwargs)
        else:
            current_obs = getattr(lf, "get_current_observation_id", None)
            root_id = _observation_id_ctx.get()
            create_event = getattr(lf, "create_event", None)
            current_id = current_obs() if callable(current_obs) else None
            if (
                current_id
                and str(current_id) != str(root_id or "")
                and callable(create_event)
            ):
                create_event(**kwargs)
            else:
                root = _root_span_ctx.get()
                create_on_root = (
                    getattr(root, "create_event", None) if root is not None else None
                )
                if callable(create_on_root):
                    create_on_root(**kwargs)
                else:
                    ctx = _resolve_trace_context(lf)
                    if not ctx or not _emit_as_otel_child(lf, kwargs, ctx):
                        logger.warning(
                            "Langfuse kind event %s dropped; no parent generation trace",
                            kind,
                        )
                        return
        flush_fn = getattr(lf, "flush", None)
        if callable(flush_fn):
            flush_fn()
    except Exception as exc:
        logger.warning("Langfuse kind event failed: %s", exc)


def _resolve_trace_context(lf: Any) -> Optional[Dict[str, str]]:
    """Look up the generation trace after LangGraph dropped the OTel span."""
    trace_id = _trace_id_ctx.get()
    parent_span_id = _observation_id_ctx.get()
    if not trace_id:
        getter = getattr(lf, "get_current_trace_id", None)
        obs_getter = getattr(lf, "get_current_observation_id", None)
        if callable(getter):
            trace_id = getter()
        if callable(obs_getter):
            parent_span_id = obs_getter()
    if not trace_id:
        try:
            from src.utils.error_logger import get_conversation_context

            session_id = get_conversation_context()
        except Exception:
            session_id = None
        cached = _trace_by_session.get(session_id) if session_id else None
        if cached:
            trace_id, parent_span_id = cached
    if not trace_id:
        return None
    ctx: Dict[str, str] = {"trace_id": str(trace_id)}
    if parent_span_id:
        ctx["parent_span_id"] = str(parent_span_id)
    return ctx
