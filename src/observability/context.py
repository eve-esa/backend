"""Request context on spans, the agent root span, kind events, trace ids.

Only the OpenTelemetry API is used here, never the SDK: with telemetry off the
tracer is the API no-op, spans are non recording, trace ids read as ``None``
and :func:`record_kind` only logs. Nothing in this module raises into the
caller.

- :class:`ContextAttributesSpanProcessor` copies the request contextvars kept
  by ``src.utils.error_logger`` (conversation, message, user) onto every span,
  under the names Langfuse and HyperDX read: ``session.id`` is the
  conversation id, ``user.id`` the user id, never an email.
- :func:`agent_span` opens the ``invoke_agent`` root of one answer. LangChain
  and LangGraph spans (OpenLLMetry) and the MCP client spans nest under it.
- :func:`record_kind` turns the agentic failure taxonomy (timeout, retry,
  fallback, ...) into a span event plus a WARNING log line.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Dict, Iterator, Optional

from opentelemetry import context as otel_context
from opentelemetry import trace

logger = logging.getLogger(__name__)

TRACER_NAME = "eve.backend"
ROOT_SPAN_NAME = "invoke_agent"

CONVERSATION_ID = "gen_ai.conversation.id"
SESSION_ID = "session.id"
USER_ID = "user.id"
MESSAGE_ID = "eve.message_id"

# Keys of ``extra`` copied onto a kind event, same set the Mongo log keeps.
_KIND_EXTRA_KEYS = ("signal", "tool", "server", "attempt")
_MAX_DESCRIPTION = 500


def get_tracer():
    """Tracer from the provider installed by ``init_telemetry`` (or a test).

    Falls back to the global API tracer, a no-op until a provider is set.
    """
    try:
        from src.observability import _state

        provider = _state.get("tracer_provider")
        if provider is not None:
            return provider.get_tracer(TRACER_NAME)
    except Exception:  # pragma: no cover - defensive
        pass
    return trace.get_tracer(TRACER_NAME)


# ─── trace ids ────────────────────────────────────────────────────────────────


def format_trace_id(span_context: Any) -> Optional[str]:
    """32 lowercase hex chars, or ``None`` for an invalid or missing context."""
    try:
        if span_context is None or not span_context.is_valid:
            return None
        return trace.format_trace_id(span_context.trace_id)
    except Exception:
        return None


def span_trace_id(span: Any) -> Optional[str]:
    """Trace id of ``span`` (``None`` when telemetry is off)."""
    if span is None:
        return None
    try:
        return format_trace_id(span.get_span_context())
    except Exception:
        return None


def current_trace_id() -> Optional[str]:
    """Trace id of the current span, ``None`` outside a trace."""
    return span_trace_id(trace.get_current_span())


# ─── request context on every span ───────────────────────────────────────────


def _request_context() -> tuple:
    """``(conversation_id, message_id, user_id)`` from the error log contextvars.

    Imported lazily: ``error_logger`` imports this module for ``record_kind``.
    """
    from src.utils import error_logger

    return (
        error_logger.conversation_id_context.get(),
        error_logger.message_id_context.get(),
        error_logger.user_id_context.get(),
    )


def _is_user_id(value: Optional[str]) -> bool:
    # user.id carries the Mongo user id; anything shaped like an email stays out.
    return bool(value) and "@" not in str(value)


def context_attributes(
    conversation_id: Optional[str],
    message_id: Optional[str],
    user_id: Optional[str],
) -> Dict[str, str]:
    attributes: Dict[str, str] = {}
    if conversation_id:
        attributes[CONVERSATION_ID] = str(conversation_id)
        attributes[SESSION_ID] = str(conversation_id)
    if message_id:
        attributes[MESSAGE_ID] = str(message_id)
    if _is_user_id(user_id):
        attributes[USER_ID] = str(user_id)
    return attributes


class ContextAttributesSpanProcessor:
    """Stamp conversation, message and user ids on every span at start.

    Duck typed against the SDK ``SpanProcessor`` so this module never imports
    the SDK. A span that sets one of these attributes later (OpenLLMetry sets
    ``gen_ai.conversation.id`` from the LangGraph thread id) overwrites the
    value, which is the same id.
    """

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        try:
            attributes = context_attributes(*_request_context())
            if attributes:
                span.set_attributes(attributes)
        except Exception:  # pragma: no cover - a processor must never break a span
            pass

    def _on_ending(self, span: Any) -> None:
        return None

    def on_end(self, span: Any) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


# ─── agent root span ──────────────────────────────────────────────────────────


def _detach(token: Any) -> None:
    if token is None:
        return
    try:
        otel_context.detach(token)
    except Exception:  # pragma: no cover - detach already swallows and logs
        pass


@contextlib.contextmanager
def child_span(name: str, attributes: Optional[Dict[str, Any]] = None) -> Iterator[Any]:
    """Start ``name`` as the current span and end it on exit.

    Yields the span (a non recording one when telemetry is off, whose trace id
    reads as ``None``). An exception from the body is recorded and re raised;
    a cancellation or a closed generator ends the span with
    ``eve.cancelled=true`` and no error status. Setup failures are swallowed:
    the body always runs.
    """
    span = None
    token = None
    try:
        span = get_tracer().start_span(
            name,
            attributes={k: v for k, v in (attributes or {}).items() if v is not None},
        )
        token = otel_context.attach(trace.set_span_in_context(span))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("span %s setup failed: %s", name, exc)
    try:
        yield span if span is not None else trace.INVALID_SPAN
    except Exception as exc:
        if span is not None:
            with contextlib.suppress(Exception):
                span.record_exception(exc)
                span.set_status(trace.Status(trace.StatusCode.ERROR, type(exc).__name__))
        raise
    except BaseException:
        if span is not None:
            with contextlib.suppress(Exception):
                span.set_attribute("eve.cancelled", True)
        raise
    finally:
        _detach(token)
        if span is not None:
            with contextlib.suppress(Exception):
                span.end()


@contextlib.contextmanager
def agent_span(
    agent_name: str,
    *,
    conversation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    message_id: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> Iterator[Any]:
    """Open the ``invoke_agent`` root span of one answer and make it current.

    ``agent_name`` goes to ``gen_ai.agent.name``; the ids are set explicitly
    because the span may start before the router has set every contextvar.
    """
    span_attributes: Dict[str, Any] = {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.agent.name": agent_name,
        **context_attributes(conversation_id, message_id, user_id),
        **(attributes or {}),
    }
    with child_span(ROOT_SPAN_NAME, span_attributes) as span:
        yield span


def set_llm_attributes(
    span: Any, endpoint: Optional[Dict[str, Any]], *, include_answered: bool = True
) -> None:
    """``eve.llm.*`` from a ``build_endpoint_metadata`` payload onto ``span``.

    ``include_answered=False`` when the span starts before the answer exists:
    the payload then only names the endpoint the run starts on.
    """
    if span is None or not endpoint:
        return
    try:
        values = {
            "eve.llm.requested": endpoint.get("requested"),
            "eve.llm.chain": [str(c) for c in (endpoint.get("chain") or [])] or None,
            "eve.llm.answered": endpoint.get("answered") if include_answered else None,
            "eve.fallback_used": (
                bool(endpoint.get("substituted")) if include_answered else None
            ),
        }
        span.set_attributes({k: v for k, v in values.items() if v is not None})
    except Exception:  # pragma: no cover - defensive
        pass


def mark_error(span: Any, exc: BaseException) -> None:
    """Record ``exc`` on ``span`` and set error status, for handled failures."""
    if span is None:
        return
    try:
        span.record_exception(exc)
        span.set_status(trace.Status(trace.StatusCode.ERROR, type(exc).__name__))
    except Exception:  # pragma: no cover - defensive
        pass


def add_span_event(span: Any, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
    if span is None:
        return
    try:
        span.add_event(
            name, {k: v for k, v in (attributes or {}).items() if v is not None}
        )
    except Exception:  # pragma: no cover - defensive
        pass


# ─── kind events ──────────────────────────────────────────────────────────────


def kind_attributes(
    kind: str,
    *,
    node: Optional[str] = None,
    graph: Optional[str] = None,
    source: Optional[str] = None,
    description: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    attributes: Dict[str, Any] = {"eve.kind": kind}
    if node:
        attributes["eve.node"] = str(node)
    if graph:
        attributes["eve.graph"] = str(graph)
    if source:
        attributes["eve.source"] = str(source)
    status = (description or "").strip()[:_MAX_DESCRIPTION]
    if status:
        attributes["eve.description"] = status
    for key in _KIND_EXTRA_KEYS:
        value = (extra or {}).get(key)
        if value is None:
            continue
        attributes[f"eve.{key}"] = (
            value if isinstance(value, (bool, int, float, str)) else str(value)
        )
    return attributes


def record_kind(
    kind: Optional[str],
    node: Optional[str] = None,
    graph: Optional[str] = None,
    source: Optional[str] = None,
    description: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Record an agentic ``kind`` as an event on the current span and a WARNING.

    The event is named after the kind (``timeout``, ``run_timeout``,
    ``retry``, ``error_handler``, ``fallback``, ``tool_error``) and carries
    ``eve.kind``, ``eve.node``, ``eve.graph``, ``eve.source``,
    ``eve.description`` and the ``signal``, ``tool``, ``server``, ``attempt``
    keys of ``extra``. Outside a trace only the log line is written. Never
    raises.
    """
    if not kind:
        return
    try:
        attributes = kind_attributes(
            kind,
            node=node,
            graph=graph,
            source=source,
            description=description,
            extra=extra,
        )
        span = trace.get_current_span()
        if span.is_recording():
            span.add_event(kind, attributes)
        logger.warning(
            "agent kind=%s node=%s graph=%s source=%s: %s",
            kind,
            node or "-",
            graph or "-",
            source or "-",
            attributes.get("eve.description", ""),
            extra={"eve_kind": kind},
        )
    except Exception as exc:  # pragma: no cover - never break the caller
        logger.debug("record_kind failed: %s", exc)
