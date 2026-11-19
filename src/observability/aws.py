"""botocore spans: an S3 key that is not there yet is not an error.

The workspace reads its state object before the first save writes it, and the
storage layer probes keys with HEAD; both answer 404, which the callers expect
and handle (``ObjectNotFoundError``). The botocore instrumentation marks every
4xx ERROR, so each agentic turn added one error span to the HyperDX views.

The instrumentation's ``response_hook`` runs after it set the status, and the
SDK ignores ``set_status(UNSET)`` on a live span, so the hook only tags the
span with ``eve.s3.expected_miss``; :class:`ExpectedMissSpanProcessor` clears
the status of tagged spans before export, the same rebuild the Stop rewrite
uses, and drops its ``exception`` event so no view counts the miss (other
events stay). Only S3 ``GetObject`` and ``HeadObject`` answering 404 are tagged: a
403, a 5xx or any other service keeps its ERROR status.
"""

import logging
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan

logger = logging.getLogger(__name__)

EXPECTED_MISS_ATTRIBUTE = "eve.s3.expected_miss"
_MISS_OPERATIONS = frozenset({"GetObject", "HeadObject"})


def s3_response_hook(span: Any, service_name: str, operation_name: str, result: Any) -> None:
    """botocore ``response_hook``: tag an S3 GetObject or HeadObject 404."""
    try:
        if service_name != "s3" or operation_name not in _MISS_OPERATIONS:
            return
        if not isinstance(result, dict) or span is None or not span.is_recording():
            return
        status = (result.get("ResponseMetadata") or {}).get("HTTPStatusCode")
        if status == 404:
            span.set_attribute(EXPECTED_MISS_ATTRIBUTE, True)
    except Exception as exc:  # pragma: no cover - a hook never breaks the S3 call
        logger.debug("s3 response hook failed: %s", exc)


def instrument_botocore(tracer_provider: Any) -> None:
    from opentelemetry.instrumentation.botocore import BotocoreInstrumentor

    BotocoreInstrumentor().instrument(
        tracer_provider=tracer_provider, response_hook=s3_response_hook
    )


def expected_miss_span(span: ReadableSpan) -> ReadableSpan:
    """``span`` with its ERROR status and exception event dropped when tagged."""
    if not (span.attributes or {}).get(EXPECTED_MISS_ATTRIBUTE):
        return span
    if span.status.status_code != trace.StatusCode.ERROR:
        return span
    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=span.attributes,
        events=[e for e in span.events if e.name != "exception"],
        links=span.links,
        kind=span.kind,
        status=trace.Status(trace.StatusCode.UNSET),
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


class ExpectedMissSpanProcessor:
    """Export processor wrapper that clears the status of tagged S3 misses."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        self._inner.on_start(span, parent_context=parent_context)

    def _on_ending(self, span: Any) -> None:
        on_ending = getattr(self._inner, "_on_ending", None)
        if on_ending is not None:
            on_ending(span)

    def on_end(self, span: Any) -> None:
        try:
            span = expected_miss_span(span)
        except Exception as exc:  # pragma: no cover - the span still goes out
            logger.debug("expected miss span rewrite failed: %s", exc)
        self._inner.on_end(span)

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._inner.force_flush(timeout_millis)
