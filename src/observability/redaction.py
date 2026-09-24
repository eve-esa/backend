"""Redaction applied where telemetry leaves the process: spans and log records.

Built on :mod:`src.utils.redaction`, so the rules match the Mongo error log.
The collector redaction processor stays the enforcement point; this is the
first line, so a secret never reaches the network in the first place.
"""

import logging
import traceback
from typing import Any, Optional, Sequence

from src.utils.redaction import REDACTED, is_secret_key, redact_secrets

# Attributes a LogRecord always has. Anything else was passed with ``extra=``.
_RECORD_ATTRS = frozenset(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


def _key_is_secret(key: str) -> bool:
    """``http.request.header.authorization`` is judged by its last segment."""
    return is_secret_key(key.rsplit(".", 1)[-1])


def redact_attribute(key: str, value: Any) -> Any:
    """Return a redacted copy of one OTel attribute value, same type family."""
    if _key_is_secret(key):
        if isinstance(value, (list, tuple)):
            return tuple(REDACTED for _ in value)
        return REDACTED if isinstance(value, str) else value
    if isinstance(value, str):
        if key.endswith("query"):
            # url.query carries no leading "?", which the query rule anchors on.
            return redact_secrets("?" + value)[1:]
        return redact_secrets(value)
    if isinstance(value, (list, tuple)):
        return tuple(redact_secrets(v) if isinstance(v, str) else v for v in value)
    return value


def redact_attributes(attributes: Optional[Any]) -> dict:
    if not attributes:
        return {}
    return {k: redact_attribute(k, v) for k, v in attributes.items()}


class RedactingSpanExporter:
    """SpanExporter wrapper that rebuilds each span with redacted attributes.

    A finished ``ReadableSpan`` cannot be mutated, so a copy is made with the
    span name, attributes, event names and attributes and link attributes run
    through the redaction rules. A redaction failure drops nothing: the span
    goes out with every string attribute replaced, never with the original.
    """

    def __init__(self, exporter):
        self._exporter = exporter

    @staticmethod
    def _rebuild(span):
        from opentelemetry.sdk.trace import Event, ReadableSpan
        from opentelemetry.trace import Link

        try:
            attributes = redact_attributes(span.attributes)
            events = [
                Event(
                    name=redact_secrets(event.name),
                    attributes=redact_attributes(event.attributes),
                    timestamp=event.timestamp,
                )
                for event in span.events
            ]
            links = [
                Link(link.context, redact_attributes(link.attributes))
                for link in span.links
            ]
            name = redact_secrets(span.name)
        except Exception:  # pragma: no cover - defensive
            attributes = {
                k: (REDACTED if isinstance(v, str) else v)
                for k, v in (span.attributes or {}).items()
            }
            events, links, name = [], [], span.name
        return ReadableSpan(
            name=name,
            context=span.context,
            parent=span.parent,
            resource=span.resource,
            attributes=attributes,
            events=events,
            links=links,
            kind=span.kind,
            status=span.status,
            start_time=span.start_time,
            end_time=span.end_time,
            instrumentation_scope=span.instrumentation_scope,
        )

    def export(self, spans: Sequence):
        return self._exporter.export([self._rebuild(span) for span in spans])

    def shutdown(self):
        return self._exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        flush = getattr(self._exporter, "force_flush", None)
        return flush(timeout_millis) if flush else True


class RedactionFilter(logging.Filter):
    """Scrub credentials and emails from a record before any handler writes it.

    Attached to every handler (console, OTLP, uvicorn and gunicorn). Arguments
    are redacted one by one when the template itself is clean, so formatters
    that read ``record.args`` (uvicorn's access log) keep working; otherwise
    the message is collapsed to its redacted text. A traceback is rendered,
    redacted and cached in ``exc_text``; the OTLP handler redacts its own
    exception attributes. Never drops a record, never raises.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            self._redact(record)
        except Exception:  # pragma: no cover - a redactor must not lose the log
            pass
        return True

    @staticmethod
    def _redact_arg(value: Any) -> Any:
        return redact_secrets(value) if isinstance(value, str) else value

    def _redact(self, record: logging.LogRecord) -> None:
        if getattr(record, "_eve_redacted", False):
            return
        record._eve_redacted = True

        message = record.getMessage()
        if redact_secrets(message) != message:
            msg = record.msg
            args = record.args
            if isinstance(msg, str) and redact_secrets(msg) == msg and args:
                if isinstance(args, tuple):
                    record.args = tuple(self._redact_arg(a) for a in args)
                elif isinstance(args, dict):
                    record.args = {k: self._redact_arg(v) for k, v in args.items()}
            if redact_secrets(record.getMessage()) != record.getMessage():
                record.msg = redact_secrets(message)
                record.args = None

        for key, value in list(vars(record).items()):
            if key in _RECORD_ATTRS or key.startswith("_eve"):
                continue
            if _key_is_secret(key):
                setattr(record, key, REDACTED)
            elif isinstance(value, str):
                setattr(record, key, redact_secrets(value))

        if record.exc_info and record.exc_info[0] is not None and not record.exc_text:
            # Formatters reuse a cached exc_text instead of rendering exc_info.
            text = "".join(traceback.format_exception(*record.exc_info)).rstrip("\n")
            record.exc_text = redact_secrets(text)
