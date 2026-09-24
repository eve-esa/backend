"""OpenTelemetry setup for the backend: traces and logs over OTLP HTTP.

Off unless ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set: then nothing from the SDK is
imported, no provider is registered, no exporter thread starts and
:func:`wrap_asgi` returns the app unchanged. :func:`init_telemetry` never
raises; a broken setup logs one warning and leaves telemetry off.

Call order in ``server.py``: ``src.config`` import (runs ``load_dotenv``),
:func:`init_telemetry`, ``configure_logging``, then :func:`wrap_asgi` on the
outermost ASGI app. Gunicorn runs without ``--preload``, so each worker runs
this after the fork and owns its exporter threads.

Instrumented: inbound HTTP (ASGI middleware), httpx, pymongo without
statements, redis, botocore, stdlib logging. Not requests or urllib3 (the
exporter itself travels there and URLs carry keys), not openai (LangChain
covers LLM calls), not FastAPIInstrumentor (misses the /v1 and /mcp
dispatchers and would double the server span).
"""

import logging
import os
import uuid
from typing import Any, Callable, Optional, Tuple

logger = logging.getLogger(__name__)

SERVICE_NAMESPACE = "eve"
SERVICE_NAME = "eve-backend"
DEFAULT_ATTRIBUTE_VALUE_LENGTH_LIMIT = 4096
# Matched with re.search against scheme://host:port/path (no query string).
EXCLUDED_URLS = r"/health(/.*)?$"
# Dispatcher paths are templated by hand: the FastAPI router never sees them.
OPENAI_PROXY_ROUTE = "/v1/{path}"
MCP_PROXY_ROUTE = "/mcp/{server}/{path}"
# Header names blanked by the ASGI middleware if header capture is ever
# switched on with OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_SERVER_REQUEST.
SANITIZED_HEADERS = [
    "authorization",
    "proxy-authorization",
    "cookie",
    "set-cookie",
    "x-api-key",
    ".*token.*",
    ".*secret.*",
    ".*key.*",
]

_TRUE = {"1", "true", "yes", "on"}

_state: dict = {"enabled": False, "tracer_provider": None, "logger_provider": None}


def is_enabled() -> bool:
    """True once :func:`init_telemetry` has installed the providers."""
    return bool(_state["enabled"])


def _endpoint() -> str:
    return os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()


def _sdk_disabled() -> bool:
    return os.getenv("OTEL_SDK_DISABLED", "").strip().lower() in _TRUE


def _env_resource_keys() -> set:
    raw = os.getenv("OTEL_RESOURCE_ATTRIBUTES", "")
    return {
        item.split("=", 1)[0].strip()
        for item in raw.split(",")
        if "=" in item and item.split("=", 1)[0].strip()
    }


def build_resource():
    """Resource for every signal of this process.

    ``service.name`` and ``service.namespace`` are fixed. The environment comes
    from ``OTEL_RESOURCE_ATTRIBUTES`` when it names
    ``deployment.environment.name``, otherwise from ``APP_ENVIRONMENT``.
    ``service.instance.id`` is new per process, so per gunicorn worker.
    """
    from opentelemetry.sdk.resources import Resource

    attributes = {
        "service.namespace": SERVICE_NAMESPACE,
        "service.name": SERVICE_NAME,
        "service.version": os.getenv("APP_VERSION", "").strip() or "unknown",
        "service.instance.id": str(uuid.uuid4()),
    }
    if "deployment.environment.name" not in _env_resource_keys():
        from src.config import APP_ENVIRONMENT

        attributes["deployment.environment.name"] = APP_ENVIRONMENT or "unknown"
    git_sha = os.getenv("APP_GIT_SHA", "").strip()
    if git_sha and git_sha != "unknown":
        attributes["vcs.ref.head.revision"] = git_sha
    # Resource.create merges OTEL_RESOURCE_ATTRIBUTES first; ours win on clashes.
    return Resource.create(attributes)


def _span_limits():
    from opentelemetry.sdk.trace import SpanLimits

    raw = os.getenv("OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT", "").strip()
    try:
        limit = int(raw) if raw else DEFAULT_ATTRIBUTE_VALUE_LENGTH_LIMIT
    except ValueError:
        limit = DEFAULT_ATTRIBUTE_VALUE_LENGTH_LIMIT
    return SpanLimits(max_attribute_length=limit)


def build_tracer_provider(exporter, *, batch: bool = True):
    """TracerProvider with the EVE resource and ``exporter`` behind redaction.

    Sampler from ``OTEL_TRACES_SAMPLER`` and ``OTEL_TRACES_SAMPLER_ARG``
    (SDK default ``parentbased_always_on``). Tests pass ``batch=False``.
    """
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

    from src.observability.redaction import RedactingSpanExporter

    provider = TracerProvider(resource=build_resource(), span_limits=_span_limits())
    processor_cls = BatchSpanProcessor if batch else SimpleSpanProcessor
    provider.add_span_processor(processor_cls(RedactingSpanExporter(exporter)))
    return provider


def _otlp_log_handler_class():
    from opentelemetry.instrumentation.logging.handler import LoggingHandler

    from src.observability.redaction import redact_attributes

    class RedactingLoggingHandler(LoggingHandler):
        """OTLP log handler whose attributes, exception ones included, are redacted."""

        _eve_otel = True

        def _get_attributes(self, record):
            attributes = super()._get_attributes(record)
            return redact_attributes(
                {
                    k: v
                    for k, v in attributes.items()
                    if not k.startswith(("_eve", "otel"))
                }
            )

    return RedactingLoggingHandler


def _instrument_clients(tracer_provider) -> None:
    """Client auto instrumentation, each independent so one failure costs one."""

    def _httpx():
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().instrument(tracer_provider=tracer_provider)

    def _pymongo():
        from opentelemetry.instrumentation.pymongo import PymongoInstrumentor

        PymongoInstrumentor().instrument(
            tracer_provider=tracer_provider, capture_statement=False
        )

    def _redis():
        from opentelemetry.instrumentation.redis import RedisInstrumentor

        RedisInstrumentor().instrument(tracer_provider=tracer_provider)

    def _botocore():
        from opentelemetry.instrumentation.botocore import BotocoreInstrumentor

        BotocoreInstrumentor().instrument(tracer_provider=tracer_provider)

    for name, fn in (
        ("httpx", _httpx),
        ("pymongo", _pymongo),
        ("redis", _redis),
        ("botocore", _botocore),
    ):
        try:
            fn()
        except Exception as exc:
            logger.warning("OpenTelemetry: %s instrumentation skipped: %s", name, exc)


def _setup_logs(resource) -> None:
    from opentelemetry._logs import set_logger_provider
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

    from src.observability.redaction import RedactionFilter

    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter()))
    set_logger_provider(logger_provider)
    _state["logger_provider"] = logger_provider

    # Trace and span ids on every record for the console formatter; the
    # handler is ours (redacting), so the instrumentor must not add its own.
    LoggingInstrumentor().instrument(
        tracer_provider=_state["tracer_provider"],
        set_logging_format=False,
        inject_trace_context=True,
        enable_log_auto_instrumentation=False,
    )
    handler = _otlp_log_handler_class()(logger_provider=logger_provider)
    handler.addFilter(RedactionFilter())
    # The exporter's own warnings must not loop back into the exporter.
    handler.addFilter(lambda record: not record.name.startswith("opentelemetry"))
    logging.getLogger().addHandler(handler)


def init_telemetry() -> bool:
    """Install tracing and OTLP logs when an endpoint is configured.

    Reads the standard ``OTEL_*`` variables: the exporters pick up
    ``OTEL_EXPORTER_OTLP_ENDPOINT`` and ``OTEL_EXPORTER_OTLP_HEADERS`` on their
    own; only ``http/protobuf`` is shipped. Returns True when telemetry is on.
    Idempotent, never raises.
    """
    if _state["enabled"]:
        return True
    try:
        if _sdk_disabled() or not _endpoint():
            return False
        protocol = os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "").strip()
        if protocol and protocol != "http/protobuf":
            logger.warning(
                "OpenTelemetry: OTEL_EXPORTER_OTLP_PROTOCOL=%s is not supported, "
                "using http/protobuf",
                protocol,
            )

        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        tracer_provider = build_tracer_provider(OTLPSpanExporter())
        trace.set_tracer_provider(tracer_provider)
        _state["tracer_provider"] = tracer_provider
        _state["enabled"] = True

        _instrument_clients(tracer_provider)
        try:
            _setup_logs(tracer_provider.resource)
        except Exception as exc:
            logger.warning("OpenTelemetry: OTLP logs disabled: %s", exc)
        return True
    except Exception as exc:
        logger.warning("OpenTelemetry: init failed, telemetry stays off: %s", exc)
        return bool(_state["enabled"])


def make_span_details(fastapi_app: Any = None) -> Callable[[dict], Tuple[str, dict]]:
    """Build the ASGI ``default_span_details`` hook for ``fastapi_app``.

    Span name is ``METHOD template`` and ``http.route`` the template: the
    matching FastAPI route path, the fixed dispatcher templates for ``/v1``
    and ``/mcp``, or just ``METHOD`` with no route when nothing matches, so a
    404 scan or an id in the path never becomes a span name.
    """
    from starlette.routing import Match, Mount

    cache: dict = {}

    def _flat_routes() -> list:
        # FastAPI 0.14x keeps included routers as _IncludedRouter entries whose
        # effective route contexts carry the full path; plain routes (and
        # older FastAPI) are used as they are. Built on first use, after
        # startup has registered everything.
        if "routes" not in cache:
            flat = []
            router = getattr(fastapi_app, "router", None)
            for route in list(getattr(router, "routes", None) or []):
                contexts = getattr(route, "effective_route_contexts", None)
                if callable(contexts):
                    flat.extend(contexts())
                else:
                    flat.append(route)
            cache["routes"] = flat
        return cache["routes"]

    def _template(scope: dict) -> Optional[str]:
        path = scope.get("path", "") or ""
        if path == "/v1" or path.startswith("/v1/"):
            return OPENAI_PROXY_ROUTE
        if path.startswith("/mcp/"):
            return MCP_PROXY_ROUTE
        partial = None
        for route in _flat_routes():
            try:
                # A copy: FastAPI's matchers write bookkeeping keys into scope.
                match, _ = route.matches(dict(scope))
            except Exception:
                continue
            if match == Match.NONE:
                continue
            route_path = getattr(route, "path", None)
            if route_path is None:
                continue
            if isinstance(getattr(route, "starlette_route", None) or route, Mount):
                route_path = route_path.rstrip("/") + "/{path}"
            if match == Match.FULL:
                return route_path
            partial = partial or route_path
        return partial

    def eve_span_details(scope: dict) -> Tuple[str, dict]:
        method = (scope.get("method") or "").strip().upper() or "HTTP"
        try:
            template = _template(scope)
        except Exception:
            template = None
        if scope.get("type") == "websocket":
            method = "WS"
        if template:
            return f"{method} {template}", {"http.route": template}
        return method, {}

    return eve_span_details


def _find_fastapi_app(app: Any) -> Any:
    """Walk ``main_app`` and ``app`` links down to the app that owns a router."""
    seen = 0
    while app is not None and seen < 10:
        if hasattr(getattr(app, "router", None), "routes"):
            return app
        app = getattr(app, "main_app", None) or getattr(app, "app", None)
        seen += 1
    return None


def wrap_asgi(app: Any, fastapi_app: Any = None, tracer_provider: Any = None) -> Any:
    """Wrap the outermost ASGI app in one server span per request.

    Returns ``app`` unchanged when telemetry is off and no ``tracer_provider``
    is given. ``send`` and ``receive`` spans are not created, so an SSE
    stream is one span however many chunks it sends; ``/health`` is skipped.
    """
    if tracer_provider is None and not _state["enabled"]:
        return app
    try:
        from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware

        return OpenTelemetryMiddleware(
            app,
            excluded_urls=EXCLUDED_URLS,
            default_span_details=make_span_details(
                fastapi_app or _find_fastapi_app(app)
            ),
            tracer_provider=tracer_provider or _state["tracer_provider"],
            http_capture_headers_sanitize_fields=SANITIZED_HEADERS,
            exclude_spans=["send", "receive"],
        )
    except Exception as exc:
        logger.warning("OpenTelemetry: ASGI middleware not installed: %s", exc)
        return app


def shutdown() -> None:
    """Flush and stop the exporters. Safe to call when telemetry is off."""
    for key in ("tracer_provider", "logger_provider"):
        provider = _state.get(key)
        if provider is None:
            continue
        try:
            provider.shutdown()
        except Exception as exc:
            logger.warning("OpenTelemetry: %s shutdown failed: %s", key, exc)
