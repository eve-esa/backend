"""OpenTelemetry wiring: off by default, one span per request, templated
routes, redaction on spans and logs, log to trace correlation, CORS.

Spans go to the SDK in memory exporter through the same RedactingSpanExporter
and ASGI middleware the server uses; no global provider is registered here.
"""

import asyncio
import logging
import threading
import warnings

import pytest
from fastapi.responses import StreamingResponse
from httpx import ASGITransport, AsyncClient
from opentelemetry import trace
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind

import server
from src import observability
from src.config import CORS_ALLOWED_ORIGINS, LOG_FORMAT_WITH_TRACE
from src.observability.redaction import RedactingSpanExporter, RedactionFilter

JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0ZXN0LXVzZXIifQ.c2lnbmF0dXJlLXNlY3JldA"
EVE_KEY = "eve_" + "a1" * 32
COOKIE_VALUE = "sessionid-cookie-value-9f8e7d"
SECRETS = (JWT, EVE_KEY, COOKIE_VALUE)


def _all_attribute_text(span: ReadableSpan) -> str:
    parts = [span.name]
    for attrs in [span.attributes] + [e.attributes for e in span.events]:
        for key, value in (attrs or {}).items():
            parts.append(f"{key}={value!r}")
    parts.extend(e.name for e in span.events)
    return "\n".join(parts)


@pytest.fixture
def exporter():
    return InMemorySpanExporter()


@pytest.fixture
def provider(exporter):
    # Same factory as production, synchronous processor so spans are ready
    # when the response returns.
    tp = observability.build_tracer_provider(exporter, batch=False)
    yield tp
    tp.shutdown()


@pytest.fixture
def traced_app(provider, monkeypatch):
    """A fresh app from create_app, with two test routes, wrapped like prod."""
    # Header capture on for everything: proves the exporter redacts even if
    # someone switches it on in an environment.
    monkeypatch.setenv("OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_SERVER_REQUEST", ".*")
    outer = server.create_app()
    assert not observability.is_enabled()
    fastapi_app = observability._find_fastapi_app(outer)

    async def stream():
        async def events():
            for i in range(5):
                yield f"data: chunk {i}\n\n"
                await asyncio.sleep(0)

        return StreamingResponse(events(), media_type="text/event-stream")

    async def log_inside(item_id: str):
        logging.getLogger("eve.test.otel").info("handled item %s", item_id)
        return {"item": item_id}

    fastapi_app.add_api_route("/otel-test/stream", stream, methods=["GET"])
    fastapi_app.add_api_route("/otel-test/log/{item_id}", log_inside, methods=["GET"])
    return observability.wrap_asgi(outer, fastapi_app=fastapi_app, tracer_provider=provider)


async def _get(app, path, **kwargs):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(path, **kwargs)


# C1: off by default


async def test_init_is_a_noop_without_endpoint(monkeypatch, caplog):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    threads_before = {t.name for t in threading.enumerate()}
    caplog.set_level(logging.DEBUG)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert observability.init_telemetry() is False
    assert not observability.is_enabled()
    # Nothing registered: the global provider is still the API proxy.
    assert not isinstance(trace.get_tracer_provider(), TracerProvider)
    new_threads = {t.name for t in threading.enumerate()} - threads_before
    assert not [n for n in new_threads if "otel" in n.lower()]
    assert not any("OtelBatch" in t.name for t in threading.enumerate())
    assert not [w for w in caught if "opentelemetry" in str(w.filename).lower()]
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    # The app the server exports is the bare dispatcher, no middleware.
    sentinel = object()
    assert observability.wrap_asgi(sentinel) is sentinel
    assert type(server.app).__name__ == "MCPProxyDispatcher"


async def test_empty_endpoint_and_sdk_disabled_are_noops(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "  ")
    assert observability.init_telemetry() is False
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    assert observability.init_telemetry() is False
    assert not observability.is_enabled()


async def test_init_never_raises(monkeypatch, caplog):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)

    def boom(*_args, **_kwargs):
        raise RuntimeError("exporter exploded")

    monkeypatch.setattr(observability, "build_tracer_provider", boom)
    with caplog.at_level(logging.WARNING, logger="src.observability"):
        assert observability.init_telemetry() is False
    assert not observability.is_enabled()
    assert "init failed" in caplog.text


async def test_resource_attributes(monkeypatch):
    monkeypatch.setenv("APP_VERSION", "9.9.9")
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "deployment.environment.name=local")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "someone-else")
    attrs = observability.build_resource().attributes
    assert attrs["service.name"] == "eve-backend"
    assert attrs["service.namespace"] == "eve"
    assert attrs["service.version"] == "9.9.9"
    assert attrs["deployment.environment.name"] == "local"
    assert attrs["service.instance.id"]

    monkeypatch.delenv("OTEL_RESOURCE_ATTRIBUTES")
    from src.config import APP_ENVIRONMENT

    attrs = observability.build_resource().attributes
    assert attrs["deployment.environment.name"] == APP_ENVIRONMENT


# C2: one server span per request, templated names


async def test_sse_stream_is_one_server_span(traced_app, exporter):
    resp = await _get(traced_app, "/otel-test/stream")
    assert resp.status_code == 200
    assert resp.text.count("data: chunk") == 5
    spans = exporter.get_finished_spans()
    assert len(spans) == 1, [s.name for s in spans]
    span = spans[0]
    assert span.kind == SpanKind.SERVER
    assert span.name == "GET /otel-test/stream"
    assert span.attributes["http.route"] == "/otel-test/stream"
    assert not [s for s in spans if "send" in s.name or "receive" in s.name]


async def test_health_is_not_traced(traced_app, exporter):
    resp = await _get(traced_app, "/health")
    assert resp.status_code == 200
    assert exporter.get_finished_spans() == ()


async def test_route_with_id_is_named_by_its_template(traced_app, exporter):
    resp = await _get(traced_app, "/conversations/65f0c0ffee0000000000abcd")
    assert resp.status_code in (401, 403)
    (span,) = exporter.get_finished_spans()
    assert span.name == "GET /conversations/{conversation_id}"
    assert span.attributes["http.route"] == "/conversations/{conversation_id}"
    assert "65f0c0ffee" not in span.name


@pytest.mark.parametrize(
    "path, template",
    [
        ("/v1/chat/completions", "/v1/{path}"),
        ("/v1/models", "/v1/{path}"),
        ("/mcp/some-server/mcp", "/mcp/{server}/{path}"),
        ("/mcp/other/tools/list", "/mcp/{server}/{path}"),
    ],
)
async def test_dispatcher_paths_get_fixed_templates(traced_app, exporter, path, template):
    await _get(traced_app, path)
    (span,) = exporter.get_finished_spans()
    assert span.name == f"GET {template}"
    assert span.attributes["http.route"] == template


async def test_static_mount_is_templated(traced_app, exporter):
    await _get(traced_app, "/demo-images/some-file.png")
    (span,) = exporter.get_finished_spans()
    assert span.name == "GET /demo-images/{path}"


async def test_unknown_path_has_no_route(traced_app, exporter):
    resp = await _get(traced_app, "/wp-admin/abc123")
    assert resp.status_code == 404
    (span,) = exporter.get_finished_spans()
    assert span.name == "GET"
    assert "http.route" not in span.attributes


# C3: no credential reaches an exported span


async def test_credentials_never_in_exported_span(traced_app, exporter):
    await _get(
        traced_app,
        f"/otel-test/log/abc?api_key={EVE_KEY}&token={JWT}&q=ok",
        headers={
            "Authorization": f"Bearer {JWT}",
            "Cookie": f"session={COOKIE_VALUE}",
            "X-API-Key": EVE_KEY,
        },
    )
    (span,) = exporter.get_finished_spans()
    text = _all_attribute_text(span)
    for secret in SECRETS:
        assert secret not in text
    # Header capture really happened; the values were blanked on export.
    assert span.attributes["http.request.header.authorization"] == ("[REDACTED]",)
    assert span.attributes["http.request.header.cookie"] == ("[REDACTED]",)
    assert span.attributes["http.request.header.x_api_key"] == ("[REDACTED]",)
    assert "q=ok" in text


async def test_redacting_exporter_rebuilds_spans():
    raw = InMemorySpanExporter()
    redacted = InMemorySpanExporter()
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(raw))
    tp.add_span_processor(SimpleSpanProcessor(RedactingSpanExporter(redacted)))
    tracer = tp.get_tracer("test")
    with tracer.start_as_current_span("call", kind=SpanKind.CLIENT) as span:
        span.set_attribute("url.full", f"https://api.example/v1?api_key={EVE_KEY}")
        span.set_attribute("url.query", f"token={JWT}&page=2")
        span.set_attribute("http.request.header.authorization", (f"Bearer {JWT}",))
        span.set_attribute("eve.count", 3)
        span.add_event("retry", {"detail": f"Authorization: Bearer {JWT}"})
        span.record_exception(ValueError(f"upstream refused key {EVE_KEY}"))
    tp.shutdown()

    (original,) = raw.get_finished_spans()
    (clean,) = redacted.get_finished_spans()
    assert clean is not original
    assert EVE_KEY in _all_attribute_text(original)
    text = _all_attribute_text(clean)
    for secret in SECRETS:
        assert secret not in text
    assert clean.attributes["eve.count"] == 3
    assert clean.attributes["url.query"].endswith("&page=2")
    assert clean.context == original.context
    assert clean.parent == original.parent
    assert (clean.start_time, clean.end_time) == (original.start_time, original.end_time)
    assert [e.name for e in clean.events] == ["retry", "exception"]


# C4: a log line inside a request carries the request's trace id


async def test_log_line_carries_request_trace_id(traced_app, exporter, provider):
    lines = []

    class Capture(logging.Handler):
        def emit(self, record):
            lines.append(self.format(record))

    handler = Capture()
    handler.setFormatter(logging.Formatter(LOG_FORMAT_WITH_TRACE))
    handler.addFilter(RedactionFilter())
    test_logger = logging.getLogger("eve.test.otel")
    test_logger.addHandler(handler)
    test_logger.setLevel(logging.INFO)
    instrumentor = LoggingInstrumentor()
    instrumentor.instrument(
        tracer_provider=provider,
        set_logging_format=False,
        inject_trace_context=True,
        enable_log_auto_instrumentation=False,
    )
    try:
        resp = await _get(traced_app, "/otel-test/log/item-42")
    finally:
        instrumentor.uninstrument()
        test_logger.removeHandler(handler)
    assert resp.status_code == 200
    (span,) = exporter.get_finished_spans()
    trace_id = format(span.context.trace_id, "032x")
    span_id = format(span.context.span_id, "016x")
    (line,) = [line for line in lines if "handled item item-42" in line]
    assert f"trace_id={trace_id}" in line
    assert f"span_id={span_id}" in line


def test_redaction_filter_on_log_records():
    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(self.format(record))

    handler = Capture()
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.addFilter(RedactionFilter())
    log = logging.getLogger("eve.test.redaction")
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False
    try:
        log.info("GET %s", f"https://serp.example/search?api_key={EVE_KEY}&q=x")
        log.info(f"Authorization: Bearer {JWT}")
        log.info("key=%s", EVE_KEY)
        try:
            raise RuntimeError(f"bad token {JWT}")
        except RuntimeError:
            log.exception("failed")
        # uvicorn's access formatter unpacks record.args: they stay a tuple.
        record = logging.makeLogRecord(
            {
                "msg": '%s - "%s %s HTTP/%s" %d',
                "args": ("1.2.3.4:5", "GET", f"/x?token={JWT}", "1.1", 200),
            }
        )
        RedactionFilter().filter(record)
        assert isinstance(record.args, tuple) and record.args[4] == 200
        assert JWT not in record.getMessage()
    finally:
        log.removeHandler(handler)
        log.propagate = True
    joined = "\n".join(records)
    for secret in (JWT, EVE_KEY):
        assert secret not in joined
    assert "q=x" in joined
    assert "RuntimeError" in joined


def test_configure_logging_puts_redaction_on_server_handlers():
    from src.config import configure_logging

    access = logging.getLogger("uvicorn.access")
    handler = logging.StreamHandler()
    access.addHandler(handler)
    try:
        configure_logging()
        assert any(isinstance(f, RedactionFilter) for f in handler.filters)
    finally:
        access.removeHandler(handler)


# C5: CORS preflight admits the trace propagation headers


async def test_cors_preflight_admits_trace_headers():
    origin = CORS_ALLOWED_ORIGINS[0]
    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.options(
            "/conversations",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "traceparent, tracestate, baggage, content-type",
            },
        )
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == origin
    allowed = {h.strip().lower() for h in resp.headers["access-control-allow-headers"].split(",")}
    assert {"traceparent", "tracestate", "baggage"} <= allowed
