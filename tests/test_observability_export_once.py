"""Each span and each log record leaves the process once, as production wires it.

FastAPI 0.142 has native telemetry: at lifespan startup, with
``OTEL_EXPORTER_OTLP_ENDPOINT`` set, it adds its own OTLP span and log
processors to the providers ``init_telemetry`` installed, so every span was
exported twice (once unwrapped, past ``StoppedTurnSpanProcessor``) and every
log record twice, and it opened a second server span per request in a trace
of its own.

The real ``server`` import (``init_telemetry``), the real app and the real ASGI
lifespan run in a child process with the OTLP exporters swapped for in-memory
ones: telemetry installs process-wide providers and instrumentation that the
rest of the suite must not inherit.
"""

import json
import os
import subprocess
import sys
import textwrap
from collections import Counter

import pytest

pytestmark = pytest.mark.no_db

_CHILD = textwrap.dedent(
    """
    import asyncio, contextlib, io, json, logging

    import opentelemetry.exporter.otlp.proto.http._log_exporter as log_exporter
    import opentelemetry.exporter.otlp.proto.http.metric_exporter as metric_exporter
    import opentelemetry.exporter.otlp.proto.http.trace_exporter as trace_exporter
    from opentelemetry import trace
    from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter
    from opentelemetry.sdk.metrics.export import ConsoleMetricExporter
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    # Every OTLP exporter built in this process, ours or another component's,
    # writes to one place, so a second pipeline shows up as a second copy.
    spans = InMemorySpanExporter()
    logs = InMemoryLogRecordExporter()
    trace_exporter.OTLPSpanExporter = lambda *a, **k: spans
    log_exporter.OTLPLogExporter = lambda *a, **k: logs
    metric_exporter.OTLPMetricExporter = lambda *a, **k: ConsoleMetricExporter(out=io.StringIO())

    import server
    from httpx import ASGITransport, AsyncClient
    from src import observability
    from src.observability.context import agent_span, mark_turn_stopped

    assert observability.is_enabled()
    fastapi_app = observability._find_fastapi_app(server.app)

    @contextlib.asynccontextmanager
    async def no_startup(app):
        # Mongo, Qdrant and the rate limiter stay out; FastAPI's own lifespan
        # wrapper (where its telemetry configures itself) still runs.
        yield

    fastapi_app.router.lifespan_context = no_startup

    async def log_once():
        with trace.get_tracer("eve.test").start_as_current_span("export once work"):
            logging.getLogger("eve.test.export_once").warning("export once marker")
        return {"ok": True}

    fastapi_app.add_api_route("/otel-test/export-once", log_once, methods=["GET"])

    async def lifespan_startup():
        queue = asyncio.Queue()
        await queue.put({"type": "lifespan.startup"})
        sent = []

        async def receive():
            return await queue.get()

        async def send(message):
            sent.append(message["type"])
            if message["type"] == "lifespan.startup.complete":
                await queue.put({"type": "lifespan.shutdown"})

        await server.app({"type": "lifespan", "asgi": {"version": "3.0"}}, receive, send)
        return sent

    async def main():
        lifespan = await lifespan_startup()
        async with AsyncClient(
            transport=ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            status = (await client.get("/otel-test/export-once")).status_code

        stop = asyncio.Event()
        with agent_span("generation_stream", stop_event=stop):
            stop.set()
            tracer = observability._state["tracer_provider"].get_tracer(
                "opentelemetry.instrumentation.langchain"
            )
            span = tracer.start_span("invoke_agent LangGraph")
            exc = GeneratorExit()
            span.set_status(trace.Status(trace.StatusCode.ERROR, "GeneratorExit"))
            span.record_exception(exc)
            span.end()
            mark_turn_stopped()
        return lifespan, status

    lifespan, status = asyncio.run(main())
    observability._state["tracer_provider"].force_flush()
    observability._state["logger_provider"].force_flush()

    print("EXPORT-RESULT " + json.dumps({
        "lifespan": lifespan,
        "status": status,
        "spans": [
            {
                "id": format(s.context.span_id, "016x"),
                "name": s.name,
                "scope": s.instrumentation_scope.name if s.instrumentation_scope else "",
                "status": s.status.status_code.name,
                "stopped": bool((s.attributes or {}).get("eve.stopped")),
                "events": [e.name for e in s.events],
                "trace": format(s.context.trace_id, "032x"),
            }
            for s in spans.get_finished_spans()
        ],
        "logs": [
            {
                "body": str(d.log_record.body),
                "trace": format(d.log_record.trace_id or 0, "032x"),
            }
            for d in logs.get_finished_logs()
        ],
    }))
    """
)


@pytest.fixture(scope="module")
def exported():
    env = dict(os.environ)
    env["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://127.0.0.1:1"
    env.pop("OTEL_SDK_DISABLED", None)
    done = subprocess.run(
        [sys.executable, "-c", _CHILD],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    lines = [l for l in done.stdout.splitlines() if l.startswith("EXPORT-RESULT ")]
    assert done.returncode == 0 and lines, done.stdout[-2000:] + done.stderr[-4000:]
    result = json.loads(lines[-1].removeprefix("EXPORT-RESULT "))
    assert result["lifespan"] == ["lifespan.startup.complete", "lifespan.shutdown.complete"]
    assert result["status"] == 200
    return result


def test_each_span_is_exported_once(exported):
    copies = Counter(span["id"] for span in exported["spans"])

    assert exported["spans"], "no span exported"
    assert [i for i, n in copies.items() if n > 1] == []


def test_the_request_server_span_is_exported_once(exported):
    server_spans = [
        s for s in exported["spans"] if s["name"] == "GET /otel-test/export-once"
    ]

    assert len(server_spans) == 1
    assert server_spans[0]["scope"] == "opentelemetry.instrumentation.asgi"
    assert [s for s in exported["spans"] if s["scope"] == "fastapi"] == []


def test_the_request_work_shares_the_server_span_trace(exported):
    """No second root trace: the route's spans and logs sit under the server span."""
    (server_span,) = [
        s for s in exported["spans"] if s["name"] == "GET /otel-test/export-once"
    ]
    (work,) = [s for s in exported["spans"] if s["name"] == "export once work"]
    (marker,) = [l for l in exported["logs"] if l["body"] == "export once marker"]

    assert server_span["trace"] != "0" * 32
    assert work["trace"] == server_span["trace"]
    assert marker["trace"] == server_span["trace"]


def test_each_log_record_is_exported_once(exported):
    assert [l["body"] for l in exported["logs"]].count("export once marker") == 1


def test_a_stopped_span_is_exported_once_and_rewritten(exported):
    library = [s for s in exported["spans"] if s["name"] == "invoke_agent LangGraph"]

    assert len(library) == 1
    assert library[0]["status"] == "UNSET"
    assert library[0]["stopped"] is True
    assert "exception" not in library[0]["events"]
