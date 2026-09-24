"""Agentic error kinds as span events plus a WARNING log (record_kind).

Replaces the old SDK kind events: the taxonomy that lands in Mongo
``error_logs`` (timeout, run_timeout, retry, error_handler, fallback,
tool_error) also becomes an event on the current span, and a WARNING line in
the log stream that carries the trace id.
"""

import logging
from unittest.mock import patch

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src import observability
from src.observability.context import record_kind
from src.utils.error_logger import _SPAN_EVENT_KINDS, ErrorLogger

pytestmark = pytest.mark.no_db

TAXONOMY = {"timeout", "run_timeout", "retry", "error_handler", "fallback", "tool_error"}


@pytest.fixture
def tracer():
    exporter = InMemorySpanExporter()
    provider = observability.build_tracer_provider(exporter, batch=False)
    yield provider.get_tracer("test"), exporter
    provider.shutdown()


def test_the_taxonomy_is_mirrored():
    assert _SPAN_EVENT_KINDS == TAXONOMY


@pytest.mark.parametrize("kind", sorted(TAXONOMY))
def test_kind_becomes_an_event_on_the_current_span(tracer, caplog, kind):
    tracer, exporter = tracer
    caplog.set_level(logging.WARNING, logger="src.observability.context")

    with tracer.start_as_current_span("execute_task agent"):
        record_kind(
            kind,
            node="agent",
            graph="react",
            source="runner",
            description="Policy on node 'agent'",
            extra={"attempt": 2, "signal": "exception", "unlisted": "dropped"},
        )

    (span,) = exporter.get_finished_spans()
    (event,) = span.events
    assert event.name == kind
    assert dict(event.attributes) == {
        "eve.kind": kind,
        "eve.node": "agent",
        "eve.graph": "react",
        "eve.source": "runner",
        "eve.description": "Policy on node 'agent'",
        "eve.signal": "exception",
        "eve.attempt": 2,
    }
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert f"kind={kind}" in warnings[0].getMessage()
    assert warnings[0].eve_kind == kind


def test_event_carries_tool_and_server_for_tool_errors(tracer):
    tracer, exporter = tracer
    with tracer.start_as_current_span("mcp.call_tool"):
        record_kind(
            "tool_error",
            node="tools",
            source="mcp",
            extra={"signal": "structured", "tool": "search", "server": "wiley"},
        )
    (span,) = exporter.get_finished_spans()
    attributes = dict(span.events[0].attributes)
    assert attributes["eve.tool"] == "search"
    assert attributes["eve.server"] == "wiley"


def test_description_is_capped(tracer):
    tracer, exporter = tracer
    with tracer.start_as_current_span("agent"):
        record_kind("timeout", description="x" * 2000)
    (span,) = exporter.get_finished_spans()
    assert len(span.events[0].attributes["eve.description"]) == 500


def test_outside_a_trace_only_logs(caplog):
    caplog.set_level(logging.WARNING, logger="src.observability.context")
    record_kind("timeout", node="agent")
    assert any("kind=timeout" in r.getMessage() for r in caplog.records)


def test_empty_kind_is_a_noop(caplog):
    caplog.set_level(logging.DEBUG, logger="src.observability.context")
    record_kind(None)
    record_kind("")
    assert not caplog.records


def test_never_raises_on_a_broken_span():
    class _Broken:
        def is_recording(self):
            raise RuntimeError("boom")

    with patch("src.observability.context.trace.get_current_span", return_value=_Broken()):
        record_kind("timeout", node="agent")


def _drain(logger: ErrorLogger) -> None:
    if logger._flush_task and not logger._flush_task.done():
        logger._flush_task.cancel()
    logger._buffer.clear()


class TestLogErrorMirrorsKind:
    @pytest.mark.asyncio
    async def test_log_error_records_kind(self):
        logger = ErrorLogger()
        with patch("src.utils.error_logger.record_kind") as mirrored:
            await logger.log_error(
                RuntimeError("idle"),
                description="Policy 'timeout' on node 'agent'",
                kind="timeout",
                node="agent",
                graph="react",
                error_extra={"attempt": 1},
            )
        mirrored.assert_called_once_with(
            "timeout",
            node="agent",
            graph="react",
            source=None,
            description="Policy 'timeout' on node 'agent'",
            extra={"attempt": 1},
        )
        assert len(logger._buffer) == 1
        assert logger._buffer[0].kind == "timeout"
        _drain(logger)

    @pytest.mark.asyncio
    async def test_policy_alias_is_recorded_as_kind(self):
        logger = ErrorLogger()
        with patch("src.utils.error_logger.record_kind") as mirrored:
            await logger.log_error(
                RuntimeError("retrying"), description="retry", policy="retry", node="agent"
            )
        assert mirrored.call_args.args[0] == "retry"
        _drain(logger)

    @pytest.mark.asyncio
    async def test_mcp_load_is_not_a_span_event(self):
        logger = ErrorLogger()
        with patch("src.utils.error_logger.record_kind") as mirrored:
            await logger.log_error(
                RuntimeError("down"),
                description="server skipped",
                kind="mcp_load",
                source="mcp_load",
            )
        mirrored.assert_not_called()
        assert len(logger._buffer) == 1
        _drain(logger)

    @pytest.mark.asyncio
    async def test_rag_still_writes_mongo(self):
        logger = ErrorLogger()
        with patch("src.utils.error_logger.record_kind") as mirrored:
            await logger.log_error(
                RuntimeError("llm"),
                description="generation failed",
                kind="rag",
                component="LLM",
                pipeline_stage="generation",
            )
        mirrored.assert_not_called()
        assert len(logger._buffer) == 1
        assert logger._buffer[0].kind == "rag"
        _drain(logger)

    @pytest.mark.asyncio
    async def test_event_lands_on_the_span_that_is_current(self, tracer):
        tracer, exporter = tracer
        logger = ErrorLogger()
        with tracer.start_as_current_span("execute_task agent"):
            await logger.log_error(
                RuntimeError("idle"),
                description="Policy 'fallback' on node 'agent'",
                kind="fallback",
                node="agent",
                graph="react",
            )
        (span,) = exporter.get_finished_spans()
        assert [e.name for e in span.events] == ["fallback"]
        _drain(logger)
