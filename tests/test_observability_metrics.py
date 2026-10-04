"""OTLP metrics: off without an endpoint, event loop lag, in-flight generations,
shutdown.

Gauges are read through the SDK in-memory reader on the same factory the
server uses; no global meter provider is registered here or in production.
"""

import asyncio
import logging
import time

import pytest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.resources import Resource

from src import observability
from src.observability.metrics import (
    IN_FLIGHT_METRIC,
    LOOP_LAG_METRIC,
    LoopLagMonitor,
    build_meter_provider,
)
from src.services.load_shedding import GenerationLimiter

pytestmark = pytest.mark.no_db

RESOURCE = Resource.create({"service.name": "eve-backend-test"})


def _points(reader, name):
    data = reader.get_metrics_data()
    if data is None:
        return []
    return [
        point
        for rm in data.resource_metrics
        for sm in rm.scope_metrics
        for metric in sm.metrics
        if metric.name == name
        for point in metric.data.data_points
    ]


@pytest.fixture
def clean_state(monkeypatch):
    """Telemetry state as a fresh worker sees it, restored afterwards."""
    saved = dict(observability._state)
    for key in observability._state:
        observability._state[key] = None
    observability._state["enabled"] = False
    yield
    observability._state.clear()
    observability._state.update(saved)


async def test_no_meter_provider_without_endpoint(monkeypatch, clean_state):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    assert observability.init_telemetry() is False
    assert observability._state["meter_provider"] is None
    # The lifespan hook is a no-op: no probe task on the loop.
    assert observability.start_runtime_metrics() is False
    assert not [t for t in asyncio.all_tasks() if t.get_name() == "eve-loop-lag-probe"]


async def test_loop_lag_gauge_reports_a_blocked_loop():
    reader = InMemoryMetricReader()
    monitor = LoopLagMonitor(interval=0.05)
    provider = build_meter_provider(
        RESOURCE,
        reader=reader,
        monitor=monitor,
        limiter_getter=lambda: GenerationLimiter(1),
    )
    assert monitor.start() is True
    try:
        await asyncio.sleep(0.3)
        quiet = [p.value for p in _points(reader, LOOP_LAG_METRIC)]
        assert quiet and quiet[0] < 0.2

        async def blocker():
            time.sleep(0.3)  # the synchronous call the gauge exists to catch

        await asyncio.create_task(blocker())
        await asyncio.sleep(0.1)
        blocked = [p.value for p in _points(reader, LOOP_LAG_METRIC)]
        assert blocked and blocked[0] > 0.2
        # Max since the last observation: the next quiet interval is small again.
        await asyncio.sleep(0.2)
        after = [p.value for p in _points(reader, LOOP_LAG_METRIC)]
        assert after and after[0] < 0.2
    finally:
        monitor.stop()
        provider.shutdown()


async def test_loop_lag_counts_a_probe_still_waiting():
    monitor = LoopLagMonitor(interval=0.05)
    assert monitor.start() is True
    try:
        await asyncio.sleep(0.1)
        # Observed from inside the block, as the exporter thread would.
        time.sleep(0.3)
        assert monitor.observe() > 0.2
    finally:
        monitor.stop()


async def test_in_flight_gauge_equals_held_slots():
    reader = InMemoryMetricReader()
    limiter = GenerationLimiter(5)
    provider = build_meter_provider(
        RESOURCE,
        reader=reader,
        monitor=LoopLagMonitor(),
        limiter_getter=lambda: limiter,
    )
    try:
        slots = [await limiter.try_acquire("test") for _ in range(3)]
        points = _points(reader, IN_FLIGHT_METRIC)
        assert [p.value for p in points] == [3]
        assert dict(points[0].attributes) == {"eve.generations.limit": 5}
        slots[0].release()
        slots[0].release()  # idempotent, still one slot back
        assert [p.value for p in _points(reader, IN_FLIGHT_METRIC)] == [2]
        for slot in slots[1:]:
            slot.release()
        assert [p.value for p in _points(reader, IN_FLIGHT_METRIC)] == [0]
    finally:
        provider.shutdown()


async def test_shutdown_flushes_and_never_raises(clean_state, caplog):
    reader = InMemoryMetricReader()
    provider = build_meter_provider(RESOURCE, reader=reader)
    observability._state["meter_provider"] = provider
    assert observability.start_runtime_metrics() is True
    await asyncio.sleep(0.15)
    with caplog.at_level(logging.WARNING, logger="src.observability"):
        observability.shutdown()
        observability.shutdown()  # second call: the SDK only logs
    assert "shutdown failed" not in caplog.text
    probes = [t for t in asyncio.all_tasks() if t.get_name() == "eve-loop-lag-probe"]
    await asyncio.sleep(0)
    assert all(t.done() for t in probes)
    # Telemetry off: nothing to flush, nothing raised.
    observability._state["meter_provider"] = None
    observability.shutdown()


async def test_init_builds_the_otlp_meter_provider(monkeypatch, clean_state):
    """The real path: OTLP exporter on an unroutable endpoint, init then shutdown."""
    from src.observability import metrics

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:1")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TIMEOUT", "1")
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    # Only the provider wiring is under test: no global providers, no client
    # instrumentation, no root log handler left behind for other tests.
    monkeypatch.setattr("opentelemetry.trace.set_tracer_provider", lambda tp: None)
    monkeypatch.setattr(observability, "_instrument_clients", lambda tp: None)
    monkeypatch.setattr(observability, "instrument_genai", lambda tp: False)
    monkeypatch.setattr(observability, "_setup_logs", lambda resource: None)
    seen = []
    real_build = metrics.build_meter_provider

    def spy(resource, *args, **kwargs):
        seen.append(resource)
        return real_build(resource, *args, **kwargs)

    monkeypatch.setattr(metrics, "build_meter_provider", spy)
    assert observability.init_telemetry() is True
    assert observability._state["meter_provider"] is not None
    # Metrics carry the same resource as traces.
    assert seen == [observability._state["tracer_provider"].resource]
    assert observability.start_runtime_metrics() is True
    started = time.monotonic()
    observability.shutdown()
    assert time.monotonic() - started < 10
    await asyncio.sleep(0)
    assert not metrics.get_loop_lag_monitor().running
