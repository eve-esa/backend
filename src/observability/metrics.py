"""OTLP metrics: event loop lag and in-flight generations, per worker.

Built by :func:`src.observability.init_telemetry` only when an OTLP endpoint is
set, so with telemetry off nothing here imports the SDK. The meter provider is
private to EVE: it is not registered as the global one, so the client
instrumentations keep emitting spans only and add no HTTP metric series.

Two observable gauges, both read on the exporter thread at each collection:

- ``eve.event_loop.lag_seconds``: the longest delay since the previous
  collection between the moment the loop should have woken a short sleep and
  the moment it did. A synchronous call on the loop (a blocking client, a CPU
  bound parse) shows up here as seconds; a healthy loop stays in milliseconds.
- ``eve.generations.in_flight``: the slots held in this worker's generation
  limiter (:mod:`src.services.load_shedding`), next to its cap.

One counter, added to on the request path:

- ``eve.rate_limit.decisions``: one per request rate limit decision
  (:mod:`src.services.request_rate_limiter`), with ``class``, ``decision``
  (allowed, limited, skipped_store_down), ``mode`` and ``subject_kind``. No
  user, key or address ever goes into an attribute.
"""

import asyncio
import logging
import threading
import time
from typing import Callable, Iterable, Optional

logger = logging.getLogger(__name__)

METER_NAME = "eve.backend"
LOOP_LAG_METRIC = "eve.event_loop.lag_seconds"
IN_FLIGHT_METRIC = "eve.generations.in_flight"
RATE_LIMIT_METRIC = "eve.rate_limit.decisions"
EXPORT_INTERVAL_MILLIS = 15_000
# The probe sleeps this long and records how late it woke. Short on purpose: a
# block only shows when it covers a wake-up, so a 1 s probe would miss most
# blocks under a second and under-report the rest by up to the interval. Ten
# wake-ups a second cost microseconds each.
LOOP_LAG_PROBE_SECONDS = 0.1


class LoopLagMonitor:
    """Background task measuring how late the event loop wakes a sleep.

    :meth:`observe` returns the largest lag seen since the previous call and
    resets it. A probe still waiting past its deadline counts too, so a loop
    wedged for longer than a collection interval reports the wedge instead of
    zero. Safe to observe from another thread.
    """

    def __init__(self, interval: float = LOOP_LAG_PROBE_SECONDS) -> None:
        self.interval = interval
        self._lock = threading.Lock()
        self._max = 0.0
        self._deadline: Optional[float] = None
        self._task: Optional[asyncio.Task] = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> bool:
        """Start on the running loop. Returns False when there is none or it runs."""
        if self.running:
            return False
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False
        self._task = loop.create_task(self._probe(), name="eve-loop-lag-probe")
        return True

    def stop(self) -> None:
        task, self._task = self._task, None
        self._deadline = None
        if task is not None and not task.done():
            task.cancel()

    async def _probe(self) -> None:
        try:
            while True:
                deadline = time.monotonic() + self.interval
                self._deadline = deadline
                await asyncio.sleep(self.interval)
                self._record(time.monotonic() - deadline)
        finally:
            self._deadline = None

    def _record(self, lag: float) -> None:
        with self._lock:
            if lag > self._max:
                self._max = lag

    def observe(self) -> float:
        now = time.monotonic()
        with self._lock:
            value, self._max = self._max, 0.0
        deadline = self._deadline
        if deadline is not None:
            value = max(value, now - deadline)
        return max(value, 0.0)


_monitor = LoopLagMonitor()


def get_loop_lag_monitor() -> LoopLagMonitor:
    return _monitor


# Set by register_instruments; None while metrics are off, so recording is a no-op.
_rate_limit_counter = None


def record_rate_limit_decision(
    route_class: str, decision: str, mode: str, subject_kind: str
) -> None:
    """Count one rate limit decision. No-op without a meter provider; never raises."""
    counter = _rate_limit_counter
    if counter is None:
        return
    try:
        counter.add(
            1,
            {
                "class": route_class,
                "decision": decision,
                "mode": mode,
                "subject_kind": subject_kind,
            },
        )
    except Exception:  # noqa: BLE001 - a metric must never fail a request
        logger.debug("rate limit counter add failed", exc_info=True)


def _default_limiter():
    from src.services.load_shedding import get_generation_limiter

    return get_generation_limiter()


def register_instruments(
    meter_provider,
    monitor: Optional[LoopLagMonitor] = None,
    limiter_getter: Optional[Callable[[], object]] = None,
) -> None:
    """Create the gauges and the counter on ``meter_provider``. Tests pass their own sources."""
    global _rate_limit_counter
    from opentelemetry.metrics import CallbackOptions, Observation

    monitor = monitor or _monitor
    limiter_getter = limiter_getter or _default_limiter
    meter = meter_provider.get_meter(METER_NAME)

    def loop_lag(_options: CallbackOptions) -> Iterable[Observation]:
        if monitor.running:
            yield Observation(monitor.observe())

    def in_flight(_options: CallbackOptions) -> Iterable[Observation]:
        limiter = limiter_getter()
        yield Observation(
            limiter.in_flight, {"eve.generations.limit": int(limiter.limit)}
        )

    meter.create_observable_gauge(
        LOOP_LAG_METRIC,
        callbacks=[loop_lag],
        unit="s",
        description="Longest event loop wake-up delay since the previous collection",
    )
    meter.create_observable_gauge(
        IN_FLIGHT_METRIC,
        callbacks=[in_flight],
        unit="{generation}",
        description="Answer generations holding a load-shedding slot in this worker",
    )
    _rate_limit_counter = meter.create_counter(
        RATE_LIMIT_METRIC,
        unit="{request}",
        description="Request rate limit decisions by class, decision, mode and subject kind",
    )


def build_meter_provider(
    resource,
    reader=None,
    monitor: Optional[LoopLagMonitor] = None,
    limiter_getter: Optional[Callable[[], object]] = None,
):
    """MeterProvider with the EVE gauges; OTLP HTTP every 15 s unless ``reader``.

    The OTLP exporter reads the endpoint and headers from the same
    ``OTEL_EXPORTER_OTLP_*`` variables as the trace exporter and appends
    ``/v1/metrics``.
    """
    from opentelemetry.sdk.metrics import MeterProvider

    if reader is None:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

        reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(), export_interval_millis=EXPORT_INTERVAL_MILLIS
        )
    provider = MeterProvider(resource=resource, metric_readers=[reader])
    register_instruments(provider, monitor, limiter_getter)
    return provider
