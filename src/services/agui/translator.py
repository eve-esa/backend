"""Legacy stream bus frames of one agentic run, translated to AG-UI 1.0 events.

Pure: no FastAPI, no I/O, no clock. The route builds one ``AguiTranslator`` per
run, calls ``start()`` before it subscribes to the bus, ``translate()`` once per
legacy frame (already parsed from its ``data:`` line into a dict), and
``finish()`` when the bus closes. Each call returns the AG-UI events to send, in
order. The translator holds the open text message, tool calls, steps,
subagents and activities, closes them before the terminal event, and drops
every frame after the first terminal one.

Input rejected before the run starts never reaches the translator: it is an
HTTP error with no event stream, shaped by ``pre_stream_error``.

Logging: event types, ids and counts only. No user text, no tool arguments,
no tool results, no tokens, no raw exception text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

# One legacy frame as the runner publishes it, parsed from its ``data:`` line.
LegacyFrame = Mapping[str, Any]
# An AG-UI event model (``ag_ui.core``); typed loosely until the SDK is pinned.
AguiEvent = Any


@dataclass(frozen=True)
class RunContext:
    """Ids and labels of one run, known to the route before the first frame."""

    thread_id: str
    run_id: str
    message_id: str
    trace_id: Optional[str] = None
    graph: Optional[str] = None


@dataclass(frozen=True)
class PreStreamError:
    """An HTTP error answered instead of opening the event stream."""

    status_code: int
    body: Dict[str, Any]
    headers: Dict[str, str] = field(default_factory=dict)


def pre_stream_error(
    code: str,
    *,
    retry_after: Optional[int] = None,
) -> PreStreamError:
    """HTTP status, ``{"detail": {"code", "message"}}`` body and headers for ``code``."""
    raise NotImplementedError


class AguiTranslator:
    """Stateful translator for the frames of exactly one run."""

    def __init__(self, context: RunContext) -> None:
        self.context = context

    @property
    def terminated(self) -> bool:
        """True once a terminal event (``RUN_FINISHED`` or ``RUN_ERROR``) was returned."""
        raise NotImplementedError

    def start(self) -> List[AguiEvent]:
        """``RUN_STARTED`` with the run ids and the ``eve.*`` metadata."""
        raise NotImplementedError

    def translate(self, frame: LegacyFrame) -> List[AguiEvent]:
        """Events for one legacy frame; empty after the terminal event."""
        raise NotImplementedError

    def finish(self) -> List[AguiEvent]:
        """Events for the bus closing: ``RUN_ERROR stream_incomplete`` if no terminal frame came."""
        raise NotImplementedError
