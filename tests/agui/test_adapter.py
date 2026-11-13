"""Red tests for the AG-UI adapter: legacy stream bus frames in, AG-UI 1.0 events out.

Every legacy frame below is built the way the agentic runner publishes it today
(backend main 0c98821); the comment on each builder names the emission site.
Fields marked "B-S2" are the additive enrichments the AG-UI contract asks the
runner for (``message_id`` on every frame, real tool status, ``server``,
``duration_ms``, ``endpoint``, ``usage``, the ``sources`` frame); the old
client ignores them.

Events are compared as JSON dicts (camelCase, ``None`` dropped), so the tests
do not depend on which event model the adapter returns. Every transcript is
also checked by the shared transcript validator.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Optional

import pytest

from src.services.agui.adapter import (
    AguiAdapter,
    encode_sse,
    RunContext,
    pre_stream_error,
)

try:
    from tests.agui.validator import validate_transcript as _validate_transcript

    def validate(transcript: str) -> List[Any]:
        return _validate_transcript(transcript, thread_id=T, run_id=R)

except ImportError:  # pragma: no cover - shim until the validator branch merges

    def validate(transcript: str) -> List[Any]:
        pytest.fail("tests.agui.validator is missing: merge the validator first")


pytestmark = pytest.mark.no_db

T = "6a3f0c2e9b1d4e7f8a2b3c4d"  # threadId, the conversation id
M = "6a3f0c4a9b1d4e7f8a2b3c55"  # the answer messageId, the Mongo message id
R = "0f8e6c1a-3b2d-4e5f-9a7b-1c2d3e4f5a6b"  # runId, minted by the client
TRACE = "4481c48c462beed158f2fe9a26f5b456"

TERMINAL_TYPES = {"RUN_FINISHED", "RUN_ERROR"}
COPY_TIMEOUT = "The model did not answer in time. Retry in a moment."
COPY_EMPTY = "The model returned an empty answer. Retry."
COPY_UPSTREAM = "Something went wrong while answering. Retry."
COPY_INCOMPLETE = "The answer stream was interrupted. Reload the conversation."

ENDPOINT = {
    "requested": None,
    "chain": ["eve_jsc", "fallback"],
    "answered": "eve_jsc",
    "substituted": False,
    "circuit_open": [],
}


# ─── Legacy frames, as the runner publishes them ──────────────────────────────


def f_status() -> Dict[str, Any]:
    # runner.py:2189 `{'type': 'status', 'content': 'Thinking…'}`
    return {"type": "status", "content": "Thinking…", "message_id": M}


def f_token(text: str) -> Dict[str, Any]:
    # runner.py:2369 `{'type': 'token', 'content': tok}`
    return {"type": "token", "content": text, "message_id": M}


def f_tool_call(
    call_id: Optional[str],
    tool: str,
    args: Dict[str, Any],
    *,
    server: str,
    label: str,
    args_truncated: bool = False,
) -> Dict[str, Any]:
    # runner.py:2306-2311: content is "<label>: <query>" or "<label>…", query
    # is args["query"] or None, args went through _stream_tool_args
    # (runner.py:215, strings cut at STREAM_TOOL_ARG_MAX_CHARS=500 plus "…",
    # runner.py:212). B-S2: server, args_truncated, message_id.
    query = args.get("query", "")
    return {
        "type": "tool_call",
        "content": f"{label}: {query}" if query else f"{label}…",
        "tool": tool,
        "label": label,
        "query": query or None,
        "id": call_id,
        "args": args,
        "server": server,
        "args_truncated": args_truncated,
        "message_id": M,
    }


def f_tool_result(
    call_id: Optional[str],
    tool: str,
    content: str,
    *,
    status: str = "ok",
    server: str,
    duration_ms: int,
) -> Dict[str, Any]:
    # runner.py:2508 `{'type': 'tool_result', 'content': preview, 'tool': name,
    # 'status': 'ok', 'id': tool_call_id}`, preview = first 200 chars
    # (runner.py:2495). B-S2: real status (ok, error, timeout), server,
    # duration_ms, message_id.
    return {
        "type": "tool_result",
        "content": content[:200],
        "tool": tool,
        "status": status,
        "id": call_id,
        "server": server,
        "duration_ms": duration_ms,
        "message_id": M,
    }


def f_summarizing(phase: str) -> Dict[str, Any]:
    # runner.py:1462-1468 `_summarizing_event`
    return {
        "type": "summarizing",
        "phase": phase,
        "first_message_id": "6a3f0b009b1d4e7f8a2b3c01",
        "last_message_id": "6a3f0b109b1d4e7f8a2b3c09",
        "message_id": M,
    }


def f_final(
    answer: str,
    *,
    latencies: Optional[Dict[str, Any]] = None,
    artifact_ids: Optional[List[str]] = None,
    endpoint: Optional[Dict[str, Any]] = None,
    usage: Optional[List[Dict[str, Any]]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    # runner.py:2164-2167 `_final_events`: answer, latencies, artifact_ids
    # (None when empty), trace_id. B-S2: endpoint, usage, message_id.
    # `error`: the late failure today persisted only in metadata.error while the
    # runner still sends `final` with latencies {} (runner.py:2708, 2741); the
    # adapter needs it on the frame to end with RUN_ERROR (contract decision 1).
    frame: Dict[str, Any] = {
        "type": "final",
        "answer": answer,
        "latencies": latencies if latencies is not None else {},
        "artifact_ids": artifact_ids,
        "trace_id": TRACE,
        "endpoint": endpoint,
        "usage": usage,
        "message_id": M,
    }
    if error is not None:
        frame["error"] = error
    return frame


def f_stopped() -> Dict[str, Any]:
    # runner.py:2186 and 2423 `{'type': 'stopped'}`; the Stop route publishes the
    # same frame through bus.stop (routers/message.py:120), without message_id.
    return {"type": "stopped"}


def f_error(code: str, message: str) -> Dict[str, Any]:
    # runner.py:2123, 2711, 2744 and the outer safety net runner.py:2863-2874:
    # `{'type': 'error', 'code': ..., 'message': ...}` from build_error_payload
    # (generate_answer.py:181-197, message is str(exc)[:500]).
    return {"type": "error", "code": code, "message": message, "message_id": M}


def f_sources(call_id: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    # New frame (B-SRC), published at tool result time.
    return {"type": "sources", "tool_call_id": call_id, "documents": documents, "message_id": M}


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _ctx(graph: str = "react") -> RunContext:
    return RunContext(thread_id=T, run_id=R, message_id=M, trace_id=TRACE, graph=graph)


def _as_dict(event: Any) -> Dict[str, Any]:
    if hasattr(event, "model_dump"):
        return event.model_dump(mode="json", by_alias=True, exclude_none=True)
    assert isinstance(event, dict), f"unexpected event object {type(event)!r}"
    return event


def _run(frames: Iterable[Dict[str, Any]], *, graph: str = "react", finish: bool = True) -> List[Dict[str, Any]]:
    adapter = AguiAdapter(_ctx(graph))
    out: List[Any] = list(adapter.start())
    for frame in frames:
        out.extend(adapter.adapt(frame))
    if finish:
        out.extend(adapter.finish())
    return [_as_dict(e) for e in out]


def _types(events: List[Dict[str, Any]]) -> List[str]:
    return [e["type"] for e in events]


def _of(events: List[Dict[str, Any]], kind: str) -> List[Dict[str, Any]]:
    return [e for e in events if e["type"] == kind]


def _one(events: List[Dict[str, Any]], kind: str) -> Dict[str, Any]:
    found = _of(events, kind)
    assert len(found) == 1, f"expected one {kind}, got {len(found)}: {_types(events)}"
    return found[0]


def _sse(events: List[Dict[str, Any]]) -> str:
    return "".join(f"data: {json.dumps(e, separators=(',', ':'))}\n\n" for e in events)


def _assert_valid(events: List[Dict[str, Any]]) -> None:
    failures = list(validate(_sse(events)))
    assert not failures, failures


def _assert_ends_with(events: List[Dict[str, Any]], kind: str) -> Dict[str, Any]:
    terminals = [e for e in events if e["type"] in TERMINAL_TYPES]
    assert len(terminals) == 1, _types(events)
    assert events[-1]["type"] == kind, _types(events)
    return events[-1]


def _assert_run_error(events: List[Dict[str, Any]], code: str, copy: str) -> Dict[str, Any]:
    err = _assert_ends_with(events, "RUN_ERROR")
    assert err["code"] == code
    assert err["metadata"]["eve.error.code"] == code
    assert err["message"] == copy
    assert not _of(events, "RUN_FINISHED")
    return err


def _assert_cancelled(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    fin = _assert_ends_with(events, "RUN_FINISHED")
    assert fin["threadId"] == T and fin["runId"] == R
    assert fin["outcome"]["type"] == "cancelled"
    assert "result" not in fin
    assert fin["metadata"]["eve.stopped"] is True
    return fin


# ─── Cases ────────────────────────────────────────────────────────────────────


def test_plain_answer():
    usage = [{"provider": "jsc", "model": "20 - EVE-Instruct", "input_tokens": 5120, "output_tokens": 64}]
    events = _run(
        [
            f_status(),
            f_token("Sentinel-2 revisits "),
            f_token("every 5 days."),
            f_final(
                "Sentinel-2 revisits every 5 days.",
                latencies={"total_s": 9.4},
                artifact_ids=["6a3f0c4b9b1d4e7f8a2b3c60"],
                endpoint=ENDPOINT,
                usage=usage,
            ),
        ]
    )

    assert _types(events) == [
        "RUN_STARTED",
        "STEP_STARTED",
        "STEP_FINISHED",
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_END",
        "CUSTOM",
        "RUN_FINISHED",
    ]
    started = events[0]
    assert started["threadId"] == T and started["runId"] == R
    assert started["protocolVersion"] == "1.0"
    assert started["metadata"] == {
        "eve.traceId": TRACE,
        "eve.conversationId": T,
        "eve.messageId": M,
        "eve.graph": "react",
    }
    assert events[1]["stepName"] == "thinking" and events[2]["stepName"] == "thinking"
    assert events[3] == {"type": "TEXT_MESSAGE_START", "messageId": M, "role": "assistant"}
    assert [e["delta"] for e in _of(events, "TEXT_MESSAGE_CONTENT")] == ["Sentinel-2 revisits ", "every 5 days."]
    assert {e["messageId"] for e in _of(events, "TEXT_MESSAGE_CONTENT")} == {M}
    end = _one(events, "TEXT_MESSAGE_END")
    assert end["messageId"] == M and end["metadata"]["eve.endpoint"] == ENDPOINT
    assert events[7]["name"] == "eve.artifacts"
    assert events[7]["value"] == {"artifactIds": ["6a3f0c4b9b1d4e7f8a2b3c60"]}

    fin = _assert_ends_with(events, "RUN_FINISHED")
    assert fin["threadId"] == T and fin["runId"] == R
    assert fin["outcome"] == {"type": "success"}
    assert fin["metadata"]["eve.latencies"] == {"total_s": 9.4}
    assert "eve.error.code" not in fin["metadata"] and "eve.stopped" not in fin["metadata"]
    (row,) = fin["usage"]
    assert (row["provider"], row["model"], row["inputTokens"], row["outputTokens"]) == ("jsc", "20 - EVE-Instruct", 5120, 64)
    _assert_valid(events)


def test_plain_answer_without_streamed_tokens_sends_one_text_triad():
    # A final with an answer and no token frame before it (a text-format turn
    # flushed at the end): the answer still travels in standard text events.
    events = _run([f_status(), f_final("Five days.", endpoint=ENDPOINT)])

    assert _types(events)[-4:] == ["TEXT_MESSAGE_START", "TEXT_MESSAGE_CONTENT", "TEXT_MESSAGE_END", "RUN_FINISHED"]
    assert _one(events, "TEXT_MESSAGE_CONTENT")["delta"] == "Five days."
    assert _assert_ends_with(events, "RUN_FINISHED")["outcome"] == {"type": "success"}
    _assert_valid(events)


def test_mcp_tool_call_with_args_and_result():
    args = {"query": "Sentinel-2 revisit time", "k": 5}
    events = _run(
        [
            f_status(),
            f_tool_call("call_a1", "eve_retrieval_retrieve", args, server="eve_retrieval", label="Searching EVE sources"),
            f_tool_result("call_a1", "eve_retrieval_retrieve", "10 documents found", server="eve_retrieval", duration_ms=2140),
            f_token("Every 5 days."),
            f_final("Every 5 days.", endpoint=ENDPOINT),
        ]
    )

    assert _types(events) == [
        "RUN_STARTED",
        "STEP_STARTED",
        "STEP_FINISHED",
        "TOOL_CALL_START",
        "TOOL_CALL_ARGS",
        "TOOL_CALL_END",
        "TOOL_CALL_RESULT",
        "STEP_STARTED",
        "STEP_FINISHED",
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_END",
        "RUN_FINISHED",
    ]
    start = _one(events, "TOOL_CALL_START")
    assert start["toolCallId"] == "call_a1"
    assert start["toolCallName"] == "eve_retrieval_retrieve"
    assert start["parentMessageId"] == M
    assert start["metadata"] == {"eve.server": "eve_retrieval", "eve.label": "Searching EVE sources"}
    tool_args = _one(events, "TOOL_CALL_ARGS")
    assert tool_args["toolCallId"] == "call_a1"
    assert json.loads(tool_args["delta"]) == args
    assert _one(events, "TOOL_CALL_END")["toolCallId"] == "call_a1"
    result = _one(events, "TOOL_CALL_RESULT")
    assert result["messageId"] == f"{M}:tool:call_a1"
    assert result["toolCallId"] == "call_a1"
    assert result["role"] == "tool"
    assert result["content"] == "10 documents found"
    assert result["metadata"]["eve.status"] == "ok"
    assert result["metadata"]["eve.server"] == "eve_retrieval"
    assert result["metadata"]["eve.durationMs"] == 2140
    assert _assert_ends_with(events, "RUN_FINISHED")["outcome"] == {"type": "success"}
    _assert_valid(events)


def test_tool_call_with_cut_args_is_flagged():
    long_query = "q" * 600
    streamed = {"query": long_query[:500] + "…"}  # what _stream_tool_args sends (runner.py:220)
    events = _run(
        [
            f_status(),
            f_tool_call("call_t1", "eve_retrieval_retrieve", streamed, server="eve_retrieval", label="Searching EVE sources", args_truncated=True),
            f_tool_result("call_t1", "eve_retrieval_retrieve", "0 documents", server="eve_retrieval", duration_ms=900),
            f_final("Nothing found.", endpoint=ENDPOINT),
        ]
    )

    assert _one(events, "TOOL_CALL_START")["metadata"]["eve.argsTruncated"] is True
    assert json.loads(_one(events, "TOOL_CALL_ARGS")["delta"]) == streamed
    _assert_valid(events)


def test_tool_error_is_reported_on_the_result_and_the_run_goes_on():
    events = _run(
        [
            f_status(),
            f_tool_call("call_e1", "gdacs_events", {"country": "IT"}, server="gdacs", label="Searching GDACS"),
            f_tool_result("call_e1", "gdacs_events", "Tool error: upstream answered 502", status="error", server="gdacs", duration_ms=310),
            f_token("GDACS is not reachable right now."),
            f_final("GDACS is not reachable right now.", endpoint=ENDPOINT),
        ]
    )

    result = _one(events, "TOOL_CALL_RESULT")
    assert result["toolCallId"] == "call_e1"
    assert result["metadata"]["eve.status"] == "error"
    assert result["metadata"]["eve.server"] == "gdacs"
    assert result["metadata"]["eve.durationMs"] == 310
    assert _types(events)[_types(events).index("TOOL_CALL_RESULT") + 1] == "STEP_STARTED"
    assert not _of(events, "RUN_ERROR")
    assert _assert_ends_with(events, "RUN_FINISHED")["outcome"] == {"type": "success"}
    _assert_valid(events)


def test_tool_timeout_is_reported_on_the_result():
    events = _run(
        [
            f_status(),
            f_tool_call("call_o1", "eotdl_search", {"query": "flood masks"}, server="eotdl", label="Searching EOTDL"),
            f_tool_result("call_o1", "eotdl_search", "Tool timed out after 30 s", status="timeout", server="eotdl", duration_ms=30000),
            f_token("EOTDL did not answer in time."),
            f_final("EOTDL did not answer in time.", endpoint=ENDPOINT),
        ]
    )

    result = _one(events, "TOOL_CALL_RESULT")
    assert result["toolCallId"] == "call_o1"
    assert result["metadata"]["eve.status"] == "timeout"
    assert result["metadata"]["eve.durationMs"] == 30000
    assert _assert_ends_with(events, "RUN_FINISHED")["outcome"] == {"type": "success"}
    _assert_valid(events)


def test_stop_mid_tool_cancels_and_closes_what_is_open():
    adapter = AguiAdapter(_ctx("deep"))
    raw: List[Any] = list(adapter.start())
    for frame in [
        f_status(),
        f_token("Let me check the archive."),
        f_tool_call("call_b2", "eotdl_search", {"query": "flood masks"}, server="eotdl", label="Searching EOTDL"),
        f_stopped(),
    ]:
        raw.extend(adapter.adapt(frame))
    assert adapter.terminated is True
    # A second stopped from the Stop route and the bus close add nothing.
    assert list(adapter.adapt(f_stopped())) == []
    assert list(adapter.finish()) == []
    events = [_as_dict(e) for e in raw]

    assert _types(events) == [
        "RUN_STARTED",
        "STEP_STARTED",
        "STEP_FINISHED",
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_END",
        "TOOL_CALL_START",
        "TOOL_CALL_ARGS",
        "TOOL_CALL_END",
        "RUN_FINISHED",
    ]
    assert _one(events, "TOOL_CALL_END")["toolCallId"] == "call_b2"
    assert not _of(events, "TOOL_CALL_RESULT")  # no ghost result on a cancel
    _assert_cancelled(events)
    _assert_valid(events)


@pytest.mark.parametrize(
    "frames",
    [
        pytest.param([f_stopped()], id="cancelled-before-status"),  # runner.py:2186
        pytest.param([f_status(), f_stopped()], id="stop-while-thinking"),  # routers/message.py:120
    ],
)
def test_stop_before_the_first_token(frames):
    events = _run(frames)

    assert not [t for t in _types(events) if t.startswith("TEXT_MESSAGE")]
    assert len(_of(events, "STEP_STARTED")) == len(_of(events, "STEP_FINISHED"))
    _assert_cancelled(events)
    _assert_valid(events)


def test_empty_answer_is_a_run_error():
    # Today the runner sends `final` with answer "" (runner.py:2649, clear_error
    # false); the user got nothing, so the run ends with empty_answer.
    events = _run([f_status(), f_final("", endpoint=ENDPOINT)])

    assert not [t for t in _types(events) if t.startswith("TEXT_MESSAGE")]
    _assert_run_error(events, "empty_answer", COPY_EMPTY)
    _assert_valid(events)


@pytest.mark.parametrize(
    "retry_after, header",
    [pytest.param(3600, {"Retry-After": "3600"}, id="reset-known"), pytest.param(None, {}, id="reset-unknown")],
)
def test_budget_429_is_an_http_error_before_the_stream(retry_after, header):
    # The legacy route raises HTTPException(429, detail=<string>) from
    # enforce_token_budget_or_raise (token_rate_limiter.py:261-270) before any
    # frame; the AG-UI route answers the same status with the wrapped detail and
    # never opens a stream with RUN_ERROR.
    error = pre_stream_error("budget_exhausted", retry_after=retry_after)

    assert error.status_code == 429
    assert error.body["detail"]["code"] == "budget_exhausted"
    assert isinstance(error.body["detail"]["message"], str) and error.body["detail"]["message"]
    assert error.headers == header


@pytest.mark.parametrize(
    "frames, keeps_text",
    [
        pytest.param(
            [
                f_status(),
                f_tool_call("call_d1", "eve_retrieval_retrieve", {"query": "SAR"}, server="eve_retrieval", label="Searching EVE sources"),
            ],
            False,
            id="no-text",
        ),
        pytest.param([f_status(), f_token("Partial answer")], True, id="after-text"),
    ],
)
def test_generation_deadline_is_run_error_timeout(frames, keeps_text):
    # The outer safety net (runner.py:2863-2874) publishes `error` with the code
    # of GenerationDeadlineExceeded (generate_answer.py:2326-2333): "timeout".
    raw = "Generation did not finish within 240 seconds"
    events = _run([*frames, f_error("timeout", raw)])

    _assert_run_error(events, "timeout", COPY_TIMEOUT)
    assert raw not in json.dumps(events)
    assert not _of(events, "TOOL_CALL_RESULT")  # ghost results only on success
    if keeps_text:
        assert _types(events)[-4:] == ["TEXT_MESSAGE_START", "TEXT_MESSAGE_CONTENT", "TEXT_MESSAGE_END", "RUN_ERROR"]
        assert _one(events, "TEXT_MESSAGE_CONTENT")["delta"] == "Partial answer"
    _assert_valid(events)


@pytest.mark.parametrize(
    "terminal_frame",
    [
        pytest.param(
            f_final("Sentinel-2 revisits", error={"code": "upstream_error", "message": "RuntimeError: boom in agent node"}),
            id="final-with-late-error",
        ),
        pytest.param(f_error("upstream_error", "RuntimeError: boom in agent node"), id="error-frame"),
    ],
)
def test_error_after_streamed_text_is_run_error_and_keeps_the_partial(terminal_frame):
    events = _run([f_status(), f_token("Sentinel-2 "), f_token("revisits"), terminal_frame])

    assert _types(events)[-5:] == [
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_END",
        "RUN_ERROR",
    ]
    assert "".join(e["delta"] for e in _of(events, "TEXT_MESSAGE_CONTENT")) == "Sentinel-2 revisits"
    _assert_run_error(events, "upstream_error", COPY_UPSTREAM)
    assert "boom" not in json.dumps(events)
    _assert_valid(events)


def test_text_format_tool_calls_get_unique_ids_and_never_leak_as_text():
    # parse_text_tool_calls mints `call_<n>_<name>` per turn
    # (agentic_utils.py:203), so the same id comes back in the next agent step;
    # the ToolMessage answering it carries the same raw id.
    raw_id = "call_0_eve_retrieval_retrieve"
    events = _run(
        [
            f_status(),
            f_tool_call(raw_id, "eve_retrieval_retrieve", {"query": "flood"}, server="eve_retrieval", label="Searching EVE sources"),
            f_tool_result(raw_id, "eve_retrieval_retrieve", "3 documents", server="eve_retrieval", duration_ms=800),
            f_tool_call(raw_id, "eve_retrieval_retrieve", {"query": "flood Po valley"}, server="eve_retrieval", label="Searching EVE sources"),
            f_tool_result(raw_id, "eve_retrieval_retrieve", "5 documents", server="eve_retrieval", duration_ms=700),
            f_token("Floods in the Po valley..."),
            f_final("Floods in the Po valley...", endpoint=ENDPOINT),
        ]
    )

    starts = _of(events, "TOOL_CALL_START")
    assert [s["toolCallId"] for s in starts] == [raw_id, f"{raw_id}~2"]
    assert [json.loads(a["delta"])["query"] for a in _of(events, "TOOL_CALL_ARGS")] == ["flood", "flood Po valley"]
    results = _of(events, "TOOL_CALL_RESULT")
    assert [r["toolCallId"] for r in results] == [raw_id, f"{raw_id}~2"]
    assert [r["messageId"] for r in results] == [f"{M}:tool:{raw_id}", f"{M}:tool:{raw_id}~2"]
    assert [r["content"] for r in results] == ["3 documents", "5 documents"]
    assert not any("[TOOL_CALLS]" in e.get("delta", "") for e in events)
    _assert_valid(events)


def test_tool_call_without_id_gets_a_unique_id():
    # runner.py:2294: `call_id = tc.get("id")` may be None; the frame keeps it.
    events = _run(
        [
            f_status(),
            f_tool_call(None, "geocode_lookup", {"place": "Frascati"}, server="geocode", label="Geocoding"),
            f_tool_result(None, "geocode_lookup", "41.8, 12.7", server="geocode", duration_ms=120),
            f_final("Frascati is at 41.8 N, 12.7 E.", endpoint=ENDPOINT),
        ]
    )

    start = _one(events, "TOOL_CALL_START")
    assert isinstance(start["toolCallId"], str) and start["toolCallId"]
    assert _one(events, "TOOL_CALL_RESULT")["toolCallId"] == start["toolCallId"]
    _assert_valid(events)


def test_summarizing_is_an_activity():
    events = _run(
        [
            f_status(),
            f_summarizing("started"),
            f_summarizing("finished"),
            f_token("Done."),
            f_final("Done.", endpoint=ENDPOINT),
        ]
    )

    snapshots = _of(events, "ACTIVITY_SNAPSHOT")
    assert [s["messageId"] for s in snapshots] == [f"{M}:summarizing:1"] * 2
    assert {s["activityType"] for s in snapshots} == {"eve.summarizing"}
    assert [s["content"]["phase"] for s in snapshots] == ["started", "finished"]
    assert snapshots[0]["content"]["firstMessageId"] == "6a3f0b009b1d4e7f8a2b3c01"
    assert snapshots[0]["content"]["lastMessageId"] == "6a3f0b109b1d4e7f8a2b3c09"
    _assert_valid(events)


def test_summary_still_running_at_the_end_is_closed_before_the_terminal_event():
    events = _run([f_status(), f_summarizing("started"), f_token("Done."), f_final("Done.", endpoint=ENDPOINT)])

    snapshots = _of(events, "ACTIVITY_SNAPSHOT")
    assert [s["content"]["phase"] for s in snapshots] == ["started", "finished"]
    types = _types(events)
    last_snapshot = max(i for i, t in enumerate(types) if t == "ACTIVITY_SNAPSHOT")
    # Closing sequence order: text end (step 3) before the summary close (step 6).
    assert types.index("TEXT_MESSAGE_END") < last_snapshot < types.index("RUN_FINISHED")
    _assert_valid(events)


def test_sources_are_a_custom_event():
    documents = [{"id": "d1", "title": "Sentinel-2 mission guide", "collection_name": "esa-rag"}]
    events = _run(
        [
            f_status(),
            f_tool_call("call_a1", "eve_retrieval_retrieve", {"query": "Sentinel-2"}, server="eve_retrieval", label="Searching EVE sources"),
            f_tool_result("call_a1", "eve_retrieval_retrieve", "1 document", server="eve_retrieval", duration_ms=1500),
            f_sources("call_a1", documents),
            f_token("See the guide."),
            f_final("See the guide.", endpoint=ENDPOINT),
        ]
    )

    custom = _one(events, "CUSTOM")
    assert custom["name"] == "eve.sources"
    assert custom["value"] == {"toolCallId": "call_a1", "documents": documents}
    assert _types(events).index("CUSTOM") > _types(events).index("TOOL_CALL_RESULT")
    _assert_valid(events)


@pytest.mark.parametrize(
    "frames, closes",
    [
        pytest.param([f_status()], "STEP_FINISHED", id="while-thinking"),
        pytest.param([f_status(), f_token("Partial ")], "TEXT_MESSAGE_END", id="after-text"),
    ],
)
def test_lost_terminal_frame_is_stream_incomplete(frames, closes):
    # The bus closes (stream_bus.py close, "[[__EOD__]]") with no terminal frame.
    events = _run(frames, finish=True)

    assert _types(events)[-2:] == [closes, "RUN_ERROR"]
    _assert_run_error(events, "stream_incomplete", COPY_INCOMPLETE)
    _assert_valid(events)


def test_no_user_text_in_any_log_record(caplog):
    secrets = [
        "PRIVATE-QUERY-flood-in-my-village",
        "PRIVATE-RESULT-coordinates-of-my-house",
        "PRIVATE-TOKEN-my-diagnosis",
        "PRIVATE-DOC-title",
        "PRIVATE-EXC-bearer-abc123",
    ]
    caplog.set_level(logging.DEBUG)
    _run(
        [
            f_status(),
            f_tool_call("call_p1", "eve_retrieval_retrieve", {"query": secrets[0]}, server="eve_retrieval", label="Searching EVE sources"),
            f_tool_result("call_p1", "eve_retrieval_retrieve", secrets[1], server="eve_retrieval", duration_ms=10),
            f_sources("call_p1", [{"id": "d1", "title": secrets[3]}]),
            f_token(secrets[2]),
            f_error("upstream_error", secrets[4]),
        ]
    )

    for record in caplog.records:
        text = record.getMessage() + repr(record.args) + repr(record.__dict__)
        for secret in secrets:
            assert secret not in text, f"{record.name}:{record.levelname} logs user text"


# ─── Wire and edge cases (B-S3) ───────────────────────────────────────────────


def test_wire_body_from_the_sdk_encoder_validates():
    # The route sends encode_sse (the SDK EventEncoder), not _sse: check that body.
    adapter = AguiAdapter(_ctx())
    raw: List[Any] = list(adapter.start())
    for frame in [
        f_status(),
        f_tool_call("call_w1", "eve_retrieval_retrieve", {"query": "SAR"}, server="eve_retrieval", label="Searching EVE sources"),
        f_tool_result("call_w1", "eve_retrieval_retrieve", "2 documents", server="eve_retrieval", duration_ms=50),
        f_token("Answer."),
        f_final("Answer.", endpoint=ENDPOINT, latencies={"total_s": 1.0}),
    ]:
        raw.extend(adapter.adapt(frame))
    body = encode_sse(raw)

    assert body.endswith("\n\n") and "\r" not in body
    assert '"requested":null' in body  # a null inside eve.endpoint stays on the wire
    assert not list(validate(body))


def test_unknown_error_code_is_upstream_error():
    events = _run([f_status(), f_error("weird_code", "x")])

    _assert_run_error(events, "upstream_error", COPY_UPSTREAM)
    _assert_valid(events)


def test_result_for_an_unannounced_call_opens_the_call_first():
    events = _run(
        [
            f_status(),
            f_tool_result("call_u1", "gdacs_events", "1 event", server="gdacs", duration_ms=40),
            f_token("One event."),
            f_final("One event.", endpoint=ENDPOINT),
        ]
    )

    assert _types(events)[3:6] == ["TOOL_CALL_START", "TOOL_CALL_END", "TOOL_CALL_RESULT"]
    assert _one(events, "TOOL_CALL_RESULT")["toolCallId"] == "call_u1"
    _assert_valid(events)


def test_empty_answer_after_a_shown_tool_result_is_success():
    # Decision 6: empty_answer only when no tool result was shown.
    events = _run(
        [
            f_status(),
            f_tool_call("call_s1", "dummy_image", {}, server="dummy_image", label="Rendering"),
            f_tool_result("call_s1", "dummy_image", "image ready", server="dummy_image", duration_ms=30),
            f_final("", artifact_ids=["6a3f0c4b9b1d4e7f8a2b3c61"], endpoint=ENDPOINT),
        ]
    )

    assert _assert_ends_with(events, "RUN_FINISHED")["outcome"] == {"type": "success"}
    _assert_valid(events)


def test_success_closes_an_unanswered_call_with_a_not_run_result():
    events = _run(
        [
            f_status(),
            f_tool_call("call_g1", "eve_retrieval_retrieve", {"query": "x"}, server="eve_retrieval", label="Searching EVE sources"),
            f_token("Answered without the tool."),
            f_final("Answered without the tool.", endpoint=ENDPOINT),
        ]
    )

    result = _one(events, "TOOL_CALL_RESULT")
    assert result["toolCallId"] == "call_g1" and result["content"] == ""
    assert result["metadata"] == {"eve.status": "not_run"}
    _assert_valid(events)


@pytest.mark.parametrize(
    "code, status, header",
    [
        pytest.param("overloaded", 429, {"Retry-After": "10"}, id="overloaded-default-retry"),
        pytest.param("run_id_conflict", 409, {}, id="run-id-conflict"),
        pytest.param("no_such_code", 500, {}, id="unknown-is-internal"),
    ],
)
def test_pre_stream_error_table(code, status, header):
    error = pre_stream_error(code)

    assert error.status_code == status
    assert error.headers == header
    assert error.body["detail"]["code"] == (code if status != 500 else "internal_error")
