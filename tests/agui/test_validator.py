"""The AG-UI transcript validator against the contract example transcripts.

The eight fixtures are the section 6 transcripts of the AG-UI contract, ids
spelled in full and written in the wire form (every event ends with a blank
line). Each bad transcript must yield exactly its expected codes, so a check
that fires too widely fails here as surely as one that never fires.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.agui.validator import parse_sse, validate_transcript, violation_codes

pytestmark = pytest.mark.no_db

FIXTURES = Path(__file__).parent / "fixtures"
THREAD_ID = "6a3f0c2e9b1d4e7f8a2b3c4d"
RUN_ID = "0f8e6c1a-3b2d-4e5f-9a7b-1c2d3e4f5a6b"
MESSAGE_ID = "6a3f0c4a9b1d4e7f8a2b3c55"

GOOD = ["good_1_retrieval_sources_artifact", "good_2_stop_during_tool"]

BAD = {
    "bad_1_event_before_run_started": {"first_event"},
    "bad_2_open_at_terminal": {"open_at_terminal"},
    "bad_3_two_terminals": {"terminal_count", "after_terminal"},
    "bad_4_result_ref_and_args_after_end": {"result_ref", "sequence"},
    "bad_5_custom_name_and_ag_ui_key": {"custom_name", "metadata_keys"},
    "bad_6_stop_as_success_and_chunk": {"outcome_cancelled", "forbidden_type"},
}


def _read(name: str) -> str:
    return (FIXTURES / f"{name}.sse").read_text(encoding="utf-8")


def _validate(body: str, **kwargs):
    return validate_transcript(body, thread_id=THREAD_ID, run_id=RUN_ID, **kwargs)


def _events(name: str) -> list[dict]:
    events, violations = parse_sse(_read(name))
    assert not violations
    return [e.data for e in events]


def _body(events: list[dict]) -> str:
    return "".join(f"data: {json.dumps(e, separators=(',', ':'))}\n\n" for e in events)


@pytest.mark.parametrize("name", GOOD)
def test_good_transcript_passes(name):
    violations = _validate(_read(name))
    assert violations == [], "\n".join(str(v) for v in violations)


@pytest.mark.parametrize("name,expected", sorted(BAD.items()))
def test_bad_transcript_yields_exactly_its_codes(name, expected):
    violations = _validate(_read(name))
    assert violation_codes(violations) == expected, "\n".join(str(v) for v in violations)


# Single-fault mutations of Good 1, one per check the fixtures do not reach.


def _good1_with(mutate) -> str:
    events = _events("good_1_retrieval_sources_artifact")
    mutate(events)
    return _body(events)


def _index(events, etype, start=0):
    return next(i for i, e in enumerate(events) if e["type"] == etype and i >= start)


def test_adjacent_data_lines_are_a_frame_failure():
    body = _read("good_1_retrieval_sources_artifact").replace("}\n\ndata: {\"type\":\"STEP_STARTED\"", "}\ndata: {\"type\":\"STEP_STARTED\"", 1)
    assert violation_codes(_validate(body)) == {"frame"}


def test_event_field_and_missing_final_blank_line_are_frame_failures():
    body = "event: message\n" + _read("good_2_stop_during_tool").rstrip("\n")
    assert violation_codes(_validate(body)) == {"frame"}


def test_metadata_null_fails_schema_and_metadata_rule():
    def mutate(events):
        events[_index(events, "STEP_STARTED")]["metadata"] = None

    assert violation_codes(_validate(_good1_with(mutate))) == {"schema", "metadata_keys"}


def test_unknown_member_fails_schema():
    def mutate(events):
        events[_index(events, "TOOL_CALL_END")]["server"] = "eve_retrieval"

    assert violation_codes(_validate(_good1_with(mutate))) == {"schema"}


def test_truncated_transcript_is_missing_terminal():
    events = _events("good_1_retrieval_sources_artifact")[:-1]
    events[-1:] = []  # drop eve.artifacts too, so nothing else is wrong
    assert violation_codes(_validate(_body(events))) == {"missing_terminal"}
    assert violation_codes(_validate(_body(events), require_terminal=False)) == set()


def test_run_started_without_protocol_version_or_trace_id():
    def mutate(events):
        del events[0]["protocolVersion"]
        del events[0]["metadata"]["eve.traceId"]

    assert violation_codes(_validate(_good1_with(mutate))) == {"first_event"}


def test_run_id_mismatch_on_run_finished():
    def mutate(events):
        events[-1]["runId"] = "another-run"

    assert violation_codes(_validate(_good1_with(mutate))) == {"run_ids"}


def test_success_with_a_tool_call_without_result():
    def mutate(events):
        del events[_index(events, "TOOL_CALL_RESULT")]

    assert violation_codes(_validate(_good1_with(mutate))) == {"pending_on_success"}


def test_tool_call_started_twice():
    def mutate(events):
        start = _index(events, "TOOL_CALL_START")
        again = [dict(events[start]), dict(events[start + 2])]
        at = _index(events, "TEXT_MESSAGE_START")
        events[at:at] = again

    assert violation_codes(_validate(_good1_with(mutate))) == {"tool_reopen"}


def test_tool_parent_must_be_the_answer():
    def mutate(events):
        events[_index(events, "TOOL_CALL_START")]["parentMessageId"] = "elsewhere"

    assert violation_codes(_validate(_good1_with(mutate))) == {"tool_parent"}


def test_wrong_step_name_and_role():
    def mutate(events):
        events[_index(events, "STEP_STARTED")]["stepName"] = "planning"
        events[_index(events, "STEP_FINISHED")]["stepName"] = "planning"
        events[_index(events, "TEXT_MESSAGE_START")]["role"] = "developer"

    assert violation_codes(_validate(_good1_with(mutate))) == {"step_name", "role"}


def test_unprefixed_activity_type():
    def mutate(events):
        events.insert(
            -1,
            {
                "type": "ACTIVITY_SNAPSHOT",
                "messageId": f"{MESSAGE_ID}:plan",
                "activityType": "plan",
                "content": {"todos": []},
            },
        )

    assert violation_codes(_validate(_good1_with(mutate))) == {"activity_type"}


def test_run_error_with_its_fixed_copy_passes_and_a_raw_message_fails():
    events = _events("good_1_retrieval_sources_artifact")[:3]
    ok = {
        "type": "RUN_ERROR",
        "code": "timeout",
        "message": "The model did not answer in time. Retry in a moment.",
        "metadata": {"eve.error.code": "timeout"},
    }
    assert violation_codes(_validate(_body(events + [ok]))) == set()
    leak = dict(ok, message="ReadTimeout('https://...')")
    assert violation_codes(_validate(_body(events + [leak]))) == {"run_error"}


def test_subagent_event_before_subagent_started():
    def mutate(events):
        at = _index(events, "TEXT_MESSAGE_START")
        events[at:at] = [
            {"type": "STEP_STARTED", "stepName": "thinking", "subagentRunId": "sa-call_a1"},
            {"type": "SUBAGENT_STARTED", "subagentRunId": "sa-call_a1", "name": "researcher", "parentToolCallId": "call_a1"},
            {"type": "STEP_FINISHED", "stepName": "thinking", "subagentRunId": "sa-call_a1"},
            {"type": "SUBAGENT_FINISHED", "subagentRunId": "sa-call_a1"},
        ]

    assert violation_codes(_validate(_good1_with(mutate))) == {"subagent"}


def test_interrupt_outcome_names_a_pending_call():
    events = _events("good_2_stop_during_tool")
    events[-1] = {
        "type": "RUN_FINISHED",
        "threadId": THREAD_ID,
        "runId": RUN_ID,
        "outcome": {
            "type": "interrupt",
            "interrupts": [{"id": "int-call_b2", "reason": "tool_approval", "toolCallId": "call_b2"}],
        },
    }
    assert violation_codes(_validate(_body(events))) == set()
    events[-1]["outcome"]["interrupts"][0]["id"] = "int-1"
    assert violation_codes(_validate(_body(events))) == {"outcome_interrupt"}


def test_tool_arguments_that_do_not_parse_are_tool_args_json():
    def mutate(events):
        at = _index(events, "TOOL_CALL_ARGS")
        events[at]["delta"] = '{"query": "Sentinel-2 revi'

    assert violation_codes(_validate(_good1_with(mutate))) == {"tool_args_json"}


def test_tool_call_on_a_later_model_call_message_passes_and_a_foreign_one_fails():
    def later(events):
        events[_index(events, "TOOL_CALL_START")]["parentMessageId"] = f"{MESSAGE_ID}:assistant:2"

    def first_as_segment(events):
        events[_index(events, "TOOL_CALL_START")]["parentMessageId"] = f"{MESSAGE_ID}:assistant:1"

    def foreign(events):
        events[_index(events, "TOOL_CALL_START")]["parentMessageId"] = "6a3f0c4a9b1d4e7f8a2b3c99:assistant:2"

    assert violation_codes(_validate(_good1_with(later))) == set()
    assert violation_codes(_validate(_good1_with(first_as_segment))) == {"tool_parent"}
    assert violation_codes(_validate(_good1_with(foreign))) == {"tool_parent"}
