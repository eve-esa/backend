"""A standard AG-UI client on the adapter output (decision 0008).

CopilotKit, used as a plain AG-UI client, showed truncated tool arguments and
results as broken JSON, and the tool card under the answer text it preceded.
These cases pin the fixes: truncation keeps the JSON valid, and the text of
each model call is its own assistant message with the tool calls of that call
attached to it by ``parentMessageId``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from tests.agui.test_adapter import (
    ENDPOINT,
    M,
    _assert_ends_with,
    _assert_valid,
    _of,
    _one,
    _run,
    _types,
    f_final,
    f_status,
    f_token,
    f_tool_call,
    f_tool_result,
)

pytestmark = pytest.mark.no_db


def _args_json(events: List[Dict[str, Any]], tool_call_id: str) -> Any:
    deltas = [e["delta"] for e in _of(events, "TOOL_CALL_ARGS") if e["toolCallId"] == tool_call_id]
    return json.loads("".join(deltas))


def _result(events: List[Dict[str, Any]], tool_call_id: str) -> Dict[str, Any]:
    (result,) = [e for e in _of(events, "TOOL_CALL_RESULT") if e["toolCallId"] == tool_call_id]
    return result


def _tool_turn(args: Dict[str, Any], content: Any, **call_kwargs: Any) -> List[Dict[str, Any]]:
    result = f_tool_result("call_1", "execute", "", server="sandbox", duration_ms=10)
    result["content"] = content  # as sent, not cut by the builder
    return _run(
        [
            f_status(),
            f_tool_call("call_1", "execute", args, server="sandbox", label="Running code", **call_kwargs),
            result,
            f_token("Done."),
            f_final("Done.", endpoint=ENDPOINT),
        ]
    )


# ─── Truncation keeps valid JSON ──────────────────────────────────────────────


def test_long_string_argument_is_shortened_inside_the_value():
    # A frame whose string argument was not cut upstream: the adapter bounds
    # it inside the value, so the argument text still parses.
    code = "x = 1\n" * 2000
    events = _tool_turn({"command": code, "timeout": 30}, "ok")

    args = _args_json(events, "call_1")
    assert set(args) == {"command", "timeout"}
    assert args["timeout"] == 30
    assert len(args["command"]) < 1000
    assert args["command"].startswith("x = 1\n")
    assert args["command"].endswith("…")
    assert _one(events, "TOOL_CALL_START")["metadata"]["eve.argsTruncated"] is True
    _assert_valid(events)


def test_long_array_argument_is_cut_with_a_count():
    ids = [f"doc-{i}" for i in range(200)]
    events = _tool_turn({"ids": ids}, "ok")

    args = _args_json(events, "call_1")
    assert args["ids"][:5] == ids[:5]
    assert len(args["ids"]) < 50
    assert "more" in args["ids"][-1] and str(200 - (len(args["ids"]) - 1)) in args["ids"][-1]
    assert _one(events, "TOOL_CALL_START")["metadata"]["eve.argsTruncated"] is True
    _assert_valid(events)


def test_short_arguments_are_unchanged_and_not_flagged():
    events = _tool_turn({"query": "Sentinel-1", "k": 5}, "ok")

    assert _args_json(events, "call_1") == {"query": "Sentinel-1", "k": 5}
    assert "eve.argsTruncated" not in (_one(events, "TOOL_CALL_START").get("metadata") or {})
    _assert_valid(events)


def test_result_preview_cut_inside_the_json_is_closed_as_valid_json():
    # What the runner sends today: the first 200 chars of the stringified
    # result, which ends in the middle of a string value.
    full = json.dumps(
        {
            "documents": [
                {"title": f"Flood mapping note {i}", "text": "Sentinel-1 backscatter thresholds " * 5}
                for i in range(10)
            ],
            "count": 10,
        }
    )
    events = _tool_turn({"query": "floods"}, full[:200])

    result = _result(events, "call_1")
    parsed = json.loads(result["content"])
    assert parsed["documents"][0]["title"] == "Flood mapping note 0"
    assert result["metadata"]["eve.contentTruncated"] is True
    _assert_valid(events)


@pytest.mark.parametrize(
    "cut",
    [
        '{"a": 1, "b": "tex',  # inside a value string
        '{"a": 1, "b": "te\\',  # inside an escape
        '{"a": 1, "b": "te\\u00',  # inside a unicode escape
        '{"a": 1, "bb',  # inside a key
        '{"a": 1, "b":',  # after the colon
        '{"a": 1, "b": tr',  # inside a literal
        '{"a": 1, "b": 12',  # a number that may go on
        '{"a": [1, 2, {"c": [',  # nested, open
        '[{"a": "x"}, {"a": "y"},',  # after a comma
    ],
)
def test_every_cut_point_of_a_json_result_closes_as_valid_json(cut):
    events = _tool_turn({"query": "q"}, cut)

    result = _result(events, "call_1")
    parsed = json.loads(result["content"])
    assert isinstance(parsed, (dict, list))
    assert result["metadata"]["eve.contentTruncated"] is True
    _assert_valid(events)


def test_long_array_result_is_cut_with_a_count_and_stays_json():
    full = json.dumps({"scenes": [{"id": f"S1A_{i:04d}", "cloud": i % 7} for i in range(150)]})
    events = _tool_turn({"area": "Po delta"}, full)

    result = _result(events, "call_1")
    parsed = json.loads(result["content"])
    scenes = parsed["scenes"]
    assert scenes[0] == {"id": "S1A_0000", "cloud": 0}
    assert len(scenes) < 50
    assert isinstance(scenes[-1], str) and "more" in scenes[-1]
    assert str(150 - (len(scenes) - 1)) in scenes[-1]
    assert result["metadata"]["eve.contentTruncated"] is True
    _assert_valid(events)


def test_non_json_tool_result_is_left_as_text():
    text = "The flooded area is 7.80 km². [Command succeeded with exit code 0]"
    events = _tool_turn({"command": "python3 -c 'print(1)'"}, text)

    result = _result(events, "call_1")
    assert result["content"] == text
    assert "eve.contentTruncated" not in result["metadata"]
    _assert_valid(events)


def test_complete_json_result_is_sent_as_is():
    content = '{"status": "ok",  "count": 3}'
    events = _tool_turn({"query": "q"}, content)

    result = _result(events, "call_1")
    assert result["content"] == content
    assert "eve.contentTruncated" not in result["metadata"]
    _assert_valid(events)


# ─── One assistant message per model call ─────────────────────────────────────


def test_two_model_calls_with_a_tool_between_are_two_text_messages_in_order():
    events = _run(
        [
            f_status(),
            f_token("Let me check the archive."),
            f_tool_call("call_b2", "eotdl_search", {"query": "floods"}, server="eotdl", label="Searching EOTDL"),
            f_tool_result("call_b2", "eotdl_search", "3 datasets", server="eotdl", duration_ms=800),
            f_token("\n\nThree datasets "),
            f_token("match."),
            f_final("Let me check the archive.\n\nThree datasets match.", endpoint=ENDPOINT),
        ]
    )

    starts = _of(events, "TEXT_MESSAGE_START")
    assert len(starts) == 2
    first, second = starts[0]["messageId"], starts[1]["messageId"]
    assert first == M
    assert second != first
    tool_start = _one(events, "TOOL_CALL_START")
    assert tool_start["parentMessageId"] == first

    # Order on the wire: first text, its tool call and result, then the second text.
    order = [
        (e["type"], e.get("messageId") or e.get("toolCallId"))
        for e in events
        if e["type"] in ("TEXT_MESSAGE_START", "TEXT_MESSAGE_END", "TOOL_CALL_START", "TOOL_CALL_RESULT")
    ]
    assert order == [
        ("TEXT_MESSAGE_START", first),
        ("TEXT_MESSAGE_END", first),
        ("TOOL_CALL_START", "call_b2"),
        ("TOOL_CALL_RESULT", f"{M}:tool:call_b2"),
        ("TEXT_MESSAGE_START", second),
        ("TEXT_MESSAGE_END", second),
    ]
    by_message: Dict[str, str] = {}
    for e in _of(events, "TEXT_MESSAGE_CONTENT"):
        by_message[e["messageId"]] = by_message.get(e["messageId"], "") + e["delta"]
    # Deltas are passed through untouched: the texts concatenate to the stored answer.
    assert by_message[first] + by_message[second] == "Let me check the archive.\n\nThree datasets match."
    ends = _of(events, "TEXT_MESSAGE_END")
    assert "eve.endpoint" not in (ends[0].get("metadata") or {})
    assert ends[1]["metadata"]["eve.endpoint"] == ENDPOINT
    assert _assert_ends_with(events, "RUN_FINISHED")["outcome"] == {"type": "success"}
    _assert_valid(events)


def test_tool_call_of_a_later_model_call_attaches_to_that_call_message():
    # Model call 1: a tool only. Model call 2: text and a second tool. Model
    # call 3: the answer. Each call has its own assistant message id.
    events = _run(
        [
            f_status(),
            f_tool_call("call_1", "search", {"query": "a"}, server="s", label="Search"),
            f_tool_result("call_1", "search", "1", server="s", duration_ms=5),
            f_token("Refining."),
            f_tool_call("call_2", "search", {"query": "b"}, server="s", label="Search"),
            f_tool_result("call_2", "search", "2", server="s", duration_ms=5),
            f_token("Answer."),
            f_final("Refining.\n\nAnswer.", endpoint=ENDPOINT),
        ]
    )

    starts = _of(events, "TOOL_CALL_START")
    texts = _of(events, "TEXT_MESSAGE_START")
    assert starts[0]["parentMessageId"] == M
    assert len(texts) == 2
    refining, answer = texts[0]["messageId"], texts[1]["messageId"]
    assert len({M, refining, answer}) == 3
    assert starts[1]["parentMessageId"] == refining
    _assert_valid(events)


def test_parallel_tool_calls_of_one_model_call_share_the_parent():
    events = _run(
        [
            f_status(),
            f_token("Checking both."),
            f_tool_call("call_1", "search", {"query": "a"}, server="s", label="Search"),
            f_tool_call("call_2", "search", {"query": "b"}, server="s", label="Search"),
            f_tool_result("call_1", "search", "1", server="s", duration_ms=5),
            f_tool_result("call_2", "search", "2", server="s", duration_ms=5),
            f_token("Both found."),
            f_final("Checking both.\n\nBoth found.", endpoint=ENDPOINT),
        ]
    )

    assert {e["parentMessageId"] for e in _of(events, "TOOL_CALL_START")} == {M}
    assert len(_of(events, "TEXT_MESSAGE_START")) == 2
    assert "TOOL_CALL_RESULT" in _types(events)
    _assert_valid(events)
