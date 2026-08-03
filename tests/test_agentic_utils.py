"""Tests for text-format tool-call parsing helpers in agentic_utils.py.

Covers the live-repro bug where a turn containing `[TOOL_CALLS]...` segments
directly followed by real answer prose leaked the raw tool-call syntax into
the persisted answer instead of being stripped (see
``split_tool_calls_and_answer_text`` and its caller in
``src/services/agents/core/runner.py``'s ``_flush_turn_buffer_to_events``).
"""

from src.services.agentic_utils import (
    has_text_tool_call,
    might_be_incomplete_text_tool_call,
    parse_text_tool_calls,
    split_tool_calls_and_answer_text,
)


class TestSplitToolCallsAndAnswerText:
    def test_no_marker_returns_content_untouched(self):
        calls, answer = split_tool_calls_and_answer_text("just a normal answer")
        assert calls == []
        assert answer == "just a normal answer"

    def test_single_call_with_args_no_trailing_text(self):
        calls, answer = split_tool_calls_and_answer_text(
            '[TOOL_CALLS]dummy_get_sample_image{"color": "blue"}'
        )
        assert len(calls) == 1
        assert calls[0]["name"] == "dummy_get_sample_image"
        assert calls[0]["args"] == {"color": "blue"}
        assert answer == ""

    def test_call_with_no_args_braces(self):
        calls, answer = split_tool_calls_and_answer_text("[TOOL_CALLS]dummy_get_sample_report")
        assert len(calls) == 1
        assert calls[0]["name"] == "dummy_get_sample_report"
        assert calls[0]["args"] == {}
        assert answer == ""

    def test_two_calls_followed_by_answer_text(self):
        """The exact live-repro shape: two marker-prefixed calls then prose."""
        content = (
            '[TOOL_CALLS]dummy_get_sample_image{"color": "blue"}'
            "[TOOL_CALLS]dummy_get_sample_report{}"
            "Here is a sample image and report for you."
        )
        calls, answer = split_tool_calls_and_answer_text(content)
        assert [c["name"] for c in calls] == [
            "dummy_get_sample_image",
            "dummy_get_sample_report",
        ]
        assert calls[0]["args"] == {"color": "blue"}
        assert calls[1]["args"] == {}
        assert answer == "Here is a sample image and report for you."
        # None of the raw tool-call syntax should survive into the answer.
        assert "[TOOL_CALLS]" not in answer
        assert "dummy_get_sample_image" not in answer

    def test_single_call_followed_by_answer_text(self):
        content = '[TOOL_CALLS]dummy_search{"query": "x"}The answer is 42.'
        calls, answer = split_tool_calls_and_answer_text(content)
        assert len(calls) == 1
        assert calls[0]["name"] == "dummy_search"
        assert answer == "The answer is 42."

    def test_malformed_marker_with_no_valid_name_returns_no_calls(self):
        calls, answer = split_tool_calls_and_answer_text("[TOOL_CALLS]123not_a_name")
        assert calls == []
        # Whatever couldn't be parsed as a call is returned as-is (still
        # carries the marker) rather than silently discarded.
        assert answer == "[TOOL_CALLS]123not_a_name"

    def test_ids_are_unique_and_ordered(self):
        content = "[TOOL_CALLS]a{}[TOOL_CALLS]b{}"
        calls, _ = split_tool_calls_and_answer_text(content)
        assert calls[0]["id"] != calls[1]["id"]
        assert calls[0]["name"] == "a"
        assert calls[1]["name"] == "b"


class TestExistingHelpersStillWork:
    """Regression coverage: this module is now runner.py's sole source for
    these names (see the import-fallback fix in runner.py), so their basic
    contracts must hold.
    """

    def test_has_text_tool_call(self):
        assert has_text_tool_call("[TOOL_CALLS]foo{}") is True
        assert has_text_tool_call("plain text") is False

    def test_might_be_incomplete_text_tool_call(self):
        assert might_be_incomplete_text_tool_call("") is False
        assert might_be_incomplete_text_tool_call("[TOOL") is True
        assert might_be_incomplete_text_tool_call("[TOOL_CALLS]foo{}") is True
        assert might_be_incomplete_text_tool_call("plain answer text") is False

    def test_parse_text_tool_calls_single(self):
        parsed = parse_text_tool_calls('[TOOL_CALLS]foo{"a": 1}')
        assert len(parsed) == 1
        assert parsed[0]["name"] == "foo"
        assert parsed[0]["args"] == {"a": 1}
