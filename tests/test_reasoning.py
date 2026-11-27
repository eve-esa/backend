"""Reasoning capture: provider fields, inline tags, and the AG-UI mapping."""

import pytest
from langchain_core.messages import AIMessageChunk

pytestmark = pytest.mark.no_db

from src.core.llm_manager import ReasoningChatOpenAI
from src.services.agents.core.reasoning import ReasoningSplitter, split_complete
from src.services.agui.adapter import AguiAdapter, RunContext

THREAD = "6a3f0c2e9b1d4e7f8a2b3c4d"
MESSAGE = "6a3f0c4a9b1d4e7f8a2b3c55"
RUN = "0f8e6c1a-3b2d-4e5f-9a7b-1c2d3e4f5a6b"


def _client() -> ReasoningChatOpenAI:
    return ReasoningChatOpenAI(api_key="test", model="m")


def _chunk(delta: dict):
    generation = _client()._convert_chunk_to_generation_chunk(
        {"choices": [{"delta": delta}]}, AIMessageChunk, None
    )
    assert generation is not None
    return generation.message


def test_chunk_keeps_reasoning_content():
    message = _chunk({"content": "", "reasoning_content": "hmm"})
    assert message.content == ""
    assert message.additional_kwargs["reasoning_content"] == "hmm"


def test_chunk_keeps_the_reasoning_field():
    message = _chunk({"content": "Hi", "reasoning": "hmm"})
    assert message.content == "Hi"
    assert message.additional_kwargs["reasoning_content"] == "hmm"


def test_chunk_lifts_mistral_thinking_parts_out_of_the_answer():
    message = _chunk(
        {
            "content": [
                {"type": "thinking", "thinking": "hmm"},
                {"type": "text", "text": "Hi"},
            ]
        }
    )
    assert message.content == "Hi"
    assert message.additional_kwargs["reasoning_content"] == "hmm"


def test_chunk_lifts_nested_mistral_thinking_text():
    # ThinkChunk.thinking is a list of text chunks, not a string.
    # https://docs.mistral.ai/inference/conversations/reasoning/
    message = _chunk(
        {
            "content": [
                {
                    "type": "thinking",
                    "thinking": [{"type": "text", "text": "hm"}, {"type": "text", "text": "m"}],
                },
                {"type": "text", "text": "Hi"},
            ]
        }
    )
    assert message.content == "Hi"
    assert message.additional_kwargs["reasoning_content"] == "hmm"


def test_chunk_without_reasoning_leaves_the_answer_alone():
    message = _chunk({"content": "Hi"})
    assert message.content == "Hi"
    assert "reasoning_content" not in message.additional_kwargs


def test_result_keeps_reasoning_content():
    result = _client()._create_chat_result(
        {
            "model": "m",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "Hi",
                        "reasoning_content": "hmm",
                    },
                }
            ],
        }
    )
    message = result.generations[0].message
    assert message.content == "Hi"
    assert message.additional_kwargs["reasoning_content"] == "hmm"


def test_result_lifts_thinking_parts_when_the_field_is_absent():
    result = _client()._create_chat_result(
        {
            "model": "m",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "hmm"},
                            {"type": "text", "text": "Hi"},
                        ],
                    },
                }
            ],
        }
    )
    message = result.generations[0].message
    assert message.content == "Hi"
    assert message.additional_kwargs["reasoning_content"] == "hmm"


def _fed(*parts: str) -> tuple[str, str]:
    splitter = ReasoningSplitter()
    reasoning = ""
    answer = ""
    for part in parts:
        more_reasoning, more_answer = splitter.feed(part)
        reasoning += more_reasoning
        answer += more_answer
    more_reasoning, more_answer = splitter.flush()
    return reasoning + more_reasoning, answer + more_answer


def test_tags_split_across_chunks():
    assert _fed("<thi", "nk>ab", "c</thi", "nk>Hi") == ("abc", "Hi")


def test_bracket_tags():
    assert split_complete("[THINK]zz[/THINK]ok") == ("zz", "ok")


def test_a_stream_that_opens_inside_a_think_block():
    # The chat template injected the opening tag, so the first tag is the close.
    assert _fed("because x</thi", "nk>\nYes") == ("because x", "Yes")


def test_plain_answer_text_is_not_held_for_a_tag():
    assert _fed("Hel", "lo") == ("", "Hello")


def test_provider_reasoning_passes_through():
    splitter = ReasoningSplitter()
    assert splitter.feed("Hi", "hmm") == ("hmm", "Hi")


def _events(frames: list[dict]) -> list[dict]:
    adapter = AguiAdapter(
        RunContext(thread_id=THREAD, run_id=RUN, message_id=MESSAGE, graph="react")
    )
    out = list(adapter.start())
    for frame in frames:
        out.extend(adapter.adapt(frame))
    return [event.model_dump(mode="json", by_alias=True, exclude_none=True) for event in out]


def test_reasoning_frames_become_one_reasoning_message_closed_by_the_answer():
    events = _events(
        [
            {"type": "reasoning", "content": "hm"},
            {"type": "reasoning", "content": "m"},
            {"type": "token", "content": "done"},
            {"type": "final", "answer": "done", "latencies": {}},
        ]
    )
    kinds = [event["type"] for event in events]
    assert kinds.index("REASONING_START") < kinds.index("REASONING_MESSAGE_CONTENT")
    assert kinds.index("REASONING_MESSAGE_END") < kinds.index("TEXT_MESSAGE_START")
    contents = [
        event["delta"] for event in events if event["type"] == "REASONING_MESSAGE_CONTENT"
    ]
    assert contents == ["hm", "m"]
    message_ids = {
        event["messageId"]
        for event in events
        if event["type"].startswith("REASONING_")
    }
    assert message_ids == {f"{MESSAGE}:reasoning:1"}
