"""Tool-call arguments with trailing junk must not be sent again as-is."""

import json

import pytest

from src.services.agents.core.tool_args import (
    drop_assistant_names,
    ensure_tool_args_are_json,
    repair_payload_tool_args,
    repair_tool_arguments,
)

pytestmark = pytest.mark.no_db


def test_extra_data_after_a_tool_argument_object_is_dropped():
    raw = (
        '{"command": "python3 -c \\"print(1)\\""}}'
    )
    repaired = repair_tool_arguments(raw)
    assert json.loads(repaired) == {"command": 'python3 -c "print(1)"'}


def test_valid_tool_arguments_stay_unchanged():
    raw = '{"command": "ls"}'
    assert repair_tool_arguments(raw) == raw


def test_arguments_that_are_not_json_become_an_empty_object():
    assert repair_tool_arguments("not json") == "{}"


def test_payload_tool_calls_are_repaired_in_place():
    payload = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call-1",
                        "function": {
                            "name": "execute",
                            "arguments": '{"command": "ls"}}',
                        },
                    }
                ],
            }
        ]
    }
    repair_payload_tool_args(payload)
    arguments = payload["messages"][0]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(arguments) == {"command": "ls"}


def test_assistant_name_is_removed_and_tool_name_stays():
    payload = {
        "messages": [
            {"role": "assistant", "name": "deep", "content": "done"},
            {"role": "tool", "name": "search", "content": "ok"},
            {"role": "user", "name": "deep", "content": "next"},
        ]
    }
    drop_assistant_names(payload)
    assert "name" not in payload["messages"][0]
    assert payload["messages"][1]["name"] == "search"
    assert payload["messages"][2]["name"] == "deep"


def test_a_model_repairs_arguments_when_it_builds_the_request():
    seen = {}

    class _Model:
        def _get_request_payload(self, input_, *, stop=None, **kwargs):
            return {
                "messages": [
                    {
                        "role": "assistant",
                        "name": "deep",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "execute",
                                    "arguments": '{"command": "pwd"}}',
                                }
                            }
                        ],
                    }
                ]
            }

    model = ensure_tool_args_are_json(_Model())
    payload = model._get_request_payload("hi")
    message = payload["messages"][0]
    assert "name" not in message
    arguments = message["tool_calls"][0]["function"]["arguments"]
    assert json.loads(arguments) == {"command": "pwd"}
    again = ensure_tool_args_are_json(model)
    assert again is model
    seen["ok"] = True
