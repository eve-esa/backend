"""A written sandbox file tells the model how the shell names it.

The model writes ``/workspace/count.py`` and then runs ``python3
/workspace/count.py``, which does not exist in the shell. The tool result the
model reads next carries the runnable form, relative to the workspace.
"""

import inspect
import json

import pytest

from src.services.agents.core.run_hint import (
    add_run_hints,
    run_hint,
    workspace_relative_path,
)
from src.services.agents.core import sandbox
from src.services.agents.core.tool_args import ensure_tool_args_are_json

pytestmark = pytest.mark.no_db

_ROOT = "/opt/amazon/genesis1p-tools/var"


@pytest.fixture(autouse=True)
def _agentcore_root(monkeypatch):
    """The session root prepare_workspace_layout records on AgentCore."""
    monkeypatch.setattr(sandbox, "_session_roots", {_ROOT})

_EXECUTE = {"type": "function", "function": {"name": "execute", "parameters": {}}}
_WRITE = {"type": "function", "function": {"name": "write_file", "parameters": {}}}


def _payload(path, content, *, tool="write_file", tools=(_EXECUTE, _WRITE)):
    return {
        "messages": [
            {"role": "user", "content": "save it"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": tool,
                            "arguments": json.dumps(
                                {"file_path": path, "content": "print(1)"}
                            ),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": content},
        ],
        "tools": list(tools),
    }


def _tool_content(payload):
    return payload["messages"][2]["content"]


_HINT = "Run it in the shell as `python3 count.py` (the shell starts in the workspace)."


@pytest.mark.parametrize(
    "path",
    [
        "count.py",
        "./count.py",
        "workspace/count.py",
        "/workspace/count.py",
        "/opt/amazon/genesis1p-tools/var/workspace/count.py",
    ],
)
def test_every_spelling_of_a_workspace_file_gets_the_relative_run_form(path):
    payload = add_run_hints(_payload(path, f"Updated file {path}"))
    assert _tool_content(payload) == f"Updated file {path}\n{_HINT}"


def test_a_nested_file_keeps_its_sub_path():
    payload = add_run_hints(
        _payload("/workspace/src/app/main.py", "Updated file /workspace/src/app/main.py")
    )
    assert _tool_content(payload).endswith(
        "Run it in the shell as `python3 src/app/main.py` "
        "(the shell starts in the workspace)."
    )


def test_a_failed_write_gets_no_hint():
    content = "Error: permission denied for write on /workspace/count.py"
    payload = add_run_hints(_payload("/workspace/count.py", content))
    assert _tool_content(payload) == content


def test_a_run_without_a_shell_is_unchanged():
    payload = add_run_hints(
        _payload("/workspace/count.py", "Updated file /workspace/count.py", tools=(_WRITE,))
    )
    assert _tool_content(payload) == "Updated file /workspace/count.py"


def test_a_successful_edit_gets_the_hint():
    content = "Successfully replaced 1 instance(s) of the string in '/workspace/count.py'"
    payload = add_run_hints(_payload("/workspace/count.py", content, tool="edit_file"))
    assert _tool_content(payload) == f"{content}\n{_HINT}"


def test_other_tools_are_untouched():
    payload = add_run_hints(
        _payload("/workspace/count.py", "Updated file /workspace/count.py", tool="read_file")
    )
    assert _tool_content(payload) == "Updated file /workspace/count.py"


def test_a_file_outside_the_workspace_gets_no_hint():
    payload = add_run_hints(_payload("/skills/x/run.py", "Updated file /skills/x/run.py"))
    assert _tool_content(payload) == "Updated file /skills/x/run.py"


def test_a_data_file_gets_its_shell_path_without_a_command():
    payload = add_run_hints(_payload("/workspace/data.csv", "Updated file /workspace/data.csv"))
    assert _tool_content(payload).endswith(
        "In the shell its path is `data.csv` (the shell starts in the workspace)."
    )


def test_a_shell_script_gets_its_path_without_a_runner():
    assert run_hint("/workspace/build.sh") == (
        "In the shell its path is `build.sh` (the shell starts in the workspace)."
    )


def test_a_name_starting_with_a_dash_is_not_read_as_an_option():
    assert run_hint("/workspace/-x.py") == (
        "Run it in the shell as `python3 ./-x.py` (the shell starts in the workspace)."
    )


def test_a_name_with_a_space_is_quoted_for_the_shell():
    assert run_hint("/workspace/my count.py") == (
        "Run it in the shell as `python3 'my count.py'` "
        "(the shell starts in the workspace)."
    )


@pytest.mark.parametrize(
    "path, expected",
    [
        ("count.py", "count.py"),
        ("/workspace/workspace/x.py", "workspace/x.py"),
        ("/tmp/x.py", "tmp/x.py"),
        ("/workspace", None),
        ("../x.py", None),
        ("/artifacts/1/x.png", None),
        ("/large_tool_results/abc", None),
        ("", None),
    ],
)
def test_workspace_relative_path(path, expected):
    assert workspace_relative_path(path) == expected


def test_a_second_pass_does_not_repeat_the_hint():
    payload = add_run_hints(add_run_hints(_payload("count.py", "Updated file count.py")))
    assert _tool_content(payload).count("Run it in the shell") == 1


def test_list_content_gets_a_text_part():
    payload = _payload("count.py", "x")
    payload["messages"][2]["content"] = [{"type": "text", "text": "Updated file count.py"}]
    add_run_hints(payload)
    assert payload["messages"][2]["content"][-1] == {"type": "text", "text": _HINT}


def test_broken_arguments_get_no_hint_and_no_error():
    payload = _payload("count.py", "Updated file count.py")
    payload["messages"][1]["tool_calls"][0]["function"]["arguments"] = "not json"
    assert _tool_content(add_run_hints(payload)) == "Updated file count.py"


def test_the_model_request_carries_the_hint():
    """The real OpenAI converter, through the wrapper every agent model gets."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_openai import ChatOpenAI

    llm = ensure_tool_args_are_json(
        ChatOpenAI(model="x", api_key="test-only", base_url="http://127.0.0.1:1/v1")
    )
    messages = [
        HumanMessage("save it"),
        AIMessage(
            "",
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "write_file",
                    "args": {"file_path": "/workspace/count.py", "content": "print(1)"},
                }
            ],
        ),
        ToolMessage("Updated file /workspace/count.py", tool_call_id="call_1"),
    ]
    payload = llm._get_request_payload(messages, tools=[_EXECUTE, _WRITE])
    tool = [m for m in payload["messages"] if m.get("role") == "tool"][0]
    assert tool["content"] == f"Updated file /workspace/count.py\n{_HINT}"
    stored = messages[2].content
    assert stored == "Updated file /workspace/count.py"


def test_deepagents_success_texts_match_what_the_hint_reads():
    """A deepagents bump that rewords the success line must fail here."""
    from deepagents.middleware.filesystem import FilesystemMiddleware

    write = inspect.getsource(FilesystemMiddleware._create_write_file_tool)
    edit = inspect.getsource(FilesystemMiddleware._create_edit_file_tool)
    assert 'content=f"Updated file {res.path}"' in write
    assert 'content=f"Successfully replaced {res.occurrences}' in edit


@pytest.mark.parametrize(
    "path",
    [
        f"{_ROOT}/count.py",
        f"{_ROOT}/skills/workspace/x.py",
        f"{_ROOT}/workspace",
    ],
)
def test_a_real_path_outside_the_workspace_gets_no_hint(path):
    assert workspace_relative_path(path) is None


def test_a_real_path_under_the_workspace_keeps_its_sub_path():
    assert workspace_relative_path(f"{_ROOT}/workspace/a/workspace/x.py") == (
        "a/workspace/x.py"
    )


def test_without_a_known_root_a_real_path_is_placed_like_the_file_tool_does(
    monkeypatch,
):
    monkeypatch.setattr(sandbox, "_session_roots", set())
    assert workspace_relative_path("/tmp/x.py") == "tmp/x.py"


def test_the_layout_records_the_session_root(monkeypatch):
    monkeypatch.setattr(sandbox, "_session_roots", set())

    class _Backend:
        def _get_cwd(self):
            return _ROOT + "/"

        def execute(self, command):
            return None

    sandbox.prepare_workspace_layout(_Backend())
    assert sandbox.known_session_roots() == {_ROOT}


def test_writes_of_earlier_turns_get_no_hint():
    payload = _payload("count.py", "Updated file count.py")
    payload["messages"].append({"role": "assistant", "content": "done"})
    payload["messages"].append({"role": "user", "content": "now something else"})
    add_run_hints(payload)
    assert _tool_content(payload) == "Updated file count.py"


def test_list_content_through_the_real_converter_leaves_the_stored_message():
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_openai import ChatOpenAI

    llm = ensure_tool_args_are_json(
        ChatOpenAI(model="x", api_key="test-only", base_url="http://127.0.0.1:1/v1")
    )
    parts = [{"type": "text", "text": "Updated file /workspace/count.py"}]
    messages = [
        HumanMessage("save it"),
        AIMessage(
            "",
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "write_file",
                    "args": {"file_path": "/workspace/count.py", "content": "x"},
                }
            ],
        ),
        ToolMessage(content=parts, tool_call_id="call_1"),
    ]
    payload = llm._get_request_payload(messages, tools=[_EXECUTE, _WRITE])
    tool = [m for m in payload["messages"] if m.get("role") == "tool"][0]
    assert _HINT in json.dumps(tool["content"])
    assert messages[2].content == [
        {"type": "text", "text": "Updated file /workspace/count.py"}
    ]
