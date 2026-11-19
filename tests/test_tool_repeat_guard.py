"""An agent that repeats one tool call is refused from the third identical call.

Dev 2026-11-19: the deep graph ran the same ``execute`` command 31 times
until the turn deadline. The guard counts per turn by tool name and
normalised arguments, for deepagents built-in tools and for MCP tools.
"""

import asyncio
import logging
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mcp.types import CallToolResult, TextContent

from src.services.agents.core import repeat_guard
from src.services.agents.core.interceptors import RepeatGuardInterceptor
from src.services.agents.core.repeat_guard import (
    check_tool_call,
    install_builtin_tool_guard,
    normalise_args,
    reset_tool_repeat_context,
    set_tool_repeat_context,
)

pytestmark = pytest.mark.no_db

DEV_COMMAND = (
    "jq '.features[] | {id, FIREDATE, AREA_HA, geometry: .geometry.coordinates[0]}' "
    "/workspace/effis_attica_2018.geojson | head -n 10"
)


@pytest.fixture
def turn():
    guard, token = set_tool_repeat_context()
    try:
        yield guard
    finally:
        reset_tool_repeat_context(token)


def _filesystem_middleware():
    from deepagents.middleware.filesystem import FilesystemMiddleware

    assert install_builtin_tool_guard() is True
    return FilesystemMiddleware()


def _tool_request(name: str, args: dict, call_id: str):
    from langchain.agents.middleware.types import ToolCallRequest

    return ToolCallRequest(
        tool_call={"name": name, "args": args, "id": call_id, "type": "tool_call"},
        tool=None,
        state={},
        runtime=None,
    )


async def _run_builtin(middleware, name: str, args: dict, call_id: str, runs: list):
    from langchain_core.messages import ToolMessage

    async def handler(request):
        runs.append(request.tool_call["id"])
        return ToolMessage(content="null\nnull", name=name, tool_call_id=call_id)

    return await type(middleware).awrap_tool_call(
        middleware, _tool_request(name, args, call_id), handler
    )


# ── built-in tools through deepagents FilesystemMiddleware ──────────────────


async def test_the_dev_sequence_runs_the_command_twice_and_then_refuses(turn):
    middleware = _filesystem_middleware()
    runs: list = []
    results = [
        await _run_builtin(middleware, "execute", {"command": DEV_COMMAND}, f"c{i}", runs)
        for i in range(31)
    ]

    assert runs == ["c0", "c1"]
    assert "You already ran this exact call 2 times" in results[2].content
    assert "change the command or the approach" in results[2].content
    assert results[2].tool_call_id == "c2"
    assert results[2].name == "execute"
    assert "change the command or the approach" in results[3].content
    for result in results[4:]:
        assert "Stop calling tools now and answer the user" in result.content


async def test_distinct_commands_all_run(turn):
    middleware = _filesystem_middleware()
    runs: list = []
    for i in range(31):
        await _run_builtin(middleware, "execute", {"command": f"echo {i}"}, f"c{i}", runs)
    assert len(runs) == 31


async def test_the_counts_start_from_zero_in_the_next_turn():
    middleware = _filesystem_middleware()
    runs: list = []
    for turn_index in range(2):
        _, token = set_tool_repeat_context()
        try:
            for i in range(3):
                await _run_builtin(
                    middleware, "execute", {"command": DEV_COMMAND}, f"t{turn_index}c{i}", runs
                )
        finally:
            reset_tool_repeat_context(token)
    assert runs == ["t0c0", "t0c1", "t1c0", "t1c1"]


async def test_a_call_outside_a_turn_always_runs():
    middleware = _filesystem_middleware()
    runs: list = []
    for i in range(5):
        await _run_builtin(middleware, "execute", {"command": DEV_COMMAND}, f"c{i}", runs)
    assert len(runs) == 5


async def test_an_edit_lets_the_same_command_run_again(turn):
    middleware = _filesystem_middleware()
    runs: list = []
    for i in range(2):
        await _run_builtin(middleware, "execute", {"command": "python plot.py"}, f"a{i}", runs)
    await _run_builtin(
        middleware,
        "edit_file",
        {"file_path": "plot.py", "old_string": "a", "new_string": "b"},
        "edit",
        runs,
    )
    await _run_builtin(middleware, "execute", {"command": "python plot.py"}, "b0", runs)
    assert runs == ["a0", "a1", "edit", "b0"]


async def test_an_mcp_call_passing_the_filesystem_middleware_is_not_counted_there(turn):
    middleware = _filesystem_middleware()
    runs: list = []
    for i in range(4):
        await _run_builtin(middleware, "effis_get_fires", {"year": 2018}, f"c{i}", runs)
    assert len(runs) == 4


async def test_the_install_is_idempotent():
    from deepagents.middleware.filesystem import FilesystemMiddleware

    assert install_builtin_tool_guard() is True
    first = FilesystemMiddleware.awrap_tool_call
    assert install_builtin_tool_guard() is True
    assert FilesystemMiddleware.awrap_tool_call is first


def test_the_sync_wrap_refuses_too(turn):
    from langchain_core.messages import ToolMessage

    middleware = _filesystem_middleware()
    runs: list = []

    def handler(request):
        runs.append(request.tool_call["id"])
        return ToolMessage(content="null", name="execute", tool_call_id=request.tool_call["id"])

    results = [
        type(middleware).wrap_tool_call(
            middleware, _tool_request("execute", {"command": DEV_COMMAND}, f"c{i}"), handler
        )
        for i in range(3)
    ]
    assert runs == ["c0", "c1"]
    assert "You already ran this exact call 2 times" in results[2].content


# ── MCP tools through the interceptor ───────────────────────────────────────


def _mcp_request(name: str, args: dict, server: str = "effis"):
    return SimpleNamespace(name=name, args=args, server_name=server)


async def test_the_mcp_interceptor_refuses_the_third_identical_call(turn):
    interceptor = RepeatGuardInterceptor()
    handler = AsyncMock(
        return_value=CallToolResult(content=[TextContent(type="text", text="[]")])
    )
    results = [
        await interceptor(_mcp_request("get_fires", {"year": 2018, "country": "GR"}), handler)
        for _ in range(6)
    ]

    assert handler.await_count == 2
    assert results[2].isError is False
    assert "change the command or the approach" in results[2].content[0].text
    assert "Stop calling tools now" in results[5].content[0].text
    # Another server with the same tool name and arguments is a distinct call.
    await interceptor(_mcp_request("get_fires", {"year": 2018, "country": "GR"}, "other"), handler)
    assert handler.await_count == 3


async def test_a_real_mcp_tool_never_reaches_the_server_on_a_refused_call(turn):
    from langchain_mcp_adapters.tools import convert_mcp_tool_to_langchain_tool
    from mcp.types import Tool as MCPTool

    session = SimpleNamespace(
        call_tool=AsyncMock(
            return_value=CallToolResult(content=[TextContent(type="text", text="null")])
        )
    )
    tool = convert_mcp_tool_to_langchain_tool(
        session,
        MCPTool(
            name="get_fires",
            description="d",
            inputSchema={"type": "object", "properties": {"year": {"type": "integer"}}},
        ),
        tool_interceptors=[RepeatGuardInterceptor()],
        server_name="effis",
        tool_name_prefix=True,
    )
    outputs = [await tool.ainvoke({"year": 2018}) for _ in range(3)]

    assert session.call_tool.await_count == 2
    assert "You already ran this exact call 2 times" in str(outputs[2])


async def test_the_loader_puts_the_guard_before_the_other_interceptors(monkeypatch):
    from unittest.mock import MagicMock, patch

    from src.services.agents.core.interceptors import ErrorLoggingInterceptor
    from src.services.mcp.tool_loader import load_mcp_tools_for_servers

    client = MagicMock()
    client.get_tools = AsyncMock(return_value=[])
    client_cls = MagicMock(return_value=client)
    server = SimpleNamespace(
        name="effis",
        config=SimpleNamespace(
            transport=SimpleNamespace(value="streamable_http"),
            url="http://example.com/mcp",
            headers={},
        ),
    )
    with patch("src.services.mcp.tool_loader.MultiServerMCPClient", client_cls), patch(
        "src.services.mcp.tool_loader.get_cognito_token_provider", return_value=None
    ):
        await load_mcp_tools_for_servers([server])

    interceptors = client_cls.call_args.kwargs["tool_interceptors"]
    kinds = [type(i) for i in interceptors]
    assert kinds.index(RepeatGuardInterceptor) < kinds.index(ErrorLoggingInterceptor)


# ── keys, log line, concurrency ─────────────────────────────────────────────


def test_whitespace_and_key_order_do_not_make_a_call_distinct():
    assert normalise_args({"b": 1, "a": "  ls -la \n"}) == normalise_args({"a": "ls -la", "b": 1})
    assert normalise_args('{"a": "x"}') == normalise_args({"a": " x "})
    assert normalise_args({"a": "ls"}) != normalise_args({"a": "ls -la"})


def test_the_log_line_names_the_tool_and_count_and_never_the_arguments(turn, caplog):
    with caplog.at_level(logging.INFO, logger=repeat_guard.__name__):
        for _ in range(3):
            check_tool_call("execute", {"command": DEV_COMMAND})
    lines = [r.getMessage() for r in caplog.records if "repeat_blocked" in r.getMessage()]
    assert lines == ["tool.repeat_blocked tool=execute count=3"]
    assert "jq" not in caplog.text


def test_concurrent_threads_let_exactly_two_identical_calls_run(turn):
    allowed: list = []
    barrier = threading.Barrier(16)
    import contextvars

    def call():
        barrier.wait()
        if check_tool_call("execute", {"command": DEV_COMMAND}) is None:
            allowed.append(1)

    threads = [
        threading.Thread(target=contextvars.copy_context().run, args=(call,)) for _ in range(16)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(allowed) == 2


async def test_parallel_tasks_in_one_turn_share_the_counts(turn):
    results = await asyncio.gather(
        *[asyncio.to_thread(check_tool_call, "execute", {"command": "ls"}) for _ in range(5)]
    )
    assert sum(r is None for r in results) == 2


# ── the runner opens one guard per streamed turn ────────────────────────────


class _RepeatingToolGraph:
    """Calls the same tool three times from inside the graph run, then answers."""

    def __init__(self, seen: list):
        self.seen = seen

    async def astream(self, *args, **kwargs):
        from langchain_core.messages import AIMessage

        async def tool_call():
            return check_tool_call("execute", {"command": DEV_COMMAND})

        for _ in range(3):
            # A tool node runs in its own task, as LangGraph does.
            self.seen.append(await asyncio.create_task(tool_call()))
        yield "messages", (AIMessage(content="done"), {"langgraph_node": "agent"})


async def test_each_streamed_turn_gets_its_own_guard():
    from unittest.mock import MagicMock

    from src.schemas.generation_request import GenerationRequest
    from src.services.agents.core.runner import generate_answer_agentic_stream_helper
    from tests.test_agentic_fallback import _patched_runner

    seen: list = []
    with _patched_runner(_build_react_graph=MagicMock(return_value=_RepeatingToolGraph(seen))):
        for turn_id in ("m1", "m2"):
            async for _event in generate_answer_agentic_stream_helper(
                GenerationRequest(query="metrics over the Attica fire", agent="deep"),
                conversation_id="c1",
                message_id=turn_id,
                user_id="test-user",
            ):
                pass

    assert [result is None for result in seen] == [True, True, False] * 2
    assert check_tool_call("execute", {"command": DEV_COMMAND}) is None
