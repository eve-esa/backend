"""An agent that repeats one tool call with the same result is refused.

Dev 2026-11-19: the deep graph ran the same ``execute`` command 31 times
until the turn deadline. The guard counts per turn by tool name, normalised
arguments and result, for deepagents built-in tools and for MCP tools, and
the runner ends a turn whose model keeps calling refused tools.
"""

import asyncio
import contextlib
import logging
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mcp.types import CallToolResult, TextContent

from src.services.agents.core import repeat_guard
from src.services.agents.core.interceptors import RepeatGuardInterceptor
from src.services.agents.core.repeat_guard import (
    END_NOTE,
    SUBAGENT_STOPPED,
    begin_tool_call,
    check_tool_call,
    finish_tool_call,
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


@pytest.fixture(autouse=True)
def restore_deepagents(monkeypatch):
    """Undo the class patch after each test, so other files see deepagents as it was."""
    from deepagents.middleware.filesystem import FilesystemMiddleware

    for hook in ("wrap_tool_call", "awrap_tool_call"):
        monkeypatch.setattr(FilesystemMiddleware, hook, vars(FilesystemMiddleware)[hook])
    monkeypatch.setattr(repeat_guard, "_DEEPAGENTS_WRAP", repeat_guard._DEEPAGENTS_WRAP)
    monkeypatch.setattr(repeat_guard, "_DEEPAGENTS_AWRAP", repeat_guard._DEEPAGENTS_AWRAP)


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


async def _run_builtin(
    middleware,
    name: str,
    args: dict,
    call_id: str,
    runs: list,
    content="null\nnull",
    status: str = "success",
):
    from langchain_core.messages import ToolMessage

    async def handler(request):
        runs.append(request.tool_call["id"])
        if isinstance(content, BaseException):
            raise content
        return ToolMessage(content=content, name=name, tool_call_id=call_id, status=status)

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
    assert results[2].content.startswith("NOT RUN.")
    assert "already ran 2 times in this turn and returned the same result" in results[2].content
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
    assert results[2].content.startswith("NOT RUN. This exact call already ran 2 times")


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
    assert "NOT RUN. This exact call already ran 2 times" in str(outputs[2])


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
    assert lines == ["tool.repeat_blocked tool=execute count=3 subagent=False end=False"]
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


# ── review of PR #401: results, state changes, subagents, the turn bound ────

EDIT = {"file_path": "plot.py", "old_string": "a", "new_string": "b"}


async def test_an_identical_edit_repeated_is_refused_like_any_call(turn):
    middleware = _filesystem_middleware()
    runs: list = []
    for i in range(5):
        await _run_builtin(middleware, "edit_file", EDIT, f"e{i}", runs, content="Updated plot.py")
    assert runs == ["e0", "e1"]


async def test_a_command_and_a_failing_edit_in_a_loop_are_both_refused(turn):
    middleware = _filesystem_middleware()
    runs: list = []
    for i in range(4):
        await _run_builtin(middleware, "execute", {"command": "python plot.py"}, f"x{i}", runs)
        await _run_builtin(
            middleware,
            "edit_file",
            EDIT,
            f"e{i}",
            runs,
            content="Error: String not found in file: 'a'",
            status="error",
        )
    assert runs == ["x0", "e0", "x1", "e1"]


async def test_a_failed_write_does_not_reset_the_other_counts(turn):
    middleware = _filesystem_middleware()
    runs: list = []
    for i in range(2):
        await _run_builtin(middleware, "execute", {"command": "python plot.py"}, f"x{i}", runs)
    await _run_builtin(
        middleware,
        "write_file",
        {"file_path": "plot.py", "content": "print(1)"},
        "w",
        runs,
        content="Cannot write to plot.py because it already exists.",
        status="error",
    )
    refused = await _run_builtin(middleware, "execute", {"command": "python plot.py"}, "x2", runs)
    assert runs == ["x0", "x1", "w"]
    assert refused.content.startswith("NOT RUN.")


async def test_a_listing_runs_again_after_a_new_command(turn):
    middleware = _filesystem_middleware()
    runs: list = []
    for i in range(2):
        await _run_builtin(middleware, "ls", {"path": "/workspace"}, f"ls{i}", runs)
        await _run_builtin(middleware, "execute", {"command": f"python step{i}.py"}, f"x{i}", runs)
    await _run_builtin(middleware, "ls", {"path": "/workspace"}, "ls2", runs)
    await _run_builtin(middleware, "read_file", {"file_path": "out.csv"}, "r0", runs)
    await _run_builtin(middleware, "read_file", {"file_path": "out.csv"}, "r1", runs)
    await _run_builtin(middleware, "execute", {"command": "python step0.py"}, "x2", runs)
    await _run_builtin(middleware, "read_file", {"file_path": "out.csv"}, "r2", runs)
    assert runs == ["ls0", "x0", "ls1", "x1", "ls2", "r0", "r1", "x2", "r2"]


async def test_polling_with_a_changing_output_is_never_refused(turn):
    middleware = _filesystem_middleware()
    runs: list = []
    for i in range(10):
        await _run_builtin(
            middleware, "execute", {"command": "tail -n 1 job.log"}, f"p{i}", runs, content=f"step {i}"
        )
    assert len(runs) == 10


async def test_the_same_error_twice_refuses_the_third_retry(turn):
    middleware = _filesystem_middleware()
    runs: list = []
    for i in range(3):
        with contextlib.suppress(TimeoutError):
            await _run_builtin(
                middleware, "execute", {"command": "curl x"}, f"t{i}", runs, content=TimeoutError("slow")
            )
    assert runs == ["t0", "t1"]


async def test_two_different_transient_errors_let_the_retry_run(turn):
    interceptor = RepeatGuardInterceptor()
    outcomes = [RuntimeError("502"), RuntimeError("503"), CallToolResult(content=[TextContent(type="text", text="ok")])]
    calls = 0

    async def handler(request):
        nonlocal calls
        outcome = outcomes[calls]
        calls += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    for _ in range(2):
        with pytest.raises(RuntimeError):
            await interceptor(_mcp_request("get_fires", {"year": 2018}), handler)
    result = await interceptor(_mcp_request("get_fires", {"year": 2018}), handler)
    assert calls == 3
    assert result.content[0].text == "ok"


async def test_parallel_copies_in_one_step_say_not_run_and_how_many_ran(turn):
    """Local run 6afecd0d: five identical execute calls in one step, three refused."""
    from langchain_core.messages import ToolMessage

    middleware = _filesystem_middleware()
    runs: list = []

    async def handler(request):
        runs.append(request.tool_call["id"])
        await asyncio.sleep(0.01)
        return ToolMessage(content="ok", name="execute", tool_call_id=request.tool_call["id"])

    results = await asyncio.gather(
        *[
            type(middleware).awrap_tool_call(
                middleware, _tool_request("execute", {"command": DEV_COMMAND}, f"c{i}"), handler
            )
            for i in range(5)
        ]
    )
    refused = [r.content for r in results if r.content != "ok"]
    assert len(runs) == 2
    assert len(refused) == 3
    for text in refused:
        assert text.startswith("NOT RUN. This exact call already ran 2 times in this turn.")
        assert "same result" not in text


async def test_a_subagent_has_its_own_counts_and_never_pushes_the_parent_to_stop(turn):
    from langchain_core.messages import ToolMessage

    middleware = _filesystem_middleware()
    runs: list = []

    async def subagent(request):
        for i in range(6):
            await _run_builtin(middleware, "execute", {"command": DEV_COMMAND}, f"s{i}", runs)
        return ToolMessage(content="subagent done", name="task", tool_call_id="task1")

    result = await type(middleware).awrap_tool_call(
        middleware, _tool_request("task", {"description": "x"}, "task1"), subagent
    )
    assert result.content == "subagent done"
    assert runs == ["s0", "s1"]
    assert turn.soft_refusals == 0 and turn.hard_refusals == 0
    # The parent still runs the same command twice on its own counts.
    for i in range(3):
        await _run_builtin(middleware, "execute", {"command": DEV_COMMAND}, f"m{i}", runs)
    assert runs[2:] == ["m0", "m1"]


async def test_a_subagent_that_ignores_the_stop_text_is_cut(turn):
    middleware = _filesystem_middleware()
    runs: list = []

    async def subagent(request):
        for i in range(31):
            await _run_builtin(middleware, "execute", {"command": DEV_COMMAND}, f"s{i}", runs)
        raise AssertionError("the subagent should have been cut")

    result = await type(middleware).awrap_tool_call(
        middleware, _tool_request("task", {"description": "x"}, "task1"), subagent
    )
    assert result.content == SUBAGENT_STOPPED
    assert result.tool_call_id == "task1"
    assert runs == ["s0", "s1"]
    assert turn.end_requested is False


async def test_a_guard_error_runs_the_original_handler(turn, monkeypatch, caplog):
    middleware = _filesystem_middleware()
    runs: list = []

    def broken(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(repeat_guard, "normalise_args", broken)
    with caplog.at_level(logging.WARNING, logger=repeat_guard.__name__):
        for i in range(3):
            await _run_builtin(middleware, "execute", {"command": DEV_COMMAND}, f"c{i}", runs)
    assert runs == ["c0", "c1", "c2"]
    assert "tool.repeat_guard_failed error=RuntimeError" in caplog.text


def test_the_install_skips_inherited_hooks(monkeypatch):
    from deepagents.middleware.filesystem import FilesystemMiddleware

    monkeypatch.delattr(FilesystemMiddleware, "wrap_tool_call")
    monkeypatch.setattr(repeat_guard, "_SKIP_LOGGED", False)
    assert install_builtin_tool_guard() is False


def test_the_end_is_requested_after_the_stop_text_is_ignored_three_times(turn):
    texts = [check_tool_call("execute", {"command": DEV_COMMAND}) for _ in range(8)]
    assert texts[:2] == [None, None]
    assert all("Stop calling tools now" in t for t in texts[4:])
    assert turn.end_requested is True
    _, token = set_tool_repeat_context()
    try:
        texts = [check_tool_call("execute", {"command": DEV_COMMAND}) for _ in range(7)]
        assert repeat_guard.current_tool_repeat_guard().end_requested is False
    finally:
        reset_tool_repeat_context(token)


async def test_a_compiled_deep_graph_routes_built_in_calls_through_the_guard(turn):
    from deepagents import create_deep_agent
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage, ToolMessage

    class _Model(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    def ls_call(i):
        return AIMessage(
            content="",
            tool_calls=[{"name": "ls", "args": {"path": "/"}, "id": f"c{i}", "type": "tool_call"}],
        )

    assert install_builtin_tool_guard() is True
    model = _Model(messages=iter([ls_call(0), ls_call(1), ls_call(2), AIMessage(content="done")]))
    graph = create_deep_agent(model=model, tools=[])
    out = await graph.ainvoke({"messages": [{"role": "user", "content": "list the files"}]})

    results = {m.tool_call_id: m.content for m in out["messages"] if isinstance(m, ToolMessage)}
    assert not results["c0"].startswith("NOT RUN.")
    assert not results["c1"].startswith("NOT RUN.")
    assert results["c2"].startswith("NOT RUN. This exact call already ran 2 times")
    assert out["messages"][-1].content == "done"


# ── the runner ends a looping turn cleanly ──────────────────────────────────


class _LoopingToolGraph:
    """Calls the same tool on every step, as the dev turn did 31 times."""

    def __init__(self, steps: list, answer_first: bool = False):
        self.steps = steps
        self.answer_first = answer_first

    async def astream(self, *args, **kwargs):
        from langchain_core.messages import AIMessageChunk, ToolMessage

        if self.answer_first:
            yield "messages", (
                AIMessageChunk(content="Partial findings.", id="a0"),
                {"langgraph_node": "model"},
            )
        for i in range(31):

            async def tool_call(call_id=f"c{i}"):
                refusal, ticket = begin_tool_call("execute", {"command": DEV_COMMAND})
                if refusal is not None:
                    return ToolMessage(content=refusal, tool_call_id=call_id)
                result = ToolMessage(content="null\nnull", tool_call_id=call_id)
                finish_tool_call(ticket, result)
                return result

            message = await asyncio.create_task(tool_call())
            self.steps.append(message.content)
            yield "updates", {"tools": {"messages": [message]}}


async def _stream_turn(message_id: str = "m1"):
    from src.schemas.generation_request import GenerationRequest
    from src.services.agents.core.runner import generate_answer_agentic_stream_helper

    frames = []
    async for event in generate_answer_agentic_stream_helper(
        GenerationRequest(query="metrics over the Attica fire", agent="deep"),
        conversation_id="c1",
        message_id=message_id,
        user_id="test-user",
    ):
        frames.append(event)
    return frames


@pytest.mark.parametrize("answer_first", [False, True])
async def test_the_31_call_turn_ends_early_with_a_stored_answer(answer_first):
    from unittest.mock import MagicMock

    from tests.test_agentic_fallback import _patched_runner

    steps: list = []
    with _patched_runner(
        _build_react_graph=MagicMock(return_value=_LoopingToolGraph(steps, answer_first))
    ) as patched:
        frames = await asyncio.wait_for(_stream_turn(), timeout=10)

    # 2 runs, 2 soft refusals, the stop text, then 3 more refusals: 8 steps.
    assert len(steps) == 8
    final_calls = [
        c.kwargs
        for c in patched["persist_message_state"].await_args_list
        if "output" in c.kwargs and "latencies" in c.kwargs
    ]
    assert len(final_calls) == 1
    stored = final_calls[0]
    assert stored["output"].endswith(END_NOTE)
    assert "error" not in stored
    if answer_first:
        assert stored["output"].startswith("Partial findings.")
    assert not any('"type": "error"' in f for f in frames)
    assert any('"type": "final"' in f for f in frames)


async def test_two_concurrent_turns_each_refuse_only_their_own_third_call():
    from unittest.mock import MagicMock

    from tests.test_agentic_fallback import _patched_runner

    class _Interleaving:
        def __init__(self, seen):
            self.seen = seen

        async def astream(self, *args, **kwargs):
            from langchain_core.messages import AIMessage

            for _ in range(3):
                await asyncio.sleep(0)
                self.seen.append(
                    await asyncio.to_thread(check_tool_call, "execute", {"command": DEV_COMMAND})
                )
            yield "messages", (AIMessage(content="done"), {"langgraph_node": "agent"})

    seen_a: list = []
    seen_b: list = []
    graphs = iter([_Interleaving(seen_a), _Interleaving(seen_b)])
    with _patched_runner(_build_react_graph=MagicMock(side_effect=lambda *a, **k: next(graphs))):
        await asyncio.gather(_stream_turn("m1"), _stream_turn("m2"))

    assert [r is None for r in seen_a] == [True, True, False]
    assert [r is None for r in seen_b] == [True, True, False]

