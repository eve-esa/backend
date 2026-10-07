"""Retrieval before the first answer on an agentic turn (``force_first_tool``).

The runner asks the react graph for one ``eve_retrieval_retrieve`` call before
the model answers only when the ``eve_retrieval`` toolkit is on and its tool
loaded, at least one collection is selected and the message does not opt out.

The last tests run the real react graph from the pinned ``eve-esa-agents``
with a scripted model: the forced call reaches the stream and the persisted
trace, and a Stop during it leaves a thread the next turn can read.
"""

import asyncio
import contextlib
import json
from types import SimpleNamespace
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver

from src.schemas.generation_request import GenerationRequest
from src.services.agents.core.registry import get_agent_graph
from src.services.agents.core.runner import (
    RETRIEVAL_OPT_OUT_PHRASES,
    _force_first_tool,
    generate_answer_agentic,
    generate_answer_agentic_stream_helper,
)

_RUNNER = "src.services.agents.core.runner"
RETRIEVE = "eve_retrieval_retrieve"

pytestmark = pytest.mark.no_db


def _tool(name: str = RETRIEVE) -> Any:
    return SimpleNamespace(name=name)


def _request(
    query: str = "What is the Doppler effect?",
    *,
    servers=("eve_retrieval",),
    public=("Wiley AI Gateway",),
    private=(),
) -> GenerationRequest:
    request = GenerationRequest(
        query=query,
        public_collections=list(public),
        private_collections=list(private),
    )
    request.mcp_server_configs = [SimpleNamespace(name=name) for name in servers]
    return request


# ─── the decision ─────────────────────────────────────────────────────────────


class TestForceFirstTool:
    def test_toolkit_on_collection_selected_forces_retrieval(self):
        assert _force_first_tool(_request(), [_tool()]) == RETRIEVE

    def test_a_private_collection_alone_is_a_selection(self):
        request = _request(public=(), private=("6aa28720dafbcdc730a04acd",))
        assert _force_first_tool(request, [_tool()]) == RETRIEVE

    def test_toolkit_off_forces_nothing(self):
        request = _request(servers=("geocode",))
        assert _force_first_tool(request, [_tool("geocode_geocode_place")]) is None

    def test_toolkit_on_but_tool_not_loaded_forces_nothing(self):
        # Tool discovery failed or timed out: the graph has nothing to call.
        assert _force_first_tool(_request(), []) is None
        assert _force_first_tool(_request(), [_tool("geocode_geocode_place")]) is None

    def test_no_collection_selected_forces_nothing(self):
        assert _force_first_tool(_request(public=(), private=()), [_tool()]) is None

    @pytest.mark.parametrize("phrase", RETRIEVAL_OPT_OUT_PHRASES)
    def test_an_opt_out_phrase_forces_nothing(self, phrase):
        query = f"Explain the Doppler effect, {phrase.upper()} please"
        assert _force_first_tool(_request(query), [_tool()]) is None

    def test_opt_out_survives_a_typographic_apostrophe_and_extra_spaces(self):
        query = "Explain the Doppler effect but don’t   retrieve anything"
        assert _force_first_tool(_request(query), [_tool()]) is None

    def test_the_opt_out_list_is_the_prompt_list(self):
        assert set(RETRIEVAL_OPT_OUT_PHRASES) == {
            "no rag",
            "do not use tools",
            "don't use tools",
            "without sources",
            "do not retrieve",
            "don't retrieve",
        }


# ─── the run config the runner hands to the graph ─────────────────────────────


class _RecordingGraph:
    """Answers at once and keeps the config of every run."""

    def __init__(self, streaming: bool):
        self.configs: List[dict] = []
        self._streaming = streaming

    async def astream(self, state, config=None, stream_mode=None):
        self.configs.append(config)
        answer = AIMessage(content="answer")
        if self._streaming:
            yield "messages", (answer, {"langgraph_node": "agent"})
        else:
            yield {"agent": {"messages": [answer]}}


def _fake_agent():
    agent = MagicMock()
    agent.instruction_text.return_value = ""
    agent.prompts = {}
    return agent


@contextlib.contextmanager
def _patched_runner(**overrides):
    defaults = {
        "_langgraph_available": True,
        "_get_agentic_checkpointer": AsyncMock(return_value=None),
        "_fetch_conversation_context": AsyncMock(return_value=([], None)),
        "_resolve_agent_graph_type": MagicMock(return_value="react"),
        "get_agent_graph": MagicMock(return_value=_fake_agent()),
        "_resolve_agentic_llm_client": AsyncMock(return_value=(MagicMock(), {})),
        "persist_policy_event": AsyncMock(),
        "persist_message_state": AsyncMock(),
        "maybe_rollup_and_trim_history": AsyncMock(),
    }
    defaults.update(overrides)
    with contextlib.ExitStack() as stack:
        applied = {
            name: stack.enter_context(patch(f"{_RUNNER}.{name}", value))
            for name, value in defaults.items()
        }
        yield applied


_CASES = {
    "rule 1, on": (_request(), [_tool()], RETRIEVE),
    "rule 2, toolkit off": (
        _request(servers=("geocode",)),
        [_tool("geocode_geocode_place")],
        None,
    ),
    "rule 3, no collection": (_request(public=()), [_tool()], None),
    "rule 4, opt out": (_request("Doppler effect, no RAG"), [_tool()], None),
    "tool not loaded": (_request(), [], None),
}


async def _run(streaming: bool, request, tools) -> dict:
    graph = _RecordingGraph(streaming)
    with _patched_runner(
        _build_tools=AsyncMock(return_value=tools),
        _build_react_graph=MagicMock(return_value=graph),
    ):
        if streaming:
            async for _event in generate_answer_agentic_stream_helper(
                request, conversation_id="c1", message_id="m1", user_id="u1"
            ):
                pass
        else:
            await generate_answer_agentic(request, user_id="u1", conversation_id="c1")
    assert len(graph.configs) == 1
    return graph.configs[0]["configurable"]


@pytest.mark.parametrize("streaming", [False, True], ids=["sync", "stream"])
@pytest.mark.parametrize("case", list(_CASES))
async def test_the_run_config_carries_the_forced_tool_only_when_the_rules_hold(
    streaming, case
):
    request, tools, expected = _CASES[case]
    configurable = await _run(streaming, request, tools)
    assert configurable["thread_id"] == "c1"
    if expected is None:
        assert "force_first_tool" not in configurable
    else:
        assert configurable["force_first_tool"] == expected


# ─── the real react graph ─────────────────────────────────────────────────────


class _ScriptedModel(BaseChatModel):
    """Returns ``answer`` and records the messages of every call."""

    calls: Any = None

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="answer"))])


def _retrieve_tool(received: List[dict], gate: asyncio.Event = None):
    async def retrieve(
        query: str,
        public_collections: List[str] = None,
        private_collections: List[str] = None,
    ) -> str:
        received.append(
            {
                "query": query,
                "public_collections": public_collections,
                "private_collections": private_collections,
            }
        )
        if gate is not None:
            await gate.wait()
        return json.dumps([])

    return StructuredTool.from_function(
        coroutine=retrieve, name=RETRIEVE, description="Search the documents."
    )


@contextlib.contextmanager
def _real_graph(model, tool, checkpointer):
    manager = MagicMock()
    manager.get_client_for_model.return_value = model
    with _patched_runner(
        get_agent_graph=get_agent_graph,
        _get_agentic_checkpointer=AsyncMock(return_value=checkpointer),
        _resolve_agentic_llm_client=AsyncMock(return_value=(model, {})),
        _load_mcp_tools_for_servers=AsyncMock(return_value=[tool]),
        get_shared_llm_manager=MagicMock(return_value=manager),
    ) as patched:
        yield patched


async def _stream(request, conversation_id, message_id, cancel_event=None):
    events = []
    async for event in generate_answer_agentic_stream_helper(
        request,
        conversation_id=conversation_id,
        message_id=message_id,
        user_id="u1",
        cancel_event=cancel_event,
    ):
        events.append(json.loads(event[len("data: "):]))
    return events


async def test_the_forced_call_streams_a_tool_step_before_the_answer():
    received: List[dict] = []
    model = _ScriptedModel(calls=[])
    with _real_graph(model, _retrieve_tool(received), InMemorySaver()) as patched:
        events = await _stream(_request(), "thread-1", "m1")

    kinds = [event["type"] for event in events]
    assert kinds.index("tool_call") < kinds.index("tool_result") < kinds.index("token")
    tool_call = next(event for event in events if event["type"] == "tool_call")
    assert tool_call["tool"] == RETRIEVE
    assert tool_call["query"] == "What is the Doppler effect?"
    # The forced call goes through the UI parameter override.
    assert received == [
        {
            "query": "What is the Doppler effect?",
            "public_collections": ["Wiley AI Gateway"],
            "private_collections": [],
        }
    ]

    persisted = patched["persist_message_state"].await_args.kwargs
    assert persisted["output"] == "answer"
    trace = persisted["trace"]
    assert [(step["node"], step["role"]) for step in trace] == [
        ("force_tool", "assistant"),
        ("tools", "tool"),
        ("agent", "assistant"),
    ]
    assert trace[0]["tool_calls"][0]["name"] == RETRIEVE
    assert trace[1]["name"] == RETRIEVE
    assert "node_force_tool_s" in persisted["latencies"]
    # One model call, after the tool result.
    assert len(model.calls) == 1
    assert any(isinstance(m, ToolMessage) or "[TOOL_RESULTS]" in str(m.content)
               for m in model.calls[0])


async def test_a_stop_during_the_forced_call_leaves_a_thread_the_next_turn_reads():
    received: List[dict] = []
    gate = asyncio.Event()
    model = _ScriptedModel(calls=[])
    checkpointer = InMemorySaver()
    cancel_event = asyncio.Event()

    with _real_graph(model, _retrieve_tool(received, gate), checkpointer) as patched:
        turn = asyncio.create_task(
            _stream(_request(), "thread-2", "m1", cancel_event=cancel_event)
        )
        while not received:
            await asyncio.sleep(0.01)
        # What the Stop path does: set the event, then cancel the running turn.
        cancel_event.set()
        turn.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await turn
        stopped = patched["persist_message_state"].await_args.kwargs
        assert stopped["stopped"] is True
        assert model.calls == []

        state = await checkpointer.aget_tuple(
            {"configurable": {"thread_id": "thread-2"}}
        )
        last = state.checkpoint["channel_values"]["messages"][-1]
        assert isinstance(last, AIMessage) and last.tool_calls  # unanswered

        gate.set()
        events = await _stream(_request("And in 2020?"), "thread-2", "m2")

    assert events[-1]["type"] == "final"
    assert events[-1]["answer"] == "answer"
    assert received[-1]["query"] == "What is the Doppler effect? And in 2020?"
    # The model never sees a structured tool call without its result: the
    # graph rewrites the history into the text tool format first.
    assert len(model.calls) == 1
    sent = model.calls[0]
    assert not any(getattr(m, "tool_calls", None) for m in sent)
    assert not any(isinstance(m, ToolMessage) for m in sent)
    assert any("[TOOL_CALLS]" in str(m.content) for m in sent)
