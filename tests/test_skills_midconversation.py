"""A skill enabled or disabled between two turns of one thread reaches the next turn.

The runner compiles the deep graph per turn with that turn's skills tree on
``/skills/``. deepagents' SkillsMiddleware caches ``skills_metadata`` in the
checkpointed thread state and skips the load while it holds a list, so the
runner asks for a reload when the turn's skill set differs from that list,
and only then (a reload lists the sandbox again).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("deepagents")

from deepagents import create_deep_agent  # noqa: E402
from deepagents.backends import CompositeBackend, StateBackend  # noqa: E402
from deepagents.backends.filesystem import FilesystemBackend  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from src.schemas.generation_request import GenerationRequest  # noqa: E402
from src.services.agents.core import runner  # noqa: E402
from src.services.agents.core import skills as skills_module  # noqa: E402
from src.services.agents.core.runner import _skills_reload_input  # noqa: E402
from src.services.agents.core.skills import build_skills_root  # noqa: E402

pytestmark = pytest.mark.no_db

THREAD = "t1"


class _RecordingModel(BaseChatModel):
    """Answers "ok" and keeps the system prompt of every call."""

    prompts: list = []

    @property
    def _llm_type(self) -> str:
        return "recording"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_RecordingModel":
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.prompts.append(
            "\n".join(str(m.content) for m in messages if m.type == "system")
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])


def _skill(name: str, description: str, body: str = "Do it.") -> SimpleNamespace:
    text = f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n"
    return SimpleNamespace(
        name=name,
        description=description,
        files=[SimpleNamespace(path="SKILL.md", content=text.encode("utf-8"))],
    )


def _graph(skills_root, model, checkpointer):
    backend = CompositeBackend(
        default=StateBackend(),
        routes={"/skills/": FilesystemBackend(root_dir=skills_root, virtual_mode=True)},
    )
    return create_deep_agent(
        model=model,
        tools=[],
        backend=backend,
        skills=["/skills/"],
        checkpointer=checkpointer,
    )


class _World:
    def __init__(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(skills_module, "_CACHE_DIR", tmp_path / "cache")
        self.packaged = tmp_path / "packaged"
        (self.packaged / "eo-basics").mkdir(parents=True)
        (self.packaged / "eo-basics" / "SKILL.md").write_text(
            "---\nname: eo-basics\ndescription: Packaged EO skill.\n---\n\nBody.\n"
        )
        self.model = _RecordingModel()
        self.model.prompts = []
        self.checkpointer = InMemorySaver()
        self.payloads: list[dict] = []

    def root(self, *skills):
        return build_skills_root(list(skills), self.packaged)

    async def turn(self, root, text: str) -> dict:
        """One turn the way the runner drives it."""
        reload = await _skills_reload_input(self.checkpointer, THREAD, root)
        payload = {"messages": [HumanMessage(content=text)], **reload}
        self.payloads.append(payload)
        await _graph(root, self.model, self.checkpointer).ainvoke(
            payload, config={"configurable": {"thread_id": THREAD}}
        )
        return reload


@pytest.fixture
def world(tmp_path, monkeypatch):
    return _World(tmp_path, monkeypatch)


async def test_skill_enabled_mid_conversation_reaches_the_next_turn(world):
    await world.turn(world.root(), "hi")
    reload = await world.turn(world.root(_skill("roll-dice", "Roll dice.")), "roll")

    assert reload == {"skills_metadata": None}
    assert "eo-basics" in world.model.prompts[0]
    assert "roll-dice" not in world.model.prompts[0]
    assert "roll-dice" in world.model.prompts[-1]


async def test_skill_disabled_mid_conversation_leaves_the_next_turn(world):
    await world.turn(world.root(_skill("roll-dice", "Roll dice.")), "roll")
    reload = await world.turn(world.root(), "bye")

    assert reload == {"skills_metadata": None}
    assert "roll-dice" in world.model.prompts[0]
    assert "roll-dice" not in world.model.prompts[-1]
    assert "eo-basics" in world.model.prompts[-1]


async def test_changed_description_reloads(world):
    await world.turn(world.root(_skill("roll-dice", "Roll dice.")), "roll")
    reload = await world.turn(world.root(_skill("roll-dice", "Roll two dice.")), "again")

    assert reload == {"skills_metadata": None}
    assert "Roll two dice." in world.model.prompts[-1]


async def test_unchanged_set_does_not_reload(world):
    dice = _skill("roll-dice", "Roll dice.")
    first = await world.turn(world.root(dice), "roll")
    second = await world.turn(world.root(dice), "again")
    # A body edit changes the tree digest but not what the model is listed:
    # the body is read from /skills/ on demand, so no reload either.
    edited = _skill("roll-dice", "Roll dice.", body="Do it twice.")
    third = await world.turn(world.root(edited), "once more")

    assert first == {}, "first turn: the middleware loads on its own"
    assert second == {}
    assert third == {}
    assert all("skills_metadata" not in p for p in world.payloads)
    assert "roll-dice" in world.model.prompts[-1]


async def test_no_skills_configured_is_untouched():
    checkpointer = MagicMock()
    checkpointer.aget_tuple = AsyncMock()

    assert await _skills_reload_input(checkpointer, THREAD, None) == {}
    assert await _skills_reload_input(None, THREAD, "/x") == {}
    assert await _skills_reload_input(checkpointer, None, "/x") == {}
    checkpointer.aget_tuple.assert_not_awaited()


async def test_thread_without_a_skill_list_does_not_reload(world, tmp_path):
    checkpointer = MagicMock()
    checkpointer.aget_tuple = AsyncMock(
        return_value=SimpleNamespace(checkpoint={"channel_values": {"messages": []}})
    )
    root = world.root(_skill("roll-dice", "Roll dice."))

    assert await _skills_reload_input(checkpointer, THREAD, root) == {}


async def test_checkpoint_read_failure_keeps_the_list(world, caplog):
    checkpointer = MagicMock()
    checkpointer.aget_tuple = AsyncMock(side_effect=RuntimeError("mongo down"))

    assert await _skills_reload_input(checkpointer, THREAD, world.root()) == {}
    assert "checkpoint read failed" in caplog.text


async def test_slow_checkpoint_read_is_bounded(world, monkeypatch):
    import asyncio

    async def _hang(_config):
        await asyncio.sleep(10)

    checkpointer = MagicMock()
    checkpointer.aget_tuple = _hang
    monkeypatch.setattr(runner, "_SKILLS_STATE_READ_TIMEOUT_S", 0.05)

    assert await _skills_reload_input(checkpointer, THREAD, world.root()) == {}


# ─── both astream inputs carry the reload ─────────────────────────────────────


class _InputRecordingGraph:
    def __init__(self, stream: bool) -> None:
        self.inputs: list = []
        self.stream = stream

    async def astream(self, graph_input, **_kwargs):
        self.inputs.append(graph_input)
        update = {"agent": {"messages": [AIMessage(content="agentic answer")]}}
        yield ("updates", update) if self.stream else update


def _patch_runner(monkeypatch, graph):
    agent = MagicMock()
    agent.instruction_text.return_value = ""
    agent.prompts = {}
    reload = AsyncMock(return_value={"skills_metadata": None})
    for name, value in {
        "_langgraph_available": True,
        "_build_tools": AsyncMock(return_value=[]),
        "_get_agentic_checkpointer": AsyncMock(return_value=object()),
        "_fetch_conversation_context": AsyncMock(return_value=([], None)),
        "_resolve_agent_graph_type": MagicMock(return_value="react"),
        "get_agent_graph": MagicMock(return_value=agent),
        "_resolve_agentic_llm_client": AsyncMock(return_value=(MagicMock(), {})),
        "_build_react_graph": MagicMock(return_value=graph),
        "_skills_reload_input": reload,
        "persist_policy_event": AsyncMock(),
        "persist_message_state": AsyncMock(),
        "maybe_rollup_and_trim_history": AsyncMock(),
    }.items():
        monkeypatch.setattr(runner, name, value)
    return reload


async def test_sync_turn_sends_the_reload(monkeypatch):
    graph = _InputRecordingGraph(stream=False)
    reload = _patch_runner(monkeypatch, graph)

    await runner.generate_answer_agentic(
        GenerationRequest(query="hi", agent="react"), user_id="u1", conversation_id="c1"
    )

    assert graph.inputs[0]["skills_metadata"] is None
    assert graph.inputs[0]["messages"]
    assert reload.await_args.args[1] == "c1"


async def test_streamed_turn_sends_the_reload(monkeypatch):
    graph = _InputRecordingGraph(stream=True)
    reload = _patch_runner(monkeypatch, graph)

    async for _event in runner.generate_answer_agentic_stream_helper(
        GenerationRequest(query="hi", agent="react"),
        conversation_id="c1",
        message_id="m1",
        user_id="u1",
    ):
        pass

    assert graph.inputs, "the stream never reached the graph"
    assert graph.inputs[0]["skills_metadata"] is None
    assert graph.inputs[0]["messages"]
    assert reload.await_args.args[1] == "c1"
