"""A skill enabled or disabled between two turns of one thread reaches the next turn.

The runner compiles the deep graph per turn with that turn's skills tree on
``/skills/``. deepagents' SkillsMiddleware caches ``skills_metadata`` in the
checkpointed thread state and skips the load while it holds a list, so the
runner asks for a reload when the turn's local skills key differs from the key
the last turn recorded in its checkpoint metadata, and only then (a reload
lists the sandbox again).
"""

from __future__ import annotations

import asyncio
import logging
import threading
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
from src.services.agents.core.skills import (  # noqa: E402
    build_skills_root,
    skills_root_key,
)

pytestmark = pytest.mark.no_db

THREAD = "t1"
KEY = runner._SKILLS_KEY_METADATA


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
    if skills_root is None:
        # The runner's turn without a skills tree: no /skills/ route and no
        # SkillsMiddleware, so the thread keeps whatever list it had.
        return create_deep_agent(
            model=model,
            tools=[],
            backend=CompositeBackend(default=StateBackend(), routes={}),
            checkpointer=checkpointer,
        )
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

    def backend_only(self, *skills):
        """A tree with no packaged skills, as when the deep graph ships none."""
        return build_skills_root(list(skills), None)

    async def turn(self, root, text: str, *, listed_root=None) -> dict:
        """One turn the way the runner drives it.

        *listed_root* is the tree the graph lists, when it differs from the
        local tree (a sandbox whose listing disagrees).
        """
        config: dict = {"configurable": {"thread_id": THREAD}}
        found = await self.checkpointer.aget_tuple(config)
        reload: dict = {}
        if root is not None:
            reload, metadata = await _skills_reload_input(found, THREAD, root)
            if metadata:
                config["metadata"] = metadata
        payload = {"messages": [HumanMessage(content=text)], **reload}
        self.payloads.append(payload)
        graph = _graph(listed_root or root, self.model, self.checkpointer)
        await graph.ainvoke(payload, config=config)
        return reload

    async def stored_key(self):
        found = await self.checkpointer.aget_tuple({"configurable": {"thread_id": THREAD}})
        return found.metadata.get(KEY)


@pytest.fixture
def world(tmp_path, monkeypatch):
    skills_module._tree_key.cache_clear()
    monkeypatch.setattr(skills_module, "_ROOT_KEYS", {})
    # Several tests edit an unaddressed tree in place between turns.
    monkeypatch.setattr(skills_module, "_UNADDRESSED_KEY_TTL_S", 0)
    return _World(tmp_path, monkeypatch)


def _write(folder, text: str):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(text)


async def test_skill_enabled_mid_conversation_reaches_the_next_turn(world):
    await world.turn(world.root(), "hi")
    reload = await world.turn(world.root(_skill("roll-dice", "Roll dice.")), "roll")

    assert reload == {"skills_metadata": None}
    assert "eo-basics" in world.model.prompts[0]
    assert "roll-dice" not in world.model.prompts[0]
    assert "roll-dice" in world.model.prompts[-1]


async def test_first_turn_without_skills_then_one_skill_lists_it(world):
    # The local e2e: no packaged skills, turn 1 with no skills (no tree, no
    # middleware, no key), turn 2 with one skill.
    await world.turn(None, "hi")
    await world.turn(world.backend_only(_skill("ndvi-helper", "Vegetation index.")), "ndvi")

    assert "ndvi-helper" not in world.model.prompts[0]
    assert "ndvi-helper" in world.model.prompts[-1]


async def test_a_turn_without_skills_does_not_hide_the_next_change(world):
    # Turn 2 runs without a tree: no middleware, no key recorded, while the
    # thread still holds turn 1's list. Turn 3 must list its own skill.
    await world.turn(world.backend_only(_skill("roll-dice", "Roll dice.")), "roll")
    await world.turn(None, "plain")
    reload = await world.turn(world.backend_only(_skill("ndvi-helper", "Vegetation index.")), "ndvi")

    assert reload == {"skills_metadata": None}
    assert "ndvi-helper" in world.model.prompts[-1]
    assert "roll-dice" not in world.model.prompts[-1]


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
    assert await world.stored_key() == skills_root_key(world.root(edited))


async def test_a_listing_that_differs_from_the_local_tree_does_not_reload(world, tmp_path):
    # The sandbox listed nothing (failed download, other cwd): the checkpointed
    # list never matches the local tree, and that must not reload every turn.
    root = world.root(_skill("roll-dice", "Roll dice."))
    empty = tmp_path / "sandbox-listing"
    empty.mkdir()

    await world.turn(root, "one", listed_root=empty)
    second = await world.turn(root, "two", listed_root=empty)
    third = await world.turn(root, "three", listed_root=empty)

    assert second == {} and third == {}


async def test_duplicate_skill_names_do_not_reload(world, tmp_path):
    root = tmp_path / "dupes"
    _write(root / "alpha", "---\nname: same\ndescription: First.\n---\n")
    _write(root / "beta", "---\nname: same\ndescription: Second.\n---\n")

    await world.turn(root, "one")
    second = await world.turn(root, "two")
    third = await world.turn(root, "three")

    assert second == {} and third == {}


async def test_a_skill_that_does_not_parse_is_left_out_once(world, tmp_path, caplog):
    root = tmp_path / "tree"
    _write(root / "good", "---\nname: good\ndescription: Fine.\n---\n")
    _write(root / "broken", "---\nname: [unclosed\n---\n")
    _write(root / "bare", "no frontmatter at all\n")
    caplog.set_level(logging.WARNING, logger=skills_module.logger.name)

    await world.turn(root, "one")
    second = await world.turn(root, "two")
    # Logged once per tree state, not once per turn.
    lines = [r for r in caplog.records if "left out of the reload check" in r.getMessage()]
    _write(root / "good", "---\nname: good\ndescription: Changed.\n---\n")
    third = await world.turn(root, "three")

    assert second == {}, "a bad file must not force a reload every turn"
    assert third == {"skills_metadata": None}, "the good skills still count"
    assert sorted(r.getMessage().split()[1] for r in lines) == ["bare", "broken"]
    assert all(r.levelno == logging.WARNING and r.exc_info is None for r in lines)


async def test_no_skills_or_no_conversation_is_untouched():
    found = SimpleNamespace(metadata={KEY: "old"})

    assert await _skills_reload_input(found, THREAD, None) == ({}, {})
    assert await _skills_reload_input(found, None, "/x") == ({}, {})


async def test_a_thread_from_before_the_check_reloads_once(world):
    # Checkpoint written before the key existed: its list may be stale, so
    # list again once and start recording.
    root = world.root(_skill("roll-dice", "Roll dice."))
    found = SimpleNamespace(
        metadata={"source": "loop"},
        checkpoint={"channel_values": {"skills_metadata": [{"name": "old"}]}},
    )

    reload, metadata = await _skills_reload_input(found, THREAD, root)

    assert reload == {"skills_metadata": None}
    assert metadata == {KEY: skills_root_key(root)}


async def test_no_checkpoint_does_not_reload(world):
    root = world.root(_skill("roll-dice", "Roll dice."))

    reload, metadata = await _skills_reload_input(None, THREAD, root)

    assert reload == {}
    assert metadata == {KEY: skills_root_key(root)}


@pytest.mark.parametrize(
    "found",
    [
        object(),
        SimpleNamespace(metadata="not a dict"),
        SimpleNamespace(metadata={KEY: 5}),
        SimpleNamespace(metadata={KEY: ""}),
        SimpleNamespace(metadata={KEY: {"nested": "value"}}),
    ],
)
async def test_an_unreadable_stored_key_counts_as_missing(world, found):
    root = world.root(_skill("roll-dice", "Roll dice."))

    reload, metadata = await _skills_reload_input(found, THREAD, root)

    assert reload == {"skills_metadata": None}
    assert metadata == {KEY: skills_root_key(root)}


async def test_a_scan_error_keeps_the_list_and_the_old_key(world, monkeypatch, caplog):
    def _boom(_root):
        raise PermissionError("/tmp/eve-skills/secret-path")

    monkeypatch.setattr(runner, "skills_root_key", _boom)
    found = SimpleNamespace(metadata={KEY: "old"})

    reload, metadata = await _skills_reload_input(found, THREAD, world.root())

    assert reload == {}
    assert metadata == {KEY: "old"}, "the next turn still compares against it"
    (record,) = [r for r in caplog.records if "skills reload check" in r.getMessage()]
    assert record.levelno == logging.WARNING
    assert record.exc_info is None
    assert "PermissionError" in record.getMessage()
    assert "secret-path" not in record.getMessage()


async def test_a_slow_scan_is_bounded_and_finishes_in_the_background(
    world, monkeypatch, caplog
):
    release = threading.Event()
    finished = threading.Event()

    def _slow(_root):
        release.wait(5)
        finished.set()
        return "new"

    monkeypatch.setattr(runner, "skills_root_key", _slow)
    monkeypatch.setattr(runner, "_SKILLS_CHECK_TIMEOUT_S", 0.05)
    found = SimpleNamespace(metadata={KEY: "old"})

    reload, metadata = await _skills_reload_input(found, THREAD, world.root())

    assert reload == {}
    assert metadata == {KEY: "old"}
    (record,) = [r for r in caplog.records if "skills reload check" in r.getMessage()]
    # Under load the wait measures the executor queue: no WARNING per turn.
    assert record.levelno == logging.INFO
    assert record.exc_info is None
    assert "timeouts=" in record.getMessage()
    assert not finished.is_set()
    release.set()
    assert await asyncio.to_thread(finished.wait, 5)


async def test_a_slow_scan_without_a_stored_key_records_nothing(world, monkeypatch):
    # Nothing to re-record: the next turn finds no key and reloads.
    monkeypatch.setattr(runner, "skills_root_key", lambda _root: threading.Event().wait(1))
    monkeypatch.setattr(runner, "_SKILLS_CHECK_TIMEOUT_S", 0.01)

    reload, metadata = await _skills_reload_input(
        SimpleNamespace(metadata={}), THREAD, world.root()
    )

    assert reload == {} and metadata == {}


async def test_an_unreadable_checkpoint_reloads_and_records_the_key(world):
    root = world.root(_skill("roll-dice", "Roll dice."))

    reload, metadata = await _skills_reload_input(
        runner._CHECKPOINT_UNREADABLE, THREAD, root
    )

    assert reload == {"skills_metadata": None}
    assert metadata == {KEY: skills_root_key(root)}


async def test_an_unreadable_checkpoint_and_no_key_records_nothing(world, monkeypatch):
    def _boom(_root):
        raise OSError("disk")

    monkeypatch.setattr(runner, "skills_root_key", _boom)

    reload, metadata = await _skills_reload_input(
        runner._CHECKPOINT_UNREADABLE, THREAD, world.root()
    )

    assert reload == {}
    assert metadata == {}, "no key recorded, so the next turn reloads"


async def test_a_listing_with_errors_reloads_on_the_next_turn(world):
    root = world.root(_skill("roll-dice", "Roll dice."))
    key = skills_root_key(root)
    failed = SimpleNamespace(
        metadata={KEY: key},
        checkpoint={"channel_values": {"skills_metadata": [], "skills_load_errors": ["x"]}},
    )
    clean = SimpleNamespace(
        metadata={KEY: key},
        checkpoint={"channel_values": {"skills_metadata": [], "skills_load_errors": []}},
    )

    assert (await _skills_reload_input(failed, THREAD, root))[0] == {"skills_metadata": None}
    assert (await _skills_reload_input(clean, THREAD, root))[0] == {}


async def test_an_addressed_tree_is_scanned_once(world, monkeypatch):
    root = world.root(_skill("roll-dice", "Roll dice."))
    calls: list = []
    real = runner.skills_root_key

    def _counting(path):
        calls.append(path)
        return real(path)

    monkeypatch.setattr(runner, "skills_root_key", _counting)
    found = SimpleNamespace(metadata={KEY: "old"})

    first = await _skills_reload_input(found, THREAD, root)
    second = await _skills_reload_input(found, THREAD, root)

    assert first == second
    assert len(calls) == 1


def test_an_unaddressed_root_is_scanned_again_after_the_ttl(
    world, monkeypatch, tmp_path
):
    root = tmp_path / "packaged-live"
    _write(root / "good", "---\nname: good\ndescription: Fine.\n---\n")
    monkeypatch.setattr(skills_module, "_UNADDRESSED_KEY_TTL_S", 30.0)
    clock = [1000.0]
    monkeypatch.setattr(skills_module.time, "monotonic", lambda: clock[0])

    key = skills_root_key(root)
    assert skills_module.cached_skills_root_key(root) == key
    clock[0] += 31
    assert skills_module.cached_skills_root_key(root) is None


# ─── the runner: one checkpoint read, reload and key on both paths ────────────


class _RecordingGraph:
    def __init__(self, stream: bool) -> None:
        self.inputs: list = []
        self.configs: list = []
        self.stream = stream

    async def astream(self, graph_input, config=None, **_kwargs):
        self.inputs.append(graph_input)
        self.configs.append(config)
        if self.stream:
            yield ("messages", (AIMessage(content="ok"), {"langgraph_node": "agent"}))
            return
        yield {"agent": {"messages": [AIMessage(content="ok")]}}

    async def aget_state(self, _config):
        return SimpleNamespace(values={})


class _DeepAgent:
    """``compile`` declares ``backend`` and ``skills``, like the deep graph."""

    def __init__(self) -> None:
        self.prompts: dict = {}
        self.name = "deep"

    def instruction_text(self, **_kwargs):
        return ""

    def compile(self, backend=None, skills=None, **kwargs):
        raise AssertionError("the test builds the graph itself")


def _patch_runner(monkeypatch, graph, root):
    checkpointer = MagicMock()
    checkpointer.aget_tuple = AsyncMock(
        return_value=SimpleNamespace(metadata={KEY: "key-of-the-last-turn"}, checkpoint={})
    )
    for name, value in {
        "_langgraph_available": True,
        "_build_tools": AsyncMock(return_value=[]),
        "_get_agentic_checkpointer": AsyncMock(return_value=checkpointer),
        "_fetch_conversation_context": AsyncMock(return_value=([], None)),
        "_resolve_agent_graph_type": MagicMock(return_value="deep"),
        "get_agent_graph": MagicMock(return_value=_DeepAgent()),
        "_resolve_agentic_llm_client": AsyncMock(return_value=(MagicMock(), {})),
        "_build_react_graph": MagicMock(return_value=graph),
        "resolve_skills_root": MagicMock(return_value=root),
        "acquire_sandbox": AsyncMock(return_value=None),
        "open_sandbox": AsyncMock(return_value=None),
        "persist_policy_event": AsyncMock(),
        "persist_message_state": AsyncMock(),
        "maybe_rollup_and_trim_history": AsyncMock(),
        "save_agent_summary": AsyncMock(),
    }.items():
        monkeypatch.setattr(runner, name, value, raising=False)
    return checkpointer


def _assert_turn(graph, checkpointer, root):
    checkpointer.aget_tuple.assert_awaited_once()
    assert graph.inputs[0]["skills_metadata"] is None
    assert graph.inputs[0]["messages"]
    assert graph.configs[0]["metadata"] == {KEY: skills_root_key(root)}
    assert graph.configs[0]["configurable"] == {"thread_id": "c1"}


async def test_sync_turn_reads_once_and_sends_the_reload(world, monkeypatch):
    root = world.root(_skill("roll-dice", "Roll dice."))
    graph = _RecordingGraph(stream=False)
    checkpointer = _patch_runner(monkeypatch, graph, root)

    result = await runner.generate_answer_agentic(
        GenerationRequest(query="hi", agent="deep"), user_id="u1", conversation_id="c1"
    )

    _assert_turn(graph, checkpointer, root)
    assert result[3]["setup_skills_check_s"] >= 0


async def test_streamed_turn_reads_once_and_sends_the_reload(world, monkeypatch):
    root = world.root(_skill("roll-dice", "Roll dice."))
    graph = _RecordingGraph(stream=True)
    checkpointer = _patch_runner(monkeypatch, graph, root)

    async for _event in runner.generate_answer_agentic_stream_helper(
        GenerationRequest(query="hi", agent="deep"),
        conversation_id="c1",
        message_id="m1",
        user_id="u1",
    ):
        pass

    assert graph.inputs, "the stream never reached the graph"
    _assert_turn(graph, checkpointer, root)


async def test_a_failed_checkpoint_read_reloads_the_list(world, monkeypatch):
    # The thread is treated as there; its list may be stale, so list again
    # instead of recording this turn's key over it.
    root = world.root(_skill("roll-dice", "Roll dice."))
    graph = _RecordingGraph(stream=False)
    checkpointer = _patch_runner(monkeypatch, graph, root)
    checkpointer.aget_tuple = AsyncMock(side_effect=RuntimeError("primary down"))

    await runner.generate_answer_agentic(
        GenerationRequest(query="hi", agent="deep"), user_id="u1", conversation_id="c1"
    )

    _assert_turn(graph, checkpointer, root)
