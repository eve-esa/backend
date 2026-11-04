"""Skills requested by name, stored as bytes, merged with the packaged tree."""

import json
import logging
import shutil
import subprocess
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from bson import ObjectId

from src.commands.seed_dice_skill import ROLL_PY, dice_skill
from src.database.models.skill import Skill, SkillFile
from src.routers.message import _prepare_agentic_request
from src.schemas.generation_request import GenerationRequest
from src.services.agentic_utils import is_agentic_generation_request
from src.services.agents.core.skills import (
    build_skills_root,
    packaged_skills_dir,
    resolve_skills_root,
    validate_skill,
)

_MESSAGE = "src.routers.message"
_USER_ID = "user-1"
_PREFIX = "zz-skill-"


def _description(name: str) -> str:
    return f"Do {name} for the user when they ask for it."


def _markdown(name: str, description: str, extra: str = "") -> bytes:
    front = f"---\nname: {name}\ndescription: {description}\n"
    if extra:
        front += extra
    front += "---\n\nSteps.\n"
    return front.encode("utf-8")


def _skill(name: str = "roll-dice", description: str | None = None, extra=(), **kwargs):
    description = _description(name) if description is None else description
    files = [SkillFile(path="SKILL.md", content=_markdown(name, description)), *extra]
    return Skill(
        user_id=None,
        name=name,
        description=description,
        enabled=True,
        files=files,
        **kwargs,
    )


def _only_markdown(name: str, description: str, extra: str = "") -> Skill:
    """A skill whose single file is the SKILL.md under test."""
    return Skill(
        user_id=None,
        name=name,
        description=description,
        enabled=True,
        files=[SkillFile(path="SKILL.md", content=_markdown(name, description, extra=extra))],
    )


@pytest.fixture
def built():
    roots: list[Path] = []

    def _build(*args, **kwargs):
        root = build_skills_root(*args, **kwargs)
        roots.append(root)
        return root

    yield _build
    for root in roots:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.no_db
def test_skills_alone_select_the_agentic_pipeline():
    request = GenerationRequest(query="q", skills=["roll-dice"])
    assert is_agentic_generation_request(request) is True


@pytest.mark.no_db
def test_seeded_dice_skill_is_valid():
    assert validate_skill(dice_skill()) is None


@pytest.mark.no_db
def test_roll_script_reports_the_total(tmp_path):
    script = tmp_path / "roll.py"
    script.write_bytes(ROLL_PY.encode("utf-8"))
    result = subprocess.run(
        [sys.executable, str(script), "3d6+2"],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["notation"] == "3d6+2"
    assert len(payload["rolls"]) == 3
    assert all(1 <= roll <= 6 for roll in payload["rolls"])
    assert payload["modifier"] == 2
    assert payload["total"] == sum(payload["rolls"]) + 2

    rejected = subprocess.run(
        [sys.executable, str(script), "101d6"],
        capture_output=True,
        text=True,
    )
    assert rejected.returncode == 2


@pytest.mark.no_db
def test_tree_keeps_bytes_including_non_utf8(built):
    blob = bytes([0xFF, 0xFE, 0x00, 0x01])
    skill = _skill(
        extra=(
            SkillFile(path="scripts/roll.py", content=b"print(1)\n"),
            SkillFile(path="assets/blob.bin", content=blob),
        )
    )
    root = built([skill], None)
    folder = root / "roll-dice"
    assert (folder / "scripts" / "roll.py").read_bytes() == b"print(1)\n"
    assert (folder / "assets" / "blob.bin").read_bytes() == blob
    assert (folder / "SKILL.md").read_bytes() == skill.files[0].content


@pytest.mark.no_db
def test_digest_is_reused(built):
    skill = _skill(name="cache-dice")
    first = built([skill], None)
    (first / "marker").write_text("keep")
    second = built([skill], None)
    assert second == first
    assert (second / "marker").read_text() == "keep"


@pytest.mark.no_db
def test_packaged_skill_of_the_same_name_wins(built, tmp_path, caplog):
    packaged = tmp_path / "packaged"
    skill_dir = packaged / "roll-dice"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_bytes(b"from packaged")
    caplog.set_level(logging.WARNING)
    root = built([_skill()], packaged)
    assert (root / "roll-dice" / "SKILL.md").read_bytes() == b"from packaged"
    assert "packaged skill wins" in caplog.text


@pytest.mark.no_db
@pytest.mark.parametrize(
    "skill",
    [
        _skill(name="Bad"),
        _skill(name="-bad"),
        _skill(name="bad--name"),
        _skill(name="a" * 65),
        _only_markdown("roll-dice", ""),
        _only_markdown("roll-dice", "x" * 1025),
        Skill(
            name="roll-dice",
            description=_description("roll-dice"),
            enabled=True,
            files=[
                SkillFile(
                    path="SKILL.md",
                    content=_markdown("other-dice", _description("roll-dice")),
                )
            ],
        ),
        Skill(
            name="roll-dice",
            description="different description for the row",
            enabled=True,
            files=[
                SkillFile(
                    path="SKILL.md",
                    content=_markdown("roll-dice", _description("roll-dice")),
                )
            ],
        ),
        Skill(
            name="roll-dice",
            description=_description("roll-dice"),
            enabled=True,
            files=[SkillFile(path="scripts/roll.py", content=b"x")],
        ),
        _skill(extra=(SkillFile(path="../escape.txt", content=b"x"),)),
        _skill(extra=(SkillFile(path="/tmp/escape.txt", content=b"x"),)),
        _only_markdown(
            "roll-dice",
            _description("roll-dice"),
            extra="compatibility: " + ("x" * 501) + "\n",
        ),
        _only_markdown(
            "roll-dice",
            _description("roll-dice"),
            extra="metadata:\n  author: 1\n",
        ),
    ],
)
def test_invalid_skill_is_skipped(built, skill, caplog):
    caplog.set_level(logging.WARNING)
    root = built([skill], None)
    assert not (root / skill.name).exists()
    assert "Skipping skill" in caplog.text


@pytest.mark.no_db
def test_empty_request_uses_the_packaged_folder():
    assert resolve_skills_root([]) == packaged_skills_dir()


@pytest.mark.no_db
def test_build_failure_falls_back_to_packaged(monkeypatch, caplog):
    def _boom(*_args, **_kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(
        "src.services.agents.core.skills.build_skills_root", _boom
    )
    caplog.set_level(logging.ERROR)
    assert resolve_skills_root([_skill()]) == packaged_skills_dir()
    assert "using packaged skills" in caplog.text


@pytest.mark.no_db
async def test_requested_names_resolve_and_missing_is_logged(caplog):
    visible = _skill()
    captured = {}

    async def _find_all(filter_dict=None, **_kwargs):
        captured["filter"] = filter_dict
        return [visible]

    caplog.set_level(logging.WARNING)
    with patch(f"{_MESSAGE}.Skill.find_all", AsyncMock(side_effect=_find_all)):
        request = GenerationRequest(query="q", skills=["roll-dice", "missing"])
        prepared = await _prepare_agentic_request(
            request, SimpleNamespace(id=_USER_ID)
        )

    assert prepared.skill_configs == [visible]
    skill_filter = captured["filter"]
    assert skill_filter["name"] == {"$in": ["roll-dice", "missing"]}
    assert skill_filter["enabled"] is True
    assert skill_filter["deleted_at"] is None
    assert {"user_id": _USER_ID} in skill_filter["$or"]
    assert {"user_id": None} in skill_filter["$or"]
    assert "missing" in caplog.text


@pytest.mark.no_db
async def test_empty_skills_list_loads_nothing():
    find_all = AsyncMock()
    with patch(f"{_MESSAGE}.Skill.find_all", find_all):
        request = GenerationRequest(query="q", skills=[])
        prepared = await _prepare_agentic_request(
            request, SimpleNamespace(id=_USER_ID)
        )
    find_all.assert_not_called()
    assert prepared.skill_configs == []
    assert resolve_skills_root(prepared.skill_configs) == packaged_skills_dir()


@pytest.mark.no_db
def test_no_sandbox_mounts_the_skills_root(tmp_path):
    pytest.importorskip("agents.graphs.deep.graph")
    from src.services.agents.core.runner import _build_react_graph

    captured = {}

    class _Agent:
        name = "deep"

        def compile(
            self,
            llm,
            tools,
            checkpointer,
            history=None,
            summary=None,
            fallback_llm=None,
            llm_run_timeout=None,
            llm_idle_timeout=None,
            on_policy=None,
            backend=None,
            permissions=None,
            skills=None,
        ):
            captured["backend"] = backend
            captured["permissions"] = permissions
            captured["skills"] = skills
            return "graph"

    _build_react_graph(
        None,
        [],
        None,
        agent=_Agent(),
        llm=object(),
        fallback_llm=object(),
        on_policy=lambda **_kwargs: None,
        skills_root=tmp_path,
    )
    assert captured["backend"] is not None
    assert captured["permissions"]
    assert captured["skills"] == ["/skills/"]


@pytest.mark.no_db
def test_compile_without_a_skills_parameter_stays_unchanged(tmp_path):
    from src.services.agents.core.runner import _build_react_graph

    captured = {}

    class _Agent:
        name = "deep"

        def compile(
            self,
            llm,
            tools,
            checkpointer,
            history=None,
            summary=None,
            fallback_llm=None,
            llm_run_timeout=None,
            llm_idle_timeout=None,
            on_policy=None,
            backend=None,
            permissions=None,
        ):
            captured["backend"] = backend
            return "graph"

    _build_react_graph(
        None,
        [],
        None,
        agent=_Agent(),
        llm=object(),
        fallback_llm=object(),
        on_policy=lambda **_kwargs: None,
        skills_root=tmp_path,
    )
    assert captured["backend"] is not None


@pytest.mark.no_db
async def test_open_sandbox_uploads_the_given_root(tmp_path, monkeypatch):
    folder = tmp_path / "roll-dice"
    (folder / "scripts").mkdir(parents=True)
    (folder / "SKILL.md").write_bytes(b"skill")
    (folder / "scripts" / "roll.py").write_bytes(b"print(1)\n")
    monkeypatch.setenv("SANDBOX_PROVIDER", "agentcore")
    monkeypatch.setenv("SANDBOX_AGENTCORE_REGION", "eu-west-1")
    monkeypatch.setenv("EVE_AGENT_SKILLS_DIR", str(tmp_path / "unused"))

    uploads: list = []
    module = types.ModuleType("langchain_agentcore_codeinterpreter")

    class _Sandbox:
        def upload_files(self, files):
            uploads.append(list(files))
            return [SimpleNamespace(error=None) for _ in files]

        def close(self):
            return None

        @staticmethod
        def create(**_kwargs):
            return _Sandbox()

    module.AgentCoreSandbox = _Sandbox
    monkeypatch.setitem(sys.modules, "langchain_agentcore_codeinterpreter", module)

    from src.services.agents.core.sandbox import close_sandbox, open_sandbox

    session = await open_sandbox(tmp_path)
    await close_sandbox(session)

    uploaded = dict(uploads[0])
    assert uploaded["/skills/roll-dice/SKILL.md"] == b"skill"
    assert uploaded["/skills/roll-dice/scripts/roll.py"] == b"print(1)\n"
    assert "/skills/unused" not in uploaded


class TestSkillDocuments:
    async def test_binary_file_roundtrip(self):
        await Skill.delete_many({"name": {"$regex": f"^{_PREFIX}"}})
        blob = bytes(range(256))
        created = await Skill.create(
            name=f"{_PREFIX}binary",
            description="binary file roundtrip",
            enabled=True,
            files=[SkillFile(path="assets/blob.bin", content=blob)],
        )
        try:
            loaded = await Skill.find_one({"_id": ObjectId(created.id)})
            assert loaded is not None
            assert loaded.files[0].content == blob
            assert isinstance(loaded.files[0].content, bytes)
        finally:
            await Skill.delete_many({"name": f"{_PREFIX}binary"})

    async def test_disabled_deleted_and_other_users_are_excluded(self, caplog):
        await Skill.delete_many({"name": {"$regex": f"^{_PREFIX}"}})
        owner = f"{_PREFIX}owner"
        stranger = f"{_PREFIX}stranger"
        names = {
            "global": f"{_PREFIX}global",
            "mine": f"{_PREFIX}mine",
            "disabled": f"{_PREFIX}disabled",
            "deleted": f"{_PREFIX}deleted",
            "other": f"{_PREFIX}other",
        }
        rows = [
            (names["global"], None, True, None),
            (names["mine"], owner, True, None),
            (names["disabled"], None, False, None),
            (names["deleted"], None, True, datetime.now(timezone.utc)),
            (names["other"], stranger, True, None),
        ]
        try:
            for name, user_id, enabled, deleted_at in rows:
                await Skill.create(
                    name=name,
                    description=name,
                    user_id=user_id,
                    enabled=enabled,
                    deleted_at=deleted_at,
                    files=[SkillFile(path="SKILL.md", content=b"x")],
                )
            caplog.set_level(logging.WARNING)
            request = GenerationRequest(
                query="q",
                skills=[*names.values(), f"{_PREFIX}missing"],
            )
            prepared = await _prepare_agentic_request(
                request, SimpleNamespace(id=owner)
            )
            assert {skill.name for skill in prepared.skill_configs} == {
                names["global"],
                names["mine"],
            }
            assert f"{_PREFIX}missing" in caplog.text
        finally:
            await Skill.delete_many({"name": {"$regex": f"^{_PREFIX}"}})
