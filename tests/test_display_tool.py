"""The sandbox ``display`` tool and where the runner attaches it."""

import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage

from src.database.models.artifact import Artifact
from src.schemas.generation_request import GenerationRequest
from src.services.agents.core.display import build_display_tool
from src.services.agents.core.runner import (
    generate_answer_agentic,
    generate_answer_agentic_stream_helper,
)
from src.services.mcp.artifact_context import (
    reset_artifact_context,
    set_artifact_context,
)
from tests.utils.fake_storage import FakeStorage

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
CSV_BYTES = b"a,b\n1,2\n"
USER_ID = "display-tool-user"
_RUNNER = "src.services.agents.core.runner"


class _Backend:
    """``adownload_files`` stand-in. One response for every requested path."""

    def __init__(self, content=None, error=None, exc=None):
        self.content = content
        self.error = error
        self.exc = exc
        self.calls = []

    async def adownload_files(self, paths):
        self.calls.append(list(paths))
        if self.exc is not None:
            raise self.exc
        from deepagents.backends.protocol import FileDownloadResponse

        return [
            FileDownloadResponse(path=path, content=self.content, error=self.error)
            for path in paths
        ]


def _parse_stub(text: str) -> dict:
    lines = text.split("\n")
    assert len(lines) == 2, text
    return json.loads(lines[1])


@pytest.fixture
def stored(monkeypatch):
    fake = FakeStorage()
    monkeypatch.setattr("src.services.mcp.artifact_ingestion.storage_service", fake)
    return fake


@pytest.fixture
def with_context():
    ctx, token = set_artifact_context(
        user_id=USER_ID, conversation_id="conv-1", message_id=None
    )
    try:
        yield ctx
    finally:
        reset_artifact_context(token)


async def _cleanup():
    await Artifact.delete_many({"user_id": USER_ID})


@pytest.mark.asyncio
async def test_png_is_stored_as_an_image_stub(stored, with_context):
    backend = _Backend(content=PNG_BYTES)
    tool = build_display_tool(backend)

    try:
        text = await tool.ainvoke({"path": "plot.png"})
        payload = _parse_stub(text)
        assert text.split("\n", 1)[0].startswith("![plot.png]")
        assert ' "MCP: sandbox/display")' in text
        assert payload["content_type"] == "image/png"
        assert payload["filename"] == "plot.png"
        assert payload["artifact_id"] in with_context.collected_artifact_ids
        assert base64.b64encode(PNG_BYTES).decode("ascii") not in text
        assert backend.calls == [["plot.png"]]
        artifact = await Artifact.find_by_id(payload["artifact_id"])
        assert artifact.source.type == "mcp_tool"
        assert artifact.source.mcp_server == "sandbox"
        assert artifact.source.tool_name == "display"
        assert artifact.key in stored.objects
    finally:
        await _cleanup()


@pytest.mark.asyncio
async def test_csv_is_stored_as_a_link_stub(stored, with_context):
    tool = build_display_tool(_Backend(content=CSV_BYTES))

    try:
        text = await tool.ainvoke({"path": "report.csv"})
        assert text.split("\n", 1)[0].startswith("[report.csv]")
        assert not text.startswith("!")
        payload = _parse_stub(text)
        assert payload["content_type"] == "text/csv"
        assert payload["artifact_id"] in with_context.collected_artifact_ids
    finally:
        await _cleanup()


@pytest.mark.asyncio
async def test_missing_file_is_an_error_and_stores_nothing(stored, with_context):
    tool = build_display_tool(_Backend(error="file_not_found"))

    text = await tool.ainvoke({"path": "missing.png"})

    assert "file_not_found" in text
    assert "relative path" in text
    assert with_context.collected_artifact_ids == []
    assert stored.objects == {}


@pytest.mark.asyncio
async def test_oversized_file_is_skipped(stored, with_context, monkeypatch):
    monkeypatch.setattr("src.services.mcp.artifact_ingestion.ARTIFACT_MAX_BYTES", 4)
    tool = build_display_tool(_Backend(content=b"0123456789"))

    text = await tool.ainvoke({"path": "big.bin"})

    assert "exceeds size limit" in text
    assert with_context.collected_artifact_ids == []
    assert stored.objects == {}


@pytest.mark.asyncio
async def test_without_artifact_context_the_bytes_are_not_returned(stored):
    tool = build_display_tool(_Backend(content=PNG_BYTES))

    text = await tool.ainvoke({"path": "plot.png"})

    assert "not stored" in text
    assert "/artifacts/" not in text
    assert base64.b64encode(PNG_BYTES).decode("ascii") not in text
    assert stored.objects == {}


@pytest.mark.asyncio
async def test_read_failure_is_an_error():
    tool = build_display_tool(_Backend(exc=RuntimeError("gone")))

    text = await tool.ainvoke({"path": "plot.png"})

    assert "read failed" in text
    assert "relative path" in text
    assert "gone" not in text


@pytest.mark.asyncio
async def test_a_space_in_the_name_still_stores(stored, with_context):
    tool = build_display_tool(_Backend(content=PNG_BYTES))

    try:
        text = await tool.ainvoke({"path": "my plot.png"})
        assert "/artifacts/" in text
        assert ' "MCP: sandbox/display")' in text
        assert with_context.collected_artifact_ids
    finally:
        await _cleanup()


class _Graph:
    def __init__(self, *, stream):
        self._stream = stream

    async def astream(self, *_args, **_kwargs):
        if self._stream:
            yield (
                "messages",
                (AIMessage(content="ok"), {"langgraph_node": "agent"}),
            )
            return
        yield {"agent": {"messages": [AIMessage(content="ok")]}}

    async def aget_state(self, _config):
        return SimpleNamespace(values={})


class _Agent:
    """``compile`` declares ``backend``, so the runner opens a sandbox."""

    def __init__(self, graph):
        self._graph = graph
        self.prompts = {}
        self.name = "deep"
        self.tools = None
        self.backend = None

    def instruction_text(self, **_kwargs):
        return ""

    def compile(self, backend=None, skills=None, **kwargs):
        self.tools = list(kwargs["tools"])
        self.backend = backend
        return self._graph


def _tool_names(agent: _Agent) -> list[str]:
    return [tool.name for tool in agent.tools]


async def _run(monkeypatch, *, stream: bool, sandbox: bool):
    backend = _Backend(error="file_not_found")
    agent = _Agent(_Graph(stream=stream))
    lease = SimpleNamespace(backend=backend) if sandbox else None

    async def _prepare(*_args, **_kwargs):
        return [], None, [], True, agent

    async def _llm(*_args, **_kwargs):
        return object(), {}

    async def _acquire(*_args, **_kwargs):
        return lease

    monkeypatch.setattr(f"{_RUNNER}._prepare_agent_context", _prepare)
    monkeypatch.setattr(f"{_RUNNER}._resolve_agentic_llm_client", _llm)
    monkeypatch.setattr(f"{_RUNNER}._get_agentic_checkpointer", AsyncMock(return_value=None))
    monkeypatch.setattr(f"{_RUNNER}.acquire_sandbox", _acquire)
    monkeypatch.setattr(f"{_RUNNER}.release_sandbox", AsyncMock())
    monkeypatch.setattr(f"{_RUNNER}.close_sandbox", AsyncMock())
    monkeypatch.setattr(f"{_RUNNER}.persist_message_state", AsyncMock())
    monkeypatch.setattr(f"{_RUNNER}.persist_policy_event", AsyncMock())
    monkeypatch.setattr(f"{_RUNNER}.save_agent_summary", AsyncMock())

    request = GenerationRequest(query="hi", llm_type="main", agent="deep")
    if stream:
        async for _event in generate_answer_agentic_stream_helper(
            request,
            conversation_id="c1",
            message_id="m1",
            user_id="test-user",
        ):
            pass
    else:
        await generate_answer_agentic(
            request, user_id="test-user", conversation_id="c1"
        )
    return agent, backend


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_nonstream_turn_includes_display_only_with_a_sandbox(monkeypatch):
    agent, backend = await _run(monkeypatch, stream=False, sandbox=True)

    assert "display" in _tool_names(agent)
    assert agent.backend.default is backend
    display = next(tool for tool in agent.tools if tool.name == "display")
    text = await display.ainvoke({"path": "plot.png"})
    assert backend.calls == [["plot.png"]]
    assert "file_not_found" in text

    agent, _backend = await _run(monkeypatch, stream=False, sandbox=False)
    assert "display" not in _tool_names(agent)


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_stream_turn_includes_display_only_with_a_sandbox(monkeypatch):
    agent, backend = await _run(monkeypatch, stream=True, sandbox=True)

    assert "display" in _tool_names(agent)
    assert agent.backend.default is backend
    display = next(tool for tool in agent.tools if tool.name == "display")
    await display.ainvoke({"path": "plot.png"})
    assert backend.calls == [["plot.png"]]

    agent, _backend = await _run(monkeypatch, stream=True, sandbox=False)
    assert "display" not in _tool_names(agent)
