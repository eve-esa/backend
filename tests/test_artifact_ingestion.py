"""Tests for the MCP artifact-ingestion interceptor.

Builds real ``mcp.types.CallToolResult`` fixtures (the raw server response,
before ``langchain_mcp_adapters`` converts it into a ``ToolMessage``) and a
fake handler, and exercises ``ArtifactInterceptor`` directly against them.
"""

import asyncio
import base64
import contextlib
import json
from types import SimpleNamespace

import pytest
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from mcp.types import (
    BlobResourceContents,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
    TextResourceContents,
)

from src.database.models.artifact import Artifact
from src.services.mcp.artifact_context import (
    reset_artifact_context,
    set_artifact_context,
)
from src.services.mcp.artifact_ingestion import ArtifactInterceptor
from tests.utils.cleaner import cleanup_models
from tests.utils.fake_storage import FakeStorage

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
PNG_B64 = base64.b64encode(PNG_BYTES).decode("ascii")

USER_ID = "u1"


def _request(name="search", server_name="wiley", args=None) -> MCPToolCallRequest:
    return MCPToolCallRequest(name=name, args=args or {}, server_name=server_name)


def _handler(result):
    async def handler(request):
        return result

    return handler


class _FakeSession:
    """Stand-in for an MCP ClientSession: only read_resource is exercised here."""

    def __init__(self, contents=None, raise_exc=None, hang=False):
        self._contents = contents or []
        self._raise = raise_exc
        self._hang = hang

    async def read_resource(self, uri):
        if self._raise is not None:
            raise self._raise
        if self._hang:
            await asyncio.sleep(3600)
        return SimpleNamespace(contents=self._contents)


class _FakeBoundClient:
    """Stand-in for MultiServerMCPClient: only .session() is exercised here."""

    def __init__(self, session_obj: _FakeSession):
        self._session_obj = session_obj

    @contextlib.asynccontextmanager
    async def session(self, server_name):
        yield self._session_obj


@pytest.fixture
def interceptor(monkeypatch):
    fake = FakeStorage()
    monkeypatch.setattr("src.services.mcp.artifact_ingestion.storage_service", fake)
    return ArtifactInterceptor(), fake


@pytest.fixture
def with_context():
    ctx, token = set_artifact_context(
        user_id=USER_ID, conversation_id="conv-1", message_id=None
    )
    try:
        yield ctx
    finally:
        reset_artifact_context(token)
        # Best-effort cleanup: don't leak artifacts across tests.


async def _cleanup():
    await Artifact.delete_many({"user_id": USER_ID})


def _parse_stub(block: TextContent) -> dict:
    """Split a stub TextContent into (markdown_line, parsed_json) and return the JSON."""
    lines = block.text.split("\n")
    assert len(lines) == 2, f"expected 2 lines in stub, got: {block.text!r}"
    return json.loads(lines[1])


# ---------------- (a) ImageContent -----------------


@pytest.mark.asyncio
async def test_image_content_persisted_and_stubbed(interceptor, with_context):
    """A small image is persisted as an Artifact and replaced with a markdown+JSON stub."""
    icept, fake = interceptor
    result = CallToolResult(
        content=[ImageContent(type="image", data=PNG_B64, mimeType="image/png")]
    )

    out = await icept(_request(), _handler(result))

    try:
        assert len(out.content) == 1
        block = out.content[0]
        assert isinstance(block, TextContent)
        assert "/artifacts/" in block.text
        payload = _parse_stub(block)
        assert payload["content_type"] == "image/png"
        assert payload["size_bytes"] == len(PNG_BYTES)
        # original base64 data must not appear anywhere in the result anymore.
        assert PNG_B64 not in json.dumps(out.model_dump())

        artifact = await Artifact.find_by_id(payload["artifact_id"])
        assert artifact is not None
        assert artifact.source.type == "mcp_tool"
        assert artifact.source.mcp_server == "wiley"
        assert artifact.source.tool_name == "search"
        assert artifact.conversation_id == "conv-1"
        assert artifact.key in fake.objects
        assert with_context.collected_artifact_ids == [artifact.id]
    finally:
        await _cleanup()


# ---------------- (b) EmbeddedResource blob -----------------


@pytest.mark.asyncio
async def test_embedded_resource_blob_persisted(interceptor, with_context):
    """An EmbeddedResource carrying a blob is persisted as an Artifact."""
    icept, fake = interceptor
    data = b"%PDF-1.4 fake pdf bytes"
    resource = EmbeddedResource(
        type="resource",
        resource=BlobResourceContents(
            uri="resource://tool/report.pdf",
            mimeType="application/pdf",
            blob=base64.b64encode(data).decode("ascii"),
        ),
    )
    result = CallToolResult(content=[resource])

    out = await icept(_request(name="make_report"), _handler(result))

    try:
        assert len(out.content) == 1
        payload = _parse_stub(out.content[0])
        assert payload["content_type"] == "application/pdf"
        assert payload["filename"] == "report.pdf"

        artifact = await Artifact.find_by_id(payload["artifact_id"])
        assert artifact is not None
        assert artifact.size_bytes == len(data)
        assert artifact.key in fake.objects
    finally:
        await _cleanup()


@pytest.mark.asyncio
async def test_embedded_resource_text_passthrough(interceptor, with_context):
    """Inline text resources stay inline in v1 (no artifact created)."""
    icept, _fake = interceptor
    resource = EmbeddedResource(
        type="resource",
        resource=TextResourceContents(
            uri="resource://tool/notes.txt",
            mimeType="text/plain",
            text="hello",
        ),
    )
    result = CallToolResult(content=[resource])

    out = await icept(_request(), _handler(result))

    assert out.content == [resource]
    assert with_context.collected_artifact_ids == []


# ---------------- (c) ResourceLink -----------------


@pytest.mark.asyncio
async def test_resource_link_resolved_and_persisted(interceptor, with_context):
    """A ResourceLink is fetched via the bound client and persisted."""
    icept, fake = interceptor
    data = b"csv,data\n1,2\n"
    session = _FakeSession(
        contents=[
            BlobResourceContents(
                uri="resource://tool/data.csv",
                mimeType="text/csv",
                blob=base64.b64encode(data).decode("ascii"),
            )
        ]
    )
    client = _FakeBoundClient(session)
    icept.bind_client(client)

    link = ResourceLink(
        type="resource_link", name="data.csv", uri="resource://tool/data.csv"
    )
    result = CallToolResult(content=[link])

    out = await icept(_request(name="export"), _handler(result))

    try:
        assert len(out.content) == 1
        payload = _parse_stub(out.content[0])
        assert payload["content_type"] == "text/csv"
        assert payload["filename"] == "data.csv"

        artifact = await Artifact.find_by_id(payload["artifact_id"])
        assert artifact is not None
        assert artifact.size_bytes == len(data)
        assert artifact.key in fake.objects
    finally:
        await _cleanup()


@pytest.mark.asyncio
async def test_resource_link_fetch_failure_leaves_block_untouched(
    interceptor, with_context
):
    """If reading the resource fails, the original ResourceLink block is untouched."""
    icept, _fake = interceptor
    session = _FakeSession(raise_exc=RuntimeError("boom"))
    client = _FakeBoundClient(session)
    icept.bind_client(client)

    link = ResourceLink(
        type="resource_link", name="data.csv", uri="resource://tool/data.csv"
    )
    result = CallToolResult(content=[link])

    out = await icept(_request(), _handler(result))

    assert out.content == [link]
    assert with_context.collected_artifact_ids == []


@pytest.mark.asyncio
async def test_resource_link_timeout_leaves_block_untouched(
    interceptor, with_context, monkeypatch
):
    """A resource read that exceeds ARTIFACT_RESOURCE_READ_TIMEOUT_S is left untouched."""
    icept, _fake = interceptor
    monkeypatch.setattr(
        "src.services.mcp.artifact_ingestion.ARTIFACT_RESOURCE_READ_TIMEOUT_S", 0.01
    )
    session = _FakeSession(hang=True)
    client = _FakeBoundClient(session)
    icept.bind_client(client)

    link = ResourceLink(
        type="resource_link", name="slow.bin", uri="resource://tool/slow.bin"
    )
    result = CallToolResult(content=[link])

    out = await icept(_request(), _handler(result))

    assert out.content == [link]
    assert with_context.collected_artifact_ids == []


@pytest.mark.asyncio
async def test_resource_link_oversized_item_leaves_block_untouched(
    interceptor, with_context, monkeypatch
):
    """An oversized resolved item aborts the whole link (no partial persistence)."""
    icept, fake = interceptor
    monkeypatch.setattr("src.services.mcp.artifact_ingestion.ARTIFACT_MAX_BYTES", 4)
    session = _FakeSession(
        contents=[
            BlobResourceContents(
                uri="resource://tool/big.bin",
                mimeType="application/octet-stream",
                blob=base64.b64encode(b"way too big").decode("ascii"),
            )
        ]
    )
    client = _FakeBoundClient(session)
    icept.bind_client(client)

    link = ResourceLink(
        type="resource_link", name="big.bin", uri="resource://tool/big.bin"
    )
    result = CallToolResult(content=[link])

    out = await icept(_request(), _handler(result))

    assert out.content == [link]
    assert with_context.collected_artifact_ids == []
    assert fake.objects == {}


@pytest.mark.asyncio
async def test_resource_link_no_bound_client_leaves_block_untouched(with_context):
    """Without bind_client(), a ResourceLink is left untouched (nothing to fetch with)."""
    # Deliberately not using the `interceptor` fixture: this one is never bound to a client.
    icept = ArtifactInterceptor()
    link = ResourceLink(
        type="resource_link", name="data.csv", uri="resource://tool/data.csv"
    )
    result = CallToolResult(content=[link])

    out = await icept(_request(), _handler(result))

    assert out.content == [link]


# ---------------- (d) oversize ImageContent -----------------


@pytest.mark.asyncio
async def test_oversize_image_skipped_with_warning(interceptor, with_context, monkeypatch):
    """An oversized image is replaced with a warning stub; nothing is persisted."""
    icept, fake = interceptor
    monkeypatch.setattr("src.services.mcp.artifact_ingestion.ARTIFACT_MAX_BYTES", 4)
    result = CallToolResult(
        content=[ImageContent(type="image", data=PNG_B64, mimeType="image/png")]
    )

    out = await icept(_request(), _handler(result))

    assert len(out.content) == 1
    assert isinstance(out.content[0], TextContent)
    assert out.content[0].text.startswith("artifact skipped:")
    assert "size limit" in out.content[0].text
    assert with_context.collected_artifact_ids == []
    assert fake.objects == {}


# ---------------- (e) ARTIFACT_MAX_PER_TOOL_CALL cap -----------------


@pytest.mark.asyncio
async def test_max_per_tool_call_cap_respected(interceptor, with_context, monkeypatch):
    """Blocks beyond ARTIFACT_MAX_PER_TOOL_CALL are replaced with a limit-reached stub."""
    icept, fake = interceptor
    monkeypatch.setattr(
        "src.services.mcp.artifact_ingestion.ARTIFACT_MAX_PER_TOOL_CALL", 1
    )
    result = CallToolResult(
        content=[
            ImageContent(type="image", data=PNG_B64, mimeType="image/png"),
            ImageContent(type="image", data=PNG_B64, mimeType="image/png"),
        ]
    )

    out = await icept(_request(), _handler(result))

    try:
        assert len(out.content) == 2
        first_payload = _parse_stub(out.content[0])
        assert "artifact_id" in first_payload
        assert out.content[1].text == "artifact skipped: limit reached"
        assert len(with_context.collected_artifact_ids) == 1
        assert len(fake.objects) == 1
    finally:
        await _cleanup()


# ---------------- (f) storage failure: fail-open -----------------


@pytest.mark.asyncio
async def test_storage_failure_fails_open(interceptor, with_context, monkeypatch):
    """If storage raises, the interceptor returns the original result unmodified."""
    icept, fake = interceptor

    async def _boom(key, body, content_type):
        raise RuntimeError("s3 unavailable")

    monkeypatch.setattr(fake, "put_object", _boom)

    original = CallToolResult(
        content=[ImageContent(type="image", data=PNG_B64, mimeType="image/png")]
    )

    out = await icept(_request(), _handler(original))

    assert out is original
    assert with_context.collected_artifact_ids == []


# ---------------- (g) contextvar unset -----------------


@pytest.mark.asyncio
async def test_no_context_passthrough(interceptor):
    """With no artifact context set, the result passes through untouched."""
    icept, fake = interceptor
    original = CallToolResult(
        content=[ImageContent(type="image", data=PNG_B64, mimeType="image/png")]
    )

    out = await icept(_request(), _handler(original))

    assert out is original
    assert fake.objects == {}


# ---------------- (h) TextContent-only result -----------------


@pytest.mark.asyncio
async def test_text_only_result_untouched(interceptor, with_context):
    """A result with only TextContent blocks passes through unchanged."""
    icept, fake = interceptor
    original = CallToolResult(content=[TextContent(type="text", text="plain answer")])

    out = await icept(_request(), _handler(original))

    assert out.content == original.content
    assert fake.objects == {}
    assert with_context.collected_artifact_ids == []


@pytest.mark.asyncio
async def test_non_calltoolresult_passthrough(interceptor, with_context):
    """Non-CallToolResult handler outputs (e.g. a ToolMessage) pass through untouched."""
    icept, _fake = interceptor
    sentinel = object()

    out = await icept(_request(), _handler(sentinel))

    assert out is sentinel


# ---------------- Wiring -----------------


@pytest.mark.asyncio
async def test_artifact_interceptor_is_last_in_tool_interceptors(monkeypatch):
    """The agentic runner appends ArtifactInterceptor last and binds it to the client."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from src.services.agents.core.runner import _load_mcp_tools_for_servers

    mock_client_instance = MagicMock()
    mock_client_instance.get_tools = AsyncMock(return_value=[])
    mock_client_cls = MagicMock(return_value=mock_client_instance)

    srv = SimpleNamespace(
        name="wiley",
        config=SimpleNamespace(
            transport=SimpleNamespace(value="streamable_http"),
            url="http://example.com/mcp",
            headers={},
        ),
    )

    with patch(
        "src.services.agents.core.runner.MultiServerMCPClient", mock_client_cls
    ), patch(
        "src.services.agents.core.runner.get_cognito_token_provider",
        return_value=None,
    ):
        await _load_mcp_tools_for_servers([srv])

    _, kwargs = mock_client_cls.call_args
    interceptors = kwargs["tool_interceptors"]
    assert isinstance(interceptors[-1], ArtifactInterceptor)
    # bind_client() was called with this exact client instance (weakref resolves to it).
    assert interceptors[-1]._client_ref() is mock_client_instance
