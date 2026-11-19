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
        # The markdown image line carries a title attributing the MCP source.
        markdown_line = block.text.split("\n", 1)[0]
        assert markdown_line.startswith("![")
        assert ' "MCP: wiley/search")' in markdown_line
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


class _Sandbox:
    def __init__(self):
        self.files: list[tuple[str, bytes]] = []

    def upload_files(self, files):
        self.files.extend(files)
        return [{} for _ in files]


@pytest.mark.asyncio
async def test_conversation_copy_is_uploaded_to_the_sandbox(interceptor, with_context):
    """The conversation folder gets the bytes, and the live sandbox does too."""
    icept, fake = interceptor
    sandbox = _Sandbox()
    with_context.sandbox_backend = sandbox
    result = CallToolResult(
        content=[ImageContent(type="image", data=PNG_B64, mimeType="image/png")]
    )

    out = await icept(_request(), _handler(result))

    try:
        payload = _parse_stub(out.content[0])
        relative = f"{payload['artifact_id']}/search-1.png"
        key = f"users/{USER_ID}/conversations/conv-1/artifacts/{relative}"
        assert payload["path"] == f"/artifacts/{relative}"
        assert list(fake.objects) == [key]
        assert fake.objects[key]["body"] == PNG_BYTES
        artifact = await Artifact.find_by_id(payload["artifact_id"])
        assert artifact is not None
        assert artifact.key == key
        assert sandbox.files == [(payload["path"], PNG_BYTES)]
    finally:
        await _cleanup()


@pytest.mark.asyncio
async def test_a_json_path_becomes_the_sandbox_path_and_the_stub_is_dropped(
    interceptor, with_context
):
    """The model sees the tool JSON with the sandbox path, not the markdown stub."""
    icept, _fake = interceptor
    data = b"report\nhello from files-test\n"
    resource = EmbeddedResource(
        type="resource",
        resource=BlobResourceContents(
            uri="file:///report.txt",
            mimeType="text/plain",
            blob=base64.b64encode(data).decode("ascii"),
        ),
    )
    result = CallToolResult(
        content=[
            TextContent(type="text", text='{"path": "report.txt"}'),
            resource,
        ],
        structuredContent={"path": "report.txt"},
    )

    out = await icept(_request(name="make_report", server_name="files_test"), _handler(result))

    try:
        assert len(out.content) == 1
        payload = json.loads(out.content[0].text)
        assert set(payload) == {"path"}
        assert payload["path"].startswith("/artifacts/")
        assert payload["path"].endswith("/report.txt")
        assert "artifact_id" not in out.content[0].text
        assert out.structuredContent == payload
    finally:
        await _cleanup()


@pytest.mark.asyncio
async def test_nested_file_paths_keep_their_fields_and_drop_the_stub(
    interceptor, with_context
):
    """A files list keeps the keys the tool sent. Only path changes, and the stub goes."""
    icept, _fake = interceptor
    csv = b"a,b\n1,2\n"
    body = {
        "ok": True,
        "files": [
            {
                "name": "severity_map.png",
                "path": "severity_map.png",
                "mime_type": "image/png",
                "bytes": len(PNG_BYTES),
            },
            {
                "name": "recovery.csv",
                "path": "recovery.csv",
                "mime_type": "text/csv",
                "bytes": len(csv),
            },
            {
                "name": "later.png",
                "path": "later.png",
                "mime_type": "image/png",
                "bytes": 99999,
            },
        ],
    }
    result = CallToolResult(
        content=[
            TextContent(type="text", text=json.dumps(body)),
            ImageContent(type="image", data=PNG_B64, mimeType="image/png"),
            EmbeddedResource(
                type="resource",
                resource=BlobResourceContents(
                    uri="file:///recovery.csv",
                    mimeType="text/csv",
                    blob=base64.b64encode(csv).decode("ascii"),
                ),
            ),
            TextContent(type="text", text="Some files were not attached."),
        ]
    )

    out = await icept(
        _request(name="compute_metrics", server_name="effis"), _handler(result)
    )

    try:
        assert [type(block) for block in out.content] == [TextContent, TextContent]
        payload = json.loads(out.content[0].text)
        image, table, pending = payload["files"]
        assert set(image) == {"name", "path", "mime_type", "bytes"}
        assert image["name"] == "severity_map.png"
        assert image["bytes"] == len(PNG_BYTES)
        assert image["path"].startswith("/artifacts/")
        assert image["path"].endswith("/severity_map.png")
        assert table["path"].startswith("/artifacts/")
        assert table["path"].endswith("/recovery.csv")
        assert pending["path"] == "later.png"
        assert "artifact_id" not in out.content[0].text
        assert out.content[1].text == "Some files were not attached."
    finally:
        await _cleanup()


@pytest.mark.asyncio
async def test_source_path_meta_replaces_the_loaded_path(interceptor, with_context):
    """_meta.eve.source_path replaces that exact string, including a non-path key."""
    icept, _fake = interceptor
    geojson = b'{"type":"FeatureCollection","features":[]}'
    image_path = "/tmp/effis/severity_map.png"
    geojson_path = "/tmp/effis/burnt_areas.geojson"
    body = {
        "ok": True,
        "geojson_path": geojson_path,
        "files": [
            {
                "name": "severity_map.png",
                "path": image_path,
                "mime_type": "image/png",
                "bytes": len(PNG_BYTES),
            }
        ],
    }
    result = CallToolResult(
        content=[
            TextContent(type="text", text=json.dumps(body)),
            ImageContent(
                type="image",
                data=PNG_B64,
                mimeType="image/png",
                _meta={"eve": {"source_path": image_path}},
            ),
            EmbeddedResource(
                type="resource",
                resource=BlobResourceContents(
                    uri="file:///burnt_areas.geojson",
                    mimeType="application/geo+json",
                    blob=base64.b64encode(geojson).decode("ascii"),
                ),
                _meta={"eve": {"source_path": geojson_path}},
            ),
        ],
        structuredContent=body,
    )

    out = await icept(
        _request(name="get_effis_burnt_areas", server_name="effis"), _handler(result)
    )

    try:
        assert [type(block) for block in out.content] == [TextContent]
        payload = json.loads(out.content[0].text)
        assert set(payload) == {"ok", "geojson_path", "files"}
        saved_image = payload["files"][0]["path"]
        saved_geojson = payload["geojson_path"]
        assert saved_image.startswith("/artifacts/")
        assert saved_image.endswith("/severity_map.png")
        assert saved_geojson.startswith("/artifacts/")
        assert saved_geojson.endswith("/burnt_areas.geojson")
        assert image_path not in out.content[0].text
        assert geojson_path not in out.content[0].text
        assert out.structuredContent["geojson_path"] == saved_geojson
        assert out.structuredContent["files"][0]["path"] == saved_image
    finally:
        await _cleanup()


@pytest.mark.asyncio
async def test_ambiguous_file_sizes_keep_the_stub(interceptor, with_context):
    """Two files of the same size are not guessed. Their stubs stay."""
    icept, _fake = interceptor
    body = {
        "files": [
            {"name": "a.png", "path": "a.png", "bytes": len(PNG_BYTES)},
            {"name": "b.png", "path": "b.png", "bytes": len(PNG_BYTES)},
        ]
    }
    image = ImageContent(type="image", data=PNG_B64, mimeType="image/png")
    result = CallToolResult(
        content=[
            TextContent(type="text", text=json.dumps(body)),
            image,
            image,
        ]
    )

    out = await icept(_request(), _handler(result))

    try:
        assert len(out.content) == 3
        payload = json.loads(out.content[0].text)
        assert payload["files"][0]["path"] == "a.png"
        assert payload["files"][1]["path"] == "b.png"
        assert _parse_stub(out.content[1])["size_bytes"] == len(PNG_BYTES)
    finally:
        await _cleanup()


@pytest.mark.asyncio
async def test_no_conversation_stores_nothing(interceptor):
    """A turn with no conversation id stores no object and returns a warning."""
    icept, fake = interceptor
    ctx, token = set_artifact_context(user_id=USER_ID)
    result = CallToolResult(
        content=[ImageContent(type="image", data=PNG_B64, mimeType="image/png")]
    )
    try:
        out = await icept(_request(), _handler(result))
        assert out.content[0].text == "artifact skipped: not stored: no conversation"
        assert PNG_B64 not in json.dumps(out.model_dump())
        assert fake.objects == {}
        assert ctx.collected_artifact_ids == []
    finally:
        reset_artifact_context(token)


@pytest.mark.asyncio
async def test_stub_title_strips_quotes_and_falls_back_to_mcp(interceptor, with_context):
    """The provenance title strips embedded quotes and degrades to a bare 'MCP'."""
    icept, _fake = interceptor

    # A server/tool name containing a double quote must not break the markdown
    # title syntax: the quote is stripped rather than escaped.
    result = CallToolResult(
        content=[ImageContent(type="image", data=PNG_B64, mimeType="image/png")]
    )
    out = await icept(
        _request(name='shady"tool', server_name='ser"ver'), _handler(result)
    )
    try:
        markdown_line = out.content[0].text.split("\n", 1)[0]
        assert ' "MCP: server/shadytool")' in markdown_line
    finally:
        await _cleanup()

    # When the request carries no resolvable name, the title degrades to a
    # bare "MCP" rather than embedding the internal "unknown"/"tool" defaults.
    result2 = CallToolResult(
        content=[ImageContent(type="image", data=PNG_B64, mimeType="image/png")]
    )
    bare_request = SimpleNamespace(args={})
    out2 = await icept(bare_request, _handler(result2))
    try:
        markdown_line2 = out2.content[0].text.split("\n", 1)[0]
        assert ' "MCP")' in markdown_line2
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


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_limit_warning_stubs_do_not_consume_the_cap(
    interceptor, with_context, monkeypatch
):
    """A ResourceLink warning stub is not a stored artifact.

    The cap counts artifacts written, not stub blocks. A link that returns
    one limit warning and a persisted count still under the cap must leave
    room for the next block.
    """
    icept, _fake = interceptor
    monkeypatch.setattr(
        "src.services.mcp.artifact_ingestion.ARTIFACT_MAX_PER_TOOL_CALL", 1
    )

    async def _resolve(
        self, block, ctx, tool_name, server_name, counter, persisted_count
    ):
        return (
            [TextContent(type="text", text="artifact skipped: limit reached")],
            persisted_count,
        )

    async def _persist(
        self, data, content_type, filename, ctx, tool_name, server_name
    ):
        artifact = SimpleNamespace(id="art-1", filename=filename)
        ctx.collected_artifact_ids.append(artifact.id)
        return artifact, None

    monkeypatch.setattr(ArtifactInterceptor, "_resolve_resource_link", _resolve)
    monkeypatch.setattr(ArtifactInterceptor, "_persist", _persist)

    link = ResourceLink(
        type="resource_link", name="data.bin", uri="resource://tool/data.bin"
    )
    result = CallToolResult(
        content=[
            link,
            ImageContent(type="image", data=PNG_B64, mimeType="image/png"),
        ]
    )

    out = await icept(_request(), _handler(result))

    assert out.content[0].text == "artifact skipped: limit reached"
    assert _parse_stub(out.content[1])["artifact_id"] == "art-1"
    assert with_context.collected_artifact_ids == ["art-1"]


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_resource_link_count_excludes_limit_warnings(
    interceptor, with_context, monkeypatch
):
    """A capped link stores two artifacts and reports that count, not three stubs."""
    icept, _fake = interceptor
    monkeypatch.setattr(
        "src.services.mcp.artifact_ingestion.ARTIFACT_MAX_PER_TOOL_CALL", 2
    )

    async def _persist(
        self, data, content_type, filename, ctx, tool_name, server_name
    ):
        artifact = SimpleNamespace(
            id=f"art-{len(ctx.collected_artifact_ids) + 1}", filename=filename
        )
        ctx.collected_artifact_ids.append(artifact.id)
        return artifact, None

    monkeypatch.setattr(ArtifactInterceptor, "_persist", _persist)
    blob = base64.b64encode(b"abc").decode("ascii")
    session = _FakeSession(
        contents=[
            BlobResourceContents(
                uri=f"resource://tool/{name}",
                mimeType="application/octet-stream",
                blob=blob,
            )
            for name in ("a.bin", "b.bin", "c.bin")
        ]
    )
    client = _FakeBoundClient(session)
    icept.bind_client(client)
    link = ResourceLink(
        type="resource_link", name="bundle", uri="resource://tool/bundle"
    )

    stubs, count = await icept._resolve_resource_link(
        link, with_context, "search", "wiley", 1, 0
    )

    assert count == 2
    assert len(stubs) == 3
    assert stubs[2].text == "artifact skipped: limit reached"
    assert with_context.collected_artifact_ids == ["art-1", "art-2"]


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
    """The MCP tool loader appends ArtifactInterceptor last and binds it to the client."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from src.services.mcp.tool_loader import load_mcp_tools_for_servers

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
        "src.services.mcp.tool_loader.MultiServerMCPClient", mock_client_cls
    ), patch(
        "src.services.mcp.tool_loader.get_cognito_token_provider",
        return_value=None,
    ):
        await load_mcp_tools_for_servers([srv])

    _, kwargs = mock_client_cls.call_args
    interceptors = kwargs["tool_interceptors"]
    assert isinstance(interceptors[-1], ArtifactInterceptor)
    # bind_client() was called with this exact client instance (weakref resolves to it).
    assert interceptors[-1]._client_ref() is mock_client_instance


# ---------------- types come from the bytes, never from the tool -----------------

SVG_BYTES = (
    b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'
)
HTML_BYTES = b"<!doctype html><html><script>alert(1)</script></html>"


@pytest.mark.asyncio
async def test_svg_declared_as_png_is_not_stored_as_an_inline_image(
    interceptor, with_context
):
    """A tool that labels SVG bytes image/png gets an opaque download, not an image."""
    icept, fake = interceptor
    result = CallToolResult(
        content=[
            ImageContent(
                type="image",
                data=base64.b64encode(SVG_BYTES).decode("ascii"),
                mimeType="image/png",
            )
        ]
    )

    out = await icept(_request(name="chart"), _handler(result))

    try:
        stub = out.content[0]
        payload = _parse_stub(stub)
        assert payload["content_type"] == "application/octet-stream"
        assert not stub.text.startswith("!")
        artifact = await Artifact.find_by_id(payload["artifact_id"])
        assert artifact.content_type == "application/octet-stream"
        assert fake.objects[artifact.key]["content_type"] == "application/octet-stream"
        assert not artifact.key.endswith((".svg", ".png"))
    finally:
        await _cleanup()


@pytest.mark.asyncio
async def test_png_declared_with_a_wrong_type_is_stored_as_png(
    interceptor, with_context
):
    icept, _fake = interceptor
    result = CallToolResult(
        content=[ImageContent(type="image", data=PNG_B64, mimeType="image/svg+xml")]
    )

    out = await icept(_request(name="chart"), _handler(result))

    try:
        stub = out.content[0]
        payload = _parse_stub(stub)
        assert payload["content_type"] == "image/png"
        assert stub.text.startswith("![chart-1.png]")
        artifact = await Artifact.find_by_id(payload["artifact_id"])
        assert artifact.content_type == "image/png"
        assert artifact.key.endswith(".png")
    finally:
        await _cleanup()


@pytest.mark.parametrize(
    "declared, uri",
    [
        ("image/svg+xml", "resource://tool/chart.svg"),
        ("text/html", "resource://tool/page.html"),
        ("image/png", "resource://tool/chart.png"),
    ],
)
@pytest.mark.asyncio
async def test_embedded_markup_is_never_an_inline_type(
    interceptor, with_context, declared, uri
):
    icept, _fake = interceptor
    body = SVG_BYTES if "svg" in uri else HTML_BYTES
    resource = EmbeddedResource(
        type="resource",
        resource=BlobResourceContents(
            uri=uri, mimeType=declared, blob=base64.b64encode(body).decode("ascii")
        ),
    )

    out = await icept(_request(name="render"), _handler(CallToolResult(content=[resource])))

    try:
        stub = out.content[0]
        payload = _parse_stub(stub)
        assert payload["content_type"] == "application/octet-stream"
        assert not stub.text.startswith("!")
    finally:
        await _cleanup()


@pytest.mark.asyncio
async def test_resource_link_html_declared_as_image_is_octet_stream(
    interceptor, with_context
):
    icept, _fake = interceptor
    session = _FakeSession(
        contents=[
            TextResourceContents(
                uri="resource://tool/page.html", mimeType="image/png", text=HTML_BYTES.decode()
            )
        ]
    )
    # Keep a strong reference: the interceptor holds the client by weakref.
    client = _FakeBoundClient(session)
    icept.bind_client(client)
    link = ResourceLink(type="resource_link", name="page", uri="resource://tool/page.html")

    out = await icept(_request(name="export"), _handler(CallToolResult(content=[link])))

    try:
        stub = out.content[0]
        payload = _parse_stub(stub)
        assert payload["content_type"] == "application/octet-stream"
        assert not stub.text.startswith("!")
    finally:
        await _cleanup()


@pytest.mark.parametrize(
    "uri_name",
    [
        "x%5D(javascript:alert(1))",
        "x%5D(https://phish.example)",
        "a%0A%0A%5Bclick%5D(https://phish.example)",
    ],
)
@pytest.mark.asyncio
async def test_filename_is_escaped_in_the_markdown_stub(
    interceptor, with_context, uri_name, monkeypatch
):
    icept, _fake = interceptor
    from urllib.parse import unquote

    raw_name = unquote(uri_name)
    resource = EmbeddedResource(
        type="resource",
        resource=BlobResourceContents(
            uri="resource://tool/report.pdf",
            mimeType="application/pdf",
            blob=base64.b64encode(b"%PDF-1.4 x").decode("ascii"),
        ),
    )
    monkeypatch.setattr(
        "src.services.mcp.artifact_ingestion._filename_from_uri", lambda uri: raw_name
    )
    out = await icept(_request(name="make"), _handler(CallToolResult(content=[resource])))

    try:
        stub = out.content[0]
        link_line, meta_line = stub.text.split("\n")
        assert "](javascript:" not in link_line
        assert "](https://phish" not in link_line
        # One link only: the text part closes exactly once, before the URL.
        assert link_line.count("](/artifacts/") == 1
        assert link_line.count("](") == 1
        assert "\\]" in link_line and "\\(" in link_line
        payload = json.loads(meta_line)
        assert "\n" not in payload["filename"]
    finally:
        await _cleanup()


@pytest.mark.asyncio
async def test_a_tool_image_linked_by_its_bare_filename_renders(
    interceptor, with_context
):
    """dummy_image on dev, 2026-11-18: the answer linked the bare filename.

    The image is stored at the key its row serves, the turn remembers its
    filename, and the answer link becomes ``/artifacts/{id}``.
    """
    from src.services.agents.core.runner import rewrite_answer_file_links

    icept, fake = interceptor
    result = CallToolResult(
        content=[ImageContent(type="image", data=PNG_B64, mimeType="image/png")]
    )

    out = await icept(
        _request(name="get_sample_image", server_name="dummy_image"), _handler(result)
    )

    try:
        payload = _parse_stub(out.content[0])
        assert payload["filename"] == "get_sample_image-1.png"
        artifact = await Artifact.find_by_id(payload["artifact_id"])
        assert fake.objects[artifact.key]["body"] == PNG_BYTES
        assert with_context.collected_artifact_files == {
            "get_sample_image-1.png": artifact.id
        }

        answer = await rewrite_answer_file_links(
            "![get_sample_image-1.png](get_sample_image-1.png)"
        )

        assert answer == f"![get_sample_image-1.png](/artifacts/{artifact.id})"
    finally:
        await _cleanup()


@pytest.mark.asyncio
async def test_two_calls_of_one_tool_in_a_turn_keep_both_images(
    interceptor, with_context
):
    """The default name restarts at 1 on every call. Inside one turn the
    second file gets its own name, so a bare link to each renders its image."""
    from src.services.agents.core.runner import rewrite_answer_file_links

    icept, _fake = interceptor
    request = _request(name="get_sample_image", server_name="dummy_image")
    first_png = PNG_BYTES + b"first"
    second_png = PNG_BYTES + b"second"

    def _image(data):
        return CallToolResult(
            content=[
                ImageContent(
                    type="image",
                    data=base64.b64encode(data).decode("ascii"),
                    mimeType="image/png",
                )
            ]
        )

    first = await icept(request, _handler(_image(first_png)))
    second = await icept(request, _handler(_image(second_png)))

    try:
        first_stub = _parse_stub(first.content[0])
        second_stub = _parse_stub(second.content[0])
        assert first_stub["filename"] == "get_sample_image-1.png"
        assert second_stub["filename"] == "get_sample_image-1-2.png"
        second_artifact = await Artifact.find_by_id(second_stub["artifact_id"])
        assert second_artifact.filename == "get_sample_image-1-2.png"
        assert with_context.collected_artifact_files == {
            "get_sample_image-1.png": first_stub["artifact_id"],
            "get_sample_image-1-2.png": second_stub["artifact_id"],
        }

        answer = await rewrite_answer_file_links(
            "![a](get_sample_image-1.png)\n![b](get_sample_image-1-2.png)"
        )

        assert answer == (
            f"![a](/artifacts/{first_stub['artifact_id']})\n"
            f"![b](/artifacts/{second_stub['artifact_id']})"
        )
    finally:
        await _cleanup()


# ---------------- what the answer append reads -----------------


@pytest.mark.asyncio
async def test_a_shown_stub_records_its_type_and_that_it_was_shown(
    interceptor, with_context
):
    """The runner appends a left-out tool image from these two records."""
    from src.services.agents.core.runner import append_unreferenced_turn_images

    icept, _fake = interceptor
    result = CallToolResult(
        content=[
            ImageContent(type="image", data=PNG_B64, mimeType="image/png"),
            TextContent(type="text", text='{"width": 512, "color": "blue"}'),
        ]
    )

    out = await icept(
        _request(name="get_sample_image", server_name="dummy_image"), _handler(result)
    )

    try:
        artifact_id = _parse_stub(out.content[0])["artifact_id"]
        assert with_context.collected_artifact_types == {artifact_id: "image/png"}
        assert with_context.shown_artifact_ids == [artifact_id]
        assert append_unreferenced_turn_images("Here it is:") == (
            f"Here it is:\n\n![get_sample_image-1.png](/artifacts/{artifact_id})"
        )
    finally:
        await _cleanup()


@pytest.mark.asyncio
async def test_a_stub_folded_into_the_tool_json_is_not_shown(
    interceptor, with_context
):
    """A file the tool names by path is data for the next step, not a picture."""
    from src.services.agents.core.runner import append_unreferenced_turn_images

    icept, _fake = interceptor
    body = {"files": [{"name": "tile.png", "path": "tile.png", "bytes": len(PNG_BYTES)}]}
    result = CallToolResult(
        content=[
            TextContent(type="text", text=json.dumps(body)),
            ImageContent(type="image", data=PNG_B64, mimeType="image/png"),
        ]
    )

    out = await icept(_request(name="fetch_tiles", server_name="eo"), _handler(result))

    try:
        assert len(out.content) == 1
        (artifact_id,) = with_context.collected_artifact_ids
        assert with_context.collected_artifact_types == {artifact_id: "image/png"}
        assert with_context.shown_artifact_ids == []
        assert append_unreferenced_turn_images("Done.") == "Done."
    finally:
        await _cleanup()
