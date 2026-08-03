"""MCP tool-call interceptor that captures files emitted by MCP servers as Artifacts.

Wraps ``execute_tool`` and sees the raw ``mcp.types.CallToolResult`` returned by
the MCP server *before* ``langchain_mcp_adapters`` converts it into a
``ToolMessage`` for the LLM. Any content block rewritten here therefore never
reaches the LLM, the LangGraph checkpointer, execution traces, or the SSE
stream — only the persisted stub (a markdown link plus a one-line JSON blob)
does.

Fail-open by design: ingestion runs in a single try/except around the whole
result, and any unexpected error returns the original, untouched result rather
than breaking the tool call.

Reference: https://github.com/langchain-ai/langchain-mcp-adapters (interceptors module)
"""

import asyncio
import base64
import json
import logging
import os
import weakref
from typing import Any, List, Optional, Tuple

from src.config import (
    ARTIFACT_MAX_BYTES,
    ARTIFACT_MAX_PER_TOOL_CALL,
    ARTIFACT_RESOURCE_READ_TIMEOUT_S,
)
from src.database.models.artifact import Artifact, ArtifactSource
from src.services.mcp.artifact_context import (
    ArtifactRequestContext,
    get_artifact_context,
)
from src.services.storage import guess_extension_from_content_type, storage_service

logger = logging.getLogger(__name__)

try:
    from mcp.types import (
        BlobResourceContents,
        CallToolResult,
        EmbeddedResource,
        ImageContent,
        ResourceLink,
        TextContent,
        TextResourceContents,
    )

    _mcp_types_available = True
except Exception:
    _mcp_types_available = False


def _filename_from_uri(uri: Any) -> Optional[str]:
    """Best-effort basename extraction from a resource URI, for a display filename."""
    try:
        raw = str(uri)
        raw = raw.split("?", 1)[0].split("#", 1)[0]
        base = os.path.basename(raw.rstrip("/"))
        return base or None
    except Exception:
        return None


def _provenance_title(server_name: Optional[str], tool_name: Optional[str]) -> str:
    """Build a plain, quote-safe markdown title attributing the MCP source.

    Double quotes are stripped from both names since markdown's title syntax
    delimits with them (`"title"`); an unescaped quote in a server/tool name
    would otherwise break the link/image syntax. Falls back to a bare "MCP"
    when either name is missing/unresolved rather than embedding one of this
    module's internal placeholder defaults ("unknown", "tool") as if it were
    real data.
    """

    def _clean(value: Optional[str]) -> Optional[str]:
        if not value or value in ("unknown", "tool"):
            return None
        return value.replace('"', "")

    server = _clean(server_name)
    tool = _clean(tool_name)
    if server and tool:
        return f"MCP: {server}/{tool}"
    return "MCP"


def _stub_text(
    *,
    is_image: bool,
    filename: str,
    artifact_id: str,
    content_type: str,
    size_bytes: int,
    server_name: Optional[str] = None,
    tool_name: Optional[str] = None,
) -> "TextContent":
    """Build the one-TextContent-per-artifact stub replacing a persisted block.

    Line 1 is a markdown reference (image embed for images, plain link
    otherwise) carrying a title that attributes the originating MCP
    server/tool; line 2 is a one-line JSON payload with the same data so a
    consumer that doesn't render markdown can still parse it out.
    """
    url = f"/artifacts/{artifact_id}"
    title = _provenance_title(server_name, tool_name)
    link_line = (
        f'![{filename}]({url} "{title}")' if is_image else f'[{filename}]({url} "{title}")'
    )
    meta_line = json.dumps(
        {
            "artifact_id": artifact_id,
            "url": url,
            "content_type": content_type,
            "filename": filename,
            "size_bytes": size_bytes,
        }
    )
    return TextContent(type="text", text=f"{link_line}\n{meta_line}")


def _warning_text(message: str) -> "TextContent":
    return TextContent(type="text", text=f"artifact skipped: {message}")


class ArtifactInterceptor:
    """Persists images/resources from MCP tool output as Artifacts.

    Follows the ``ToolCallInterceptor`` protocol from ``langchain-mcp-adapters``.
    A single instance may be reused across requests (the client it's attached
    to can be cached per user), so it never reads per-request state from
    ``self`` — only from the ``artifact_context`` contextvar, set by the caller
    around each agentic run.
    """

    def __init__(self) -> None:
        self._client_ref: Optional["weakref.ReferenceType[Any]"] = None

    def bind_client(self, client: Any) -> None:
        """Bind the ``MultiServerMCPClient`` used to resolve ``ResourceLink`` blocks.

        Stored as a weakref so the interceptor never keeps the client (and its
        transport connections) alive past its normal lifetime.
        """
        self._client_ref = weakref.ref(client)

    async def __call__(self, request: Any, handler: Any) -> Any:
        result = await handler(request)

        if not _mcp_types_available or not isinstance(result, CallToolResult):
            return result

        ctx = get_artifact_context()
        if ctx is None:
            return result

        try:
            return await self._ingest(result, request, ctx)
        except Exception:
            logger.warning(
                "Artifact ingestion failed for tool %r on server %r; "
                "passing through the original result",
                getattr(request, "name", "?"),
                getattr(request, "server_name", "?"),
                exc_info=True,
            )
            return result

    async def _ingest(
        self, result: "CallToolResult", request: Any, ctx: ArtifactRequestContext
    ) -> "CallToolResult":
        tool_name = getattr(request, "name", "tool")
        server_name = getattr(request, "server_name", "unknown")

        new_content: List[Any] = []
        persisted_count = 0
        counter = 0

        for block in result.content:
            if isinstance(block, TextContent):
                new_content.append(block)
                continue

            if isinstance(block, ImageContent):
                counter += 1
                if persisted_count >= ARTIFACT_MAX_PER_TOOL_CALL:
                    new_content.append(_warning_text("limit reached"))
                    continue

                data = base64.b64decode(block.data)
                if len(data) > ARTIFACT_MAX_BYTES:
                    new_content.append(_warning_text("exceeds size limit"))
                    continue

                content_type = block.mimeType or "application/octet-stream"
                ext = guess_extension_from_content_type(content_type)
                filename = f"{tool_name}-{counter}.{ext}"
                artifact = await self._persist(
                    data, content_type, filename, ctx, tool_name, server_name
                )
                new_content.append(
                    _stub_text(
                        is_image=True,
                        filename=filename,
                        artifact_id=artifact.id,
                        content_type=content_type,
                        size_bytes=len(data),
                        server_name=server_name,
                        tool_name=tool_name,
                    )
                )
                persisted_count += 1
                continue

            if isinstance(block, EmbeddedResource):
                resource = block.resource
                # Inline text resources stay inline in v1 (no artifact created).
                if isinstance(resource, TextResourceContents) or getattr(
                    resource, "text", None
                ) is not None:
                    new_content.append(block)
                    continue

                blob_b64 = getattr(resource, "blob", None)
                if blob_b64 is None:
                    new_content.append(block)
                    continue

                counter += 1
                if persisted_count >= ARTIFACT_MAX_PER_TOOL_CALL:
                    new_content.append(_warning_text("limit reached"))
                    continue

                data = base64.b64decode(blob_b64)
                if len(data) > ARTIFACT_MAX_BYTES:
                    new_content.append(_warning_text("exceeds size limit"))
                    continue

                content_type = resource.mimeType or "application/octet-stream"
                ext = guess_extension_from_content_type(content_type)
                filename = _filename_from_uri(getattr(resource, "uri", None)) or (
                    f"{tool_name}-{counter}.{ext}"
                )
                artifact = await self._persist(
                    data, content_type, filename, ctx, tool_name, server_name
                )
                new_content.append(
                    _stub_text(
                        is_image=content_type.startswith("image/"),
                        filename=filename,
                        artifact_id=artifact.id,
                        content_type=content_type,
                        size_bytes=len(data),
                        server_name=server_name,
                        tool_name=tool_name,
                    )
                )
                persisted_count += 1
                continue

            if isinstance(block, ResourceLink):
                counter += 1
                stubs = await self._resolve_resource_link(
                    block, ctx, tool_name, server_name, counter, persisted_count
                )
                if stubs is None:
                    # Timeout, error, or an oversized item: leave the link untouched.
                    new_content.append(block)
                else:
                    new_content.extend(stubs)
                    persisted_count += len(stubs)
                continue

            # Unknown/future block types passthrough untouched.
            new_content.append(block)

        return result.model_copy(update={"content": new_content})

    async def _persist(
        self,
        data: bytes,
        content_type: str,
        filename: str,
        ctx: ArtifactRequestContext,
        tool_name: str,
        server_name: str,
    ) -> Artifact:
        ext = guess_extension_from_content_type(content_type)
        key = storage_service.build_user_key(ctx.user_id, ext, prefix="artifacts")
        await storage_service.put_object(key, data, content_type)

        artifact = await Artifact.create(
            user_id=ctx.user_id,
            key=key,
            filename=filename,
            content_type=content_type,
            size_bytes=len(data),
            source=ArtifactSource(
                type="mcp_tool", mcp_server=server_name, tool_name=tool_name
            ),
            conversation_id=ctx.conversation_id,
            message_id=ctx.message_id,
        )
        ctx.collected_artifact_ids.append(artifact.id)
        return artifact

    async def _resolve_resource_link(
        self,
        block: "ResourceLink",
        ctx: ArtifactRequestContext,
        tool_name: str,
        server_name: str,
        counter: int,
        persisted_count: int,
    ) -> Optional[List["TextContent"]]:
        """Fetch a ResourceLink's contents and persist them as artifacts.

        Returns the list of stub TextContent blocks to substitute in, or None
        if the link should be left untouched (fetch failure, timeout, or any
        returned item exceeding the size limit — the whole link aborts rather
        than partially persisting).
        """
        client = self._client_ref() if self._client_ref is not None else None
        if client is None:
            return None

        try:
            async with asyncio.timeout(ARTIFACT_RESOURCE_READ_TIMEOUT_S):
                async with client.session(server_name) as session:
                    read_result = await session.read_resource(block.uri)
        except Exception:
            logger.warning(
                "Failed to read MCP resource %r from server %r",
                getattr(block, "uri", "?"),
                server_name,
                exc_info=True,
            )
            return None

        contents = getattr(read_result, "contents", None) or []
        if not contents:
            return None

        # Validate every item's size before persisting any of them: a single
        # oversized item aborts the whole link rather than persisting a subset.
        prepared: List[Tuple[bytes, str, str]] = []
        for item in contents:
            if isinstance(item, BlobResourceContents) or getattr(item, "blob", None):
                data = base64.b64decode(item.blob)
            elif isinstance(item, TextResourceContents) or getattr(
                item, "text", None
            ) is not None:
                data = item.text.encode("utf-8")
            else:
                continue

            if len(data) > ARTIFACT_MAX_BYTES:
                return None

            content_type = getattr(item, "mimeType", None) or "application/octet-stream"
            prepared.append((data, content_type, _filename_from_uri(getattr(item, "uri", None))))

        if not prepared:
            return None

        stubs: List["TextContent"] = []
        for idx, (data, content_type, uri_filename) in enumerate(prepared, start=1):
            if persisted_count >= ARTIFACT_MAX_PER_TOOL_CALL:
                stubs.append(_warning_text("limit reached"))
                continue

            ext = guess_extension_from_content_type(content_type)
            filename = uri_filename or f"{tool_name}-{counter}-{idx}.{ext}"
            artifact = await self._persist(
                data, content_type, filename, ctx, tool_name, server_name
            )
            stubs.append(
                _stub_text(
                    is_image=content_type.startswith("image/"),
                    filename=filename,
                    artifact_id=artifact.id,
                    content_type=content_type,
                    size_bytes=len(data),
                    server_name=server_name,
                    tool_name=tool_name,
                )
            )
            persisted_count += 1

        return stubs
