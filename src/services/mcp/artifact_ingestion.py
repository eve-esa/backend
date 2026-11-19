"""MCP tool-call interceptor that captures files emitted by MCP servers as Artifacts.

Wraps ``execute_tool`` and sees the raw ``mcp.types.CallToolResult`` returned by
the MCP server *before* ``langchain_mcp_adapters`` converts it into a
``ToolMessage`` for the LLM. Any content block rewritten here therefore never
reaches the LLM, the LangGraph checkpointer, execution traces, or the SSE
stream. A file the tool JSON already describes keeps that object: its ``path``
becomes the sandbox path and the stub is dropped. A block that carries
``_meta.eve.source_path`` replaces that exact string, in any field, with the
sandbox path. A file with neither keeps the stub (a markdown link plus a
one-line JSON blob).

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
import re
import unicodedata
import weakref
from typing import Any, List, Optional, Tuple

from bson import ObjectId

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
from src.services.mcp.conversation_artifacts import (
    file_path_for,
    object_key,
    sandbox_path,
    upload_artifact,
)
from src.services.storage import (
    ARTIFACT_TYPE_CONTENT_TYPES,
    INLINE_IMAGE_CONTENT_TYPES,
    OPAQUE_CONTENT_TYPE,
    guess_extension_from_content_type,
    sniff_artifact_type,
    storage_service,
)

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


_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]+")
# Longest filename stored or shown. A longer one keeps its extension (up to
# ``_MAX_KEPT_EXTENSION`` characters) and loses the end of its stem.
MAX_FILENAME_CHARS = 255
_MAX_KEPT_EXTENSION = 16
_MARKDOWN_SPECIALS = re.compile(r'([\\\[\]()<>"])')


def detect_content_type(data: bytes, filename: Optional[str] = None) -> str:
    """Content type from the bytes, through the upload allowlist.

    The type a tool declares, and a filename extension alone, are never
    trusted: anything the allowlist cannot classify is stored opaque.
    Text types still need a matching extension plus clean UTF-8, as on upload.
    """
    kind = sniff_artifact_type(data[:16], filename, data)
    if kind is None:
        return OPAQUE_CONTENT_TYPE
    return ARTIFACT_TYPE_CONTENT_TYPES.get(kind, OPAQUE_CONTENT_TYPE)


def strip_format_chars(text: str) -> str:
    """Drop Unicode format characters (category Cf).

    Bidi overrides and isolates (U+202A to U+202E, U+2066 to U+2069) would
    show ``report<U+202E>fdp.exe`` as ``reportexe.pdf``; zero-width ones hide
    characters. None of them belongs in a filename.
    """
    return "".join(char for char in text if unicodedata.category(char) != "Cf")


def clean_filename(filename: str) -> str:
    """A filename safe to store and show: one line, no format characters, capped.

    Control characters (new lines included) become spaces, Unicode format
    characters go, and a name longer than ``MAX_FILENAME_CHARS`` is cut
    before its extension.
    """
    name = _CONTROL_CHARS.sub(" ", strip_format_chars(str(filename))).strip() or "file"
    if len(name) > MAX_FILENAME_CHARS:
        stem, ext = os.path.splitext(name)
        if len(ext) > _MAX_KEPT_EXTENSION:
            ext = ""
        name = stem[: MAX_FILENAME_CHARS - len(ext)] + ext
    return name


def _turn_file_key(filename: str) -> str:
    """The name the stored file has in its folder, the one an answer links."""
    return file_path_for("", filename)[1:]


def _unique_in_turn(filename: str, ctx: ArtifactRequestContext) -> str:
    """``filename``, or ``stem-2.ext``, ``stem-3.ext`` when the turn has it."""
    taken = ctx.collected_artifact_files
    if _turn_file_key(filename) not in taken:
        return filename
    stem, ext = os.path.splitext(filename)
    n = 2
    while _turn_file_key(f"{stem}-{n}{ext}") in taken:
        n += 1
    return f"{stem}-{n}{ext}"


def md_escape(text: str) -> str:
    """Make *text* safe as markdown link text: one line, no link syntax."""
    return _MARKDOWN_SPECIALS.sub(r"\\\1", clean_filename(text))


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

    Server names are user-registered, so both names lose control characters
    (a blank line would end the paragraph), double quotes (the title
    delimiter) and backslashes (which would escape the closing quote). Falls
    back to a bare "MCP" when either name is missing/unresolved rather than
    embedding one of this module's internal placeholder defaults ("unknown",
    "tool") as if it were real data.
    """

    def _clean(value: Optional[str]) -> Optional[str]:
        if not value or value in ("unknown", "tool"):
            return None
        cleaned = _CONTROL_CHARS.sub(" ", str(value)).replace('"', "").replace("\\", "")
        return cleaned.strip() or None

    server = _clean(server_name)
    tool = _clean(tool_name)
    if server and tool:
        return f"MCP: {server}/{tool}"
    return "MCP"


def artifact_markdown_line(
    *,
    artifact_id: str,
    filename: Optional[str],
    content_type: Optional[str],
    server_name: Optional[str] = None,
    tool_name: Optional[str] = None,
    provenance: bool = True,
) -> str:
    """The one markdown reference to a stored artifact.

    An image embed only for the raster allowlist, a plain link for anything
    else (SVG and HTML included). Used by the tool stub, by the stubs the
    runner appends to an answer and by the upload response, so all agree.
    ``provenance=False`` drops the MCP title, for a user's own upload.
    """
    url = f"/artifacts/{artifact_id}"
    target = f'{url} "{_provenance_title(server_name, tool_name)}"' if provenance else url
    label = md_escape(filename or "file")
    if content_type in INLINE_IMAGE_CONTENT_TYPES:
        return f"![{label}]({target})"
    return f"[{label}]({target})"


def _stub_text(
    *,
    filename: str,
    artifact_id: str,
    content_type: str,
    size_bytes: int,
    server_name: Optional[str] = None,
    tool_name: Optional[str] = None,
    output_path: Optional[str] = None,
) -> "TextContent":
    """Build the one-TextContent-per-artifact stub replacing a persisted block.

    Line 1 is a markdown reference (image embed for images, plain link
    otherwise) carrying a title that attributes the originating MCP
    server/tool; line 2 is a one-line JSON payload with the same data so a
    consumer that doesn't render markdown can still parse it out. ``path``
    in that JSON is the sandbox location when a conversation copy was stored.
    """
    url = f"/artifacts/{artifact_id}"
    link_line = artifact_markdown_line(
        artifact_id=artifact_id,
        filename=filename,
        content_type=content_type,
        server_name=server_name,
        tool_name=tool_name,
    )
    meta: dict[str, Any] = {
        "artifact_id": artifact_id,
        "url": url,
        "content_type": content_type,
        "filename": clean_filename(filename),
        "size_bytes": size_bytes,
    }
    # The sandbox path, without the users/.../conversations/... prefix.
    if output_path:
        meta["path"] = output_path
    meta_line = json.dumps(meta)
    return TextContent(type="text", text=f"{link_line}\n{meta_line}")


def _stub_record(block: Any) -> Optional[dict]:
    """Sandbox path and size from a markdown stub, if this block is one."""
    if not isinstance(block, TextContent):
        return None
    lines = block.text.split("\n", 1)
    if len(lines) != 2:
        return None
    try:
        meta = json.loads(lines[1])
    except json.JSONDecodeError:
        return None
    if not isinstance(meta, dict):
        return None
    path = meta.get("path")
    if not isinstance(path, str) or not path.startswith("/artifacts/"):
        return None
    size = meta.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int):
        size = None
    return {"path": path, "size": size}


def _stub_output_path(block: Any) -> Optional[str]:
    """The sandbox path carried by a markdown stub, if this block is one."""
    record = _stub_record(block)
    if record is None:
        return None
    return record["path"]


_STUB_ID = re.compile(r"^/artifacts/([0-9a-fA-F]{24})/")


def _shown_stub_ids(blocks: List[Any]) -> List[str]:
    """Artifact ids of the stubs left in a tool result after folding."""
    ids = []
    for block in blocks:
        record = _stub_record(block)
        match = _STUB_ID.match(record["path"]) if record else None
        if match:
            ids.append(match.group(1))
    return ids


def _eve_source_paths(block: Any) -> list[str]:
    """Exact path strings the MCP middleware loaded for this block."""
    meta = getattr(block, "meta", None)
    if not isinstance(meta, dict):
        return []
    eve = meta.get("eve")
    if not isinstance(eve, dict):
        return []
    many = eve.get("source_paths")
    if isinstance(many, list):
        found = [item for item in many if isinstance(item, str) and item]
        if found:
            return found
    one = eve.get("source_path")
    if isinstance(one, str) and one:
        return [one]
    return []


def _named_from_sources(sources: list[str], used_names: set) -> Optional[str]:
    """File name from the path the middleware loaded, when it is still free."""
    if not sources:
        return None
    name = os.path.basename(sources[0])
    if not name or name in {".", ".."} or name in used_names:
        return None
    used_names.add(name)
    return name


def _rewrite_source_paths(
    blocks: List[Any], structured: Any, pairs: List[Tuple[str, str]]
) -> Tuple[List[Any], Any]:
    """Replace each loaded path with the sandbox path, then drop that stub.

    The replacement walks tool text and structured content. Stub text is left
    alone, then removed when its sandbox path was written into the tool output.
    """
    if not pairs:
        return blocks, structured
    ordered = sorted(pairs, key=lambda item: len(item[0]), reverse=True)

    def rewrite(value: Any) -> Any:
        if isinstance(value, str):
            for old, new in ordered:
                if old and old != new:
                    value = value.replace(old, new)
            return value
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        return value

    rewritten: List[Any] = []
    for block in blocks:
        if isinstance(block, TextContent) and _stub_output_path(block) is None:
            rewritten.append(TextContent(type="text", text=rewrite(block.text)))
        else:
            rewritten.append(block)
    structured_out = rewrite(structured) if structured is not None else None
    visible: List[str] = []
    for block in rewritten:
        if isinstance(block, TextContent) and _stub_output_path(block) is None:
            visible.append(block.text)
    if structured_out is not None:
        visible.append(json.dumps(structured_out))
    joined = "\n".join(visible)
    used = {new for _, new in ordered if new and new in joined}
    kept = [block for block in rewritten if _stub_output_path(block) not in used]
    return kept, structured_out


def _iter_path_nodes(payload: Any):
    """Every object that already has a string ``path``, at any depth."""
    if isinstance(payload, dict):
        if isinstance(payload.get("path"), str):
            yield payload
        for value in payload.values():
            yield from _iter_path_nodes(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from _iter_path_nodes(item)


def _node_size(node: dict) -> Optional[int]:
    """A size the tool already reported on this object."""
    for key in ("bytes", "size_bytes"):
        value = node.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _json_payloads(blocks: List[Any], structured: Any) -> List[Any]:
    payloads: List[Any] = []
    for block in blocks:
        if not isinstance(block, TextContent) or _stub_output_path(block) is not None:
            continue
        try:
            payloads.append(json.loads(block.text))
        except json.JSONDecodeError:
            continue
    if structured is not None:
        payloads.append(structured)
    return payloads


def _unique_names_by_size(blocks: List[Any], structured: Any) -> dict:
    """Tool filename for a byte size, when exactly one ``path`` has that size.

    The extension must agree, so a PNG is not stored under a ``.csv`` name.
    """
    found: dict = {}
    for payload in _json_payloads(blocks, structured):
        for node in _iter_path_nodes(payload):
            size = _node_size(node)
            base = os.path.basename(node["path"])
            if size is None or not base or base in {".", ".."}:
                continue
            names = found.setdefault(size, [])
            if base not in names:
                names.append(base)
    return {size: names[0] for size, names in found.items() if len(names) == 1}


def _name_for_size(
    size: int, ext: str, names_by_size: dict, used: set
) -> Optional[str]:
    """The tool's own filename for these bytes, when the match is unique."""
    name = names_by_size.get(size)
    if not name or name in used:
        return None
    suffix = os.path.splitext(name)[1].lstrip(".").lower()
    if suffix and suffix != ext.lower():
        return None
    return name


def _pair_path_nodes(nodes: List[dict], stubs: List[dict]) -> List[Tuple[str, str]]:
    """Point each existing ``path`` at one sandbox file. Return the paths used.

    A name match wins. A byte size the tool already stored pairs a file when
    that size belongs to one object and one sandbox file. One leftover object
    and one leftover file pair too. Nothing else is written onto the object.
    """
    used_nodes: set = set()
    used_paths: set = set()
    assignments: List[Tuple[str, str]] = []

    def take(node: dict, stub: dict) -> None:
        original = node["path"]
        node["path"] = stub["path"]
        used_nodes.add(id(node))
        used_paths.add(stub["path"])
        assignments.append((original, stub["path"]))

    by_name: dict = {}
    for node in nodes:
        by_name.setdefault(os.path.basename(node["path"]), []).append(node)
    stubs_by_name: dict = {}
    for stub in stubs:
        stubs_by_name.setdefault(os.path.basename(stub["path"]), []).append(stub)
    for name, group in by_name.items():
        matches = stubs_by_name.get(name, [])
        if matches and len(group) == len(matches):
            for node, stub in zip(group, matches):
                take(node, stub)

    for node in nodes:
        if id(node) in used_nodes:
            continue
        size = _node_size(node)
        if size is None:
            continue
        node_mates = [
            item
            for item in nodes
            if id(item) not in used_nodes and _node_size(item) == size
        ]
        stub_mates = [
            item
            for item in stubs
            if item["path"] not in used_paths and item["size"] == size
        ]
        if len(node_mates) == 1 and len(stub_mates) == 1:
            take(node, stub_mates[0])

    left_nodes = [node for node in nodes if id(node) not in used_nodes]
    left_stubs = [stub for stub in stubs if stub["path"] not in used_paths]
    if len(left_nodes) == 1 and len(left_stubs) == 1:
        take(left_nodes[0], left_stubs[0])
    return assignments


def _replay_assignments(nodes: List[dict], assignments: List[Tuple[str, str]]) -> None:
    """Apply pairings from another copy of the same tool JSON."""
    groups: dict = {}
    for original, sandbox in assignments:
        groups.setdefault(original, []).append(sandbox)
    cursors = {key: 0 for key in groups}
    for node in nodes:
        slots = groups.get(node["path"])
        if not slots:
            continue
        index = cursors[node["path"]]
        if index >= len(slots):
            continue
        node["path"] = slots[index]
        cursors[node["path"]] = index + 1


def _fold_output_paths(
    blocks: List[Any], structured: Any
) -> tuple[List[Any], Any]:
    """Put the sandbox path in the tool's own JSON and drop the stub.

    Any object that already has a ``path``, at any depth, can receive the
    sandbox path of the file that belongs to it. No key is added. A file with
    no such object keeps its stub, which is how a bare image renders.
    """
    stubs = [record for block in blocks if (record := _stub_record(block))]
    if not stubs:
        return blocks, structured

    text_payloads: List[Any] = []
    for block in blocks:
        if isinstance(block, TextContent) and _stub_output_path(block) is None:
            try:
                text_payloads.append(json.loads(block.text))
            except json.JSONDecodeError:
                text_payloads.append(None)
        else:
            text_payloads.append(None)
    folded_structured = None
    if structured is not None:
        folded_structured = json.loads(json.dumps(structured))

    text_nodes = [
        node
        for payload in text_payloads
        if payload is not None
        for node in _iter_path_nodes(payload)
    ]
    structured_nodes = (
        list(_iter_path_nodes(folded_structured))
        if folded_structured is not None
        else []
    )
    if text_nodes:
        assignments = _pair_path_nodes(text_nodes, stubs)
        _replay_assignments(structured_nodes, assignments)
        used_paths = {sandbox for _, sandbox in assignments}
        leftover = [stub for stub in stubs if stub["path"] not in used_paths]
        pending = [
            node
            for node in structured_nodes
            if not str(node.get("path", "")).startswith("/artifacts/")
        ]
        if leftover and pending:
            assignments.extend(_pair_path_nodes(pending, leftover))
    else:
        assignments = _pair_path_nodes(structured_nodes, stubs)
    if not assignments:
        return blocks, structured
    used = {sandbox for _, sandbox in assignments}

    rewritten: List[Any] = []
    for block, payload in zip(blocks, text_payloads):
        if payload is not None and any(
            isinstance(node.get("path"), str) and node["path"] in used
            for node in _iter_path_nodes(payload)
        ):
            rewritten.append(TextContent(type="text", text=json.dumps(payload)))
            continue
        rewritten.append(block)
    kept = [block for block in rewritten if _stub_output_path(block) not in used]
    structured_out = structured
    if folded_structured is not None and any(
        isinstance(node.get("path"), str) and node["path"] in used
        for node in _iter_path_nodes(folded_structured)
    ):
        structured_out = folded_structured
    return kept, structured_out


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
        names_by_size = _unique_names_by_size(
            result.content, getattr(result, "structuredContent", None)
        )
        used_names: set = set()
        replacements: List[Tuple[str, str]] = []

        def filename_for(size: int, ext: str, fallback: str) -> str:
            chosen = _name_for_size(size, ext, names_by_size, used_names)
            if chosen is None:
                return fallback
            used_names.add(chosen)
            return chosen

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

                # block.mimeType is the tool's claim; only the bytes count.
                content_type = detect_content_type(data)
                ext = guess_extension_from_content_type(content_type)
                filename = _named_from_sources(_eve_source_paths(block), used_names)
                if filename is None:
                    filename = filename_for(
                        len(data), ext, f"{tool_name}-{counter}.{ext}"
                    )
                stored = await self._persist(
                    data, content_type, filename, ctx, tool_name, server_name
                )
                if stored is None:
                    new_content.append(_warning_text("not stored: no conversation"))
                    continue
                artifact, output_path = stored
                if output_path:
                    replacements.extend(
                        (source, output_path) for source in _eve_source_paths(block)
                    )
                new_content.append(
                    _stub_text(
                        filename=artifact.filename,
                        artifact_id=artifact.id,
                        content_type=content_type,
                        size_bytes=len(data),
                        server_name=server_name,
                        tool_name=tool_name,
                        output_path=output_path,
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

                uri_filename = _filename_from_uri(getattr(resource, "uri", None))
                content_type = detect_content_type(data, uri_filename)
                ext = guess_extension_from_content_type(content_type)
                filename = _named_from_sources(_eve_source_paths(block), used_names)
                if filename is None:
                    filename = uri_filename
                    if filename:
                        used_names.add(filename)
                    else:
                        filename = filename_for(
                            len(data), ext, f"{tool_name}-{counter}.{ext}"
                        )
                stored = await self._persist(
                    data, content_type, filename, ctx, tool_name, server_name
                )
                if stored is None:
                    new_content.append(_warning_text("not stored: no conversation"))
                    continue
                artifact, output_path = stored
                if output_path:
                    replacements.extend(
                        (source, output_path) for source in _eve_source_paths(block)
                    )
                new_content.append(
                    _stub_text(
                        filename=artifact.filename,
                        artifact_id=artifact.id,
                        content_type=content_type,
                        size_bytes=len(data),
                        server_name=server_name,
                        tool_name=tool_name,
                        output_path=output_path,
                    )
                )
                persisted_count += 1
                continue

            if isinstance(block, ResourceLink):
                counter += 1
                resolved = await self._resolve_resource_link(
                    block, ctx, tool_name, server_name, counter, persisted_count
                )
                if resolved is None:
                    # Timeout, error, or an oversized item: leave the link untouched.
                    new_content.append(block)
                else:
                    # `resolved` carries the count of artifacts stored. Warning
                    # stubs in the list are not stored artifacts, so the length
                    # of that list must not advance the cap.
                    stubs, persisted_count = resolved
                    new_content.extend(stubs)
                continue

            # Unknown/future block types passthrough untouched.
            new_content.append(block)

        content, structured = _rewrite_source_paths(
            new_content, getattr(result, "structuredContent", None), replacements
        )
        content, structured = _fold_output_paths(content, structured)
        ctx.shown_artifact_ids.extend(_shown_stub_ids(content))
        update: dict[str, Any] = {"content": content}
        if structured is not None:
            update["structuredContent"] = structured
        return result.model_copy(update=update)

    async def _persist(
        self,
        data: bytes,
        content_type: str,
        filename: str,
        ctx: ArtifactRequestContext,
        tool_name: str,
        server_name: str,
    ) -> Optional[tuple[Artifact, str]]:
        """Store one object under the conversation and insert its record.

        ``None`` when the turn has no conversation: nothing is written. The
        object is written first, under a preallocated id, and deleted again
        when the record insert fails. The sandbox upload is best-effort.
        """
        if not ctx.conversation_id:
            return None
        # Unique within the turn, reserved before the first await: two calls
        # of one tool both default to ``{tool}-1.png``, and the answer links
        # a file by that name.
        filename = _unique_in_turn(clean_filename(filename), ctx)
        artifact_oid = ObjectId()
        relative = file_path_for(str(artifact_oid), filename)
        turn_key = _turn_file_key(filename)
        ctx.collected_artifact_files[turn_key] = str(artifact_oid)
        key = object_key(ctx.user_id, ctx.conversation_id, relative)
        try:
            await storage_service.put_object(key, data, content_type)
        except Exception:
            ctx.collected_artifact_files.pop(turn_key, None)
            raise
        artifact = Artifact(
            id=str(artifact_oid),
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
        document = artifact.to_dict()
        document["_id"] = artifact_oid
        try:
            await Artifact.get_collection().insert_one(document)
        except Exception:
            ctx.collected_artifact_files.pop(turn_key, None)
            logger.warning(
                "artifacts.record_failed key=%s", key, exc_info=True
            )
            try:
                await storage_service.delete_object(key)
            except Exception:
                logger.warning(
                    "artifacts.record_cleanup_failed key=%s", key, exc_info=True
                )
            raise
        ctx.collected_artifact_ids.append(artifact.id)
        ctx.collected_artifact_types[artifact.id] = content_type
        path = sandbox_path(relative)
        await upload_artifact(ctx.sandbox_backend, path, data)
        return artifact, path

    async def _resolve_resource_link(
        self,
        block: "ResourceLink",
        ctx: ArtifactRequestContext,
        tool_name: str,
        server_name: str,
        counter: int,
        persisted_count: int,
    ) -> Optional[tuple[List["TextContent"], int]]:
        """Fetch a ResourceLink's contents and persist them as artifacts.

        Returns the stub blocks to substitute in, plus the updated count of
        artifacts stored for this tool call. Limit warnings are in the list
        and are not included in the count. None means leave the link
        untouched (fetch failure, timeout, or any returned item exceeding
        the size limit — the whole link aborts rather than partially
        persisting).
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

            uri_filename = _filename_from_uri(getattr(item, "uri", None))
            content_type = detect_content_type(data, uri_filename)
            prepared.append((data, content_type, uri_filename))

        if not prepared:
            return None

        stubs: List["TextContent"] = []
        for idx, (data, content_type, uri_filename) in enumerate(prepared, start=1):
            if persisted_count >= ARTIFACT_MAX_PER_TOOL_CALL:
                stubs.append(_warning_text("limit reached"))
                continue

            ext = guess_extension_from_content_type(content_type)
            filename = uri_filename or f"{tool_name}-{counter}-{idx}.{ext}"
            stored = await self._persist(
                data, content_type, filename, ctx, tool_name, server_name
            )
            if stored is None:
                stubs.append(_warning_text("not stored: no conversation"))
                continue
            artifact, output_path = stored
            stubs.append(
                _stub_text(
                    filename=artifact.filename,
                    artifact_id=artifact.id,
                    content_type=content_type,
                    size_bytes=len(data),
                    server_name=server_name,
                    tool_name=tool_name,
                    output_path=output_path,
                )
            )
            persisted_count += 1

        return stubs, persisted_count
