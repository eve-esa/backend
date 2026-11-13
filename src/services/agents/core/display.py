"""Show a sandbox file in the chat.

The model writes images and other files inside the sandbox. Those files never
reach the UI, because the sandbox filesystem is not the artifact store. This
tool reads one file and hands the bytes to ``ArtifactInterceptor``, the same
path MCP tools already use. The interceptor stores an Artifact and returns the
markdown stub the chat renders: an inline image, or a download chip.

The tool is built only for a turn that has a sandbox, and it reads that
sandbox directly. It does not go through the composite backend, so a path
cannot land on conversation history or an offloaded tool result.

A file another tool already returned is an Artifact in object storage, not a
sandbox file. Models still pass its ``/artifacts/{id}`` URL or its filename
here. Those resolve to the stored Artifact of the same user and conversation,
and the tool returns its existing stub instead of a read error.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import re
import urllib.parse
from types import SimpleNamespace
from typing import Any, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, ValidationError

from src.database.models.artifact import Artifact
from src.services.mcp.artifact_context import (
    ArtifactRequestContext,
    get_artifact_context,
)
from src.services.mcp.artifact_ingestion import ArtifactInterceptor, _stub_text
from src.services.storage import ARTIFACT_TYPE_CONTENT_TYPES, sniff_artifact_type

logger = logging.getLogger(__name__)

try:
    from mcp.types import (
        BlobResourceContents,
        CallToolResult,
        EmbeddedResource,
        TextContent,
    )

    _mcp_types_available = True
except Exception:
    _mcp_types_available = False

_DESCRIPTION = (
    "Show an image or other file you created in the sandbox to the user. "
    "Pass path relative to the working directory, for example plot.png. "
    "To show again a file another tool already returned, pass its "
    "/artifacts/{id} URL. "
    "The result is a markdown line. Copy that line into your answer verbatim. "
    "Do not rewrite the URL."
)

# Accepts "/artifacts/{id}", "artifacts/{id}", and an absolute URL that ends in
# one. Artifact ids are Mongo ObjectIds.
_ARTIFACT_REF = re.compile(r"(?:^|/)artifacts/([0-9a-fA-F]{24})/?$")


class DisplayArgs(BaseModel):
    path: str = Field(
        ...,
        description=(
            "Path of the file in the sandbox, relative to the working directory."
        ),
    )


def _read_error(path: str, detail: str) -> str:
    """Tell the model to write the file where the sandbox can read it.

    AgentCore ``readFiles`` only resolves a path under the session working
    directory. An absolute path, or a file left somewhere else, comes back
    as not found.
    """
    return (
        f"Could not read {path}: {detail}. Save the file in the working "
        "directory and pass a relative path."
    )


def _unknown_artifact(path: str) -> str:
    return (
        f"Could not display {path}: no such artifact in this conversation. "
        "Copy the /artifacts/{id} URL from the tool result that produced it."
    )


async def _stored_artifact(
    wanted: str, ctx: Optional[ArtifactRequestContext]
) -> Optional[Artifact]:
    """The Artifact *wanted* names, owned by this user and conversation.

    An ``/artifacts/{id}`` reference is looked up by id. Anything else is a
    filename, and the newest Artifact with that name wins.
    """
    if ctx is None or not ctx.conversation_id:
        return None
    try:
        match = _ARTIFACT_REF.search(wanted)
        if match:
            artifact = await Artifact.find_by_id(match.group(1))
            if (
                artifact is None
                or artifact.user_id != ctx.user_id
                or artifact.conversation_id != ctx.conversation_id
            ):
                return None
            return artifact
        filename = wanted.rstrip("/").rsplit("/", 1)[-1]
        if not filename:
            return None
        found = await Artifact.find_all(
            limit=1,
            sort=[("timestamp", -1)],
            filter_dict={
                "user_id": ctx.user_id,
                "conversation_id": ctx.conversation_id,
                "filename": filename,
            },
        )
        return found[0] if found else None
    except Exception:
        logger.warning("display could not look up artifact %s", wanted, exc_info=True)
        return None


def _reuse(artifact: Artifact, ctx: ArtifactRequestContext) -> str:
    """The existing stub for *artifact*, linked to the current message."""
    if artifact.id not in ctx.collected_artifact_ids:
        ctx.collected_artifact_ids.append(artifact.id)
    source = artifact.source
    return _stub_text(
        is_image=(artifact.content_type or "").startswith("image/"),
        filename=artifact.filename,
        artifact_id=artifact.id,
        content_type=artifact.content_type,
        size_bytes=artifact.size_bytes,
        server_name=getattr(source, "mcp_server", None),
        tool_name=getattr(source, "tool_name", None),
    ).text


def _not_stored(path: str) -> str:
    return (
        f"Could not display {path}. The file was read but not stored, "
        "so it cannot be shown."
    )


def _content_type(data: bytes, filename: str) -> str:
    """MIME type from the bytes, then the filename, then a generic type."""
    kind = sniff_artifact_type(data[:16], filename, data)
    if kind:
        mapped = ARTIFACT_TYPE_CONTENT_TYPES.get(kind)
        if mapped:
            return mapped
    guessed, _encoding = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"


def _resource_uri(path: str) -> str:
    """URI whose last segment is the display filename.

    ``file://{path}`` is what a normal relative path needs. A space or other
    character that the URI type rejects is percent-encoded under ``file:///``,
    which still ends in the filename.
    """
    direct = f"file://{path}"
    quoted = urllib.parse.quote(path.lstrip("/") or "file", safe="/")
    return direct if _uri_ok(direct) else f"file:///{quoted}"


def _uri_ok(uri: str) -> bool:
    if not _mcp_types_available:
        return False
    try:
        BlobResourceContents(uri=uri, mimeType="application/octet-stream", blob="")
    except ValidationError:
        return False
    return True


def _embedded(path: str, content_type: str, blob: str) -> Any:
    return EmbeddedResource(
        type="resource",
        resource=BlobResourceContents(
            uri=_resource_uri(path),
            mimeType=content_type,
            blob=blob,
        ),
    )


def build_display_tool(backend: Any) -> StructuredTool:
    """One ``display`` tool bound to *backend*. The model cannot pick another."""

    async def display(path: str) -> str:
        wanted = str(path or "").strip()
        if not wanted:
            return (
                "Pass the path of a file in the sandbox, relative to the "
                "working directory."
            )
        if not _mcp_types_available:
            return _not_stored(wanted)

        ctx = get_artifact_context()
        if _ARTIFACT_REF.search(wanted):
            artifact = await _stored_artifact(wanted, ctx)
            if artifact is None:
                return _unknown_artifact(wanted)
            return _reuse(artifact, ctx)

        try:
            responses = await backend.adownload_files([wanted])
        except Exception:
            logger.warning("display could not read %s", wanted, exc_info=True)
            responses, failure = None, "read failed"
        else:
            failure = None

        response = responses[0] if responses else None
        data = getattr(response, "content", None)
        error = getattr(response, "error", None)
        if response is None or error or data is None:
            artifact = await _stored_artifact(wanted, ctx)
            if artifact is not None:
                return _reuse(artifact, ctx)
            return _read_error(wanted, failure or str(error or "file_not_found"))

        filename = wanted.rstrip("/").rsplit("/", 1)[-1] or "file"
        try:
            block = _embedded(
                wanted,
                _content_type(data, filename),
                base64.b64encode(data).decode("ascii"),
            )
        except ValidationError:
            logger.warning("display rejected the path %s", wanted, exc_info=True)
            return _not_stored(wanted)

        result = CallToolResult(content=[block])
        request = SimpleNamespace(name="display", server_name="sandbox")

        async def handler(_request: Any) -> Any:
            return result

        ingested = await ArtifactInterceptor()(request, handler)
        texts: list[str] = []
        for item in getattr(ingested, "content", None) or []:
            if not isinstance(item, TextContent):
                # Fail-open leaves the original blob in place. Never hand
                # those bytes to the model, the checkpointer, or the stream.
                return _not_stored(wanted)
            texts.append(item.text)
        if not texts:
            return _not_stored(wanted)
        return "\n".join(texts)

    return StructuredTool.from_function(
        coroutine=display,
        name="display",
        description=_DESCRIPTION,
        args_schema=DisplayArgs,
    )
