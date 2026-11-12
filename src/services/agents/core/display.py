"""Show a sandbox file in the chat.

The model writes images and other files inside the sandbox. Those files never
reach the UI, because the sandbox filesystem is not the artifact store. This
tool reads one file and hands the bytes to ``ArtifactInterceptor``, the same
path MCP tools already use. The interceptor stores an Artifact and returns the
markdown stub the chat renders: an inline image, or a download chip.

The tool is built only for a turn that has a sandbox, and it reads that
sandbox directly. It does not go through the composite backend, so a path
cannot land on conversation history or an offloaded tool result.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import urllib.parse
from types import SimpleNamespace
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, ValidationError

from src.services.mcp.artifact_ingestion import ArtifactInterceptor
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
    "The result is a markdown line. Copy that line into your answer verbatim. "
    "Do not rewrite the URL."
)


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

        try:
            responses = await backend.adownload_files([wanted])
        except Exception:
            logger.warning("display could not read %s", wanted, exc_info=True)
            return _read_error(wanted, "read failed")

        response = responses[0] if responses else None
        data = getattr(response, "content", None)
        error = getattr(response, "error", None)
        if response is None or error or data is None:
            return _read_error(wanted, str(error or "file_not_found"))

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
