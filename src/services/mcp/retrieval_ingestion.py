"""MCP interceptor that keeps retrieval documents out of the model's context.

The ``eve_retrieval_retrieve`` tool returns the backend's ``POST /retrieve``
response verbatim: every chunk carries its Qdrant point id, scores, the
owner's ``user_id`` and ``collection_id``, plus the text twice. The agent graph
feeds the whole thing to the model as the tool result, and the model has been
seen copying the ids into the answer as if they were citations (staging,
2026-09-22: "...radiometersf6378a16-e113-4bbc-8ba8-7fd23b4b7e65,c58b5f5e-...").

This interceptor splits the two audiences. The full documents go to the
per-request ``retrieval_context`` for the Sources panel and the persisted
message; the model gets the same JSON shape reduced to what it needs to
answer: collection name, chunk text and the descriptive metadata (title, url,
doi, year, authors, ...).

Follows the ``ToolCallInterceptor`` protocol from ``langchain-mcp-adapters``.
Fail-open: any error leaves the original result untouched.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from src.services.mcp.retrieval_context import get_retrieval_context
from src.utils.helpers import extract_documents_from_retrieval_payload

logger = logging.getLogger(__name__)

try:
    from mcp.types import CallToolResult, TextContent

    _mcp_types_available = True
except Exception:  # pragma: no cover - mcp is a hard dependency in practice
    _mcp_types_available = False

# Keys that identify or rank a chunk, or describe how it was ingested. None of
# them helps the model answer, and the ids are exactly what leaked.
_INTERNAL_KEYS = frozenset(
    {
        "id",
        "version",
        "score",
        "reranking_score",
        "distance",
        "user_id",
        "collection_id",
        "document_id",
        "env",
        "file_path",
        "pipeline_metadata",
        "created_at",
        "last_update",
        "upload_time",
    }
)
_TEXT_KEYS = ("text", "content")
_RESPONSE_KEYS_KEPT = ("original_query", "requery")


def _merge_metadata(target: Dict[str, Any], source: Any) -> None:
    if not isinstance(source, dict):
        return
    for key, value in source.items():
        if key in _INTERNAL_KEYS or key in _TEXT_KEYS or key == "metadata":
            continue
        target.setdefault(key, value)


def slim_document_for_model(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce one retrieval document to collection, text and descriptive metadata.

    ``text`` keeps whatever the tool sent (a string, or the Wiley envelope
    object) so the model reads the same passage as before, only without the
    identifiers around it.
    """
    payload = doc.get("payload") or doc.get("document") or {}
    if not isinstance(payload, dict):
        payload = {}

    text: Any = ""
    for candidate in (doc.get("text"), payload.get("text"), payload.get("content")):
        if candidate not in (None, ""):
            text = candidate
            break

    metadata: Dict[str, Any] = {}
    _merge_metadata(metadata, payload)
    _merge_metadata(metadata, doc.get("metadata"))
    _merge_metadata(metadata, payload.get("metadata"))

    slim: Dict[str, Any] = {"collection_name": doc.get("collection_name"), "text": text}
    if metadata:
        slim["metadata"] = metadata
    return slim


def slim_retrieval_response_for_model(response: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild a ``/retrieve`` response with only what the model needs."""
    docs = response.get("retrieved_docs") or []
    slim: Dict[str, Any] = {
        "retrieved_docs": [
            slim_document_for_model(d) for d in docs if isinstance(d, dict)
        ]
    }
    for key in _RESPONSE_KEYS_KEPT:
        if response.get(key) not in (None, ""):
            slim[key] = response[key]
    return slim


def _parse_retrieval_response(text: Any) -> Optional[Dict[str, Any]]:
    """Return the parsed ``/retrieve`` response, or None for anything else."""
    if not isinstance(text, str) or not text.lstrip().startswith("{"):
        return None
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict) or not isinstance(
        parsed.get("retrieved_docs"), list
    ):
        return None
    return parsed


class RetrievalContextInterceptor:
    """Stash the full retrieval documents, hand the model a reduced copy.

    Only text blocks holding a ``{"retrieved_docs": [...]}`` object are
    rewritten. Error payloads, other tools' output and non-JSON text pass
    through, so ``is_retrieval_error_payload`` and the error logging
    interceptor keep seeing what they saw before.

    Registered in ``tool_loader.py`` between the error logger and
    ``ArtifactInterceptor``: the artifact stubs are markdown, never a
    retrieval payload, and the error logger sees the reduced result, which is
    still an error payload whenever the server sent one.
    """

    async def __call__(self, request: Any, handler: Any) -> Any:
        result = await handler(request)

        if not _mcp_types_available or not isinstance(result, CallToolResult):
            return result
        if get_retrieval_context() is None:
            return result

        try:
            return self._reduce(result)
        except Exception:
            logger.warning(
                "Retrieval result reduction failed for tool %r on server %r; "
                "passing through the original result",
                getattr(request, "name", "?"),
                getattr(request, "server_name", "?"),
                exc_info=True,
            )
            return result

    def _reduce(self, result: "CallToolResult") -> "CallToolResult":
        ctx = get_retrieval_context()
        if ctx is None:
            return result

        new_content: List[Any] = []
        rewritten = False
        for block in result.content:
            if not isinstance(block, TextContent):
                new_content.append(block)
                continue
            response = _parse_retrieval_response(block.text)
            if response is None:
                new_content.append(block)
                continue

            ctx.documents.extend(extract_documents_from_retrieval_payload(response))
            slim = slim_retrieval_response_for_model(response)
            new_content.append(
                block.model_copy(
                    update={"text": json.dumps(slim, ensure_ascii=False)}
                )
            )
            rewritten = True

        if not rewritten:
            return result
        # structuredContent would carry the full response to the model by
        # another route if the graph ever read the tool artifact.
        return result.model_copy(
            update={"content": new_content, "structuredContent": None}
        )
