"""
Utility functions for file handling and embeddings generation.

This module provides utility functions for handling file uploads,
getting embedding models based on model names, and making API requests
to RunPod for embeddings generation.
"""

import logging
import tempfile
from enum import Enum
from typing import Any, Optional, List, Dict

from fastapi import UploadFile

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
import tiktoken

from src.config import (
    MONGO_HOST,
    MONGO_PORT,
    MONGO_USERNAME,
    MONGO_PASSWORD,
    MONGO_DATABASE,
    MONGO_PARAMS,
)
from src.constants import TOKEN_OVERFLOW_LIMIT

# Configure logging
logger = logging.getLogger(__name__)
class EmbeddingModelType(Enum):
    """Supported embedding model types."""

    QWEN_3_4B = "Qwen/Qwen3-Embedding-4B"
    QWEN_3_4B_INFERENCE = "qwen/qwen3-embedding-4b"

async def save_upload_file_to_temp(upload_file: UploadFile) -> str:
    """
    Save an uploaded file to a temporary file.

    Args:
        upload_file: The uploaded file object from FastAPI

    Returns:
        str: The path to the temporary file

    Raises:
        IOError: If there's an error reading or writing the file
    """
    try:
        # Use suffix based on original file extension if possible
        original_filename = upload_file.filename or ""
        suffix = (
            f".{original_filename.split('.')[-1]}"
            if "." in original_filename
            else ".pdf"
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            contents = await upload_file.read()
            temp_file.write(contents)
            temp_file.flush()

            file_path = temp_file.name
            logger.info(f"Saved uploaded file to temporary path: {file_path}")
            return file_path

    except Exception as e:
        logger.error(f"Failed to save uploaded file: {str(e)}")
        raise IOError(f"Failed to save uploaded file: {str(e)}") from e

def _field(obj: Any, key: str, default: Any = None) -> Any:
    """Return value for key from dict-like or attribute-like object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _to_int(value: Any) -> Any:
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def _to_float(value: Any) -> Any:
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


def extract_year_range_from_filters(filters: Any) -> Optional[List[int]]:
    """Extract [start_year, end_year] from request.filters structure.

    Expected shape:
      {
        "must": [
          {"key": "year", "range": {"gte": <start>, "lte": <end>}},
          ...
        ]
      }
    Returns None if not found or values are invalid.
    """
    try:
        if not isinstance(filters, dict):
            return None
        conditions = filters.get("must") or []
        if not isinstance(conditions, list):
            return None
        for cond in conditions:
            if not isinstance(cond, dict):
                continue
            if cond.get("key") != "year":
                continue
            rng = cond.get("range") or {}
            if not isinstance(rng, dict):
                continue
            start = _to_int(rng.get("gte"))
            end = _to_int(rng.get("lte"))
            if start is None and end is None:
                return None
            if start is not None and end is not None:
                return [start, end]
            if start is not None:
                return [start, start]
            if end is not None:
                return [end, end]
        return None
    except Exception:
        return None


def extract_document_data(result: Any) -> Dict[str, Any]:
    result_id = _field(result, "id")
    result_version = _to_int(_field(result, "version"))
    result_score = _to_float(_field(result, "score") or _field(result, "distance"))
    result_rerank = _to_float(_field(result, "reranking_score"))
    result_payload = (
        _field(result, "payload", {}) or _field(result, "document", {}) or {}
    )
    collection_name = _field(result, "collection_name")
    if not collection_name and isinstance(result_payload, dict):
        collection_name = result_payload.get("collection_name") or (
            result_payload.get("metadata") or {}
        ).get("collection_name")

    result_text = _field(result, "text", "") or ""
    result_metadata = _field(result, "metadata", {}) or {}

    # Fallbacks from payload
    if not result_text and isinstance(result_payload, dict):
        result_text = (
            result_payload.get("text", result_payload.get("content", "")) or ""
        )
    if not result_metadata and isinstance(result_payload, dict):
        result_metadata = result_payload.get("metadata", {}) or {}

    return {
        "id": str(result_id) if result_id is not None else None,
        "version": result_version,
        "score": result_score,
        "reranking_score": result_rerank,
        "collection_name": collection_name,
        "payload": result_payload,
        "text": result_text,
        "metadata": result_metadata,
    }


def str_token_counter(text: str) -> int:
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4


def tiktoken_counter(messages: List[BaseMessage]) -> int:
    """Custom token counter for messages using tiktoken."""
    num_tokens = 3  # every reply is primed with <|start|>assistant<|message|>
    tokens_per_message = 3
    tokens_per_name = 1
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, ToolMessage):
            role = "tool"
        elif isinstance(msg, SystemMessage):
            role = "system"
        else:
            raise ValueError(f"Unsupported messages type {msg.__class__}")
        num_tokens += (
            tokens_per_message
            + str_token_counter(role)
            + str_token_counter(msg.content)
        )
        if hasattr(msg, "name") and msg.name:
            num_tokens += tokens_per_name + str_token_counter(msg.name)
    return num_tokens


def trim_text_to_token_limit(text: str, max_tokens: int) -> str:
    """Trim text to be at most max_tokens using tiktoken, with a char fallback."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        toks = enc.encode(text)
        if len(toks) <= max_tokens:
            return text
        return enc.decode(toks[:max_tokens])
    except Exception:
        est_chars = max(0, max_tokens * 4)
        return text[:est_chars]


def trim_context_to_token_limit(parts: List[str], max_tokens: int = 7000) -> str:
    """
    Trim context by removing entire documents (parts) when token limit is exceeded.

    This function ensures that we don't show trimmed documents to users by removing
    entire documents from the context rather than truncating individual documents.

    Args:
        parts: List of context parts (documents) to join
        max_tokens: Maximum number of tokens allowed (default: 7000)

    Returns:
        str: Trimmed context string that fits within the token limit
    """
    if not parts:
        return ""

    # Filter out None parts and join
    context = "\n".join([p for p in parts if p is not None])
    context_len = str_token_counter(context)

    # Check if the context goes beyond the limit
    # If so remove an entire document from the context, we do not want to show to the user trimmed docs
    while context_len > max_tokens and len(parts) > 1:
        # We are assuming that parts are ordered from most relevant to least
        parts.pop()
        context = "\n".join([p for p in parts if p is not None])
        context_len = str_token_counter(context)

    return context


def build_context(items: List[Any]) -> str:
    """Build a context string from items that may be strings or {text, metadata} dicts.

    - If items are strings, join non-empty with newlines (backward compatible).
    - If items are dicts with keys 'text' and optional 'metadata', format as:

      Document metadata\n
      Key: value\n
      ...\n
      Content:\n
      {text}\n
    """
    if not items:
        return ""

    # Detect structured items
    try:
        if isinstance(items[0], dict):
            parts: List[str] = []
            for item in items:
                if not isinstance(item, dict):
                    # Mixed types; fall back to string rendering
                    text_val = str(item)
                    if text_val:
                        parts.append(text_val)
                    continue

                text_val = (
                    item.get("text")
                    or item.get("document")
                    or item.get("content")
                    or ""
                )
                metadata_val = item.get("metadata") or {}

                # Render metadata block if present
                if isinstance(metadata_val, dict) and metadata_val:
                    parts.append("Document metadata")
                    # Stable key order for deterministic output
                    for key in sorted(metadata_val.keys()):
                        value = metadata_val.get(key)
                        parts.append(f"{key}: {value}")
                else:
                    # Still include a header for consistency when no metadata
                    parts.append("Document metadata")

                parts.append("Content:")
                if text_val:
                    parts.append(str(text_val))
                # Blank line between documents
                parts.append("")
            return trim_context_to_token_limit(parts, TOKEN_OVERFLOW_LIMIT)
    except Exception:
        # On any error, fall back to simple join of strings
        pass

    # Default: treat as list of strings
    return trim_context_to_token_limit(items, TOKEN_OVERFLOW_LIMIT)


def build_conversation_context(
    conversation_history: List[Any], summary: Optional[str] = None
) -> str:
    """
    Build conversation context from message history and optional summary.

    Args:
        conversation_history: List of previous messages
        summary: Optional conversation summary

    Returns:
        Formatted conversation context string
    """
    context_parts = []

    # Add summary if available
    if summary and summary.strip():
        context_parts.append(f"Conversation Summary: {summary}")
        context_parts.append("")  # Empty line for separation

    # Add recent message history
    if conversation_history:
        for msg in conversation_history:
            try:
                # Handle LangChain message objects
                if hasattr(msg, "content") and hasattr(msg, "__class__"):
                    role = "User" if "Human" in msg.__class__.__name__ else "Assistant"
                    content = msg.content
                # Handle dict-style messages
                elif isinstance(msg, dict):
                    role = msg.get("role", "User").title()
                    content = msg.get("content", "")
                else:
                    continue

                if content.strip():
                    context_parts.append(f"{role}: {content}")
            except Exception:
                continue

    return "\n".join(context_parts)


def normalize_markdown_tables(text: str) -> str:
    """Normalize Markdown tables: trim cell padding, ensure separator row, and avoid space-filled cells.

    This function scans for contiguous lines that look like Markdown table rows (lines starting with '|'),
    trims excessive whitespace inside cells, rebuilds rows with a single space around pipes,
    and guarantees a standard separator row ("| --- | ... |") right after the header.

    It is a best-effort normalizer and leaves non-table content unchanged.
    """
    try:
        if not text or "|" not in text:
            return text

        lines = text.splitlines()
        normalized_lines: List[str] = []
        i = 0

        def _is_table_line(s: str) -> bool:
            ss = s.strip()
            return ss.startswith("|") and ("|" in ss[1:])

        def _only_dashes_colons(s: str) -> bool:
            return all(ch in "-:| " for ch in s)

        while i < len(lines):
            if _is_table_line(lines[i]):
                start = i
                while i < len(lines) and _is_table_line(lines[i]):
                    i += 1
                block = lines[start:i]
                if not block:
                    continue

                # Parse header cells from first line
                header_cells = [
                    c.strip() for c in block[0].strip().strip("|").split("|")
                ]
                header_cells = [c for c in header_cells]

                normalized_block: List[str] = []

                # Rebuild header
                header_line = "| " + " | ".join(header_cells) + " |"
                normalized_block.append(header_line)

                # Build or normalize separator
                sep_line: Optional[str] = None
                if len(block) >= 2 and _only_dashes_colons(block[1].strip()):
                    # Keep a standard separator matching number of header cells
                    sep_line = (
                        "| " + " | ".join(["---"] * max(1, len(header_cells))) + " |"
                    )
                    # Skip the original sep and insert standardized one
                    row_start_idx = 2
                else:
                    sep_line = (
                        "| " + " | ".join(["---"] * max(1, len(header_cells))) + " |"
                    )
                    row_start_idx = 1
                normalized_block.append(sep_line)

                # Normalize remaining rows
                for j in range(row_start_idx, len(block)):
                    raw = block[j].strip()
                    cells = [c.strip() for c in raw.strip("|").split("|")]
                    # If row has fewer or more cells, just rebuild with what we have
                    row_line = "| " + " | ".join(cells) + " |"
                    normalized_block.append(row_line)

                normalized_lines.extend(normalized_block)
            else:
                normalized_lines.append(lines[i])
                i += 1

        return "\n".join(normalized_lines)
    except Exception:
        return text


class MarkdownTableStreamNormalizer:
    """Incremental normalizer for Markdown tables during streaming.

    Accumulates lines that look like table rows and emits normalized blocks once
    a non-table line is encountered or at flush(). Non-table lines are passed through.
    """

    def __init__(self):
        self._buffer = ""
        self._in_table = False
        self._table_lines: List[str] = []

    @staticmethod
    def _is_table_line(s: str) -> bool:
        ss = s.strip()
        return ss.startswith("|") and ("|" in ss[1:])

    def ingest(self, chunk: str) -> List[str]:
        outputs: List[str] = []
        if not isinstance(chunk, str) or chunk == "":
            return outputs
        self._buffer += chunk

        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if self._is_table_line(line):
                self._in_table = True
                self._table_lines.append(line)
                continue
            # Flush table block first if any
            if self._in_table and self._table_lines:
                normalized_block = normalize_markdown_tables(
                    "\n".join(self._table_lines)
                )
                outputs.append(normalized_block + "\n")
                self._table_lines = []
                self._in_table = False
            # Emit non-table line
            outputs.append(line + "\n")
        return outputs

    def flush(self) -> List[str]:
        outputs: List[str] = []
        if self._in_table and self._table_lines:
            normalized_block = normalize_markdown_tables(
                "\n".join(self._table_lines + ([self._buffer] if self._buffer else []))
            )
            outputs.append(normalized_block)
        elif self._buffer:
            outputs.append(self._buffer)
        self._buffer = ""
        self._table_lines = []
        self._in_table = False
        return outputs

def pluralize(n: int, singular: str, plural: str) -> str:
    """Return singular or plural form based on count."""
    return singular if n == 1 else plural


def get_co2_usage_kg(total_chars: int) -> float:
    """Calculate CO2 usage from total characters."""
    # Calculate CO2 usage from total characters
    # Assumptions:
    # - Roughly 4 characters per token
    # - ~0.000078 grams CO2 per token
    # - Convert grams to kg
    CHARS_PER_TOKEN = 4
    CO2_PER_TOKEN_GRAM = 0.000078
    GRAM_PER_KG = 1000
    return (
        (total_chars / CHARS_PER_TOKEN) * (CO2_PER_TOKEN_GRAM / GRAM_PER_KG)
        if isinstance(total_chars, (int, float)) and total_chars > 0
        else 0.0
    )
    

def get_mongodb_uri() -> str:
    """Build MongoDB URI from environment, appending MONGO_PARAMS if provided."""
    params = MONGO_PARAMS or ""
    if params and not params.startswith("?"):
        params = f"?{params}"
    return f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/{MONGO_DATABASE}{params}"
