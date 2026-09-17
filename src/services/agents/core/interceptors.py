"""MCP tool-call interceptor for backend error logging.

Persists tool-call failures to MongoDB via :class:`ErrorLogger`.
Designed to compose with :class:`LatencyInterceptor` from
``src.services.agents.graphs.base`` which handles latency tracking independently.

Does not re-raise returned MCP errors (``isError``, structured, text). Exceptions
are logged and re-raised so the tools node can swallow them into a ToolMessage.

Reference:
    https://github.com/langchain-ai/langchain-mcp-adapters (interceptors module)
    https://docs.langchain.com/oss/python/langgraph/fault-tolerance
"""

import json
import re
import time
from typing import Any, Optional, Tuple

from src.utils.error_logger import get_error_logger, redact_secrets

try:
    from langchain_mcp_adapters.interceptors import (
        MCPToolCallRequest,
        MCPToolCallResult,
    )

    _interceptors_available = True
except Exception:
    _interceptors_available = False


class ToolReturnedError(Exception):
    """MCP tool completed but reported failure in the result payload."""


# Tight heuristic for tools that return auth/HTTP failures as plain text.
# Avoid matching ordinary phrases like "there was an error in the calculation".
_TEXT_ERROR_RE = re.compile(
    r"(?i)(?:error\s*[:\-]?\s*\d{3}|unauthorized|authentication (?:failed|error)|"
    r"invalid (?:access )?token|forbidden)"
)


def _is_error_payload(value: Any) -> bool:
    """True when a structured ``error`` field is a real failure, not ``false``/empty."""
    if value is None or value is False:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return True


def _summarize_error_payload(value: Any) -> str:
    """Keep code/message only — drop JSON-RPC ``data`` and other blobs."""
    if isinstance(value, dict):
        parts = []
        if value.get("code") is not None:
            parts.append(f"code={value.get('code')}")
        message = value.get("message")
        if message:
            parts.append(redact_secrets(str(message))[:120])
        return ", ".join(parts) or "error object"
    return redact_secrets(str(value))[:120]


def _result_is_error_flag(result: Any) -> bool:
    flag = getattr(result, "isError", None)
    if flag is None:
        flag = getattr(result, "is_error", None)
    if flag is None and isinstance(result, dict):
        flag = result.get("isError", result.get("is_error"))
    return bool(flag)


def _structured_payload(result: Any) -> Any:
    structured = getattr(result, "structuredContent", None)
    if structured is None:
        structured = getattr(result, "structured_content", None)
    if structured is None and isinstance(result, dict):
        structured = result.get("structuredContent") or result.get("structured_content")
        if structured is None and "error" in result and _is_error_payload(result.get("error")):
            return result
    return structured


def _text_from_result(result: Any) -> str:
    content = getattr(result, "content", None)
    if content is None and isinstance(result, dict):
        content = result.get("content")
        if content is None and isinstance(result.get("result"), dict):
            content = result["result"].get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue
            text = getattr(block, "text", None)
            if text is None and isinstance(block, dict):
                text = block.get("text")
            if text:
                parts.append(str(text))
        return "\n".join(parts)
    return ""


def classify_mcp_tool_result(result: Any) -> Optional[Tuple[str, str]]:
    """Return ``(signal, description)`` when *result* is a tool failure.

    Signals: ``is_error`` (MCP ``isError`` / JSON-RPC error), ``structured``
    (``{error: ...}``), ``text`` (tight 401-ish heuristic). ``None`` on success.
    """
    if result is None:
        return None

    if isinstance(result, dict) and _is_error_payload(result.get("error")) and (
        "jsonrpc" in result or "id" in result or "result" in result
    ):
        return "is_error", f"JSON-RPC error: {_summarize_error_payload(result.get('error'))}"

    nested = result.get("result") if isinstance(result, dict) else None
    if isinstance(nested, dict) and nested.get("isError"):
        return "is_error", "MCP CallToolResult.isError=true"

    if _result_is_error_flag(result):
        text = _text_from_result(result)
        return "is_error", (
            _summarize_error_payload(text) if text else "MCP CallToolResult.isError=true"
        )

    structured = _structured_payload(result)
    if isinstance(structured, dict) and _is_error_payload(structured.get("error")):
        return "structured", f"structured error: {_summarize_error_payload(structured.get('error'))}"

    text = _text_from_result(result)
    if text:
        stripped = text.strip()
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, dict) and _is_error_payload(parsed.get("error")):
            return "structured", f"structured error: {_summarize_error_payload(parsed.get('error'))}"
        if _TEXT_ERROR_RE.search(stripped):
            preview = redact_secrets(stripped)[:120]
            return "text", preview

    return None


class ErrorLoggingInterceptor:
    """Persists MCP tool-call failures to the backend error log (MongoDB).

    Follows the ``ToolCallInterceptor`` protocol from ``langchain-mcp-adapters``.
    Exceptions are logged as ``kind=tool_error`` / ``signal=exception`` then
    re-raised (the tools node still swallows them). Returned failures
    (``isError``, structured ``{error: ...}``, 401-ish text) are logged and
    the result is returned so the graph continues.

    Compose with :class:`~src.services.agents.graphs.base.LatencyInterceptor` for
    latency tracking::

        from src.services.agents.graphs.base import LatencyInterceptor

        client = MultiServerMCPClient(
            connections,
            tool_interceptors=[LatencyInterceptor(), ErrorLoggingInterceptor()],
        )
    """

    async def __call__(
        self,
        request: "MCPToolCallRequest",
        handler: Any,
    ) -> "MCPToolCallResult":
        tool_name = getattr(request, "name", "unknown")
        server_name = getattr(request, "server_name", "unknown")
        start = time.perf_counter()

        try:
            result = await handler(request)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            await self._log_tool_error(
                exc,
                tool_name=tool_name,
                server_name=server_name,
                signal="exception",
                elapsed=elapsed,
            )
            raise

        elapsed = time.perf_counter() - start
        classified = classify_mcp_tool_result(result)
        if classified is not None:
            signal, description = classified
            await self._log_tool_error(
                ToolReturnedError(description),
                tool_name=tool_name,
                server_name=server_name,
                signal=signal,
                elapsed=elapsed,
            )
        return result

    async def _log_tool_error(
        self,
        exc: Exception,
        *,
        tool_name: str,
        server_name: str,
        signal: str,
        elapsed: float,
    ) -> None:
        error_logger = get_error_logger()
        await error_logger.log_error(
            error=exc,
            description=(
                f"MCP tool '{tool_name}' on server '{server_name}' "
                f"{signal} after {elapsed:.2f}s"
            ),
            kind="tool_error",
            node="tools",
            source="mcp",
            logger_name="src.services.agents.core.interceptors",
            error_extra={
                "signal": signal,
                "tool": tool_name,
                "server": server_name,
            },
        )


# Backward-compatible alias
ObservabilityInterceptor = ErrorLoggingInterceptor
