"""MCP tool-call interceptor for backend error logging.

Persists tool-call failures to MongoDB via :class:`ErrorLogger`.
Designed to compose with :class:`LatencyInterceptor` from
``src.services.agents.graphs.base`` which handles latency tracking independently.

Reference:
    https://github.com/langchain-ai/langchain-mcp-adapters (interceptors module)
"""

import time
from typing import Any

from src.utils.error_logger import Component, PipelineStage, get_error_logger

try:
    from langchain_mcp_adapters.interceptors import (
        MCPToolCallRequest,
        MCPToolCallResult,
    )

    _interceptors_available = True
except Exception:
    _interceptors_available = False


class ErrorLoggingInterceptor:
    """Persists MCP tool-call failures to the backend error log (MongoDB).

    Follows the ``ToolCallInterceptor`` protocol from ``langchain-mcp-adapters``.
    On success the call passes through transparently.  On failure the error is
    logged to MongoDB via :func:`get_error_logger` and then re-raised.

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
        tool_name = request.name
        server_name = request.server_name
        start = time.perf_counter()

        try:
            return await handler(request)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            error_logger = get_error_logger()
            await error_logger.log_error(
                error=exc,
                component=Component.MCP_TOOL,
                pipeline_stage=PipelineStage.TOOL_EXECUTION,
                description=(
                    f"MCP tool '{tool_name}' on server '{server_name}' "
                    f"failed after {elapsed:.2f}s"
                ),
                error_type=type(exc).__name__,
                logger_name="src.services.agents.core.interceptors",
            )
            raise


# Backward-compatible alias
ObservabilityInterceptor = ErrorLoggingInterceptor
