"""MCP tool-call interceptor for latency and error tracking.

Wraps every MCP tool invocation with timing, structured error logging (via the
existing :class:`ErrorLogger` infrastructure), and debug-level logging of
arguments / results.

Reference:
    https://github.com/langchain-ai/langchain-mcp-adapters (interceptors module)
"""

import logging
import time
from typing import Any

from src.utils.error_logger import Component, PipelineStage, get_error_logger

logger = logging.getLogger(__name__)

try:
    from langchain_mcp_adapters.interceptors import (
        MCPToolCallRequest,
        MCPToolCallResult,
    )

    _interceptors_available = True
except Exception:
    _interceptors_available = False


class ObservabilityInterceptor:
    """Captures latency and errors for every MCP tool call.

    Follows the ``ToolCallInterceptor`` protocol from ``langchain-mcp-adapters``:
    the ``__call__`` method wraps the *handler* with timing and error handling.

    Logged to MongoDB via :func:`get_error_logger` on failure, and always emits
    structured ``logger.info`` for latency tracking.
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
            result = await handler(request)
            elapsed = time.perf_counter() - start
            logger.info(
                "MCP tool %s (server=%s) completed in %.10fs",
                tool_name,
                server_name,
                elapsed,
            )
            return result

        except Exception as exc:
            elapsed = time.perf_counter() - start
            logger.error(
                "MCP tool %s (server=%s) failed after %.10fs: %s",
                tool_name,
                server_name,
                elapsed,
                exc,
            )
            error_logger = get_error_logger()
            await error_logger.log_error(
                error=exc,
                component=Component.MCP_TOOL,
                pipeline_stage=PipelineStage.TOOL_EXECUTION,
                description=f"MCP tool '{tool_name}' on server '{server_name}' failed after {elapsed:.2f}s",
                error_type=type(exc).__name__,
                logger_name="src.services.mcp.tool_interceptor",
            )
            raise
