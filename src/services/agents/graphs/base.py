"""AgentGraph base class — the contract for pluggable agent graphs.

Only depends on langchain-core + langgraph + standard library.
No backend (src.*) imports. Safe to use in standalone scripts/notebooks.
"""

import logging
import time
from typing import Any, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import MessagesState
from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


# ─── Standalone MCP interceptor (no backend dependencies) ─────────────────────


class LatencyInterceptor:
    """MCP tool-call interceptor that tracks latency via standard logging.

    Follows the ``ToolCallInterceptor`` protocol from ``langchain-mcp-adapters``.
    No backend dependencies — safe to use in standalone scripts/notebooks.

    Usage with ``MultiServerMCPClient``::

        client = MultiServerMCPClient(
            connections,
            tool_interceptors=[LatencyInterceptor()],
        )
    """

    async def __call__(self, request: Any, handler: Any) -> Any:
        tool_name = getattr(request, "name", "unknown")
        server_name = getattr(request, "server_name", "unknown")
        start = time.perf_counter()

        try:
            result = await handler(request)
            elapsed = time.perf_counter() - start
            logger.info(
                "MCP tool %s (server=%s) completed in %.3fs",
                tool_name,
                server_name,
                elapsed,
            )
            return result
        except Exception as exc:
            elapsed = time.perf_counter() - start
            logger.error(
                "MCP tool %s (server=%s) failed after %.3fs: %s",
                tool_name,
                server_name,
                elapsed,
                exc,
            )
            raise


# ─── AgentGraph base class ────────────────────────────────────────────────────


class AgentGraph:
    """Base class for pluggable agent graphs.

    Inherit and implement ``compile()``.  Use the helper methods for free
    node/tool latency tracking and error handling.

    Only depends on langchain-core + langgraph.  No backend imports.
    """

    name: str = "base"

    def compile(
        self,
        *,
        llm: BaseChatModel,
        tools: List[BaseTool],
        system_prompt: Optional[str],
        checkpointer: Any,
    ) -> CompiledStateGraph:
        """Build and return the compiled StateGraph.  Override in subclass."""
        raise NotImplementedError

    # ── Helper: instrumented tools node ────────────────────────────────────────

    def make_tools_node(self, tools: List[BaseTool]):
        """Create a tools node with per-tool latency tracking and error handling.

        Usage::

            builder.add_node("tools", self.make_tools_node(tools))
        """
        tool_map = {t.name: t for t in tools}

        async def tools_node(state: MessagesState):
            last = state["messages"][-1]
            results: List[ToolMessage] = []
            for tc in getattr(last, "tool_calls", []):
                name, args, call_id = tc["name"], tc["args"], tc["id"]
                tool = tool_map.get(name)
                start = time.perf_counter()
                if tool is None:
                    result = f"Unknown tool: {name}"
                    logger.warning("Tool %s not found in tool_map", name)
                else:
                    try:
                        result = await tool.ainvoke(args)
                        elapsed = time.perf_counter() - start
                        logger.info(
                            "Tool %s completed in %.3fs (%d chars)",
                            name,
                            elapsed,
                            len(str(result)),
                        )
                    except Exception as exc:
                        elapsed = time.perf_counter() - start
                        logger.error(
                            "Tool %s failed after %.3fs: %s", name, elapsed, exc
                        )
                        result = f"Tool error: {exc}"
                results.append(
                    ToolMessage(content=str(result), tool_call_id=call_id, name=name)
                )
            return {"messages": results}

        return tools_node

    # ── Helper: timed node wrapper ─────────────────────────────────────────────

    def timed_node(self, node_name: str, fn):
        """Wrap any async node function with latency logging.

        Usage::

            builder.add_node("agent", self.timed_node("agent", my_agent_fn))
        """

        async def wrapper(state):
            start = time.perf_counter()
            result = await fn(state)
            elapsed = time.perf_counter() - start
            logger.info("Node '%s' completed in %.3fs", node_name, elapsed)
            return result

        wrapper.__name__ = node_name
        return wrapper
