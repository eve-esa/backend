"""ReAct agent graph — manual tool-calling loop via LangGraph StateGraph.

Uses shared utilities from ``src.services.agents.graphs.utils`` for text-format
tool-call parsing and message sanitisation.  No other backend imports
beyond the base class.
"""

import logging
from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph

from src.services.agents.graphs.base import AgentGraph
from src.services.agents.graphs.utils import (
    parse_text_tool_calls,
    reformat_messages_for_text_tool_model,
    strip_content_from_tool_call_messages,
    tiktoken_counter,
)

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS = 16_384


class ReactAgent(AgentGraph):
    """Manual ReAct loop: agent -> tools -> agent, with text-format fallback.

    Supports models with native function calling (OpenAI-style) and models
    that emit tool calls as text (Mistral/EVE-Instruct ``[TOOL_CALLS]`` format).
    """

    name = "react"

    def compile(
        self,
        *,
        llm,
        tools: List[BaseTool],
        system_prompt: Optional[str],
        checkpointer: Any,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        **kwargs,
    ):
        llm_with_tools = llm.bind_tools(tools) if tools else llm

        # ── agent node ─────────────────────────────────────────────────────
        async def agent_fn(state: MessagesState):
            messages = list(state["messages"])
            if system_prompt:
                messages = [SystemMessage(content=system_prompt)] + messages

            if trim_messages is not None:
                messages = trim_messages(
                    messages,
                    max_tokens=max_tokens,
                    strategy="last",
                    token_counter=tiktoken_counter,
                    include_system=True,
                    start_on="human",
                    end_on=("human", "tool"),
                )

            messages = strip_content_from_tool_call_messages(messages)

            has_synthetic = any(
                (
                    isinstance(m, AIMessage)
                    and not m.content
                    and getattr(m, "tool_calls", None)
                )
                or isinstance(m, ToolMessage)
                for m in messages
            )
            if has_synthetic:
                messages = reformat_messages_for_text_tool_model(messages)

            response = await llm_with_tools.ainvoke(messages)

            if not getattr(response, "tool_calls", None) and isinstance(
                response.content, str
            ):
                parsed = parse_text_tool_calls(response.content)
                if parsed:
                    logger.info(
                        "Parsed %d text-format tool call(s) from model response",
                        len(parsed),
                    )
                    response = AIMessage(
                        content="",
                        tool_calls=parsed,
                        id=getattr(response, "id", None),
                    )

            return {"messages": [response]}

        # ── routing ────────────────────────────────────────────────────────
        def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
            last = state["messages"][-1]
            if getattr(last, "tool_calls", None):
                return "tools"
            return END

        # ── build graph ────────────────────────────────────────────────────
        builder = StateGraph(MessagesState)
        builder.add_node("agent", self.timed_node("agent", agent_fn))
        builder.add_node("tools", self.make_tools_node(tools))
        builder.add_edge(START, "agent")
        builder.add_conditional_edges(
            "agent", should_continue, {"tools": "tools", END: END}
        )
        builder.add_edge("tools", "agent")
        return builder.compile(checkpointer=checkpointer)
