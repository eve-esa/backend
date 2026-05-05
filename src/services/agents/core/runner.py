"""Agent runner — all backend integration logic for agent graph execution.

Handles: LLM instantiation, MCP tool loading, system prompt resolution,
conversation history, langfuse tracing, SSE streaming, persistence,
cancellation, error logging, token consumption, and stream bus publishing.
"""

import asyncio
import contextlib
import json
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks

from src.config import AGENTIC_LLM_TYPE, MODEL_TIMEOUT
from src.database.models.message import Message
from src.database.models.user import User
from src.services.generate_answer import (
    GenerationRequest,
    _get_conversation_history_from_db,
    _resolve_system_prompt,
    get_mcp_context,
    get_shared_llm_manager,
    maybe_rollup_and_trim_history,
    persist_message_state,
)
from src.services.agents.core.interceptors import ErrorLoggingInterceptor
from src.services.agents.core.registry import get_agent_graph
from src.services.agents.graphs.base import LatencyInterceptor
from src.services.mcp_auth import get_cognito_token_provider
from src.services.stream_bus import get_stream_bus
from src.services.token_rate_limiter import (
    consume_tokens_for_user,
    count_tokens_for_texts,
)
from src.utils.error_logger import Component, PipelineStage, get_error_logger
from src.utils.helpers import get_mongodb_uri
from src.utils.langfuse_helper import get_callbacks, langfuse_context

logger = logging.getLogger(__name__)

# ─── Optional imports ──────────────────────────────────────────────────────────

_langgraph_available = False
try:
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )
    from langchain_core.tools import StructuredTool
    from langgraph.checkpoint.mongodb import MongoDBSaver

    _langgraph_available = True
except Exception:
    AIMessage = HumanMessage = SystemMessage = ToolMessage = None  # type: ignore
    StructuredTool = None  # type: ignore
    MongoDBSaver = None  # type: ignore

try:
    from langgraph.checkpoint.memory import InMemorySaver
except Exception:
    InMemorySaver = None  # type: ignore

_mcp_adapters_available = False
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    _mcp_adapters_available = True
except Exception:
    MultiServerMCPClient = None  # type: ignore

try:
    from src.services.agents.graphs.utils import (
        SearchWileyInput,
        has_text_tool_call,
        parse_text_tool_calls,
        tool_call_label,
    )
except Exception:
    SearchWileyInput = None  # type: ignore
    tool_call_label = None  # type: ignore
    has_text_tool_call = None  # type: ignore
    parse_text_tool_calls = None  # type: ignore


# ─── Trace serialisation ──────────────────────────────────────────────────────


def _serialise_trace_entry(
    msg: Any, *, node: str = "", latency_s: Optional[float] = None
) -> Dict[str, Any]:
    """Convert a LangChain message into a JSON-serialisable trace dict."""
    entry: Dict[str, Any] = {"node": node}
    if latency_s is not None:
        entry["latency_s"] = latency_s

    if AIMessage and isinstance(msg, AIMessage):
        entry["role"] = "assistant"
        entry["content"] = (
            msg.content if isinstance(msg.content, str) else str(msg.content)
        )
        tc = getattr(msg, "tool_calls", None)
        if tc:
            entry["tool_calls"] = [
                {"name": c.get("name", ""), "args": c.get("args", {})} for c in tc
            ]
    elif ToolMessage and isinstance(msg, ToolMessage):
        entry["role"] = "tool"
        entry["name"] = getattr(msg, "name", "tool")
        entry["content"] = str(msg.content)
    elif HumanMessage and isinstance(msg, HumanMessage):
        entry["role"] = "user"
        entry["content"] = (
            msg.content if isinstance(msg.content, str) else str(msg.content)
        )
    elif SystemMessage and isinstance(msg, SystemMessage):
        entry["role"] = "system"
        entry["content"] = (
            msg.content if isinstance(msg.content, str) else str(msg.content)
        )
    else:
        entry["role"] = "unknown"
        entry["content"] = str(msg)
    return entry


# ─── MCP tool loader ──────────────────────────────────────────────────────────


async def _load_mcp_tools_for_servers(mcp_server_configs: List[Any]) -> List[Any]:
    """Connect to each MCP server, authenticate, and load its tools."""
    if not _mcp_adapters_available or not mcp_server_configs:
        return []

    token_provider = get_cognito_token_provider()
    auth_header: Optional[str] = None
    if token_provider:
        try:
            token = await token_provider.get_token()
            auth_header = f"Bearer {token}"
        except Exception as exc:
            logger.warning("Failed to obtain Cognito token for MCP auth: %s", exc)

    connections: Dict[str, Any] = {}
    for srv in mcp_server_configs:
        transport = (
            srv.config.transport.value if srv.config.transport else "streamable_http"
        )
        if transport not in ("streamable_http", "sse"):
            logger.warning(
                "Skipping MCP server %r: unsupported transport %r", srv.name, transport
            )
            continue

        headers: Dict[str, str] = dict(srv.config.headers or {})
        if auth_header and "Authorization" not in headers:
            headers["Authorization"] = auth_header

        connections[srv.name] = {
            "transport": "streamable_http" if transport == "streamable_http" else "sse",
            "url": srv.config.url,
            "headers": headers,
        }

    if not connections:
        return []

    try:
        client = MultiServerMCPClient(
            connections,
            tool_name_prefix=True,
            tool_interceptors=[LatencyInterceptor(), ErrorLoggingInterceptor()],
        )
        tools = await client.get_tools()
        logger.info(
            "Loaded %d MCP tool(s) from %d server(s): %s",
            len(tools),
            len(connections),
            [t.name for t in tools],
        )
        return tools
    except Exception as exc:
        logger.error("Failed to load MCP tools: %s", exc, exc_info=True)
        return []


# ─── Tool factory ─────────────────────────────────────────────────────────────


async def _build_tools(
    request: GenerationRequest,
    cancel_event: Optional[asyncio.Event] = None,
) -> List[Any]:
    """Return LangChain StructuredTools bound to the current request context."""
    if not _langgraph_available:
        return []

    tools: List[Any] = []

    if getattr(request, "mcp_server_configs", None):
        mcp_tools = await _load_mcp_tools_for_servers(request.mcp_server_configs)
        tools.extend(mcp_tools)

    if "Wiley AI Gateway" in (request.public_collections or []):

        async def _search_wiley_gateway(
            query: str,
            start_year: Optional[int] = None,
            end_year: Optional[int] = None,
        ) -> str:
            """Search the Wiley AI Gateway for peer-reviewed scientific articles."""
            try:
                temp = GenerationRequest(
                    query=query,
                    k=request.k,
                    score_threshold=request.score_threshold,
                    public_collections=["Wiley AI Gateway"],
                )
                temp.year = (
                    [start_year, end_year] if start_year and end_year else request.year
                )
                results, _ = await get_mcp_context(temp, cancel_event=cancel_event)
                if not results:
                    return "No articles found in the Wiley AI Gateway for this query."
                parts: List[str] = []
                for i, r in enumerate(results[: request.k]):
                    if isinstance(r, dict):
                        title = (
                            r.get("title")
                            or (r.get("metadata") or {}).get("title")
                            or f"Article {i + 1}"
                        )
                        text = (
                            r.get("text")
                            or r.get("page_content")
                            or r.get("content")
                            or ""
                        )
                        parts.append(f"[{i + 1}] {title}\n{str(text)[:800]}")
                return "\n\n".join(parts) if parts else "No articles found."
            except Exception as exc:
                logger.warning("Wiley MCP tool error: %s", exc)
                return f"Wiley Gateway search failed: {exc}"

        tools.append(
            StructuredTool.from_function(
                coroutine=_search_wiley_gateway,
                name="search_wiley_gateway",
                description=(
                    "Search the Wiley AI Gateway for peer-reviewed scientific articles. "
                    "Use when the query requires academic publications or research papers."
                ),
                args_schema=SearchWileyInput,
            )
        )

    return tools


# ─── Shared checkpointer ──────────────────────────────────────────────────────

_agentic_checkpointer: Optional[Any] = None
_agentic_checkpointer_lock = asyncio.Lock()


async def _get_agentic_checkpointer() -> Optional[Any]:
    global _agentic_checkpointer
    if _agentic_checkpointer is not None:
        return _agentic_checkpointer
    async with _agentic_checkpointer_lock:
        if _agentic_checkpointer is not None:
            return _agentic_checkpointer
        try:
            from pymongo import MongoClient

            _agentic_checkpointer = MongoDBSaver(MongoClient(get_mongodb_uri()))
            logger.info("Agentic agent using MongoDB checkpointer")
            return _agentic_checkpointer
        except Exception as exc:
            logger.warning(
                "MongoDB checkpointer unavailable, using in-memory: %s", exc
            )
            try:
                _agentic_checkpointer = InMemorySaver()
                return _agentic_checkpointer
            except Exception:
                return None


# ─── Resolve LLM ──────────────────────────────────────────────────────────────


def _get_llm(llm_type: Optional[str]):
    """Get the LLM instance, respecting AGENTIC_LLM_TYPE override."""
    effective_llm_type = AGENTIC_LLM_TYPE or llm_type
    if AGENTIC_LLM_TYPE and AGENTIC_LLM_TYPE != llm_type:
        logger.info(
            "Agentic graph: overriding llm_type %r -> %r (AGENTIC_LLM_TYPE)",
            llm_type,
            AGENTIC_LLM_TYPE,
        )
    return get_shared_llm_manager().get_client_for_model(effective_llm_type)


# ─── Resolve system prompt + conversation summary ─────────────────────────────


async def _resolve_full_system_prompt(
    llm_type: Optional[str], conversation_id: Optional[str]
) -> Optional[str]:
    """Build the system prompt with injected conversation summary."""
    system_prompt = _resolve_system_prompt(llm_type)

    if conversation_id:
        _, conversation_summary = await _get_conversation_history_from_db(
            conversation_id
        )
        if conversation_summary:
            summary_prefix = (
                f"Previous conversation summary:\n{conversation_summary}\n\n"
                "Please continue the conversation using this summary as context.\n\n"
            )
            system_prompt = (
                (summary_prefix + system_prompt) if system_prompt else summary_prefix
            )

    return system_prompt


# ─── Non-streaming generation ─────────────────────────────────────────────────


async def generate_answer_agentic(
    request: GenerationRequest,
    conversation_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> tuple[
    str,
    List[Dict[str, Any]],
    bool,
    Dict[str, Optional[float]],
    Dict[str, Any],
    List[Dict[str, Any]],
]:
    """Run the full agentic generation pipeline without streaming.

    Returns (answer, tool_results, use_rag, latencies, prompts, trace).
    """
    if not _langgraph_available:
        raise RuntimeError("LangGraph is not available — cannot run agentic generation")

    error_logger = get_error_logger()
    total_start = time.perf_counter()

    try:
        tools = await _build_tools(request)
        checkpointer = await _get_agentic_checkpointer()
        system_prompt = await _resolve_full_system_prompt(
            request.llm_type, conversation_id
        )
        llm = _get_llm(request.llm_type)

        from src.config import AGENT_GRAPH_TYPE

        agent = get_agent_graph(AGENT_GRAPH_TYPE)
        graph = agent.compile(
            llm=llm,
            tools=tools,
            system_prompt=system_prompt,
            checkpointer=checkpointer,
        )

        config = {
            "configurable": {"thread_id": conversation_id or "default"},
            "callbacks": get_callbacks(),
        }

        gen_start = time.perf_counter()
        trace_entries: List[Dict[str, Any]] = []
        all_messages: List[Any] = []
        node_latencies: Dict[str, float] = {}

        with langfuse_context(
            user_id=user_id,
            session_id=conversation_id,
            tags=["agentic", request.llm_type or "default"],
            trace_name="agentic_generation",
        ):
            async for update in graph.astream(
                {"messages": [HumanMessage(content=request.query)]},
                config=config,
                stream_mode="updates",
            ):
                step_time = time.perf_counter()
                for node_name, node_output in update.items():
                    step_latency_s = step_time - gen_start
                    node_latencies.setdefault(node_name, 0.0)

                    msgs = node_output.get("messages", [])
                    for msg in msgs:
                        entry = _serialise_trace_entry(
                            msg, node=node_name, latency_s=step_latency_s
                        )
                        trace_entries.append(entry)
                        all_messages.append(msg)

                    node_latencies[node_name] = step_latency_s

        gen_latency = time.perf_counter() - gen_start

        final_answer = ""
        for msg in reversed(all_messages):
            if (
                isinstance(msg, AIMessage)
                and msg.content
                and not getattr(msg, "tool_calls", None)
            ):
                final_answer = (
                    msg.content if isinstance(msg.content, str) else str(msg.content)
                )
                break

        tool_results: List[Dict[str, Any]] = []
        use_rag = False
        for msg in all_messages:
            if isinstance(msg, ToolMessage):
                use_rag = True
                tool_results.append(
                    {"tool": getattr(msg, "name", "tool"), "content": msg.content}
                )

        total_latency = time.perf_counter() - total_start
        latencies: Dict[str, Optional[float]] = {
            "generation_latency": gen_latency,
            "total_latency": total_latency,
            **{f"node_{k}_s": v for k, v in node_latencies.items()},
        }
        prompts: Dict[str, Any] = {
            "query": request.query,
            "system_prompt": system_prompt,
        }

        return final_answer, tool_results, use_rag, latencies, prompts, trace_entries

    except Exception as exc:
        await error_logger.log_error(
            error=exc,
            component=Component.LLM,
            pipeline_stage=PipelineStage.GENERATION,
            description="Agentic generation failed",
            error_type=type(exc).__name__,
        )
        raise


# ─── Streaming generation ─────────────────────────────────────────────────────


async def generate_answer_agentic_stream_helper(
    request: GenerationRequest,
    conversation_id: str,
    message_id: str,
    output_format: str = "json",
    background_tasks: Optional[BackgroundTasks] = None,
    cancel_event: Optional[asyncio.Event] = None,
    user_id: Optional[str] = None,
):
    """Stream agentic generation as SSE events.

    Event types emitted:
      tool_call   — agent invoked a tool (query shown)
      tool_result — tool returned (preview)
      token       — LLM final-answer token
      final       — complete answer + latencies
      stopped     — cancelled by client
      error       — unhandled exception
    """
    if not _langgraph_available:
        yield f"data: {json.dumps({'type': 'error', 'message': 'LangGraph not available'})}\n\n"
        return

    error_logger = get_error_logger()
    total_start = time.perf_counter()
    accumulated: List[str] = []

    def cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    try:
        if cancelled():
            await persist_message_state(message_id, stopped=True)
            yield f"data: {json.dumps({'type': 'stopped'})}\n\n"
            return

        tools = await _build_tools(request, cancel_event=cancel_event)
        checkpointer = await _get_agentic_checkpointer()
        system_prompt = await _resolve_full_system_prompt(
            request.llm_type, conversation_id
        )
        llm = _get_llm(request.llm_type)

        from src.config import AGENT_GRAPH_TYPE

        agent = get_agent_graph(AGENT_GRAPH_TYPE)
        graph = agent.compile(
            llm=llm,
            tools=tools,
            system_prompt=system_prompt,
            checkpointer=checkpointer,
        )

        config = {
            "configurable": {"thread_id": conversation_id},
            "callbacks": get_callbacks(),
        }

        gen_start = time.perf_counter()
        first_token_latency: Optional[float] = None
        tokens_yielded = 0
        use_rag = False
        trace_entries: List[Dict[str, Any]] = []
        node_start_time: float = gen_start
        node_latencies: Dict[str, float] = {}

        turn_buffer: List[str] = []
        current_node: Optional[str] = None

        def _flush_turn_buffer_to_events() -> List[str]:
            nonlocal tokens_yielded, first_token_latency
            if not turn_buffer:
                return []
            items = list(turn_buffer)
            joined = "".join(items)
            turn_buffer.clear()

            if has_text_tool_call and has_text_tool_call(joined):
                parsed = parse_text_tool_calls(joined) if parse_text_tool_calls else []
                if not parsed:
                    return []
                tc = parsed[0]
                tname = tc.get("name", "tool")
                args = tc.get("args", {})
                query_used = args.get("query", "")
                label = tool_call_label(tname) if tool_call_label else f"Calling {tname}"
                msg = f"{label}: {query_used}" if query_used else f"{label}…"
                return [
                    f"data: {json.dumps({'type': 'tool_call', 'content': msg})}\n\n"
                ]

            if tokens_yielded == 0:
                elapsed = time.perf_counter() - gen_start
                if elapsed > MODEL_TIMEOUT:
                    raise TimeoutError(
                        "No final-answer token received within MODEL_TIMEOUT"
                    )
                first_token_latency = time.perf_counter() - total_start

            events: List[str] = []
            for tok in items:
                if not tok:
                    continue
                tokens_yielded += 1
                accumulated.append(tok)
                if output_format == "json":
                    events.append(
                        f"data: {json.dumps({'type': 'token', 'content': tok})}\n\n"
                    )
                else:
                    events.append(f"data: {tok}\n\n")
            return events

        with langfuse_context(
            user_id=user_id,
            session_id=conversation_id,
            tags=["agentic", "stream", request.llm_type or "default"],
            trace_name="agentic_generation_stream",
        ):
            async for chunk, metadata in graph.astream(
                {"messages": [HumanMessage(content=request.query)]},
                config=config,
                stream_mode="messages",
            ):
                if cancelled():
                    await persist_message_state(
                        message_id, stopped=True, output="".join(accumulated)
                    )
                    yield f"data: {json.dumps({'type': 'stopped'})}\n\n"
                    return

                node = metadata.get("langgraph_node", "")
                if node != current_node:
                    if current_node:
                        elapsed_s = time.perf_counter() - node_start_time
                        node_latencies[current_node] = (
                            node_latencies.get(current_node, 0.0) + elapsed_s
                        )
                    for event in _flush_turn_buffer_to_events():
                        yield event
                    node_start_time = time.perf_counter()
                    current_node = node

                if ToolMessage and isinstance(chunk, ToolMessage):
                    use_rag = True
                    preview = str(chunk.content)[:200]
                    step_s = time.perf_counter() - node_start_time
                    trace_entries.append(
                        _serialise_trace_entry(chunk, node=node, latency_s=step_s)
                    )
                    yield f"data: {json.dumps({'type': 'tool_result', 'content': preview})}\n\n"
                    continue

                if AIMessage and isinstance(chunk, AIMessage):
                    if getattr(chunk, "tool_calls", None):
                        tc = chunk.tool_calls[0]
                        tname = (
                            tc.get("name", "tool")
                            if isinstance(tc, dict)
                            else getattr(tc, "name", "tool")
                        )
                        args = (
                            tc.get("args", {})
                            if isinstance(tc, dict)
                            else getattr(tc, "args", {})
                        )
                        query_used = args.get("query", "")
                        label = (
                            tool_call_label(tname)
                            if tool_call_label
                            else f"Calling {tname}"
                        )
                        msg = f"{label}: {query_used}" if query_used else f"{label}…"
                        step_s = time.perf_counter() - node_start_time
                        trace_entries.append(
                            _serialise_trace_entry(chunk, node=node, latency_s=step_s)
                        )
                        yield f"data: {json.dumps({'type': 'tool_call', 'content': msg})}\n\n"
                        continue

                    content = chunk.content
                    if isinstance(content, list):
                        content = "".join(
                            c.get("text", "") if isinstance(c, dict) else str(c)
                            for c in content
                        )
                    if not content:
                        continue

                    turn_buffer.append(content)

            if current_node:
                elapsed_s = time.perf_counter() - node_start_time
                node_latencies[current_node] = (
                    node_latencies.get(current_node, 0.0) + elapsed_s
                )
            for event in _flush_turn_buffer_to_events():
                yield event

        gen_latency = time.perf_counter() - gen_start
        answer = "".join(accumulated)
        total_latency = time.perf_counter() - total_start

        latencies: Dict[str, Optional[float]] = {
            "first_token_latency": first_token_latency,
            "generation_latency": gen_latency,
            "total_latency": total_latency,
            **{f"node_{k}_s": v for k, v in node_latencies.items()},
        }

        if answer:
            agent_s = node_latencies.get("agent", gen_latency)
            trace_entries.append(
                {
                    "role": "assistant",
                    "node": "agent",
                    "content": answer,
                    "latency_s": agent_s,
                }
            )

        await persist_message_state(
            message_id,
            output=answer,
            use_rag=use_rag,
            latencies=latencies,
            prompts={"query": request.query, "system_prompt": system_prompt},
            trace=trace_entries if trace_entries else None,
        )

        if background_tasks:
            background_tasks.add_task(maybe_rollup_and_trim_history, conversation_id)
        else:
            asyncio.create_task(maybe_rollup_and_trim_history(conversation_id))

        if output_format == "json":
            yield f"data: {json.dumps({'type': 'final', 'answer': answer, 'latencies': latencies})}\n\n"
        else:
            yield "data: [DONE]\n\n"

    except asyncio.CancelledError:
        logger.info("Agentic generation cancelled")
        await persist_message_state(
            message_id, output="".join(accumulated), stopped=True
        )
        return

    except TimeoutError as exc:
        logger.warning("Agentic generation timed out: %s", exc)
        await error_logger.log_error(
            error=exc,
            component=Component.LLM,
            pipeline_stage=PipelineStage.GENERATION,
            description="Agentic generation timed out",
            error_type=type(exc).__name__,
        )
        answer = "".join(accumulated)
        if answer:
            await persist_message_state(message_id, output=answer)
            if output_format == "json":
                yield f"data: {json.dumps({'type': 'final', 'answer': answer, 'latencies': {}})}\n\n"
            else:
                yield "data: [DONE]\n\n"
        else:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Generation timed out'})}\n\n"

    except Exception as exc:
        logger.error("Agentic streaming error: %s", exc)
        await error_logger.log_error(
            error=exc,
            component=Component.LLM,
            pipeline_stage=PipelineStage.GENERATION,
            description="Agentic streaming error",
            error_type=type(exc).__name__,
        )
        with contextlib.suppress(Exception):
            await persist_message_state(message_id, output="".join(accumulated))
        yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"


# ─── SSE wrappers ─────────────────────────────────────────────────────────────


async def generate_answer_agentic_stream(
    request: GenerationRequest,
    conversation_id: str,
    message_id: str,
    background_tasks: Optional[BackgroundTasks] = None,
    cancel_event: Optional[asyncio.Event] = None,
    user_id: Optional[str] = None,
):
    """Plain-text SSE wrapper around the agentic stream helper."""
    async for chunk in generate_answer_agentic_stream_helper(
        request,
        conversation_id,
        message_id,
        "plain",
        background_tasks,
        cancel_event,
        user_id,
    ):
        yield chunk


async def generate_answer_agentic_json_stream(
    request: GenerationRequest,
    conversation_id: str,
    message_id: str,
    background_tasks: Optional[BackgroundTasks] = None,
    cancel_event: Optional[asyncio.Event] = None,
    user_id: Optional[str] = None,
):
    """JSON SSE wrapper around the agentic stream helper."""
    async for chunk in generate_answer_agentic_stream_helper(
        request,
        conversation_id,
        message_id,
        "json",
        background_tasks,
        cancel_event,
        user_id,
    ):
        yield chunk


# ─── Bus-decoupled entry point ────────────────────────────────────────────────


async def run_agentic_generation_to_bus(
    request: GenerationRequest,
    conversation_id: str,
    message_id: str,
    background_tasks: Optional[BackgroundTasks] = None,
    cancel_event: Optional[asyncio.Event] = None,
    user_id: Optional[str] = None,
):
    """Run agentic generation in background, publishing chunks to stream bus."""
    bus = get_stream_bus()
    try:
        async for chunk in generate_answer_agentic_json_stream(
            request=request,
            conversation_id=conversation_id,
            message_id=message_id,
            background_tasks=background_tasks,
            cancel_event=cancel_event,
            user_id=user_id,
        ):
            await bus.publish(message_id, chunk)
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        await bus.publish(
            message_id,
            f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n",
        )
    finally:
        if user_id:
            try:
                user = await User.find_by_id(user_id)
                message = await Message.find_by_id(message_id)
                if user and message:
                    token_count = count_tokens_for_texts(message.input, message.output)
                    await consume_tokens_for_user(user, token_count)
            except Exception as consume_error:
                logger.warning(
                    "Failed to apply token usage for agentic generation: %s",
                    consume_error,
                )
        await bus.close(message_id)
        with contextlib.suppress(Exception):
            from src.services.cancel_manager import get_cancel_manager

            cm = get_cancel_manager()
            cm.clear_mapping_for(conversation_id, message_id)
            cm.clear(message_id)
