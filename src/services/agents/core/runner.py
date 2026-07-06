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

from src.config import AGENTIC_LLM_TYPE, AGENTIC_TIMEOUT, MODEL_TIMEOUT
from src.core.llm_manager import LLMType
from src.database.models.message import Message
from src.database.models.user import User
from src.services.custom_model_secrets import get_custom_model_api_key
from src.services.custom_model_service import get_owned_custom_model
from src.services.generate_answer import (
    GenerationRequest,
    _get_conversation_history_from_db,
    get_shared_llm_manager,
    maybe_rollup_and_trim_history,
    persist_message_state,
)
from src.services.agents.core.interceptors import ErrorLoggingInterceptor
from src.services.agents.core.registry import get_agent_graph
from src.services.agents.graphs_bundle import graphs_base_module, graphs_utils_module
from src.services.mcp.proxy_url import backend_mcp_proxy_url
from src.services.mcp_auth import get_cognito_token_provider
from src.services.stream_bus import get_stream_bus
from src.services.token_rate_limiter import (
    consume_tokens_for_user,
    count_tokens_for_texts,
)
from src.utils.error_logger import Component, PipelineStage, get_error_logger
from src.utils.helpers import get_mongodb_uri
from src.utils.langfuse_helper import get_callbacks, langfuse_context

LatencyInterceptor = graphs_base_module().LatencyInterceptor

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
    from langgraph.checkpoint.mongodb import MongoDBSaver

    _langgraph_available = True
except Exception:
    AIMessage = HumanMessage = SystemMessage = ToolMessage = None  # type: ignore
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
    _graphs_utils = graphs_utils_module()
    has_text_tool_call = _graphs_utils.has_text_tool_call
    might_be_incomplete_text_tool_call = (
        _graphs_utils.might_be_incomplete_text_tool_call
    )
    parse_text_tool_calls = _graphs_utils.parse_text_tool_calls
    tool_call_label = _graphs_utils.tool_call_label
except Exception:
    tool_call_label = None  # type: ignore
    has_text_tool_call = None  # type: ignore
    might_be_incomplete_text_tool_call = None  # type: ignore
    parse_text_tool_calls = None  # type: ignore


# ─── Trace serialisation ──────────────────────────────────────────────────────


def _coerce_trace_metadata(value: Any) -> Optional[Dict[str, Any]]:
    """Normalise LangChain metadata objects into a plain dict for persistence."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value or None
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(exclude_none=True)
        return dumped or None
    return None


def _enrich_trace_entry_with_message_metadata(
    entry: Dict[str, Any], msg: Any
) -> None:
    """Attach provider/model metadata and correlation ids to a trace entry."""
    msg_id = getattr(msg, "id", None)
    if msg_id:
        entry["id"] = msg_id

    response_metadata = _coerce_trace_metadata(
        getattr(msg, "response_metadata", None)
    )
    if response_metadata:
        entry["response_metadata"] = response_metadata

    usage_metadata = _coerce_trace_metadata(getattr(msg, "usage_metadata", None))
    if usage_metadata:
        entry["usage_metadata"] = usage_metadata

    if ToolMessage and isinstance(msg, ToolMessage):
        tool_call_id = getattr(msg, "tool_call_id", None)
        if tool_call_id:
            entry["tool_call_id"] = tool_call_id
        status = getattr(msg, "status", None)
        if status:
            entry["status"] = status

    if AIMessage and isinstance(msg, AIMessage):
        invalid_tool_calls = getattr(msg, "invalid_tool_calls", None)
        if invalid_tool_calls:
            entry["invalid_tool_calls"] = invalid_tool_calls


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
            tool_calls: List[Dict[str, Any]] = []
            for c in tc:
                tc_entry: Dict[str, Any] = {
                    "name": c.get("name", ""),
                    "args": c.get("args", {}),
                }
                if c.get("id"):
                    tc_entry["id"] = c["id"]
                tool_calls.append(tc_entry)
            entry["tool_calls"] = tool_calls
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

    _enrich_trace_entry_with_message_metadata(entry, msg)
    return entry


# ─── MCP tool loader ──────────────────────────────────────────────────────────


async def _load_mcp_tools_for_servers(
    mcp_server_configs: List[Any],
    *,
    mcp_proxy_bearer_token: Optional[str] = None,
) -> List[Any]:
    """Connect to each MCP server, authenticate, and load its tools.

    When ``mcp_proxy_bearer_token`` is set and the proxy base URL is configured,
    routes through ``/mcp/{name}`` so listing shares the proxy cache and Cognito
    egress auth. Otherwise falls back to a direct AgentCore connection.
    """
    if not _mcp_adapters_available or not mcp_server_configs:
        return []

    token_provider = get_cognito_token_provider()
    cognito_auth_header: Optional[str] = None
    if token_provider:
        try:
            token = await token_provider.get_token()
            cognito_auth_header = f"Bearer {token}"
        except Exception as exc:
            logger.warning("Failed to obtain Cognito token for MCP auth: %s", exc)

    connections: Dict[str, Any] = {}
    for srv in mcp_server_configs:
        transport = (
            srv.config.transport.value if srv.config.transport else "streamable_http"
        )
        if transport not in ("streamable_http", "sse"):
            raise ValueError(
                f"MCP server {srv.name!r} uses unsupported transport {transport!r}. "
                "Only 'streamable_http' and 'sse' are supported."
            )

        if not srv.config.url:
            logger.warning(
                "Skipping MCP server %r: missing URL in config", srv.name
            )
            continue

        headers: Dict[str, str] = dict(srv.config.headers or {})
        proxy_http_url = (
            backend_mcp_proxy_url(srv.name) if mcp_proxy_bearer_token else None
        )
        if proxy_http_url:
            headers["Authorization"] = f"Bearer {mcp_proxy_bearer_token}"
            url = proxy_http_url
        else:
            if cognito_auth_header and "Authorization" not in headers:
                headers["Authorization"] = cognito_auth_header
            url = srv.config.url

        connections[srv.name] = {
            "transport": "streamable_http" if transport == "streamable_http" else "sse",
            "url": url,
            "headers": headers,
        }

    if not connections:
        return []

    client = MultiServerMCPClient(
        connections,
        tool_name_prefix=True,
        tool_interceptors=[LatencyInterceptor(), ErrorLoggingInterceptor()],
    )
    tools: List[Any] = []
    failed_servers: List[str] = []
    for server_name in connections:
        try:
            server_tools = await client.get_tools(server_name=server_name)
            tools.extend(server_tools)
            logger.info(
                "Loaded %d MCP tool(s) from server %r: %s",
                len(server_tools),
                server_name,
                [t.name for t in server_tools],
            )
        except Exception as exc:
            failed_servers.append(server_name)
            logger.error(
                "Failed to load MCP tools from server %r: %s",
                server_name,
                exc,
                exc_info=True,
            )

    if tools:
        logger.info(
            "Loaded %d MCP tool(s) total from %d/%d MCP server(s)",
            len(tools),
            len(connections) - len(failed_servers),
            len(connections),
        )
    elif failed_servers:
        logger.warning(
            "No MCP tools loaded; all %d configured server(s) failed: %s",
            len(failed_servers),
            failed_servers,
        )
    return tools


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
        try:
            mcp_tools = await _load_mcp_tools_for_servers(
                request.mcp_server_configs,
                mcp_proxy_bearer_token=request.mcp_proxy_bearer_token,
            )
            tools.extend(mcp_tools)
        except Exception:
            logger.error("MCP tool loading failed; proceeding without MCP tools", exc_info=True)

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


def _resolve_agentic_llm_type(
    llm_type: Optional[str], *, override: Optional[str] = None
) -> Optional[str]:
    """Resolve the effective LLM type for agentic generation.

    Precedence: explicit ``override`` (e.g. forcing the Fallback model) wins,
    then the request's ``llm_type``, then the ``AGENTIC_LLM_TYPE`` env default.
    Centralising this here keeps resolution consistent for every caller of
    :func:`_build_react_graph`.
    """
    if override is not None:
        return override
    if llm_type and llm_type != AGENTIC_LLM_TYPE:
        logger.info(
            "Agentic graph: overriding AGENTIC_LLM_TYPE %r -> %r llm_type",
            AGENTIC_LLM_TYPE,
            llm_type,
        )
    return llm_type or AGENTIC_LLM_TYPE


async def _resolve_agentic_llm_client(
    request: GenerationRequest,
    *,
    user_id: Optional[str],
    llm_type_override: Optional[str] = None,
) -> tuple[Any, Dict[str, Any]]:
    """Resolve the LLM client and metadata for agentic generation."""
    if request.custom_model_id:
        if not user_id:
            raise ValueError("custom_model_id requires an authenticated user")
        model = await get_owned_custom_model(request.custom_model_id, user_id)
        api_key = await get_custom_model_api_key(model.secret_arn)
        llm = get_shared_llm_manager().build_custom_client(
            base_url=model.base_url,
            model_name=model.model_name,
            api_key=api_key,
        )
        return llm, {
            "custom_model_id": model.id,
            "custom_model_display_name": model.display_name,
            "custom_model_name": model.model_name,
        }

    effective_type = _resolve_agentic_llm_type(
        request.llm_type, override=llm_type_override
    )
    llm = get_shared_llm_manager().get_client_for_model(effective_type)
    return llm, {"agentic_llm_resolved": effective_type}


def _resolve_agent_graph_type(request: GenerationRequest) -> str:
    """Return request-selected graph type, else AGENT_GRAPH_TYPE from env."""
    from src.config import AGENT_GRAPH_TYPE

    raw = getattr(request, "agent", None)
    if raw is None:
        return AGENT_GRAPH_TYPE

    s = str(raw).strip().lstrip("\ufeff").strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    return s or AGENT_GRAPH_TYPE


# ─── Graph compile helpers ────────────────────────────────────────────────────


def _build_react_graph(
    llm_type: Optional[str],
    tools: List[Any],
    checkpointer: Any,
    *,
    agent: Any,
    history: Optional[List[Any]] = None,
    summary: Optional[str] = None,
    llm_type_override: Optional[str] = None,
    fallback_llm: Any = None,
    llm: Any = None,
) -> Any:
    """Compile the agent graph using the resolved LLM type.

    ``llm_type_override`` forces a specific model (e.g. ``LLMType.Fallback.value``)
    regardless of the request's ``llm_type``.  ``fallback_llm`` is forwarded to
    ``agent.compile()`` for in-graph node-level fallback.
    When ``llm`` is provided it is used directly (e.g. user custom models).
    """
    if llm is None:
        effective_type = _resolve_agentic_llm_type(llm_type, override=llm_type_override)
        llm = get_shared_llm_manager().get_client_for_model(effective_type)
    if fallback_llm is None:
        fallback_llm = get_shared_llm_manager().get_client_for_model(
            LLMType.Fallback.value
        )
    return agent.compile(
        llm=llm,
        tools=tools,
        checkpointer=checkpointer,
        history=history,
        summary=summary,
        fallback_llm=fallback_llm,
        llm_run_timeout=MODEL_TIMEOUT,
    )


# ─── Conversation context ─────────────────────────────────────────────────────


async def _fetch_conversation_context(
    conversation_id: Optional[str],
) -> tuple[List[Any], Optional[str]]:
    """Return ``(history, summary)`` for the given conversation.

    *history* is a list of LangChain messages (HumanMessage / AIMessage) from the
    most-recent turn stored in the DB; *summary* is the rolling summary string if
    one exists.  Both are ``[]`` / ``None`` when there is no conversation yet.
    The agent is responsible for deciding how to integrate them via
    :meth:`~AgentGraph.format_history` and :meth:`~AgentGraph.instruction_text`.
    """
    if not conversation_id:
        return [], None
    history, summary = await _get_conversation_history_from_db(conversation_id)
    return history, summary


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
        history, summary = await _fetch_conversation_context(conversation_id)

        agent_graph_type = _resolve_agent_graph_type(request)
        agent = get_agent_graph(agent_graph_type)
        resolved_instruction = agent.instruction_text(history=history, summary=summary)
        llm, llm_prompts = await _resolve_agentic_llm_client(
            request, user_id=user_id
        )
        graph = _build_react_graph(
            request.llm_type,
            tools,
            checkpointer,
            agent=agent,
            history=history,
            summary=summary,
            llm=llm,
        )

        config = {
            "configurable": {"thread_id": conversation_id or "default"},
            "callbacks": get_callbacks(),
        }

        async def _run_graph(
            g: Any,
        ) -> tuple[List[Any], List[Dict[str, Any]], Dict[str, float], float, bool]:
            """Stream the graph and collect results.

            Returns ``(raw_messages, trace_entries, node_latencies, duration,
            in_graph_fallback)`` where *in_graph_fallback* is ``True`` when the
            graph's node-level ``error_handler`` switched to the fallback model
            mid-run (signalled via the ``use_fallback_llm`` state flag).
            """
            raw_msgs: List[Any] = []
            trace: List[Dict[str, Any]] = []
            latency_map: Dict[str, float] = {}
            in_graph_fallback = False
            start = time.perf_counter()
            with langfuse_context(
                user_id=user_id,
                session_id=conversation_id,
                tags=[
                    "agentic",
                    request.custom_model_id or request.llm_type or "default",
                ],
                trace_name="agentic_generation",
            ):
                async for update in g.astream(
                    {"messages": [HumanMessage(content=request.query)]},
                    config=config,
                    stream_mode="updates",
                ):
                    step_time = time.perf_counter()
                    for node_name, node_output in update.items():
                        if node_name == "agent_fallback":
                            in_graph_fallback = True
                        step_latency_s = step_time - start
                        latency_map.setdefault(node_name, 0.0)
                        msgs = (
                            node_output.get("messages", [])
                            if isinstance(node_output, dict)
                            else []
                        )
                        for msg in msgs:
                            trace.append(
                                _serialise_trace_entry(
                                    msg, node=node_name, latency_s=step_latency_s
                                )
                            )
                            raw_msgs.append(msg)
                        latency_map[node_name] = step_latency_s
            return raw_msgs, trace, latency_map, time.perf_counter() - start, in_graph_fallback

        gen_start = time.perf_counter()
        (
            all_messages,
            trace_entries,
            node_latencies,
            gen_latency,
            used_fallback_llm,
        ) = await _run_graph(graph)

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
            "instruction": resolved_instruction,
            "agent_prompts": dict(agent.prompts),
            "agent_graph_type": agent_graph_type,
            "used_fallback_llm": used_fallback_llm,
            **llm_prompts,
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
    used_fallback_llm = False

    def cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    try:
        if cancelled():
            await persist_message_state(message_id, stopped=True)
            yield f"data: {json.dumps({'type': 'stopped'})}\n\n"
            return

        tools = await _build_tools(request, cancel_event=cancel_event)
        checkpointer = await _get_agentic_checkpointer()
        history, summary = await _fetch_conversation_context(conversation_id)

        agent_graph_type = _resolve_agent_graph_type(request)
        agent = get_agent_graph(agent_graph_type)
        resolved_instruction = agent.instruction_text(history=history, summary=summary)
        llm, llm_prompts = await _resolve_agentic_llm_client(
            request, user_id=user_id
        )

        graph = _build_react_graph(
            request.llm_type,
            tools,
            checkpointer,
            agent=agent,
            history=history,
            summary=summary,
            llm=llm,
        )

        config = {
            "configurable": {"thread_id": conversation_id},
            "callbacks": get_callbacks(),
        }

        gen_start = time.perf_counter()
        first_token_latency: Optional[float] = None
        tokens_yielded = 0
        use_rag = False
        in_graph_fallback_used = False
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
                if elapsed > AGENTIC_TIMEOUT:
                    raise TimeoutError(
                        "No final-answer token received within AGENTIC_TIMEOUT"
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
            tags=[
                "agentic",
                "stream",
                request.custom_model_id or request.llm_type or "default",
            ],
            trace_name="agentic_generation_stream",
        ):
            async for mode, payload in graph.astream(
                {"messages": [HumanMessage(content=request.query)]},
                config=config,
                stream_mode=["messages", "updates"],
            ):
                if cancelled():
                    await persist_message_state(
                        message_id, stopped=True, output="".join(accumulated)
                    )
                    yield f"data: {json.dumps({'type': 'stopped'})}\n\n"
                    return

                if mode == "updates":
                    if "agent_fallback" in payload:
                        in_graph_fallback_used = True
                    continue

                chunk, metadata = payload
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
                    joined = "".join(turn_buffer)
                    if might_be_incomplete_text_tool_call and (
                        might_be_incomplete_text_tool_call(joined)
                    ):
                        continue
                    for event in _flush_turn_buffer_to_events():
                        yield event

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
            answer_node = current_node or "agent"
            agent_s = node_latencies.get(answer_node, gen_latency)
            trace_entries.append(
                {
                    "role": "assistant",
                    "node": answer_node,
                    "content": answer,
                    "latency_s": agent_s,
                }
            )

        await persist_message_state(
            message_id,
            output=answer,
            use_rag=use_rag,
            latencies=latencies,
            prompts={
                "query": request.query,
                "instruction": resolved_instruction,
                "agent_prompts": dict(agent.prompts),
                "agent_graph_type": agent_graph_type,
                "used_fallback_llm": used_fallback_llm or in_graph_fallback_used,
                **llm_prompts,
            },
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
