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
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import BackgroundTasks

from src.config import AGENTIC_LLM_TYPE, AGENTIC_TIMEOUT, MODEL_TIMEOUT
from src.core.llm_health import is_endpoint_failure
from src.core.llm_manager import LLMType, endpoint_timeout
from src.database.models.message import Message
from src.database.models.user import User
from src.services.custom_model_service import (
    build_custom_model_llm,
    custom_model_metadata,
    ensure_custom_model_has_credentials,
    get_owned_custom_model,
)
from src.schemas.generation_request import GenerationRequest
from src.services.generate_answer import (
    _get_conversation_history_from_db,
    build_endpoint_metadata,
    build_error_payload,
    endpoint_answered_by,
    get_shared_llm_manager,
    maybe_rollup_and_trim_history,
    persist_message_state,
)
from src.services.agents.core.registry import get_agent_graph
from src.services.mcp.artifact_context import (
    reset_artifact_context,
    set_artifact_context,
)
from src.services.mcp.tool_loader import (
    _MCPToolsWithClient,
    load_mcp_tools_for_servers as _load_mcp_tools_for_servers,
)
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
    from langgraph.checkpoint.mongodb import MongoDBSaver

    _langgraph_available = True
except Exception:
    AIMessage = HumanMessage = SystemMessage = ToolMessage = None  # type: ignore
    MongoDBSaver = None  # type: ignore

try:
    from langgraph.checkpoint.memory import InMemorySaver
except Exception:
    InMemorySaver = None  # type: ignore

# Text-format tool-call helpers come from this backend's OWN
# `src.services.agentic_utils` rather than the external `graphs_utils_module()`
# package: the installed `eve-esa-agents` package's `utils.py` doesn't define
# `might_be_incomplete_text_tool_call` at all, so importing that single
# attribute used to raise inside the `try` block below and silently set
# *every* name in this group (including `has_text_tool_call` and
# `parse_text_tool_calls`, which DO exist there) to None — permanently
# disabling text-format tool-call detection in the streaming path with no
# error surfaced anywhere. `agentic_utils.py` is vendored in this repo and
# kept in sync with what the streaming/turn-buffer logic below needs.
try:
    from src.services.agentic_utils import (
        has_text_tool_call,
        might_be_incomplete_text_tool_call,
        parse_text_tool_calls,
        split_tool_calls_and_answer_text,
        tool_call_label,
    )
except Exception:
    tool_call_label = None  # type: ignore
    has_text_tool_call = None  # type: ignore
    might_be_incomplete_text_tool_call = None  # type: ignore
    parse_text_tool_calls = None  # type: ignore
    split_tool_calls_and_answer_text = None  # type: ignore


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


def _recoverable_after_agent_fallback(*, fallback_used: bool, has_answer: bool) -> bool:
    """Return True when agent_fallback produced an answer but the graph still raised."""
    return fallback_used and has_answer


# ─── MCP tool loader ──────────────────────────────────────────────────────────
# Implemented in src.services.mcp.tool_loader; re-exported for backward compat.


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
                mcp_user_id=request.mcp_user_id,
            )
            tools.extend(mcp_tools)
            # `tools.extend(...)` above only copies the individual tool
            # references, not `mcp_tools`'s own `_mcp_client` attribute — carry
            # it over explicitly so the client stays alive for as long as the
            # caller holds onto this function's return value (see
            # _MCPToolsWithClient docstring in _load_mcp_tools_for_servers).
            mcp_client = getattr(mcp_tools, "_mcp_client", None)
            if mcp_client is not None:
                tools = _MCPToolsWithClient(tools, mcp_client)
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
    user_id: str,
) -> tuple[Any, Dict[str, Any]]:
    """Resolve the LLM client and metadata for agentic generation."""
    if request.custom_model_id:
        model = request.resolved_custom_model
        if model is None:
            model = await get_owned_custom_model(
                request.custom_model_id, user_id, action="use"
            )
        else:
            ensure_custom_model_has_credentials(model)
        llm = await build_custom_model_llm(model)
        return llm, custom_model_metadata(model)

    manager = get_shared_llm_manager()
    effective_type = _resolve_agentic_llm_type(request.llm_type)
    chain = manager.resolve_chain(effective_type)
    circuit_open = [name for name in chain if manager.health.is_open(name)]
    # resolve_chain has already sunk open circuits to the back, and kept the
    # configured order when every one of them is open, so the head is the pick.
    candidate = chain[0]
    # client_for_candidate raises instead of quietly substituting Mistral, so
    # agentic_llm_resolved reports the endpoint that will actually be called.
    llm = manager.client_for_candidate(candidate)
    return llm, {
        "agentic_llm_resolved": candidate,
        "endpoint": build_endpoint_metadata(
            requested=request.llm_type,
            chain=chain,
            answered=candidate,
            circuit_open=circuit_open,
        ),
    }


class _AgenticBudgetTimeout(TimeoutError):
    """The whole-run AGENTIC_TIMEOUT guard fired.

    Distinct type on purpose: that budget includes tool execution, so it says
    nothing about the LLM endpoint's health and must never open its circuit
    (slow MCP tools would otherwise park a healthy endpoint for everyone).
    """


def _record_endpoint_failure(
    endpoint: Optional[Dict[str, Any]], exc: BaseException
) -> None:
    """Open the answering endpoint's circuit when *exc* means it is down."""
    if isinstance(exc, _AgenticBudgetTimeout):
        return
    answered = (endpoint or {}).get("answered")
    if answered and is_endpoint_failure(exc):
        get_shared_llm_manager().health.record_failure(answered, exc)


def _record_in_graph_fallback(
    endpoint: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Re-attribute the answer to the fallback model.

    Deliberately does NOT open the primary's circuit: the graph's error_handler
    swallows the node's exception, so the failure cannot be classified here,
    and a prompt-level 400 (the strict-template class) reproducing on every
    send would otherwise park a healthy endpoint indefinitely. Transport-level
    failures still open the circuit through the classified paths.
    """
    return endpoint_answered_by(endpoint, LLMType.Fallback.value)


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


# ─── Artifact-reproduction instruction ────────────────────────────────────────

# The ReAct graph's own lead-in instruction lives in `prompts.yaml` bundled
# with the `eve-esa-agents` package resolved by graphs_bundle.py (see its
# `_prompts_yaml_path`) — that package is an external git dependency, not part
# of this repo, so it can't be edited here. This backend-owned instruction is
# injected as an extra SystemMessage ahead of the user's query instead: it
# ends up positioned *after* the graph's own system prompt in every `agent`
# node invocation (ReactAgent._invoke prepends its instruction fresh each
# call), so it stays visible for the whole tool-calling loop.
_ARTIFACT_MARKDOWN_INSTRUCTION = (
    "Tool results may contain a markdown stub for a file the tool produced: "
    'an image embed (`![name](/artifacts/{id} "MCP: server/tool")`) or a '
    'link (`[name](/artifacts/{id} "MCP: server/tool")`), each followed on '
    "the next line by a one-line JSON blob with the same data. When your "
    "answer references that file, reproduce the markdown line VERBATIM — "
    "do not rewrite, reformat, translate, or invent a different URL for it. "
    "`/artifacts/{id}` URLs are the ONLY valid way to reference a "
    "tool-produced file; never emit a `resource://` URI or any other URL "
    "(e.g. a `storage.googleapis.com` link) for it, even if one appears "
    "elsewhere in the tool output."
)


def _build_initial_messages(
    request: GenerationRequest, tools: List[Any], history: Optional[List[Any]] = None
) -> List[Any]:
    """Return the graph's initial ``messages`` state for this turn.

    Prepends :data:`_ARTIFACT_MARKDOWN_INSTRUCTION` as a ``SystemMessage``
    when MCP tools are in play, since only then can a tool result carry an
    artifact stub worth guarding against being rewritten.

    First turn only: the checkpointed thread state keeps the turn-one copy
    visible on later turns, and injecting another one mid-conversation puts a
    ``system`` message after an ``assistant`` one, which strict chat templates
    refuse (Blablador: 400 "Unexpected role 'system' after role 'assistant'").
    """
    messages: List[Any] = []
    if tools and SystemMessage and not history:
        messages.append(SystemMessage(content=_ARTIFACT_MARKDOWN_INSTRUCTION))
    messages.append(HumanMessage(content=request.query))
    return messages


async def append_missing_artifact_stubs(
    answer: str, artifact_ids: Optional[List[str]]
) -> str:
    """Append markdown stubs for collected artifacts the answer never references.

    The instruction above asks the model to reproduce `/artifacts/{id}` stubs
    verbatim, but models routinely paraphrase them into invented URLs (seen
    live with EVE-Instruct emitting `storage.googleapis.com` links). The
    captured files are too valuable to lose to prompt non-compliance, so this
    deterministic pass re-attaches every collected artifact whose serving URL
    is absent from the final answer. Fail-open: any lookup error leaves the
    answer untouched.
    """
    if not answer or not artifact_ids:
        return answer
    try:
        from src.database.models.artifact import Artifact

        lines: List[str] = []
        for artifact_id in artifact_ids:
            url = f"/artifacts/{artifact_id}"
            if url in answer:
                continue
            artifact = await Artifact.find_by_id(artifact_id)
            if not artifact:
                continue
            source = getattr(artifact, "source", None)
            server = getattr(source, "mcp_server", None) or "unknown"
            tool = getattr(source, "tool_name", None) or "tool"
            title = f"MCP: {server}/{tool}"
            if (artifact.content_type or "").startswith("image/"):
                lines.append(f'![{artifact.filename}]({url} "{title}")')
            else:
                lines.append(f'[{artifact.filename}]({url} "{title}")')
        if lines:
            answer = answer.rstrip() + "\n\n" + "\n\n".join(lines) + "\n"
    except Exception:
        logger.warning(
            "Failed to append missing artifact stubs to the answer", exc_info=True
        )
    return answer


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
    llm_run_timeout: Optional[int] = None,
) -> Any:
    """Compile the agent graph using the resolved LLM type.

    ``llm_type_override`` forces a specific model (e.g. ``LLMType.Fallback.value``)
    regardless of the request's ``llm_type``.  ``fallback_llm`` is forwarded to
    ``agent.compile()`` for in-graph node-level fallback.
    ``llm`` is the primary graph model (platform or user custom).  When
    ``fallback_llm`` is not passed, the platform fallback model is wired
    automatically.  ``llm_run_timeout`` is the resolved endpoint's own node
    budget; without one the generic MODEL_TIMEOUT applies.
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
        llm_run_timeout=(
            llm_run_timeout if llm_run_timeout is not None else MODEL_TIMEOUT
        ),
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
    *,
    user_id: str,
    conversation_id: Optional[str] = None,
) -> tuple[
    str,
    List[Dict[str, Any]],
    bool,
    Dict[str, Optional[float]],
    Dict[str, Any],
    List[Dict[str, Any]],
    List[str],
]:
    """Run the full agentic generation pipeline without streaming.

    Returns (answer, tool_results, use_rag, latencies, prompts, trace, artifact_ids).
    ``artifact_ids`` lists any Artifacts the MCP artifact interceptor persisted
    from tool output during this run (see ``src.services.mcp.artifact_ingestion``).
    """
    if not _langgraph_available:
        raise RuntimeError("LangGraph is not available — cannot run agentic generation")

    error_logger = get_error_logger()
    total_start = time.perf_counter()
    # message_id is unset here: Message.create() in the router runs before this
    # call and there's no clean way to thread it through this signature; artifacts
    # are still linked via conversation_id, and the router attaches artifact_ids
    # to the Message itself once this call returns.
    artifact_ctx, artifact_token = set_artifact_context(
        user_id=user_id, conversation_id=conversation_id
    )
    endpoint_metadata: Optional[Dict[str, Any]] = None

    try:
        tools = await _build_tools(request)
        checkpointer = await _get_agentic_checkpointer()
        history, summary = await _fetch_conversation_context(conversation_id)

        agent_graph_type = _resolve_agent_graph_type(request)
        agent = get_agent_graph(agent_graph_type)
        resolved_instruction = agent.instruction_text(history=history, summary=summary)
        llm, llm_metadata = await _resolve_agentic_llm_client(
            request, user_id=user_id
        )
        endpoint_metadata = llm_metadata.pop("endpoint", None)
        graph = _build_react_graph(
            request.llm_type,
            tools,
            checkpointer,
            agent=agent,
            history=history,
            summary=summary,
            llm=llm,
            llm_run_timeout=endpoint_timeout((endpoint_metadata or {}).get("answered")),
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
            graph_exc: Optional[Exception] = None
            with langfuse_context(
                user_id=user_id,
                session_id=conversation_id,
                tags=[
                    "agentic",
                    request.custom_model_id or request.llm_type or "default",
                ],
                trace_name="agentic_generation",
            ):
                try:
                    async for update in g.astream(
                        {"messages": _build_initial_messages(request, tools, history)},
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
                except Exception as exc:
                    graph_exc = exc

            if graph_exc is not None:
                has_answer = any(
                    isinstance(msg, AIMessage)
                    and msg.content
                    and not getattr(msg, "tool_calls", None)
                    for msg in raw_msgs
                )
                if not _recoverable_after_agent_fallback(
                    fallback_used=in_graph_fallback, has_answer=has_answer
                ):
                    raise graph_exc
                logger.warning(
                    "Agent graph raised after successful agent_fallback; "
                    "returning fallback answer: %s",
                    graph_exc,
                )
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
        final_answer = await append_missing_artifact_stubs(
            final_answer, list(artifact_ctx.collected_artifact_ids)
        )

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
        if used_fallback_llm:
            endpoint_metadata = _record_in_graph_fallback(endpoint_metadata)
            # Same re-point as the streaming path: the footer reads this key.
            llm_metadata["agentic_llm_resolved"] = LLMType.Fallback.value
        prompts: Dict[str, Any] = {
            "query": request.query,
            "instruction": resolved_instruction,
            "agent_prompts": dict(agent.prompts),
            "agent_graph_type": agent_graph_type,
            "used_fallback_llm": used_fallback_llm,
            **llm_metadata,
            # This function persists nothing itself: the router lifts the
            # payload out into metadata.endpoint.
            **({"endpoint": endpoint_metadata} if endpoint_metadata else {}),
        }

        return (
            final_answer,
            tool_results,
            use_rag,
            latencies,
            prompts,
            trace_entries,
            list(artifact_ctx.collected_artifact_ids),
        )

    except Exception as exc:
        _record_endpoint_failure(endpoint_metadata, exc)
        await error_logger.log_error(
            error=exc,
            component=Component.LLM,
            pipeline_stage=PipelineStage.GENERATION,
            description="Agentic generation failed",
            error_type=type(exc).__name__,
        )
        raise
    finally:
        reset_artifact_context(artifact_token)


# ─── Streaming generation ─────────────────────────────────────────────────────


async def generate_answer_agentic_stream_helper(
    request: GenerationRequest,
    conversation_id: str,
    message_id: str,
    user_id: str,
    output_format: str = "json",
    background_tasks: Optional[BackgroundTasks] = None,
    cancel_event: Optional[asyncio.Event] = None,
) -> AsyncGenerator[str, None]:
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
        error_info = build_error_payload(RuntimeError("LangGraph not available"))
        with contextlib.suppress(Exception):
            await persist_message_state(message_id, error=error_info)
        yield f"data: {json.dumps({'type': 'error', 'code': error_info['code'], 'message': error_info['message']})}\n\n"
        return

    error_logger = get_error_logger()
    total_start = time.perf_counter()
    accumulated: List[str] = []
    used_fallback_llm = False
    endpoint_metadata: Optional[Dict[str, Any]] = None
    artifact_ctx, artifact_token = set_artifact_context(
        user_id=user_id, conversation_id=conversation_id, message_id=message_id
    )

    def cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    def _collected_artifact_ids() -> Optional[List[str]]:
        return list(artifact_ctx.collected_artifact_ids) or None

    try:
        if cancelled():
            await persist_message_state(
                message_id, stopped=True, artifact_ids=_collected_artifact_ids()
            )
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
        endpoint_metadata = llm_prompts.pop("endpoint", None)

        graph = _build_react_graph(
            request.llm_type,
            tools,
            checkpointer,
            agent=agent,
            history=history,
            summary=summary,
            llm=llm,
            llm_run_timeout=endpoint_timeout((endpoint_metadata or {}).get("answered")),
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

            events: List[str] = []
            answer_items = items

            if has_text_tool_call and has_text_tool_call(joined):
                # A turn can contain one or more `[TOOL_CALLS]name{...}` segments
                # (each call gets its own marker) directly followed by real
                # answer prose in the SAME turn — split_tool_calls_and_answer_text
                # walks every recognized call and returns whatever text is left
                # over so it's emitted as an answer instead of silently dropped
                # or, worse, leaking the raw "[TOOL_CALLS]..." syntax verbatim
                # into `accumulated`/the persisted answer.
                calls, answer_text = (
                    split_tool_calls_and_answer_text(joined)
                    if split_tool_calls_and_answer_text
                    else (parse_text_tool_calls(joined) if parse_text_tool_calls else [], "")
                )
                if not calls:
                    return []
                for tc in calls:
                    tname = tc.get("name", "tool")
                    args = tc.get("args", {})
                    query_used = args.get("query", "")
                    label = tool_call_label(tname) if tool_call_label else f"Calling {tname}"
                    msg = f"{label}: {query_used}" if query_used else f"{label}…"
                    events.append(
                        f"data: {json.dumps({'type': 'tool_call', 'content': msg})}\n\n"
                    )
                if not answer_text:
                    return events
                # Streaming granularity is inherently lost for this leftover
                # text: we can't know the tool-call syntax has ended until the
                # whole turn is buffered, so it's emitted as a single chunk
                # rather than token-by-token.
                answer_items = [answer_text]

            if tokens_yielded == 0:
                elapsed = time.perf_counter() - gen_start
                if elapsed > AGENTIC_TIMEOUT:
                    raise _AgenticBudgetTimeout(
                        "No final-answer token received within AGENTIC_TIMEOUT"
                    )
                first_token_latency = time.perf_counter() - total_start

            for tok in answer_items:
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

        graph_exc: Optional[Exception] = None
        try:
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
                    {"messages": _build_initial_messages(request, tools, history)},
                    config=config,
                    stream_mode=["messages", "updates"],
                ):
                    if cancelled():
                        await persist_message_state(
                            message_id,
                            stopped=True,
                            output="".join(accumulated),
                            artifact_ids=_collected_artifact_ids(),
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
        except Exception as exc:
            answer = "".join(accumulated)
            if not _recoverable_after_agent_fallback(
                fallback_used=in_graph_fallback_used, has_answer=bool(answer)
            ):
                raise
            graph_exc = exc
            logger.warning(
                "Agent graph raised after successful agent_fallback; "
                "persisting streamed answer: %s",
                exc,
            )

        gen_latency = time.perf_counter() - gen_start
        answer = await append_missing_artifact_stubs(
            "".join(accumulated), _collected_artifact_ids()
        )
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

        if graph_exc is not None:
            await error_logger.log_error(
                error=graph_exc,
                component=Component.LLM,
                pipeline_stage=PipelineStage.GENERATION,
                description=(
                    "Primary agent node failed after agent_fallback produced an answer"
                ),
                error_type=type(graph_exc).__name__,
            )

        if in_graph_fallback_used:
            endpoint_metadata = _record_in_graph_fallback(endpoint_metadata)
            # The footer prefers agentic_llm_resolved when a fallback ran; it
            # must name the model that answered, not the one that died.
            llm_prompts["agentic_llm_resolved"] = LLMType.Fallback.value
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
            endpoint=endpoint_metadata,
            trace=trace_entries if trace_entries else None,
            artifact_ids=_collected_artifact_ids(),
        )

        if background_tasks:
            background_tasks.add_task(maybe_rollup_and_trim_history, conversation_id)
        else:
            asyncio.create_task(maybe_rollup_and_trim_history(conversation_id))

        if output_format == "json":
            yield f"data: {json.dumps({'type': 'final', 'answer': answer, 'latencies': latencies, 'artifact_ids': _collected_artifact_ids()})}\n\n"
        else:
            yield "data: [DONE]\n\n"

    except asyncio.CancelledError:
        logger.info("Agentic generation cancelled")
        await persist_message_state(
            message_id,
            output="".join(accumulated),
            stopped=True,
            artifact_ids=_collected_artifact_ids(),
        )
        return

    except TimeoutError as exc:
        logger.warning("Agentic generation timed out: %s", exc)
        _record_endpoint_failure(endpoint_metadata, exc)
        await error_logger.log_error(
            error=exc,
            component=Component.LLM,
            pipeline_stage=PipelineStage.GENERATION,
            description="Agentic generation timed out",
            error_type=type(exc).__name__,
        )
        answer = await append_missing_artifact_stubs(
            "".join(accumulated), _collected_artifact_ids()
        )
        if answer:
            await persist_message_state(
                message_id, output=answer, artifact_ids=_collected_artifact_ids()
            )
            if output_format == "json":
                yield f"data: {json.dumps({'type': 'final', 'answer': answer, 'latencies': {}, 'artifact_ids': _collected_artifact_ids()})}\n\n"
            else:
                yield "data: [DONE]\n\n"
        else:
            error_info = build_error_payload(exc)
            with contextlib.suppress(Exception):
                await persist_message_state(
                    message_id,
                    output="",
                    error=error_info,
                    artifact_ids=_collected_artifact_ids(),
                )
            yield f"data: {json.dumps({'type': 'error', 'code': error_info['code'], 'message': error_info['message']})}\n\n"

    except Exception as exc:
        logger.error("Agentic streaming error: %s", exc)
        _record_endpoint_failure(endpoint_metadata, exc)
        await error_logger.log_error(
            error=exc,
            component=Component.LLM,
            pipeline_stage=PipelineStage.GENERATION,
            description="Agentic streaming error",
            error_type=type(exc).__name__,
        )
        error_info = build_error_payload(exc)
        with contextlib.suppress(Exception):
            await persist_message_state(
                message_id,
                output="".join(accumulated),
                error=error_info,
                artifact_ids=_collected_artifact_ids(),
            )
        yield f"data: {json.dumps({'type': 'error', 'code': error_info['code'], 'message': error_info['message']})}\n\n"

    finally:
        reset_artifact_context(artifact_token)


# ─── SSE wrappers ─────────────────────────────────────────────────────────────


async def generate_answer_agentic_stream(
    request: GenerationRequest,
    conversation_id: str,
    message_id: str,
    user_id: str,
    background_tasks: Optional[BackgroundTasks] = None,
    cancel_event: Optional[asyncio.Event] = None,
):
    """Plain-text SSE wrapper around the agentic stream helper."""
    async for chunk in generate_answer_agentic_stream_helper(
        request,
        conversation_id,
        message_id,
        user_id,
        "plain",
        background_tasks,
        cancel_event,
    ):
        yield chunk


async def generate_answer_agentic_json_stream(
    request: GenerationRequest,
    conversation_id: str,
    message_id: str,
    user_id: str,
    background_tasks: Optional[BackgroundTasks] = None,
    cancel_event: Optional[asyncio.Event] = None,
):
    """JSON SSE wrapper around the agentic stream helper."""
    async for chunk in generate_answer_agentic_stream_helper(
        request,
        conversation_id,
        message_id,
        user_id,
        "json",
        background_tasks,
        cancel_event,
    ):
        yield chunk


# ─── Bus-decoupled entry point ────────────────────────────────────────────────


async def run_agentic_generation_to_bus(
    request: GenerationRequest,
    conversation_id: str,
    message_id: str,
    user_id: str,
    background_tasks: Optional[BackgroundTasks] = None,
    cancel_event: Optional[asyncio.Event] = None,
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
        # Outer safety net: it fires exactly when the inner handlers did not,
        # so it must persist the marker itself or the turn stays a blank shell.
        error_info = build_error_payload(exc)
        with contextlib.suppress(Exception):
            await persist_message_state(message_id, error=error_info)
        await bus.publish(
            message_id,
            f"data: {json.dumps({'type': 'error', 'code': error_info['code'], 'message': error_info['message']})}\n\n",
        )
    finally:
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
