"""Endpoint to generate an answer using a language model and vector store."""

import json
import logging
import asyncio
import time
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any, List, Optional, TypedDict
from pydantic import BaseModel, Field

from src.core.vector_store_manager import VectorStoreManager
from src.core.llm_manager import LLMManager
from src.constants import (
    DEFAULT_QUERY,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM,
    DEFAULT_K,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    POLICY_NOT_ANSWER,
    POLICY_PROMPT,
    RERANKER_MODEL,
)
from src.utils.helpers import get_mongodb_uri
from src.utils.runpod_utils import get_reranked_documents_from_runpod
from src.services.mcp_client_service import MultiServerMCPClientService
from src.config import config
from src.hallucination_pipeline.loop import run_hallucination_loop
from src.hallucination_pipeline.schemas import generation_schema

logger = logging.getLogger(__name__)


# No direct OpenAI client usage; we use Runpod-backed ChatOpenAI via LLMManager


class PolicyCheck(BaseModel):
    violates_policy: bool = Field(
        description="True if the input violates policies otherwise False"
    )


class GenerationRequest(BaseModel):
    query: str = DEFAULT_QUERY
    year: Optional[List[int]] = None
    filters: Optional[Dict[str, Any]] = None
    collection_ids: List[str] = Field(default_factory=lambda: [], exclude=True)
    llm: str = DEFAULT_LLM  # or openai
    embeddings_model: str = DEFAULT_EMBEDDING_MODEL
    k: int = DEFAULT_K
    temperature: float = Field(DEFAULT_TEMPERATURE, ge=0.0, le=1.0)
    score_threshold: float = Field(DEFAULT_SCORE_THRESHOLD, ge=0.0, le=1.0)
    max_new_tokens: int = Field(DEFAULT_MAX_NEW_TOKENS, ge=100, le=8192)
    public_collections: List[str] = Field(
        default=[],
        description="List of public collection names to include in the search",
    )
    hallucination_loop_flag: bool = False  # For testing purposes


# -------- Output normalization helpers --------
import re


def _normalize_ai_output(text: str) -> str:
    """Clean AI output for end-user display.

    - Collapse multiple newlines to a single space
    - Remove leading ordered-list markers like "1." or "1)" at line starts
    - Collapse excessive whitespace
    """
    if not text:
        return text
    s = text.replace("\r\n", "\n")
    # Remove ordered list markers at the start of lines
    lines = [re.sub(r"^\s*(?:\d+\.|\d+\))\s+", "", ln) for ln in s.splitlines()]
    s = " ".join(lines)
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_final_assistant_content(messages_out: Any) -> Optional[str]:
    """Extract the last assistant response content from LangGraph/LC messages."""
    content: Optional[str] = None
    try:
        if isinstance(messages_out, list):
            # Prefer LangChain AIMessage instances
            if AIMessage:
                for msg in reversed(messages_out):
                    try:
                        if isinstance(msg, AIMessage):
                            content = getattr(msg, "content", None)
                            if content:
                                break
                    except Exception:
                        continue
            # Fallback for dict-style messages
            if not content:
                for msg in reversed(messages_out):
                    if isinstance(msg, dict) and msg.get("role") in (
                        "assistant",
                        "ai",
                    ):
                        content = msg.get("content")
                        if content:
                            break
        elif hasattr(messages_out, "content"):
            content = getattr(messages_out, "content", None)
    except Exception:
        content = None
    return content


# LangGraph / LangGraph MongoDB checkpointer (optional dependency)
# We use short-term memory per conversation via a thread_id equal to the conversation_id.
_langgraph_available = False
try:  # type: ignore
    from langgraph.graph import StateGraph, MessagesState, START  # noqa: F401
    from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver  # noqa: F401
    from langgraph.graph.message import add_messages  # noqa: F401

    _langgraph_available = True
except Exception:
    # If not installed in the environment, we will fall back to non-memory generation.
    pass

# LangChain message classes for compatibility
try:
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
except Exception:
    HumanMessage = None  # type: ignore
    SystemMessage = None  # type: ignore
    AIMessage = None  # type: ignore


# Unified message factory to avoid branching at call sites
def _make_message(role: str, content: str) -> Any:
    """Create a message compatible with LangChain if available, else dict style."""
    try:
        if role == "system" and SystemMessage:
            return SystemMessage(content=content)
        if role == "user" and HumanMessage:
            return HumanMessage(content=content)
        if role == "assistant" and AIMessage:
            return AIMessage(content=content)
    except Exception:
        pass
    return {"role": role, "content": content}


# --- Lazy singletons for compiled LangGraph and checkpointers ---
_compiled_graph = None  # Mongo-backed compiled graph
_compiled_graph_mode = None  # "mongo" or "memory"
_mongo_checkpointer_cm = None  # context manager kept open for process lifetime
_mongo_checkpointer = None  # active AsyncMongoDBSaver obtained from __aenter__
_inmemory_compiled_graph = None  # Fallback compiled graph using InMemorySaver
_graph_init_lock = asyncio.Lock()


async def _get_or_create_compiled_graph():
    """Lazily compile LangGraph once and reuse across calls.

    Tries MongoDB checkpointer first and keeps it open for the process lifetime.
    Falls back to a single in-memory compiled graph if Mongo is unavailable.
    Returns a tuple: (graph, mode) where mode in {"mongo", "memory"}.
    """
    global _compiled_graph, _compiled_graph_mode
    global _mongo_checkpointer_cm, _mongo_checkpointer
    global _inmemory_compiled_graph

    if not _langgraph_available:
        return None, None

    async with _graph_init_lock:
        # If already compiled, reuse it.
        if _compiled_graph is not None and _compiled_graph_mode == "mongo":
            return _compiled_graph, _compiled_graph_mode
        if _inmemory_compiled_graph is not None and _compiled_graph_mode == "memory":
            return _inmemory_compiled_graph, _compiled_graph_mode

        # Build the simple one-node graph
        llm = LLMManager().get_model()

        # Define a custom state so we can carry sampling params alongside messages
        class GenerationState(TypedDict):
            messages: MessagesState
            temperature: Optional[float]
            max_tokens: Optional[int]

        async def call_model(state: "GenerationState"):
            # Allow per-invocation temperature and max tokens via state
            bound_llm = llm
            try:
                bind_kwargs = {}
                try:
                    temperature_val = state.get("temperature")
                except Exception:
                    temperature_val = None
                try:
                    max_tokens_val = state.get("max_tokens")
                except Exception:
                    max_tokens_val = None

                if temperature_val is not None:
                    bind_kwargs["temperature"] = temperature_val
                if max_tokens_val is not None:
                    bind_kwargs["max_tokens"] = max_tokens_val
                if bind_kwargs:
                    bound_llm = llm.bind(**bind_kwargs)
            except Exception:
                bound_llm = llm
            response = await bound_llm.ainvoke(
                state["messages"]
            )  # returns an AIMessage
            return {"messages": response}

        builder = StateGraph(GenerationState)
        builder.add_node(call_model)
        builder.add_edge(START, "call_model")

        # Try Mongo-backed checkpointer first, keep it open
        try:
            uri = get_mongodb_uri()
            cm = AsyncMongoDBSaver.from_conn_string(uri)
            checkpointer = await cm.__aenter__()
            graph = builder.compile(checkpointer=checkpointer)

            _mongo_checkpointer_cm = cm
            _mongo_checkpointer = checkpointer
            _compiled_graph = graph
            _compiled_graph_mode = "mongo"
            return graph, "mongo"
        except Exception:
            # Fall back to a single shared in-memory saver (persists within process)
            try:
                from langgraph.checkpoint.memory import InMemorySaver

                if _inmemory_compiled_graph is None:
                    _inmemory_compiled_graph = builder.compile(
                        checkpointer=InMemorySaver()
                    )
                _compiled_graph_mode = "memory"
                return _inmemory_compiled_graph, "memory"
            except Exception:
                # As a last resort, no graph
                return None, None


async def _summarize_history_with_runpod(
    transcript: str, max_tokens: int = 50000
) -> str:
    """Summarize entire conversation history."""
    if not transcript:
        return ""
    llm = LLMManager().get_model()
    system = (
        "You are an AI assistant specialized in summarizing chat histories. "
        "Your role is to read a transcript of a conversation and produce a clear, "
        "concise, and neutral summary of the main points."
    )
    messages = [
        _make_message("system", system),
        _make_message("user", f"Conversation transcript:\n{transcript}"),
    ]
    try:
        resp = await llm.bind(max_tokens=max_tokens).ainvoke(messages)
        return getattr(resp, "content", str(resp))
    except Exception:
        return ""


def _extract_result_text_for_dedup(item: Any) -> str:
    """Extract a normalized text key from heterogeneous result items for deduplication.

    Supports objects with `.payload`, plain dicts with `text`/`page_content`/`content`,
    and falls back to common attributes. Returns a whitespace-collapsed string.
    """
    try:
        # Dict-shaped item
        if isinstance(item, dict):
            value = item.get("text") or item.get("page_content") or item.get("content")
            if not value:
                payload = item.get("payload") or {}
                if isinstance(payload, dict):
                    value = (
                        payload.get("page_content")
                        or payload.get("text")
                        or payload.get("content")
                        or (payload.get("metadata") or {}).get("page_content")
                    )
            if value:
                return " ".join(str(value).strip())

        # Object with `.payload`
        payload = getattr(item, "payload", None)
        if isinstance(payload, dict):
            value = (
                payload.get("page_content")
                or payload.get("text")
                or payload.get("content")
                or (payload.get("metadata") or {}).get("page_content")
            )
            if value:
                return " ".join(str(value).strip())

        # Common attributes as a fallback
        for attr in ("text", "page_content", "content", "document"):
            v = getattr(item, attr, None)
            if v:
                return " ".join(str(v).strip())
    except Exception:
        pass
    return ""


def _deduplicate_results(items: List[Any]) -> List[Any]:
    """Remove duplicate items by comparing their extracted text content key.

    Two items are considered duplicates if their extracted content (from
    `text`, `payload.page_content`, `page_content`, or `content`) matches
    after trimming and collapsing whitespace (case-sensitive by default).
    """
    seen: set[str] = set()
    deduped: List[Any] = []
    for it in items:
        key = _extract_result_text_for_dedup(it)
        # Keep at most one empty-key item
        if key in seen:
            continue
        seen.add(key)
        deduped.append(it)
    return deduped


def _build_context(items: List[Any]) -> str:
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

            return "\n".join([p for p in parts if p is not None])
    except Exception:
        # On any error, fall back to simple join of strings
        pass

    # Default: treat as list of strings
    return "\n".join([str(t) for t in items if str(t)])


async def _maybe_rerank(candidate_texts: List[str], query: str) -> List[dict] | None:
    """Call reranker if configured."""
    endpoint_id = config.get_reranker_id()
    if not (endpoint_id) or candidate_texts is None or len(candidate_texts) == 0:
        return None

    try:
        results = await get_reranked_documents_from_runpod(
            endpoint_id=endpoint_id,
            docs=candidate_texts,
            query=query,
            model=RERANKER_MODEL or "BAAI/bge-reranker-large",
            timeout=config.get_reranker_timeout(),
        )
        # sort results by relevance_score
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results
    except Exception as e:
        logger.warning(f"Reranker failed, using vector similarity order: {e}")
        return None


async def get_mcp_context(
    request: GenerationRequest,
) -> tuple[list, Dict[str, Optional[float]]]:
    """Call MCP semantic search and return (results, latencies)."""
    mcp_client = MultiServerMCPClientService.get_shared()

    # If frontend provided filters that include anything other than year,
    # we must NOT call Wiley MCP. In that case, return empty context/results.
    def _has_non_year_filters(filters: Any) -> bool:
        try:
            if not isinstance(filters, dict) or not filters:
                return False
            for top_key, value in filters.items():
                if top_key != "must" and value:
                    return True
            conditions = filters.get("must") or []
            if not isinstance(conditions, list):
                return True
            for cond in conditions:
                key = cond.get("key")
                if key != "year":
                    return True
            return False
        except Exception:
            return True

    if _has_non_year_filters(getattr(request, "filters", None)):
        return (
            [],
            {
                "mcp_retrieval_latency": None,
                "mcp_docs_reranking_latency": None,
            },
        )

    # Build tool arguments
    args: Dict[str, Any] = {
        "query": request.query,
        "topN": request.k * 2,
        "threshold": request.score_threshold,
    }
    if isinstance(request.year, list) and len(request.year) >= 2:
        args["start_year"] = request.year[0]
        args["end_year"] = request.year[1]

    # Call tool with latency measurement (with one retry on auth/session expiry)
    def _is_auth_error(raw_payload: Any) -> bool:
        """Detect auth/session expiry messages in MCP tool response.

        Handles both top-level and nested {result: {content: [...]}} payloads and
        items shaped as {type: 'text', text: '...'} or plain dicts/strings.
        """
        try:
            if not isinstance(raw_payload, dict):
                return False

            # Support responses wrapped under 'result'
            payload = (
                raw_payload["result"]
                if isinstance(raw_payload.get("result"), dict)
                else raw_payload
            )
            items = payload.get("content") or []

            def _extract_text(item: Any) -> str:
                if isinstance(item, str):
                    return item
                if isinstance(item, dict):
                    # Prefer explicit text for typed content items
                    if item.get("type") == "text" and isinstance(item.get("text"), str):
                        return item["text"]
                    # Fallbacks commonly seen in various MCP adapters
                    for key in ("text", "content", "document"):
                        val = item.get(key)
                        if isinstance(val, str):
                            return val
                return ""

            auth_markers = (
                "oauth 2.0 token has expired",
                "unauthorized",
                "401",
            )

            for it in items:
                lower = _extract_text(it).lower()
                if lower and any(marker in lower for marker in auth_markers):
                    return True
        except Exception:
            return False
        return False

    mcp_start = time.perf_counter()
    raw = await mcp_client.call_tool_on_server("eve-mcp-demo", "semanticSearch", args)
    mcp_retrieval_latency = time.perf_counter() - mcp_start

    # If auth expired, re-establish connection and retry once
    if _is_auth_error(raw):
        try:
            await mcp_client.close()
        except Exception:
            pass
        try:
            # Reset shared instance to force reconnection with fresh token
            logger.info(
                "Resetting shared instance to force reconnection with fresh token"
            )
            from src.services.mcp_client_service import (
                MultiServerMCPClientService as _Svc,
            )

            _Svc._shared_instance = None
        except Exception:
            pass
        mcp_client = MultiServerMCPClientService.get_shared()
        mcp_start = time.perf_counter()
        raw = await mcp_client.call_tool_on_server(
            "eve-mcp-demo", "semanticSearch", args
        )
        mcp_retrieval_latency = time.perf_counter() - mcp_start

    # Normalize response payload
    content_items = []
    is_error = False
    if isinstance(raw, dict):
        is_error = bool(raw.get("is_error"))
        content_items = raw.get("content", []) or []
    if is_error:
        return (
            [],
            {
                "mcp_retrieval_latency": mcp_retrieval_latency,
            },
        )

    def _ensure_list(obj: Any) -> List[Any]:
        if obj is None:
            return []
        if isinstance(obj, list):
            return obj
        return [obj]

    # Extract result items from various possible shapes
    extracted: List[Dict[str, Any]] = []
    for item in content_items:
        data = item
        if isinstance(item, dict) and "text" in item:
            data = item["text"]
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                # treat as plain text record
                data = {"text": data}

        # common wrappers
        if isinstance(data, dict):
            for key in ("results", "documents", "items", "data"):
                if isinstance(data.get(key), list):
                    extracted.extend(_ensure_list(data[key]))
                    break
            else:
                extracted.extend(_ensure_list(data))
        elif isinstance(data, list):
            extracted.extend(data)

    latencies: Dict[str, Optional[float]] = {
        "mcp_retrieval_latency": mcp_retrieval_latency,
    }

    return extracted, latencies


async def get_rag_context(
    vector_store: VectorStoreManager, request: GenerationRequest
) -> tuple[list, list, Dict[str, Optional[float]]]:
    """Get RAG context from vector store."""
    # Retrieve with latency measurements
    results, vs_latencies = await vector_store.retrieve_documents_with_latencies(
        collection_names=request.collection_ids,
        query=request.query,
        k=request.k * 2,
        score_threshold=request.score_threshold,
        embeddings_model=request.embeddings_model,
        filters=request.filters,
    )

    if not results:
        logger.warning(f"No documents found for query: {request.query}")

    return results, vs_latencies


async def setup_rag_and_context(request: GenerationRequest):
    """Setup RAG and get context for the request."""
    # Check if we need to use RAG using LLMManager
    is_rag = await LLMManager().should_use_rag(request.query)

    # Get context if using RAG
    latencies: Dict[str, Optional[float]] = {}
    context, results = "", []
    if is_rag:
        try:
            vector_store = VectorStoreManager(embeddings_model=request.embeddings_model)
            rag_lat: Dict[str, Optional[float]] = {}
            mcp_lat: Dict[str, Optional[float]] = {}

            results, rag_lat = await get_rag_context(vector_store, request)
            mcp_results, mcp_lat = await get_mcp_context(request)

            merged_results = list(results) + list(mcp_results)
            # Build candidate texts with mapping to original indices
            candidate_texts: List[str] = []
            candidate_indices: List[int] = []
            index_to_text: Dict[int, str] = {}
            for idx, res in enumerate(merged_results):
                text_val: Optional[str] = None
                try:
                    payload = getattr(res, "payload", None)
                    if isinstance(payload, dict):
                        text_val = payload.get("text") or payload.get("content")
                    elif isinstance(res, dict):
                        text_val = res.get("text")
                except Exception:
                    text_val = None

                if text_val:
                    text_str = str(text_val).strip()
                    if text_str and "API call failed" not in text_str:
                        candidate_indices.append(idx)
                        candidate_texts.append(text_str)
                        index_to_text[idx] = text_str

            # Rerank candidates if configured; otherwise keep original order
            latencies["reranking_latency"] = time.perf_counter()
            reranked = await _maybe_rerank(candidate_texts, request.query)
            latencies["reranking_latency"] = (
                time.perf_counter() - latencies["reranking_latency"]
            )
            if reranked:
                # Map reranked indices (relative to candidate_texts) back to merged_results indices
                ordered_indices = [
                    candidate_indices[item.get("index", i)]
                    for i, item in enumerate(reranked)
                    if isinstance(item, dict)
                ]
            else:
                ordered_indices = list(candidate_indices)

            # Trim to top-k
            top_k = int(getattr(request, "k", 5) or 5)
            selected_indices = ordered_indices[:top_k]

            # Build context and filter original results to the selected set
            context_list = [
                index_to_text[i] for i in selected_indices if i in index_to_text
            ]
            context_list = list(set(context_list))
            context = _build_context(context_list)
            results = [merged_results[i] for i in selected_indices]
            results = _deduplicate_results(results)

            latencies.update(rag_lat or {})
            latencies.update(mcp_lat or {})
        except Exception as e:
            logger.warning(
                f"Failed to get RAG context and MCP context and merge them: {e}"
            )

    return context, results, is_rag, latencies


async def generate_answer(
    request: GenerationRequest,
    conversation_id: Optional[str] = None,
) -> tuple[str, list, bool, dict, Dict[str, Optional[float]]]:
    """Generate an answer using RAG and LLM."""
    llm_manager = LLMManager()

    try:
        # Check if the query violates EO policies
        policy_prompt = POLICY_PROMPT.format(question=request.query)
        base_llm = llm_manager.get_mistral_model()
        structured_llm = base_llm.bind(temperature=0).with_structured_output(
            PolicyCheck
        )
        policy_result = await structured_llm.ainvoke(policy_prompt)
        logger.info(f"policy_result: {policy_result}")
        if policy_result.violates_policy:
            return POLICY_NOT_ANSWER, [], False, {}, {}

        total_start = time.perf_counter()
        context, results, is_rag, latencies = await setup_rag_and_context(request)

        # Build messages for LangGraph memory + generation
        messages_for_turn: List[Any] = []

        # Optionally fetch rolling summary to embed in user message content
        summary_text: Optional[str] = None
        if conversation_id:
            try:
                from src.database.models.conversation import (
                    Conversation as ConversationModel,
                )

                convo = await ConversationModel.find_by_id(conversation_id)
                if convo and getattr(convo, "summary", None):
                    summary_text = convo.summary or None
            except Exception:
                # Non-critical: proceed without summary if retrieval fails
                summary_text = None

        # Build user message including optional summary and raw RAG context (when available)
        user_parts: List[str] = []
        if summary_text:
            user_parts.append(f"Conversation summary up to now:\n{summary_text}")
        user_parts.append(request.query)
        if is_rag and context:
            user_parts.append(f"Context:\n{context}")
        user_content = "\n\n".join(user_parts)

        # Append the user message
        messages_for_turn.append(_make_message("user", user_content))

        # Use LangGraph with MongoDB checkpointer for short-term memory if available
        final_answer: Optional[str] = None
        base_gen_latency: Optional[float] = None
        mistral_gen_latency: Optional[float] = None
        if _langgraph_available and conversation_id:
            try:
                # Single invoke to append assistant response to memory (async)
                gen_start = time.perf_counter()
                # Pass temperature to the graph via state config
                graph, mode = await _get_or_create_compiled_graph()
                if graph is not None:
                    config = {"configurable": {"thread_id": conversation_id}}
                    state = {
                        "messages": messages_for_turn,
                        "temperature": request.temperature,
                        "max_tokens": request.max_new_tokens,
                    }
                    result = await graph.ainvoke(state, config)
                else:
                    result = None
                base_gen_latency = time.perf_counter() - gen_start
                messages_out = result.get("messages")
                final_answer = _extract_final_assistant_content(messages_out) or ""
            except Exception as e:
                # If LangGraph/Runpod path fails, fallback to direct generation (with internal Mistral fallback)
                logger.warning(
                    f"LangGraph invocation failed, falling back to direct generation: {e}"
                )
                gen_start = time.perf_counter()
                final_answer = await llm_manager.generate_answer_mistral(
                    query=request.query,
                    context=context,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                )
                mistral_gen_latency = time.perf_counter() - gen_start
        else:
            # Fallback: use existing direct generation path without memory persistence
            gen_start = time.perf_counter()
            final_answer = await llm_manager.generate_answer_mistral(
                query=request.query,
                context=context,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
            )
            mistral_gen_latency = time.perf_counter() - gen_start

        answer = _normalize_ai_output(final_answer or "")

        model = llm_manager.get_model()
        generation_response = generation_schema(question=request.query, answer=answer)
        loop_result: Optional[dict] = None
        hallucination_latency: Optional[dict] = None
        if len(context.strip()) != 0 and request.hallucination_loop_flag:
            logger.info("starting to run hallucination loop")
            try:
                loop_result, hallucination_latency = await run_hallucination_loop(
                    model,
                    context,
                    generation_response,
                    request.collection_ids,
                )
                answer = loop_result["final_answer"]
            except Exception as e:
                logger.warning(f"Failed to run hallucination loop: {e}")
                logger.info("falling back to mistral model for hallucination loop")
                model = llm_manager.get_mistral_model()
                loop_result, hallucination_latency = await run_hallucination_loop(
                    model,
                    context,
                    generation_response,
                    request.collection_ids,
                )
                answer = loop_result["final_answer"]

        total_latency = time.perf_counter() - total_start
        latencies = {
            **(latencies or {}),
            "base_generation_latency": base_gen_latency,
            "fallback_latency": mistral_gen_latency,
            "hallucination_latency": hallucination_latency,
            "total_latency": total_latency,
        }
        return answer, results, is_rag, loop_result, latencies

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def maybe_rollup_and_trim_history(conversation_id: str, summary_every: int = 10):
    """Every N turns, summarize entire history and reset short-term memory to only the summary.

    Note: We derive the turn count from the number of Message documents in this conversation.
    """
    try:
        from src.database.models.message import Message as MessageModel
    except Exception:
        return

    try:
        total_turns = await MessageModel.count_documents(
            {"conversation_id": conversation_id}
        )
        if total_turns == 0 or total_turns % summary_every != 0:
            return

        logger.info(
            "starting to summarize history for conversation id: %s", conversation_id
        )
        # Build transcript from the last `summary_every` turns (each Message is one turn)
        skip = max(0, total_turns - summary_every)
        messages = await MessageModel.find_all(
            filter_dict={"conversation_id": conversation_id},
            sort=[("timestamp", 1)],
            skip=skip,
            limit=summary_every,
        )
        transcript_parts: List[str] = []
        for m in messages:
            if getattr(m, "input", None):
                transcript_parts.append(f"User: {m.input}")
            if getattr(m, "output", None):
                transcript_parts.append(f"Assistant: {m.output}")
        transcript = "\n".join(transcript_parts)

        # Include existing rolling summary if available
        try:
            from src.database.models.conversation import (
                Conversation as ConversationModel,
            )
        except Exception:
            ConversationModel = None  # type: ignore

        prior_summary: str = ""
        if ConversationModel is not None:
            convo = await ConversationModel.find_by_id(conversation_id)
            if convo and getattr(convo, "summary", None):
                prior_summary = convo.summary or ""

        summarizer_input = (
            f"Current summary (may be empty):\n{prior_summary}\n\nRecent turns:\n{transcript}"
            if prior_summary
            else f"Recent turns:\n{transcript}"
        )

        summary_text = await _summarize_history_with_runpod(summarizer_input)
        if not summary_text:
            return

        # Persist the new rolling summary on the Conversation document
        if ConversationModel is not None:
            if convo is None:
                convo = await ConversationModel.find_by_id(conversation_id)
            if convo is not None:
                convo.summary = summary_text
                try:
                    await convo.save()
                except Exception:
                    pass

    except Exception:
        # Non-critical path; ignore errors
        return


async def generate_answer_stream_generator_helper(
    request: GenerationRequest, output_format: str = "plain"
):
    """Helper function to generate streaming answer with different output formats."""
    llm_manager = LLMManager()

    try:
        context, results, is_rag, _ = await setup_rag_and_context(request)

        # Send initial metadata for JSON format
        if output_format == "json":
            yield f"data: {json.dumps({'type': 'start', 'use_rag': is_rag, 'documents_count': len(results)})}\n\n"

        # Generate streaming answer
        full_answer = ""
        async for chunk in llm_manager.generate_answer_stream(
            query=request.query,
            context=context,
            llm=request.llm,
            max_new_tokens=request.max_new_tokens,
        ):
            if output_format == "json":
                full_answer += chunk
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
            else:
                yield f"data: {chunk}\n\n"

        # Send final metadata for JSON format
        if output_format == "json":
            yield f"data: {json.dumps({'type': 'end', 'full_answer': full_answer})}\n\n"

    except Exception as e:
        error_msg = (
            f"data: Error: {str(e)}\n\n"
            if output_format == "plain"
            else f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        )
        yield error_msg


async def generate_answer_stream_generator(
    request: GenerationRequest,
):
    """Generate streaming answer using RAG and LLM."""
    async for chunk in generate_answer_stream_generator_helper(request, "plain"):
        yield chunk


async def generate_answer_json_stream_generator(
    request: GenerationRequest,
):
    """Generate streaming answer using RAG and LLM with JSON format."""
    async for chunk in generate_answer_stream_generator_helper(request, "json"):
        yield chunk


async def generate_answer_stream_json(
    request: GenerationRequest,
) -> StreamingResponse:
    """Generate a streaming answer using RAG and LLM with JSON format."""
    return StreamingResponse(
        generate_answer_json_stream_generator(request),
        media_type="application/json",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        },
    )
