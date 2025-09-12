"""Endpoint to generate an answer using a language model and vector store."""

import json
import logging
import asyncio
import time
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from src.core.vector_store_manager import VectorStoreManager
from src.database.models.collection import Collection
from src.core.llm_manager import LLMManager
from src.constants import (
    DEFAULT_QUERY,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM,
    DEFAULT_K,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_GET_UNIQUE_DOCS,
    DEFAULT_MAX_NEW_TOKENS,
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


class GenerationRequest(BaseModel):
    query: str = DEFAULT_QUERY
    year: Optional[List[int]] = None
    filters: Optional[Dict[str, Any]] = None
    collection_ids: List[str] = Field(default_factory=lambda: [], exclude=True)
    llm: str = DEFAULT_LLM  # or openai
    embeddings_model: str = DEFAULT_EMBEDDING_MODEL
    k: int = DEFAULT_K
    score_threshold: float = Field(DEFAULT_SCORE_THRESHOLD, ge=0.0, le=1.0)
    max_new_tokens: int = Field(DEFAULT_MAX_NEW_TOKENS, ge=100, le=8192)
    use_rag: bool = True
    hallucination_loop_flag: bool = True  # For testing purposes


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

        async def call_model(state: "MessagesState"):
            response = await llm.ainvoke(state["messages"])  # returns an AIMessage
            return {"messages": response}

        builder = StateGraph(MessagesState)
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


async def _ainvoke_with_langgraph(messages_for_turn: List[Any], thread_id: str) -> Any:
    """Invoke a minimal LangGraph with Async MongoDB checkpointer and return result dict."""
    if not _langgraph_available:
        return None
    try:
        graph, mode = await _get_or_create_compiled_graph()
        if graph is None:
            return None
        config = {"configurable": {"thread_id": thread_id}}
        return await graph.ainvoke({"messages": messages_for_turn}, config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _summarize_history_with_runpod(transcript: str, max_tokens: int = 50000) -> str:
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
        resp = llm.bind(max_tokens=max_tokens).invoke(messages)
        return getattr(resp, "content", str(resp))
    except Exception:
        return ""


def _extract_candidate_texts(results: List[Any]) -> List[str]:
    """Extract plain text strings from vector-store results payloads."""
    texts: List[str] = []
    for item in results:
        payload = getattr(item, "payload", {}) or {}
        text = (
            payload.get("page_content")
            or payload.get("text")
            or payload.get("metadata", {}).get("page_content")
            or ""
        )
        texts.append(text)
    return texts


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
        return await get_reranked_documents_from_runpod(
            endpoint_id=endpoint_id,
            docs=candidate_texts,
            query=query,
            model=RERANKER_MODEL or "BAAI/bge-reranker-large",
            timeout=config.get_reranker_timeout(),
        )
    except Exception as e:
        logger.warning(f"Reranker failed, using vector similarity order: {e}")
        return None


async def get_mcp_context(
    request: GenerationRequest,
) -> tuple[list, list, Dict[str, Optional[float]]]:
    """Call MCP semantic search and return (context, results) like get_rag_context."""
    mcp_client = MultiServerMCPClientService.get_shared()

    # Build tool arguments
    args: Dict[str, Any] = {
        "query": request.query,
        "topN": request.k,
        "threshold": request.score_threshold,
    }
    if isinstance(request.year, list) and len(request.year) >= 2:
        args["start_year"] = request.year[0]
        args["end_year"] = request.year[1]

    # Call tool with latency measurement
    mcp_start = time.perf_counter()
    raw = await mcp_client.call_tool_on_server("eve-mcp-demo", "semanticSearch", args)
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
            [],
            {
                "mcp_retrieval_latency": mcp_retrieval_latency,
                "mcp_docs_reranking_latency": None,
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

    # Helper to extract displayable text
    def _extract_text(obj: Dict[str, Any]) -> str:
        if not isinstance(obj, dict):
            return str(obj)
        return (
            obj.get("document")
            or obj.get("content")
            or obj.get("text")
            or obj.get("snippet")
            or obj.get("abstract")
            or obj.get("page_content")
            or ""
        )

    # Reduce to {text, metadata}
    simplified_all: List[Dict[str, Any]] = []
    for r in extracted:
        text_val = _extract_text(r)
        metadata_val: Dict[str, Any] = {}
        if isinstance(r, dict):
            metadata_val = r.get("metadata") or r.get("meta") or {}
        simplified_all.append({"text": text_val, "metadata": metadata_val})

    # Prepare candidate texts for optional reranking
    candidate_texts = [item["text"] for item in simplified_all]

    mcp_rerank_latency: Optional[float] = None
    rerank_start = time.perf_counter()
    reranked = await _maybe_rerank(candidate_texts, request.query)
    if isinstance(reranked, list) and reranked:
        mcp_rerank_latency = time.perf_counter() - rerank_start
        try:
            reranked_sorted = sorted(
                reranked, key=lambda x: x.get("relevance_score", 0), reverse=True
            )
            top_reranked = reranked_sorted[: request.k]

            # Map reranked documents back to original items to preserve metadata
            from collections import defaultdict

            text_to_indices: Dict[str, List[int]] = defaultdict(list)
            for idx, item in enumerate(simplified_all):
                text_to_indices[item["text"]].append(idx)

            mapped_results: List[Dict[str, Any]] = []
            for r in top_reranked:
                doc_text = r.get("document", "")
                indices = text_to_indices.get(doc_text, [])
                if indices:
                    mapped_idx = indices.pop(0)
                    mapped_results.append(simplified_all[mapped_idx])
                else:
                    mapped_results.append({"text": doc_text, "metadata": {}})

            context_list = [item["text"] for item in mapped_results]
            return (
                context_list,
                mapped_results,
                {
                    "mcp_retrieval_latency": mcp_retrieval_latency,
                    "mcp_docs_reranking_latency": mcp_rerank_latency,
                },
            )
        except Exception as e:
            logger.warning(
                f"Failed to process reranker output for MCP, using original order: {e}"
            )

    # Fallback: use the first k results as-is
    trimmed = simplified_all[: request.k]
    context_list = [item["text"] for item in trimmed]
    return (
        context_list,
        trimmed,
        {
            "mcp_retrieval_latency": mcp_retrieval_latency,
            "mcp_docs_reranking_latency": mcp_rerank_latency,
        },
    )


async def get_rag_context(
    vector_store: VectorStoreManager, request: GenerationRequest
) -> tuple[list, list, Dict[str, Optional[float]]]:
    """Get RAG context from vector store."""
    # Retrieve with latency measurements
    results, vs_latencies = await vector_store.retrieve_documents_with_latencies(
        collection_names=request.collection_ids,
        query=request.query,
        k=request.k,
        score_threshold=request.score_threshold,
        embeddings_model=request.embeddings_model,
        filters=request.filters,
    )

    if not results:
        logger.warning(f"No documents found for query: {request.query}")
        return [], [], {**(vs_latencies or {}), "qdrant_docs_reranking_latency": None}

    # Extract plain text for reranker input (support both 'page_content' and 'text')
    candidate_texts = _extract_candidate_texts(results)

    # Optionally rerank and build context directly from reranker documents
    qdrant_rerank_latency: Optional[float] = None
    rerank_start = time.perf_counter()
    reranked = await _maybe_rerank(candidate_texts, request.query)
    if isinstance(reranked, list) and reranked:
        qdrant_rerank_latency = time.perf_counter() - rerank_start
        try:
            reranked_sorted = sorted(
                reranked, key=lambda x: x.get("relevance_score", 0), reverse=True
            )
            top_reranked = reranked_sorted[: request.k]
            trimmed = results[: request.k]
            context_list = [r.get("document", "") for r in top_reranked]
            return (
                context_list,
                trimmed,
                {
                    **vs_latencies,
                    "qdrant_docs_reranking_latency": qdrant_rerank_latency,
                },
            )
        except Exception as e:
            logger.warning(
                f"Failed to process reranker output, using vector similarity order: {e}"
            )

    # Fallback: use the first k results (already sorted by vector score upstream)
    trimmed = results[: request.k]
    trimmed_texts = _extract_candidate_texts(trimmed)
    return (
        trimmed_texts,
        trimmed,
        {
            **vs_latencies,
            "qdrant_docs_reranking_latency": qdrant_rerank_latency,
        },
    )


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

            context_list, results, rag_lat = await get_rag_context(
                vector_store, request
            )
            mcp_context_list, mcp_results, mcp_lat = await get_mcp_context(request)

            # Merge RAG and MCP contexts and results
            context_list = context_list + mcp_context_list
            # deduplicate context_list
            context_list = list(set(context_list))
            context = _build_context(context_list)
            results = results + mcp_results
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
        gen_latency: Optional[float] = None
        if _langgraph_available and conversation_id:
            try:
                # Single invoke to append assistant response to memory (async)
                gen_start = time.perf_counter()
                result = await _ainvoke_with_langgraph(
                    messages_for_turn, conversation_id
                )
                gen_latency = time.perf_counter() - gen_start
                messages_out = result.get("messages")
                final_answer = _extract_final_assistant_content(messages_out) or ""
            except Exception as e:
                # If LangGraph/Runpod path fails, fallback to direct generation (with internal Mistral fallback)
                logger.warning(
                    f"LangGraph invocation failed, falling back to direct generation: {e}"
                )
                gen_start = time.perf_counter()
                final_answer = llm_manager.generate_answer_mistral(
                    query=request.query,
                    context=context,
                    max_new_tokens=request.max_new_tokens,
                )
                gen_latency = time.perf_counter() - gen_start
        else:
            # Fallback: use existing direct generation path without memory persistence
            gen_start = time.perf_counter()
            final_answer = llm_manager.generate_answer_mistral(
                query=request.query,
                context=context,
                max_new_tokens=request.max_new_tokens,
            )
            gen_latency = time.perf_counter() - gen_start

        answer = _normalize_ai_output(final_answer or "")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        "generation_latency": gen_latency,
        "hallucination_latency": hallucination_latency,
        "total_latency": total_latency,
    }
    return answer, results, is_rag, loop_result, latencies


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

        summary_text = _summarize_history_with_runpod(summarizer_input)
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
