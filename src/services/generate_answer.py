"""Endpoint to generate an answer using a language model and vector store."""

import json
import logging
import asyncio
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from src.core.vector_store_manager import VectorStoreManager
from src.database.models.collection import Collection
from src.core.llm_manager import LLMManager
from src.constants import (
    DEFAULT_QUERY,
    DEFAULT_COLLECTION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM,
    DEFAULT_K,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_GET_UNIQUE_DOCS,
    DEFAULT_MAX_NEW_TOKENS,
)

logger = logging.getLogger(__name__)


# No direct OpenAI client usage; we use Runpod-backed ChatOpenAI via LLMManager


class GenerationRequest(BaseModel):
    query: str = DEFAULT_QUERY
    collection_ids: List[str] = Field(default_factory=lambda: [], exclude=True)
    llm: str = DEFAULT_LLM  # or openai
    embeddings_model: str = DEFAULT_EMBEDDING_MODEL
    k: int = DEFAULT_K
    score_threshold: float = Field(DEFAULT_SCORE_THRESHOLD, ge=0.0, le=1.0)
    get_unique_docs: bool = DEFAULT_GET_UNIQUE_DOCS  # Fixed typo
    max_new_tokens: int = Field(DEFAULT_MAX_NEW_TOKENS, ge=100, le=8192)
    use_rag: bool = True


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


def _build_mongodb_uri() -> str:
    """Build MongoDB URI from environment, matching src.database.mongo defaults."""
    import os

    mongo_host = os.getenv("MONGO_HOST", "localhost")
    mongo_port = os.getenv("MONGO_PORT", "27017")
    mongo_database = os.getenv("MONGO_DATABASE", "eve_backend")
    return f"mongodb://{mongo_host}:{mongo_port}/{mongo_database}"


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
        llm = LLMManager().get_langchain_chat_llm()

        async def call_model(state: "MessagesState"):
            response = await llm.ainvoke(state["messages"])  # returns an AIMessage
            return {"messages": response}

        builder = StateGraph(MessagesState)
        builder.add_node(call_model)
        builder.add_edge(START, "call_model")

        # Try Mongo-backed checkpointer first, keep it open
        try:
            uri = _build_mongodb_uri()
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


def _summarize_context_with_runpod(context: str, max_tokens: int = 600) -> str:
    """Summarize retrieved documents into a compact system message using Runpod ChatOpenAI."""
    if not context:
        return ""
    llm = LLMManager().get_langchain_chat_llm()
    system = (
        "Summarize the following retrieved passages into a concise, factual brief. "
        "Preserve key entities, numbers, and references. Maximize information density."
    )
    if SystemMessage and HumanMessage:
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=f"Retrieved context to summarize:\n{context}"),
        ]
    else:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Retrieved context to summarize:\n{context}"},
        ]
    try:
        resp = llm.bind(max_tokens=max_tokens).invoke(messages)
        return getattr(resp, "content", str(resp))
    except Exception:
        return ""


def _summarize_history_with_runpod(transcript: str, max_tokens: int = 400) -> str:
    """Summarize entire conversation history."""
    if not transcript:
        return ""
    llm = LLMManager().get_langchain_chat_llm()
    system = (
        "Create a rolling summary of the dialogue capturing goals, constraints, "
        "named entities, preferences, and decisions. Keep it compact and actionable."
    )
    if SystemMessage and HumanMessage:
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=f"Conversation transcript:\n{transcript}"),
        ]
    else:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Conversation transcript:\n{transcript}"},
        ]
    try:
        resp = llm.bind(max_tokens=max_tokens).invoke(messages)
        return getattr(resp, "content", str(resp))
    except Exception:
        return ""


async def get_rag_context(
    vector_store: VectorStoreManager, request: GenerationRequest
) -> tuple[str, list]:
    """Get RAG context from vector store."""
    # Remove duplicate vector_store initialization
    results = await vector_store.retrieve_documents_from_query(
        query=request.query,
        # todo extend support for multiple collections
        collection_name=request.collection_ids[0],
        embeddings_model=request.embeddings_model,
        score_threshold=request.score_threshold,
        get_unique_docs=request.get_unique_docs,
        k=request.k,
    )

    if not results:
        print(f"No documents found for query: {request.query}")
        return "", []

    retrieved_documents = [
        result.payload.get("page_content") or result.payload.get("text") or ""
        for result in results
    ]
    context = "\n".join(retrieved_documents)
    return context, results


async def setup_rag_and_context(request: GenerationRequest):
    """Setup RAG and get context for the request."""
    # Check if we need to use RAG using LLMManager
    try:
        is_rag = await LLMManager().should_use_rag(request.query)
    except Exception as e:
        logger.warning(f"Failed to determine RAG usage, defaulting to no RAG: {e}")
        is_rag = False

    # Get context if using RAG
    if is_rag:
        vector_store = VectorStoreManager(embeddings_model=request.embeddings_model)
        context, results = await get_rag_context(vector_store, request)
    else:
        context, results = "", []

    return context, results, is_rag


async def generate_answer(
    request: GenerationRequest,
    conversation_id: Optional[str] = None,
) -> tuple[str, list, bool]:
    """Generate an answer using RAG and LLM."""
    llm_manager = LLMManager()

    try:
        context, results, is_rag = await setup_rag_and_context(request)

        # Build messages for LangGraph memory + generation
        messages_for_turn: List[Any] = []

        # If RAG was used and we have context, add a system summary of the context
        if is_rag and context:
            ctx_summary = _summarize_context_with_runpod(context)
            if ctx_summary:
                if SystemMessage:
                    messages_for_turn.append(
                        SystemMessage(content=f"Context summary:\n{ctx_summary}")
                    )
                else:
                    messages_for_turn.append(
                        {
                            "role": "system",
                            "content": f"Context summary:\n{ctx_summary}",
                        }
                    )

        # Append the user message
        if HumanMessage:
            messages_for_turn.append(HumanMessage(content=request.query))
        else:
            messages_for_turn.append({"role": "user", "content": request.query})

        # Use LangGraph with MongoDB checkpointer for short-term memory if available
        final_answer: Optional[str] = None
        if _langgraph_available and conversation_id:
            # Single invoke to append assistant response to memory (async)
            result = await _ainvoke_with_langgraph(messages_for_turn, conversation_id)
            # result should contain {"messages": List[BaseMessage]} or similar
            try:
                messages_out = result.get("messages")
                final_answer = _extract_final_assistant_content(messages_out) or ""
            except Exception:
                # Fallback to direct generation if unexpected structure
                final_answer = llm_manager.generate_answer(
                    query=request.query,
                    context=context,
                    llm=request.llm,
                    max_new_tokens=request.max_new_tokens,
                )
        else:
            # Fallback: use existing direct generation path without memory persistence
            final_answer = llm_manager.generate_answer(
                query=request.query,
                context=context,
                llm=request.llm,
                max_new_tokens=request.max_new_tokens,
            )

        answer = _normalize_ai_output(final_answer or "")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return answer, results, is_rag


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

        # Build transcript from stored messages (input/output pairs)
        messages = await MessageModel.find_all(
            filter_dict={"conversation_id": conversation_id}, sort=[("timestamp", 1)]
        )
        transcript_parts: List[str] = []
        for m in messages:
            if getattr(m, "input", None):
                transcript_parts.append(f"User: {m.input}")
            if getattr(m, "output", None):
                transcript_parts.append(f"Assistant: {m.output}")
        transcript = "\n".join(transcript_parts)

        summary_text = _summarize_history_with_runpod(transcript)
        if not summary_text:
            return

        # Reset LangGraph memory for this thread and seed with the summary as a system message
        if _langgraph_available:
            # Prefer the shared Mongo checkpointer when available; otherwise best-effort local context
            try:
                graph, mode = await _get_or_create_compiled_graph()
                if mode == "mongo" and _mongo_checkpointer is not None:
                    try:
                        await _mongo_checkpointer.delete_thread(conversation_id)
                    except Exception:
                        pass
                else:
                    try:
                        uri = _build_mongodb_uri()
                        async with AsyncMongoDBSaver.from_conn_string(
                            uri
                        ) as checkpointer:
                            await checkpointer.delete_thread(conversation_id)
                    except Exception:
                        pass
            except Exception:
                pass
            await _ainvoke_with_langgraph(
                [
                    {
                        "role": "system",
                        "content": f"Conversation summary up to now:\n{summary_text}",
                    }
                ],
                conversation_id,
            )
    except Exception:
        # Non-critical path; ignore errors
        return


async def generate_answer_stream_generator_helper(
    request: GenerationRequest, output_format: str = "plain"
):
    """Helper function to generate streaming answer with different output formats."""
    llm_manager = LLMManager()

    try:
        context, results, is_rag = await setup_rag_and_context(request)

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
