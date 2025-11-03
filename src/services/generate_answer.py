"""Endpoint to generate an answer using a language model and vector store."""

import json
import logging
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from fastapi import BackgroundTasks, HTTPException
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, PrivateAttr

from src.core.vector_store_manager import VectorStoreManager
from src.core.llm_manager import LLMManager, LLMType
from src.constants import (
    DEFAULT_QUERY,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM,
    DEFAULT_K,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    MCP_MAX_TOP_N,
    MODEL_CONTEXT_SIZE,
    POLICY_NOT_ANSWER,
    POLICY_PROMPT,
    RERANKER_MODEL,
    TOKEN_OVERFLOW_LIMIT,
)
from src.utils.helpers import (
    build_conversation_context,
    extract_document_data,
    get_mongodb_uri,
    trim_context_to_token_limit,
)
from src.utils.runpod_utils import get_reranked_documents_from_runpod
from src.services.mcp_client_service import MultiServerMCPClientService
from src.config import (
    DEEPINFRA_API_TOKEN,
    IS_PROD,
    SCRAPING_DOG_API_KEY,
    SILICONFLOW_API_TOKEN,
    SATCOM_QDRANT_URL,
    SATCOM_QDRANT_API_KEY,
    config,
)
from src.hallucination_pipeline.loop import run_hallucination_loop
from src.hallucination_pipeline.schemas import generation_schema
from src.utils.deepinfra_reranker import DeepInfraReranker
from src.utils.template_loader import get_template
from src.utils.siliconflow_reranker import SiliconFlowReranker
from src.utils.helpers import (
    get_mongodb_uri,
    tiktoken_counter,
)
import contextlib

from src.utils.scraping_dog_crawler import ScrapingDogCrawler
from src.constants import SCRAPING_DOG_ALL_URLS


logger = logging.getLogger(__name__)

# Shared LLMManager instance for this module
_shared_llm_manager: Optional[LLMManager] = None


def get_shared_llm_manager() -> LLMManager:
    """Return a process-wide shared LLMManager instance."""
    global _shared_llm_manager
    if _shared_llm_manager is None:
        _shared_llm_manager = LLMManager()
    return _shared_llm_manager


# No direct OpenAI client usage; we use Runpod-backed ChatOpenAI via LLMManager


class ShouldUseRagDecision(BaseModel):
    """Schema for deciding whether to use RAG."""

    use_rag: bool = Field(
        description="True if the query should use RAG; False for casual/generic queries."
    )
    reason: str = Field(description="Brief justification for the decision.")
    requery: str = Field(
        description="Get new rewritten query from last query and conversation history."
    )


class PolicyCheck(BaseModel):
    violation: bool = Field(
        description="True if the input violates policies otherwise False"
    )
    reason: str = Field(description="Reason for the violation")


class GenerationRequest(BaseModel):
    query: str = DEFAULT_QUERY
    year: Optional[List[int]] = None
    filters: Optional[Dict[str, Any]] = None
    llm_type: Optional[str] = Field(
        default=None,
        description=(
            "LLM type to use. Options: 'runpod', 'mistral', 'satcom_small', 'satcom_large'. "
            "Defaults to None, which means environment-based behavior."
        ),
    )
    embeddings_model: str = DEFAULT_EMBEDDING_MODEL
    k: int = Field(DEFAULT_K, ge=0, le=10)
    temperature: float = Field(DEFAULT_TEMPERATURE, ge=0.0, le=1.0)
    score_threshold: float = Field(DEFAULT_SCORE_THRESHOLD, ge=0.0, le=1.0)
    max_new_tokens: int = Field(DEFAULT_MAX_NEW_TOKENS, ge=100, le=100_000)
    public_collections: List[str] = Field(
        default_factory=list,
        description="List of public collection names to include in the search",
    )
    hallucination_loop_flag: bool = False  # For testing purposes

    _collection_ids: List[str] = PrivateAttr(default_factory=list)
    _private_collections_map: Dict[str, str] = PrivateAttr(default_factory=dict)

    @property
    def collection_ids(self) -> List[str]:
        return self._collection_ids

    @collection_ids.setter
    def collection_ids(self, value: List[str]) -> None:
        self._collection_ids = list(value) if value else []

    @property
    def private_collections_map(self) -> Dict[str, str]:
        return self._private_collections_map

    @private_collections_map.setter
    def private_collections_map(self, value: Dict[str, str]) -> None:
        self._private_collections_map = value


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
    from langchain_core.messages import (
        trim_messages,
        HumanMessage,
        AIMessage,
        SystemMessage,
    )

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


# --- System prompt resolution (default vs. satcom) ---
try:
    _DEFAULT_SYSTEM_PROMPT: Optional[str] = get_template(
        "system_prompt", filename="system.yaml"
    )
except Exception:
    _DEFAULT_SYSTEM_PROMPT = None


def _resolve_system_prompt(llm_type: Optional[str]) -> Optional[str]:
    """Return the appropriate system prompt for the given llm_type.

    Uses satcom-specific system prompt when llm_type is satcom,
    otherwise falls back to the default system prompt loaded above.
    """
    try:
        if llm_type in (LLMType.Satcom_Small.value, LLMType.Satcom_Large.value):
            return get_template("system_prompt", filename="satcom/system.yaml")
    except Exception:
        # Fallback to default if satcom template is missing or fails to load
        pass
    return _DEFAULT_SYSTEM_PROMPT


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

        # Define a custom state so we can carry sampling params alongside messages
        class GenerationState(MessagesState):
            temperature: Optional[float]
            max_tokens: Optional[int]
            conversation_summary: Optional[str]
            is_streaming: Optional[bool]
            llm_type: Optional[str]

        async def call_model(state: GenerationState):
            # Get conversation summary and all messages
            conversation_summary = state.get("conversation_summary")
            all_messages = state["messages"]

            available_tokens = DEFAULT_MAX_NEW_TOKENS

            system_prompt = _resolve_system_prompt(state.get("llm_type"))
            if system_prompt:
                all_messages = [_make_message("system", system_prompt)] + all_messages

            if conversation_summary:
                summary_context = f"""Previous conversation summary: {conversation_summary}

Please continue the conversation using this summary as context for understanding the conversation history."""
                all_messages = [_make_message("user", summary_context)] + all_messages

            context_messages = trim_messages(
                all_messages,
                max_tokens=available_tokens,
                strategy="last",
                token_counter=tiktoken_counter,
                include_system=False,
                start_on="human",
                end_on=("human", "tool"),
            )
            bind_kwargs = {
                # "max_tokens": MODEL_CONTEXT_SIZE - tiktoken_counter(context_messages)
            }
            try:
                temperature_val = state.get("temperature")
                bind_kwargs["temperature"] = temperature_val
            except Exception:
                pass

            # Optimize for streaming performance
            try:
                is_streaming = state.get("is_streaming")
                if is_streaming:
                    bind_kwargs.update(
                        {
                            "stream": True,  # Enable streaming
                        }
                    )
            except Exception:
                pass

            logger.info(f"bind_kwargs: {bind_kwargs}")
            llm_manager = get_shared_llm_manager()
            llm = llm_manager.get_client_for_model(state.get("llm_type"))
            bound_llm = llm.bind(**bind_kwargs)

            final_messages = context_messages
            system_prompt = _resolve_system_prompt(state.get("llm_type"))
            if system_prompt:
                final_messages = [_make_message("system", system_prompt)] + list(
                    context_messages
                )

            response = await bound_llm.ainvoke(final_messages)
            return {"messages": [response]}

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


async def _get_conversation_history_from_db(
    conversation_id: str,
) -> tuple[List[Any], Optional[str]]:
    """Get conversation history and summary from database for fallback when LangGraph is not available."""
    try:
        from src.database.models.message import Message as MessageModel
        from src.database.models.conversation import Conversation as ConversationModel

        summary = None
        try:
            conversation = await ConversationModel.find_by_id(conversation_id)
            if conversation and getattr(conversation, "summary", None):
                summary = conversation.summary
        except Exception:
            summary = None

        # Get recent messages from the conversation (limit to last 10 for context)
        messages = await MessageModel.find_all(
            filter_dict={"conversation_id": conversation_id},
            sort=[("timestamp", -1)],
            limit=1,
            skip=1,
        )
        # Convert to LangChain-compatible message format
        history = []
        for msg in messages:
            if getattr(msg, "input", None):
                history.append(_make_message("user", msg.input))
            if getattr(msg, "output", None):
                history.append(_make_message("assistant", msg.output))

        return history, summary
    except Exception as e:
        logger.warning(f"Failed to get conversation history from database: {e}")
        return [], None


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
            return trim_context_to_token_limit(parts, TOKEN_OVERFLOW_LIMIT)
    except Exception:
        # On any error, fall back to simple join of strings
        pass

    # Default: treat as list of strings
    return trim_context_to_token_limit(items, TOKEN_OVERFLOW_LIMIT)


async def _maybe_rerank_runpod(
    candidate_texts: List[str], query: str
) -> List[dict] | None:
    """Call RunPod reranker if configured.

    Returns results with 'relevance_score' and 'index' fields.
    Use _sort_runpod_reranked_results() to process these results.
    """
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


def _sort_runpod_reranked_results(
    reranked: List[dict], candidate_indices: List[int]
) -> List[int]:
    """Sort results from RunPod reranker.

    RunPod returns results with 'relevance_score' and 'index' fields.
    Results are already sorted by relevance_score in descending order.
    """
    ordered_indices = [
        candidate_indices[item.get("index", i)]
        for i, item in enumerate(reranked)
        if isinstance(item, dict)
    ]
    return ordered_indices


def _sort_deepinfra_reranked_results(
    reranked: dict, candidate_indices: List[int]
) -> List[int]:
    """Sort results from DeepInfra reranker.

    DeepInfra returns results with 'scores' array where each score corresponds
    to the relevance of the document at that index.
    """
    if not isinstance(reranked, dict) or "scores" not in reranked:
        return list(candidate_indices)

    scores = reranked["scores"]
    if not isinstance(scores, list) or len(scores) != len(candidate_indices):
        return list(candidate_indices)

    # Create list of (score, original_index) pairs and sort by score descending
    scored_indices = [(scores[i], candidate_indices[i]) for i in range(len(scores))]
    scored_indices.sort(key=lambda x: x[0], reverse=True)

    # Extract the ordered indices
    ordered_indices = [idx for _, idx in scored_indices]
    return ordered_indices


def _maybe_rerank_deepinfra(
    candidate_texts: List[str], query: str, timeout: int = 5
) -> dict | None:
    """Call DeepInfra reranker if configured with timeout, else fall back to SiliconFlow."""
    if not candidate_texts:
        return None

    # --- Try DeepInfra first ---
    api_token = DEEPINFRA_API_TOKEN
    if not api_token:
        logger.warning("DEEPINFRA_API_TOKEN environment variable not set")
    else:
        reranker = DeepInfraReranker(api_token)
        executor = ThreadPoolExecutor(max_workers=1)
        logger.info("Using DeepInfra reranker")
        future = executor.submit(reranker.rerank, [query], candidate_texts)
        try:
            results = future.result(timeout=timeout)
            return results
        except FutureTimeoutError:
            logger.warning("DeepInfra reranker timed out after %s seconds", timeout)
            future.cancel()
        except Exception:
            logger.warning("DeepInfra reranker failed", exc_info=True)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    # --- Fallback: SiliconFlow ---
    logger.info("Using SiliconFlow reranker")
    api_token = SILICONFLOW_API_TOKEN
    if not api_token:
        logger.warning("SILICONFLOW_API_TOKEN environment variable not set")
        return None

    try:
        reranker = SiliconFlowReranker(api_token)
        results = reranker.rerank([query], candidate_texts)
        return {"scores": results} if not isinstance(results, dict) else results
    except Exception:
        logger.warning("SiliconFlow reranker failed", exc_info=True)
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
        "topN": min(MCP_MAX_TOP_N, request.k * 2),
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

    for it in extracted:
        if isinstance(it, dict):
            it["collection_name"] = "Wiley AI Gateway"

    latencies: Dict[str, Optional[float]] = {
        "mcp_retrieval_latency": mcp_retrieval_latency,
    }

    return extracted, latencies


async def should_use_rag(
    llm_manager: LLMManager,
    query: str,
    conversation: str,
    llm_type: Optional[str] = None,
) -> tuple[ShouldUseRagDecision, str]:
    """Decide whether to use RAG for the given query using the Runpod-backed ChatOpenAI.

    Returns True for scientific/technical queries; False for casual/generic ones.
    Defaults to True on uncertainty/errors.
    """
    try:
        if llm_type in (LLMType.Satcom_Small.value, LLMType.Satcom_Large.value):
            tmpl = get_template("is_rag_prompt", filename="satcom/prompts.yaml")
        else:
            tmpl = get_template("is_rag_prompt", filename="prompts.yaml")
        prompt = tmpl.format(conversation=conversation, query=query)
        base_llm = llm_manager.get_client_for_model(llm_type)
        logger.info(
            f"deciding should_use_rag with selected model: {llm_manager.get_selected_llm_type()}"
        )

        structured_llm = base_llm.bind(temperature=0).with_structured_output(
            ShouldUseRagDecision
        )
        result = await structured_llm.ainvoke(prompt)
        logger.info(f"should_use_rag result: {result}")
        # with_structured_output returns a Pydantic object matching the schema
        if isinstance(result, ShouldUseRagDecision):
            return result, prompt
        return None, prompt
    except Exception as e:
        logger.error(f"Failed to decide should_use_rag: {e}")
        mistral_llm = llm_manager.get_mistral_model()
        structured_mistral_llm = mistral_llm.bind(temperature=0).with_structured_output(
            ShouldUseRagDecision
        )
        result = await structured_mistral_llm.ainvoke(prompt)
        logger.info(f"should_use_rag result from mistral: {result}")
        if isinstance(result, ShouldUseRagDecision):
            return result, prompt
        return None, prompt


async def get_rag_context(
    vector_store: VectorStoreManager, request: GenerationRequest
) -> tuple[list, Dict[str, Optional[float]]]:
    """Get RAG context from vector store."""
    # Retrieve with latency measurements
    results, vs_latencies = await vector_store.retrieve_documents_with_latencies(
        collection_names=request.collection_ids,
        query=request.query,
        k=request.k * 2,
        score_threshold=request.score_threshold,
        embeddings_model=request.embeddings_model,
        filters=request.filters,
        private_collections_map=request.private_collections_map,
    )

    if not results:
        logger.warning(f"No documents found for query: {request.query}")

    return results, vs_latencies


async def setup_rag_and_context(request: GenerationRequest):
    """Setup RAG and get context for the request."""
    latencies: Dict[str, Optional[float]] = {}
    # Get RAG context
    context, results = "", []
    try:
        vector_store = VectorStoreManager(embeddings_model=request.embeddings_model)
        rag_lat: Dict[str, Optional[float]] = {}
        mcp_results: List[Any] = []
        mcp_lat: Dict[str, Optional[float]] = {}
        satcom_results: List[Any] = []
        satcom_lat: Dict[str, Optional[float]] = {}

        # temporary for satcom collection
        if not IS_PROD and "satcom-chunks-collection" in request.public_collections:
            satcom_vector_store = VectorStoreManager(
                embeddings_model=request.embeddings_model,
                qdrant_url=SATCOM_QDRANT_URL,
                qdrant_api_key=SATCOM_QDRANT_API_KEY,
            )
            collection_ids = request.collection_ids
            request.collection_ids = ["satcom-chunks-collection"]
            satcom_results, satcom_lat = await get_rag_context(
                satcom_vector_store, request
            )
            request.collection_ids = [
                cid for cid in collection_ids if cid != "satcom-chunks-collection"
            ]
            latencies.update(satcom_lat or {})

        results, rag_lat = await get_rag_context(vector_store, request)
        if "Wiley AI Gateway" in request.public_collections:
            mcp_results, mcp_lat = await get_mcp_context(request)

        merged_results = list(results) + list(mcp_results) + list(satcom_results)
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

        # Rerank candidates using DeepInfra reranker
        latency = time.perf_counter()
        reranked = _maybe_rerank_deepinfra(candidate_texts, request.query)
        latencies["reranking_latency"] = time.perf_counter() - latency
        if reranked:
            ordered_indices = _sort_deepinfra_reranked_results(
                reranked, candidate_indices
            )
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
        raise Exception(
            f"Failed to get RAG context and MCP context and merge them: {e}"
        )

    return context, results, latencies


async def check_policy(
    request: GenerationRequest, llm_manager: LLMManager
) -> tuple[PolicyCheck, str]:
    """Check if the query violates EO policies."""
    policy_prompt = POLICY_PROMPT.format(question=request.query)
    try:
        base_llm = llm_manager.get_client_for_model(request.llm_type)
        logger.info(
            f"checking policy with selected model: {llm_manager.get_selected_llm_type()}"
        )

        structured_llm = base_llm.bind(temperature=0).with_structured_output(
            PolicyCheck
        )
        policy_result = await structured_llm.ainvoke(policy_prompt)
        logger.info(f"policy_result: {policy_result}")
        return policy_result, policy_prompt
    except Exception as e:
        logger.warning(f"Failed to check policy with main model: {e}")
        base_llm = llm_manager.get_mistral_model()
        structured_llm = base_llm.bind(temperature=0).with_structured_output(
            PolicyCheck
        )
        policy_result = await structured_llm.ainvoke(policy_prompt)
        logger.info(f"policy_result from mistral: {policy_result}")
        return policy_result, policy_prompt


async def generate_answer(
    request: GenerationRequest,
    conversation_id: Optional[str] = None,
) -> tuple[str, list, bool, dict, Dict[str, Optional[float]], Dict[str, str]]:
    """Generate an answer using RAG and LLM."""
    llm_manager = get_shared_llm_manager()

    try:
        # Check if the query violates EO policies
        total_start = time.perf_counter()
        policy_result, policy_prompt = await check_policy(request, llm_manager)
        guardrail_latency = time.perf_counter() - total_start
        if policy_result.violation:
            return POLICY_NOT_ANSWER, [], False, {}, {}, {}

        # Get conversation history and summary for multi-turn context
        conversation_history, conversation_summary = (
            await _get_conversation_history_from_db(conversation_id)
            if conversation_id
            else (None, None)
        )
        conversation_context = build_conversation_context(
            conversation_history, conversation_summary
        )

        # Check if we need to use RAG using LLMManager
        rag_decision_start = time.perf_counter()
        rag_decision_result, rag_decision_prompt = await should_use_rag(
            llm_manager, request.query, conversation_context, request.llm_type
        )
        request.query = rag_decision_result.requery
        rag_decision_latency = time.perf_counter() - rag_decision_start
        context, results, latencies = "", [], {}
        if (
            len(request.collection_ids) > 0
            and request.k > 0
            and rag_decision_result.use_rag
        ):
            try:
                context, results, latencies = await setup_rag_and_context(request)
            except Exception as e:
                logger.warning(f"Failed to setup RAG and context")
                scraping_dog_start = time.perf_counter()
                scraping_dog_crawler = ScrapingDogCrawler(
                    all_urls=SCRAPING_DOG_ALL_URLS, api_key=SCRAPING_DOG_API_KEY
                )
                results = await scraping_dog_crawler.run(request.query, request.k)
                context = _build_context(results)
                latencies = {
                    "scraping_dog_latency": time.perf_counter() - scraping_dog_start,
                }

        # Build messages for LangGraph memory + generation
        messages_for_turn: List[Any] = []

        # Build user message using template conversation_prompt_with_context
        if rag_decision_result.use_rag:
            if request.llm_type in (
                LLMType.Satcom_Small.value,
                LLMType.Satcom_Large.value,
            ):
                tmpl = get_template(
                    "rag_prompt_for_langgraph", filename="satcom/prompts.yaml"
                )
            else:
                tmpl = get_template("rag_prompt_for_langgraph", filename="prompts.yaml")
        else:
            if request.llm_type in (
                LLMType.Satcom_Small.value,
                LLMType.Satcom_Large.value,
            ):
                tmpl = get_template(
                    "no_rag_prompt_for_langgraph", filename="satcom/prompts.yaml"
                )
            else:
                tmpl = get_template(
                    "no_rag_prompt_for_langgraph", filename="prompts.yaml"
                )
        user_content = tmpl.format(
            context=context or "",
            query=request.query or "",
        )

        # Append the templated user message
        messages_for_turn.append(_make_message("user", user_content))

        # Use LangGraph with MongoDB checkpointer for short-term memory if available
        final_answer: Optional[str] = None
        base_gen_latency: Optional[float] = None
        mistral_gen_latency: Optional[float] = None
        use_langgraph = False
        if _langgraph_available and conversation_id:
            try:
                # Single invoke to append assistant response to memory (async)
                gen_start = time.perf_counter()
                # Pass temperature to the graph via state config
                graph, mode = await _get_or_create_compiled_graph()
                if graph is not None:
                    config = {"configurable": {"thread_id": conversation_id}}
                    state = {
                        "messages": add_messages([], messages_for_turn),
                        "temperature": request.temperature,
                        "max_tokens": request.max_new_tokens,
                        "conversation_summary": conversation_summary,
                        "is_streaming": False,
                        "llm_type": request.llm_type,
                    }
                    result = await graph.ainvoke(state, config)
                else:
                    result = None
                base_gen_latency = time.perf_counter() - gen_start
                messages_out = result.get("messages")
                final_answer = _extract_final_assistant_content(messages_out) or ""
                use_langgraph = True
            except Exception as e:
                # If LangGraph/Runpod path fails, fallback to direct generation with conversation history
                logger.warning(
                    f"LangGraph invocation failed, falling back to direct generation: {e}"
                )
                use_langgraph = False
        if not use_langgraph:
            logger.info("starting to fallback using mistral model")
            gen_start = time.perf_counter()

            final_answer, generation_prompt = await llm_manager.generate_answer_mistral(
                query=request.query,
                context=context,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                conversation_context=conversation_context,
            )
            mistral_gen_latency = time.perf_counter() - gen_start
            user_content = generation_prompt

        model_client = llm_manager.get_client_for_model(request.llm_type)
        generation_response = generation_schema(
            question=request.query, answer=final_answer
        )
        loop_result: Optional[dict] = None
        hallucination_latency: Optional[dict] = None
        if len(context.strip()) != 0 and request.hallucination_loop_flag:
            logger.info("starting to run hallucination loop")
            try:
                loop_result, hallucination_latency = await run_hallucination_loop(
                    model_client,
                    context,
                    generation_response,
                    request.collection_ids,
                )
                final_answer = loop_result["final_answer"]
            except Exception as e:
                logger.warning(f"Failed to run hallucination loop: {e}")
                logger.info("falling back to mistral model for hallucination loop")
                model_client = llm_manager.get_mistral_model()
                loop_result, hallucination_latency = await run_hallucination_loop(
                    model_client,
                    context,
                    generation_response,
                    request.collection_ids,
                )
                final_answer = loop_result["final_answer"]

        total_latency = time.perf_counter() - total_start
        latencies = {
            "guardrail_latency": guardrail_latency,
            "rag_decision_latency": rag_decision_latency,
            **(latencies or {}),
            "base_generation_latency": base_gen_latency,
            "fallback_latency": mistral_gen_latency,
            "hallucination_latency": hallucination_latency,
            "total_latency": total_latency,
        }
        prompts = {
            "guardrail_prompt": policy_prompt,
            "guardrail_result": policy_result,
            "is_rag_prompt": rag_decision_prompt,
            "rag_decision_result": rag_decision_result,
            "generation_prompt": user_content,
        }
        return (
            final_answer,
            results,
            rag_decision_result.use_rag,
            loop_result,
            latencies,
            prompts,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def maybe_rollup_and_trim_history(conversation_id: str, summary_every: int = 2):
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

        summary_text = await get_shared_llm_manager().summarize_context_in_all(
            transcript=summarizer_input, is_force=True
        )
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
    request: GenerationRequest,
    conversation_id: str,
    message_id: str,
    output_format: str = "plain",
    background_tasks: BackgroundTasks = None,
):
    """Stream tokens as Server-Sent Events while accumulating and persisting the final result."""
    llm_manager = get_shared_llm_manager()
    llm_manager.set_selected_llm_type(request.llm_type)

    try:
        total_start = time.perf_counter()
        # # Guardrail check
        # policy_result, policy_prompt = await check_policy(request, llm_manager)
        # guardrail_latency = time.perf_counter() - total_start
        # if policy_result.violation:
        #     yield f"data: {json.dumps({'type': 'final', 'answer': POLICY_NOT_ANSWER})}\n\n"
        #     try:
        #         from src.database.models.message import Message as MessageModel

        #         # Find message by id
        #         message = await MessageModel.find_by_id(message_id)
        #         if message is not None:
        #             message.output = POLICY_NOT_ANSWER
        #             existing_metadata = dict(getattr(message, "metadata", {}) or {})
        #             existing_metadata.update(
        #                 {
        #                     "latencies": {"guardrail_latency": guardrail_latency},
        #                     "prompts": {
        #                         "guardrail_prompt": policy_prompt,
        #                         "guardrail_result": policy_result,
        #                     },
        #                 }
        #             )
        #             message.metadata = existing_metadata
        #             await message.save()
        #     except Exception as e:
        #         logger.warning(f"Failed to persist streamed message: {e}")
        #     return

        # Prepare conversation history for fallback path
        conversation_history, conversation_summary = (
            await _get_conversation_history_from_db(conversation_id)
            if conversation_id
            else (None, None)
        )
        conversation_context = build_conversation_context(
            conversation_history, conversation_summary
        )
        # Setup RAG context
        rag_decision_start = time.perf_counter()
        rag_decision_result, rag_decision_prompt = await should_use_rag(
            llm_manager, request.query, conversation_context, request.llm_type
        )
        if rag_decision_result.requery:
            if rag_decision_result.use_rag:
                yield f"data: {json.dumps({'type': 'requery', 'content': 'Searched for: '+rag_decision_result.requery})}\n\n"
            request.query = rag_decision_result.requery
        rag_decision_latency = time.perf_counter() - rag_decision_start
        context, results, latencies = "", [], {}
        if (
            len(request.collection_ids) > 0
            and request.k > 0
            and rag_decision_result.use_rag
        ):
            yield f"data: {json.dumps({'type': 'status', 'content': 'Retrieving relevant documents…'})}\n\n"
            try:
                context, results, latencies = await setup_rag_and_context(request)
                if len(results) == 0:
                    raise Exception("No RAG results found for the query")
            except Exception as e:
                logger.warning(f"Failed to setup RAG and context: {e}")
                scraping_dog_start = time.perf_counter()
                scraping_dog_crawler = ScrapingDogCrawler(
                    all_urls=SCRAPING_DOG_ALL_URLS, api_key=SCRAPING_DOG_API_KEY
                )
                results = await scraping_dog_crawler.run(request.query, request.k)
                context = _build_context(results)
                latencies = {
                    "scraping_dog_latency": time.perf_counter() - scraping_dog_start,
                }

        if rag_decision_result.use_rag:
            if request.llm_type in (
                LLMType.Satcom_Small.value,
                LLMType.Satcom_Large.value,
            ):
                tmpl = get_template(
                    "rag_prompt_for_langgraph", filename="satcom/prompts.yaml"
                )
            else:
                tmpl = get_template("rag_prompt_for_langgraph", filename="prompts.yaml")
        else:
            if request.llm_type in (
                LLMType.Satcom_Small.value,
                LLMType.Satcom_Large.value,
            ):
                tmpl = get_template(
                    "no_rag_prompt_for_langgraph", filename="satcom/prompts.yaml"
                )
            else:
                tmpl = get_template(
                    "no_rag_prompt_for_langgraph", filename="prompts.yaml"
                )
        user_content = tmpl.format(
            context=context or "",
            query=request.query or "",
        )

        # Stream generation
        accumulated = []

        # Choose streaming source (LangGraph memory if available → fallback mistral stream)
        mistral_gen_latency: Optional[float] = None
        base_gen_latency: Optional[float] = None
        first_token_latency: Optional[float] = None

        used_stream = False
        tokens_yielded = 0

        # Try optimized LangGraph streaming first
        if _langgraph_available and conversation_id:
            try:
                gen_start = time.perf_counter()
                graph, mode = await _get_or_create_compiled_graph()
                if graph is not None:
                    logger.info(
                        f"Using optimized LangGraph streaming with mode: {mode}"
                    )
                    config = {"configurable": {"thread_id": conversation_id}}

                    state = {
                        "messages": add_messages(
                            [], [_make_message("user", user_content)]
                        ),
                        "temperature": request.temperature,
                        "max_tokens": request.max_new_tokens,
                        "conversation_summary": conversation_summary,
                        "is_streaming": True,
                        "llm_type": request.llm_type,
                    }

                    logger.info("Using LangGraph astream for streaming")
                    llm_instruct_timeout = llm_manager.config.get_instruct_llm_timeout()
                    try:
                        # Create the async generator once so we can pull the first token with a timeout
                        astream = graph.astream(
                            state, config=config, stream_mode="messages"
                        )

                        # Enforce timeout only for the first token
                        async with asyncio.timeout(5000):
                            first_chunk, first_metadata = await astream.__anext__()
                            tokens_yielded += 1
                            if tokens_yielded == 1:
                                first_token_latency = time.perf_counter() - total_start

                        first_text = getattr(first_chunk, "content", None)
                        if first_text:
                            if output_format == "json":
                                yield f"data: {json.dumps({'type':'token','content':first_text})}\n\n"
                            else:
                                yield f"data: {first_text}\n\n"
                            accumulated.append(first_text)

                        # Continue streaming remaining tokens without a timeout
                        async for chunk, metadata in astream:
                            text = getattr(chunk, "content", None)
                            if not text:
                                continue
                            if output_format == "json":
                                yield f"data: {json.dumps({'type':'token','content':text})}\n\n"
                            else:
                                yield f"data: {text}\n\n"
                            accumulated.append(text)
                            tokens_yielded += 1

                        base_gen_latency = time.perf_counter() - gen_start
                        logger.info(
                            f"LangGraph streaming completed. Tokens yielded: {tokens_yielded}, Latency: {base_gen_latency}"
                        )
                        used_stream = True
                    except TimeoutError:
                        logger.warning(
                            f"LangGraph streaming timed out, falling back to Mistral streaming"
                        )
                        used_stream = False
                    finally:
                        # Ensure background tasks are torn down to avoid “Task was destroyed but it is pending!”
                        with contextlib.suppress(Exception):
                            await graph.aclose()
                else:
                    logger.warning(
                        "LangGraph graph is None, falling back to Mistral streaming"
                    )
            except Exception as e:
                logger.warning(
                    f"Optimized LangGraph streaming failed, falling back: {e}"
                )

        # Fallback to Mistral streaming if LangGraph didn't work or yielded no tokens
        mistral_token_yielded = 0
        mistral_first_token_latency: Optional[float] = None
        if not used_stream or tokens_yielded == 0:
            gen_start = time.perf_counter()
            logger.info(f"Using Mistral streaming fallback")
            async for (
                token,
                mistral_generation_prompt,
            ) in llm_manager.generate_answer_mistral_stream(
                query=request.query,
                context=context,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                conversation_context=conversation_context,
            ):
                if output_format == "json":
                    yield f"data: {json.dumps({'type':'token','content':token})}\n\n"
                else:
                    yield f"data: {token}\n\n"
                accumulated.append(str(token))
                mistral_token_yielded += 1
                if mistral_token_yielded == 1:
                    mistral_first_token_latency = time.perf_counter() - total_start

            mistral_gen_latency = time.perf_counter() - gen_start
            used_stream = True
            user_content = mistral_generation_prompt

        answer = "".join(accumulated)
        # Final accumulated answer without post-processing normalization

        # Optional hallucination loop after stream completes
        model_client = llm_manager.get_client_for_model(request.llm_type)
        generation_response = generation_schema(question=request.query, answer=answer)
        loop_result: Optional[dict] = None
        hallucination_latency: Optional[dict] = None
        if len((context or "").strip()) != 0 and request.hallucination_loop_flag:
            try:
                loop_result, hallucination_latency = await run_hallucination_loop(
                    model_client, context, generation_response, request.collection_ids
                )
                answer = loop_result.get("final_answer", answer)
            except Exception as e:
                logger.warning(f"Hallucination loop failed: {e}")

        total_latency = time.perf_counter() - total_start
        latencies = {
            # "guardrail_latency": guardrail_latency,
            "rag_decision_latency": rag_decision_latency,
            **(latencies or {}),
            "first_token_latency": first_token_latency,
            "mistral_first_token_latency": mistral_first_token_latency,
            "base_generation_latency": base_gen_latency,
            "fallback_latency": mistral_gen_latency,
            "hallucination_latency": hallucination_latency,
            "total_latency": total_latency,
        }
        prompts = {
            # "guardrail_prompt": policy_prompt,
            # "guardrail_result": policy_result,
            "is_rag_prompt": rag_decision_prompt,
            "rag_decision_result": rag_decision_result,
            "generation_prompt": user_content,
        }
        # Persist results to MongoDB Message entry created by the router
        try:
            from src.database.models.message import Message as MessageModel

            # Find message by id
            message = await MessageModel.find_by_id(message_id)
            if message is not None:
                documents_data = [extract_document_data(r) for r in (results or [])]

                message.output = answer
                message.documents = documents_data
                message.use_rag = rag_decision_result.use_rag
                existing_metadata = dict(getattr(message, "metadata", {}) or {})
                existing_metadata.update({"latencies": latencies, "prompts": prompts})
                message.metadata = existing_metadata
                await message.save()
        except Exception as e:
            logger.warning(f"Failed to persist streamed message: {e}")

        if background_tasks:
            background_tasks.add_task(maybe_rollup_and_trim_history, conversation_id)
        else:
            asyncio.create_task(maybe_rollup_and_trim_history(conversation_id))

        # Final event
        if output_format == "json":
            final_payload = {
                "type": "final",
                "answer": answer,
                "latencies": latencies,
            }
            yield f"data: {json.dumps(final_payload)}\n\n"
        else:
            yield f"data: [DONE]\n\n"

    except Exception as e:
        err_payload = {"type": "error", "message": str(e)}
        try:
            from src.database.models.message import Message as MessageModel

            message = await MessageModel.find_by_id(message_id)
            if message:
                message.metadata = {"error": str(e)}
                await message.save()
        except Exception:
            pass
        yield f"data: {json.dumps(err_payload)}\n\n"


async def generate_answer_stream_generator(
    request: GenerationRequest,
    conversation_id: str,
    message_id: str,
    background_tasks: BackgroundTasks = None,
):
    """Generate streaming answer using RAG and LLM."""
    async for chunk in generate_answer_stream_generator_helper(
        request, conversation_id, message_id, "plain", background_tasks
    ):
        yield chunk


async def generate_answer_json_stream_generator(
    request: GenerationRequest,
    conversation_id: str,
    message_id: str,
    background_tasks: BackgroundTasks = None,
):
    """Generate streaming answer using RAG and LLM with JSON format."""
    async for chunk in generate_answer_stream_generator_helper(
        request, conversation_id, message_id, "json", background_tasks
    ):
        yield chunk
