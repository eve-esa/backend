"""Endpoint to generate an answer using a language model and vector store."""

import json
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any, List
from openai import AsyncOpenAI  # Use AsyncOpenAI for async operations
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

from src.core.vector_store_manager import VectorStoreManager
from src.core.llm_manager import LLMManager
from src.config import OPENAI_API_KEY, config
from src.constants import (
    DEFAULT_QUERY,
    DEFAULT_COLLECTION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM,
    DEFAULT_K,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_GET_UNIQUE_DOCS,
    DEFAULT_MAX_NEW_TOKENS,
    FALLBACK_LLM,
    RERANKER_MODEL,
)

from src.utils.runpod_utils import get_reranked_documents_from_runpod

# Setup
router = APIRouter()
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)  # Use AsyncOpenAI


class GenerationRequest(BaseModel):
    query: str = DEFAULT_QUERY
    year: List[int] = []
    keywords: List[str] = []
    collection_names: List[str] = [DEFAULT_COLLECTION]
    llm: str = DEFAULT_LLM  # or openai
    embeddings_model: str = DEFAULT_EMBEDDING_MODEL
    k: int = DEFAULT_K
    score_threshold: float = Field(DEFAULT_SCORE_THRESHOLD, ge=0.0, le=1.0)
    get_unique_docs: bool = DEFAULT_GET_UNIQUE_DOCS  # Fixed typo
    max_new_tokens: int = Field(DEFAULT_MAX_NEW_TOKENS, ge=100, le=8192)
    fallback_llm: str = FALLBACK_LLM  # Fallback LLM when primary fails


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


def _build_context(texts: List[str]) -> str:
    """Join non-empty texts into a single context string."""
    return "\n".join([t for t in texts if t])


async def _maybe_rerank(
    candidate_texts: List[str], query: str, k: int
) -> List[dict] | None:
    """Call reranker if configured and if reranking is useful (more candidates than k)."""
    endpoint_id = config.get_reranker_id()
    if not (endpoint_id and len(candidate_texts) > k):
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


async def get_rag_context(
    vector_store: VectorStoreManager, request: GenerationRequest
) -> tuple[str, list]:
    """Get RAG context from vector store."""
    # Retrieve a larger candidate set so reranker can select the top-k
    candidate_k = max(request.k * 10, request.k)
    results = await vector_store.retrieve_documents_from_query(
        collection_names=request.collection_names,
        query=request.query,
        year=request.year,
        keywords=request.keywords,
        k=candidate_k,
        score_threshold=request.score_threshold,
        get_unique_docs=request.get_unique_docs,
        embeddings_model=request.embeddings_model,
    )

    if not results:
        logger.info(f"No documents found for query: {request.query}")
        return "", []

    # Extract plain text for reranker input (support both 'page_content' and 'text')
    candidate_texts = _extract_candidate_texts(results)

    # Optionally rerank and build context directly from reranker documents
    reranked = await _maybe_rerank(candidate_texts, request.query, request.k)
    if isinstance(reranked, list) and reranked:
        try:
            reranked_sorted = sorted(
                reranked, key=lambda x: x.get("relevance_score", 0), reverse=True
            )
            top_reranked = reranked_sorted[: request.k]
            context = _build_context([r.get("document", "") for r in top_reranked])
            return context, top_reranked
        except Exception as e:
            logger.warning(
                f"Failed to process reranker output, using vector similarity order: {e}"
            )

    # Fallback: use the first k results (already sorted by vector score upstream)
    trimmed = results[: request.k]
    trimmed_texts = _extract_candidate_texts(trimmed)
    return _build_context(trimmed_texts), trimmed


async def setup_rag_and_context(request: GenerationRequest):
    """Setup RAG and get context for the request."""
    vector_store = VectorStoreManager(embeddings_model=request.embeddings_model)

    # Check if we need to use RAG
    try:
        is_rag = await vector_store.use_rag(request.query)
    except Exception as e:
        logger.warning(f"Failed to determine RAG usage, defaulting to no RAG: {e}")
        is_rag = False

    # Get context if using RAG
    if is_rag:
        try:
            context, results = await get_rag_context(vector_store, request)
        except Exception as e:
            logger.warning(f"Failed to get RAG context, falling back to no RAG: {e}")
            context, results = "", []
            is_rag = False
    else:
        context, results = "", []

    return context, results, is_rag


@router.post("/generate_answer", response_model=Dict[str, Any])
async def generate_answer(
    request: GenerationRequest,
) -> Dict[str, Any]:  # Renamed from create_collection
    """Generate an answer using RAG and LLM."""
    llm_manager = LLMManager()

    try:
        context, results, is_rag = await setup_rag_and_context(request)

        # Generate answer
        answer = llm_manager.generate_answer(
            query=request.query,
            context=context,
            llm=request.llm,
            max_new_tokens=request.max_new_tokens,
            fallback_llm=request.fallback_llm,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"answer": answer, "documents": results, "use_rag": is_rag}


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
            fallback_llm=request.fallback_llm,
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


@router.post("/generate_answer_stream")
async def generate_answer_stream(
    request: GenerationRequest,
) -> StreamingResponse:
    """Generate a streaming answer using RAG and LLM."""
    return StreamingResponse(
        generate_answer_stream_generator(request),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        },
    )


@router.post("/generate_answer_stream_json")
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
