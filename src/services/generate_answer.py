"""Endpoint to generate an answer using a language model and vector store."""

import json
import logging
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any
from openai import AsyncOpenAI  # Use AsyncOpenAI for async operations
from pydantic import BaseModel, Field

from src.core.vector_store_manager import VectorStoreManager
from src.database.models.collection import Collection
from src.core.llm_manager import LLMManager
from src.config import OPENAI_API_KEY
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


# Setup
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)  # Use AsyncOpenAI


class GenerationRequest(BaseModel):
    query: str = DEFAULT_QUERY
    collection_id: str = DEFAULT_COLLECTION
    llm: str = DEFAULT_LLM  # or openai
    embeddings_model: str = DEFAULT_EMBEDDING_MODEL
    k: int = DEFAULT_K
    score_threshold: float = Field(DEFAULT_SCORE_THRESHOLD, ge=0.0, le=1.0)
    get_unique_docs: bool = DEFAULT_GET_UNIQUE_DOCS  # Fixed typo
    max_new_tokens: int = Field(DEFAULT_MAX_NEW_TOKENS, ge=100, le=8192)
    use_rag: bool = True


async def get_rag_context(
    vector_store: VectorStoreManager, request: GenerationRequest
) -> tuple[str, list]:
    """Get RAG context from vector store."""
    # Remove duplicate vector_store initialization
    results = await vector_store.retrieve_documents_from_query(
        query=request.query,
        collection_name=request.collection_id,
        embeddings_model=request.embeddings_model,
        score_threshold=request.score_threshold,
        get_unique_docs=request.get_unique_docs,
        k=request.k,
    )

    if not results:
        print(f"No documents found for query: {request.query}")
        return "", []

    retrieved_documents = [result.payload.get("page_content", "") for result in results]
    context = "\n".join(retrieved_documents)
    return context, results


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


async def generate_answer(
    request: GenerationRequest,
) -> tuple[str, list, bool]:
    """Generate an answer using RAG and LLM."""
    llm_manager = LLMManager()

    try:
        context, results, is_rag = await setup_rag_and_context(request)

        answer = llm_manager.generate_answer(
            query=request.query,
            context=context,
            llm=request.llm,
            max_new_tokens=request.max_new_tokens,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return answer, results, is_rag


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
