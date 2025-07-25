"""Endpoint to generate an answer using a language model and vector store."""

import json
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any
from openai import AsyncOpenAI  # Use AsyncOpenAI for async operations
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

from src.core.vector_store_manager import VectorStoreManager
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

# Setup
router = APIRouter()
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)  # Use AsyncOpenAI


class GenerationRequest(BaseModel):
    query: str = DEFAULT_QUERY
    collection_name: str = DEFAULT_COLLECTION
    llm: str = DEFAULT_LLM  # or openai
    embeddings_model: str = DEFAULT_EMBEDDING_MODEL
    k: int = DEFAULT_K
    score_threshold: float = Field(DEFAULT_SCORE_THRESHOLD, ge=0.0, le=1.0)
    get_unique_docs: bool = DEFAULT_GET_UNIQUE_DOCS  # Fixed typo
    max_new_tokens: int = Field(DEFAULT_MAX_NEW_TOKENS, ge=100, le=8192)


async def get_rag_context(
    vector_store: VectorStoreManager, request: GenerationRequest
) -> tuple[str, list]:
    """Get RAG context from vector store."""
    # Remove duplicate vector_store initialization
    results = await vector_store.retrieve_documents_from_query(
        query=request.query,
        collection_name=request.collection_name,
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


@router.post("/generate_answer", response_model=Dict[str, Any])
async def generate_answer(
    request: GenerationRequest,
) -> Dict[str, Any]:  # Renamed from create_collection
    """Generate an answer using RAG and LLM."""
    llm_manager = LLMManager()

    try:
        vector_store = VectorStoreManager(embeddings_model=request.embeddings_model)

        # Check if we need to use RAG
        is_rag = await vector_store.use_rag(
            request.query
        )  # Make sure this is awaited if async

        # Get context if using RAG
        if is_rag:
            context, results = await get_rag_context(vector_store, request)
        else:
            context, results = "", []

        # Generate answer
        answer = llm_manager.generate_answer(
            query=request.query,
            context=context,
            llm=request.llm,
            max_new_tokens=request.max_new_tokens,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"answer": answer, "documents": results, "use_rag": is_rag}


async def generate_answer_stream_generator(
    request: GenerationRequest,
):
    """Generate streaming answer using RAG and LLM."""
    llm_manager = LLMManager()

    try:
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
                logger.warning(
                    f"Failed to get RAG context, falling back to no RAG: {e}"
                )
                context, results = "", []
                is_rag = False
        else:
            context, results = "", []

        # Generate streaming answer
        async for chunk in llm_manager.generate_answer_stream(
            query=request.query,
            context=context,
            llm=request.llm,
            max_new_tokens=request.max_new_tokens,
        ):
            yield f"data: {chunk}\n\n"

    except Exception as e:
        yield f"data: Error: {str(e)}\n\n"


async def generate_answer_json_stream_generator(
    request: GenerationRequest,
):
    """Generate streaming answer using RAG and LLM with JSON format."""
    llm_manager = LLMManager()

    try:
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
                logger.warning(
                    f"Failed to get RAG context, falling back to no RAG: {e}"
                )
                context, results = "", []
                is_rag = False
        else:
            context, results = "", []

        # Send initial metadata
        yield f"data: {json.dumps({'type': 'start', 'use_rag': is_rag, 'documents_count': len(results)})}\n\n"

        # Generate streaming answer
        full_answer = ""
        async for chunk in llm_manager.generate_answer_stream(
            query=request.query,
            context=context,
            llm=request.llm,
            max_new_tokens=request.max_new_tokens,
        ):
            full_answer += chunk
            yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"

        # Send final metadata
        yield f"data: {json.dumps({'type': 'end', 'full_answer': full_answer})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"


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
