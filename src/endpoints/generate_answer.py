"""Endpoint to generate an answer using a language model and vector store."""

import asyncio
from src.database.models.user import User
from src.middlewares.auth import get_current_user
from src.database.models.conversation import Conversation
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from openai import AsyncOpenAI  # Use AsyncOpenAI for async operations
from pydantic import BaseModel, Field

from src.services.vector_store_manager import VectorStoreManager
from src.services.llm_manager import LLMManager
from src.config import QDRANT_URL, QDRANT_API_KEY, OPENAI_API_KEY

# Constants
DEFAULT_QUERY = "What is ESA?"
DEFAULT_COLLECTION = "esa-nasa-workshop"
DEFAULT_EMBEDDINGS = "nasa-impact/nasa-smd-ibm-st-v2"
DEFAULT_LLM = "eve-instruct-v0.1"  # or openai
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 0
DEFAULT_K = 3
DEFAULT_SCORE_THRESHOLD = 0.7
DEFAULT_MAX_NEW_TOKENS = 1500
DEFAULT_GET_UNIQUE_DOCS = True  # Fixed typo: was DEFAUL_GET_UNIQUE_DOCS

# Setup
router = APIRouter()
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)  # Use AsyncOpenAI


class GenerationRequest(BaseModel):
    query: str = DEFAULT_QUERY
    collection_name: str = DEFAULT_COLLECTION
    llm: str = DEFAULT_LLM  # or openai
    embeddings_model: str = DEFAULT_EMBEDDINGS
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


@router.post("/generate_answer", response_model=Conversation)
async def generate_answer(
    request: GenerationRequest,
    request_user: User = Depends(get_current_user),
):
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

    conversation = await Conversation.create(
        input=request.query,
        output=answer,
        user_id=request_user.id,
        metadata={
            "api_used": "generate_answer",
            "llm": request.llm,
            "embeddings_model": request.embeddings_model,
            "collection_name": request.collection_name,
            "k": request.k,
            "score_threshold": request.score_threshold,
            "get_unique_docs": request.get_unique_docs,
            "max_new_tokens": request.max_new_tokens,
            "is_rag": is_rag,
        },
    )

    return conversation
