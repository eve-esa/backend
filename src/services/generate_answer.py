"""Endpoint to generate an answer using a language model and vector store."""

from typing import Optional
from openai import AsyncOpenAI  # Use AsyncOpenAI for async operations
from pydantic import BaseModel, Field

from src.services.vector_store_manager import VectorStoreManager
from src.database.models.collection import Collection
from src.services.llm_manager import LLMManager
from src.config import OPENAI_API_KEY

# Constants
DEFAULT_QUERY = "What is ESA?"
DEFAULT_EMBEDDINGS = "nasa-impact/nasa-smd-ibm-st-v2"
DEFAULT_LLM = "eve-instruct-v0.1"  # or openai
DEFAULT_COLLECTION = "esa-nasa-workshop"
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 0
DEFAULT_K = 3
DEFAULT_SCORE_THRESHOLD = 0.7
DEFAULT_MAX_NEW_TOKENS = 1500
DEFAULT_GET_UNIQUE_DOCS = True  # Fixed typo: was DEFAUL_GET_UNIQUE_DOCS

# Setup
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)  # Use AsyncOpenAI


class GenerationRequest(BaseModel):
    query: str = DEFAULT_QUERY
    collection_id: Optional[str] = None
    llm: str = DEFAULT_LLM  # or openai
    embeddings_model: str = DEFAULT_EMBEDDINGS
    k: int = DEFAULT_K
    score_threshold: float = Field(DEFAULT_SCORE_THRESHOLD, ge=0.0, le=1.0)
    get_unique_docs: bool = DEFAULT_GET_UNIQUE_DOCS  # Fixed typo
    max_new_tokens: int = Field(DEFAULT_MAX_NEW_TOKENS, ge=100, le=8192)


async def get_rag_context(
    vector_store: VectorStoreManager, collection_id: str, request: GenerationRequest
) -> tuple[str, list]:
    """Get RAG context from vector store."""
    # Remove duplicate vector_store initialization
    results = await vector_store.retrieve_documents_from_query(
        query=request.query,
        collection_name=collection_id,
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


async def generate_answer(
    request: GenerationRequest,
) -> tuple[str, list, bool]:  # Renamed from create_collection
    """Generate an answer using RAG and LLM."""
    llm_manager = LLMManager()

    try:
        collection = (
            (await Collection.find_by_id(request.collection_id))
            if request.collection_id
            else None
        )
        if not collection:
            collection = {"id": DEFAULT_COLLECTION}

        qdrant_collection = collection.get("id", DEFAULT_COLLECTION)

        vector_store = VectorStoreManager(embeddings_model=request.embeddings_model)

        # Check if we need to use RAG
        is_rag = await vector_store.use_rag(request.query)

        # Get context if using RAG
        if is_rag:
            context, results = await get_rag_context(
                vector_store, qdrant_collection, request
            )
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
        raise Exception(e)

    return answer, results, is_rag
