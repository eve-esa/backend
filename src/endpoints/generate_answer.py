"""Endpoint to generate an answer using a language model and vector store."""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from openai import Client
from pydantic import BaseModel, Field

from src.services.vector_store_manager import VectorStoreManager
from src.services.llm_manager import LLMManager 
from src.config import QDRANT_URL, QDRANT_API_KEY, OPENAI_API_KEY

# Constants
DEFAULT_QUERY = "What is ESA?"
DEFAULT_COLLECTION = "esa-nasa-workshop"
DEFAULT_EMBEDDINGS = "nasa-impact/nasa-smd-ibm-v0.1"
DEFAULT_LLM = "eve-instruct-v0.1"  # or openai
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 0
DEFAULT_K = 3
DEFAULT_SCORE_THRESHOLD = 0.7
DEFAULT_MAX_NEW_TOKENS = 1500
DEFAUL_GET_UNIQUE_DOCS = True

# Setup
router = APIRouter()
openai_client = Client(api_key=OPENAI_API_KEY)

class GenerationRequest(BaseModel):
    query: str = DEFAULT_QUERY
    collection_name: str = DEFAULT_COLLECTION
    llm: str = DEFAULT_LLM  # or openai
    embeddings_model: str = DEFAULT_EMBEDDINGS
    k: int = DEFAULT_K
    score_threshold: float = Field(DEFAULT_SCORE_THRESHOLD, ge=0.0, le=1.0)
    get_unique_docs: bool = DEFAUL_GET_UNIQUE_DOCS
    max_new_tokens: int = Field(DEFAULT_MAX_NEW_TOKENS, ge=100, le=8192)


def get_rag_context(
    vector_store: VectorStoreManager, request: GenerationRequest
) -> str:
    vector_store = VectorStoreManager(embeddings_model=request.embeddings_model)
    results = vector_store.retrieve_documents_from_query(
        query=request.query,
        collection_name=request.collection_name,
        embeddings_model=request.embeddings_model,
        score_threshold=request.score_threshold,
        get_unique_docs=request.get_unique_docs,
        k=request.k,
    )
    if not results:
        print(f"No documents found for query : {request.query}")

    retrieved_documents = [result.payload.get("page_content", "") for result in results]
    context = "\n".join(retrieved_documents)
    return context, results


@router.post("/generate_answer", response_model=Dict[str, Any])
def create_collection(request: GenerationRequest) -> Dict[str, Any]:
    llm_manager = LLMManager()
    
    try:
        vector_store = VectorStoreManager(embeddings_model=request.embeddings_model)
        is_rag = vector_store.use_rag(request.query)
        context, results = (get_rag_context(vector_store, request) if is_rag else ("", []))
        
        answer = llm_manager.generate_answer(
            query=request.query,
            context=context,
            llm=request.llm,
            max_new_tokens=request.max_new_tokens,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"answer": answer, "documents": results, "use_rag": is_rag}


