from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from pydantic import BaseModel, Field
from src.services.vector_store_manager import VectorStoreManager
from src.config import QDRANT_URL, QDRANT_API_KEY

router = APIRouter()


class GenerationRequest(BaseModel):
    query: str = "What is ESA?"
    collection_name: str = "esa-nasa-workshop"
    llm: str = "openai"
    embeddings_model: str = "text-embedding-3-small"
    k: int = 3
    score_threshold: float = Field(0.7, ge=0.0, le=1.0)  # Ensure it's between 0 and 1
    get_unique_docs: bool = True
    max_new_tokens: int = Field(1500, ge=100, le=8192)


@router.post("/generate_answer", response_model=Dict[str, Any])
def create_collection(request: GenerationRequest) -> Dict[str, Any]:
    try:
        vector_store = VectorStoreManager(
            QDRANT_URL, QDRANT_API_KEY, embeddings_model=request.embeddings_model
        )
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

        retrieved_documents = [
            result.payload.get("page_content", "") for result in results
        ]
        context = "\n".join(retrieved_documents)

        answer = vector_store.generate_answer(
            query=request.query,
            context=context,
            llm=request.llm,
            max_new_tokens=request.max_new_tokens,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"answer": answer, "documents": results}
