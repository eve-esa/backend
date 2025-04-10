from fastapi import APIRouter, HTTPException
from src.services.vector_store_manager import VectorStoreManager
from pydantic import BaseModel
from src.config import QDRANT_URL, QDRANT_API_KEY

router = APIRouter()


class RetrieveRequest(BaseModel):
    query: str = "What is ESA?"
    collection_name: str = "esa-nasa-workshop"
    embeddings_model: str = "text-embedding-3-small"
    score_threshold: float = 0.7
    k: int = 3


@router.post("/retrieve_documents")
async def retrieve_documents(request: RetrieveRequest):
    try:
        # Initialize VectorStoreManager with the embeddings model provided in the request
        vector_store = VectorStoreManager(embeddings_model=request.embeddings_model)

        # Retrieve documents using the parameters from the request body
        results = vector_store.retrieve_documents_from_query(
            query=request.query,
            embeddings_model=request.embeddings_model,
            collection_name=request.collection_name,
            k=request.k,
            score_threshold=request.score_threshold,
        )

        if results == []:
            raise HTTPException(status_code=404, detail="No documents found.")
        return results

    except Exception as e:
        # Log or handle the exception (logging omitted for brevity)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
