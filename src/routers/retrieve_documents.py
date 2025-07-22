from fastapi import APIRouter, HTTPException
from src.utils.vector_store_manager import VectorStoreManager
from pydantic import BaseModel

router = APIRouter()

# Constants
DEFAULT_QUERY = "What is ESA?"
DEFAULT_COLLECTION = "esa-nasa-workshop"
DEFAULT_EMBEDDING_MODEL = "nasa-impact/nasa-smd-ibm-st-v2"
DEFAULT_SCORE_THRESHOLD = 0.7
DEFAULT_K = 3

class RetrieveRequest(BaseModel):
    query: str = DEFAULT_QUERY
    collection_name: str = DEFAULT_COLLECTION
    embeddings_model: str = DEFAULT_EMBEDDING_MODEL
    score_threshold: float = DEFAULT_SCORE_THRESHOLD
    k: int = DEFAULT_K


@router.post("/retrieve_documents")
async def retrieve_documents(request: RetrieveRequest):
    try:
        vector_store = VectorStoreManager(embeddings_model=request.embeddings_model)
        results = await vector_store.retrieve_documents_from_query(
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
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
