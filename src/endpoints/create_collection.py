from fastapi import APIRouter, HTTPException
from src.services.vector_store_manager import VectorStoreManager
from pydantic import BaseModel
from src.config import QDRANT_URL, QDRANT_API_KEY


router = APIRouter()


class CollectionRequest(BaseModel):
    collection_name: str = "esa-nasa-workshop"
    embeddings_model: str = "text-embedding-3-small" #nasa-impact/nasa-smd-ibm-v0.1


@router.post("/create_collection")
def create_collection(request: CollectionRequest):
    collection_name = request.collection_name
    vector_store = VectorStoreManager(
        QDRANT_URL, QDRANT_API_KEY, embeddings_model=request.embeddings_model
    )

    collections_name_list = vector_store.list_collections_names()
    if collection_name in collections_name_list:
        return {"message": f"Collection '{collection_name}' already exists"}
    try:
        vector_store.create_collection(collection_name)
        return {"message": f"Collection '{collection_name}' created successfully"}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create collection '{collection_name}': {str(e)}",
        )
