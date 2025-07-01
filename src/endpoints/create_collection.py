"""Endpoint for creating a collection in Qdrant."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.services.vector_store_manager import VectorStoreManager
from src.config import QDRANT_URL, QDRANT_API_KEY

router = APIRouter()

class CollectionRequest(BaseModel):
    collection_name: str = "test_collection"
    embeddings_model: str =  "nasa-impact/nasa-smd-ibm-st-v2"


@router.post("/create_collection")
def create_collection(request: CollectionRequest):
    collection_name = request.collection_name
    vector_store = VectorStoreManager(embeddings_model=request.embeddings_model)

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
