from fastapi import APIRouter, HTTPException
from src.services.vector_store_manager import VectorStoreManager
from dotenv import load_dotenv
from pydantic import BaseModel
import os

load_dotenv()
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

router = APIRouter()


class CollectionRequest(BaseModel):
    collection_name: str


@router.post("/create_collection")
def create_collection(request: CollectionRequest):
    collection_name = request.collection_name
    vector_store = VectorStoreManager(
        qdrant_url, qdrant_api_key, embeddings_model="text-embedding-3-small"
    )

    collections_name_list = vector_store.list_collections_names()
    if collection_name in collections_name_list:
        raise HTTPException(
            status_code=400, detail=f"Collection '{collection_name}' already exists"
        )
    try:
        vector_store.create_collection(collection_name)
        return {"message": f"Collection '{collection_name}' created successfully"}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create collection '{collection_name}': {str(e)}",
        )
