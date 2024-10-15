from fastapi import APIRouter, HTTPException
from src.services.vector_store_manager import VectorStoreManager
from dotenv import load_dotenv
import os

load_dotenv()
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

router = APIRouter()


@router.post("/create_collection")
def create_collection(collection_name: str):
    vector_store = VectorStoreManager(
        qdrant_url, qdrant_api_key, embeddings_model="text-embedding-3-small"
    )

    # Check if the collection already exists
    collections = vector_store.list_collections()
    if collection_name in collections:
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
