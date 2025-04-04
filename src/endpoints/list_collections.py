from fastapi import APIRouter, HTTPException
from src.services.vector_store_manager import VectorStoreManager
from src.config import QDRANT_URL, QDRANT_API_KEY

router = APIRouter()


@router.get("/list-collections")
def list_collections():
    try:
        vector_store_manager = VectorStoreManager(QDRANT_URL, QDRANT_API_KEY)
        collection_names: list[str] = vector_store_manager.list_collections_names()

        if not collection_names:
            raise HTTPException(status_code=404, detail="No collections found")

        return {"collections": collection_names}

    except Exception as e:
        # Optionally log the error here
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
