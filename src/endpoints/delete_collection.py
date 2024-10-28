from fastapi import APIRouter
from fastapi import APIRouter, HTTPException
from src.services.vector_store_manager import VectorStoreManager
from src.config import QDRANT_URL, QDRANT_API_KEY


router = APIRouter()


@router.delete("/delete_collection")
def delete_collection(collection_name: str):
    vector_store = VectorStoreManager(
        QDRANT_URL, QDRANT_API_KEY, embeddings_model="fake"
    )
    try:
        vector_store.delete_collection(collection_name)
        return {"message": f"Collection '{collection_name}' deleted successfully."}
    except ValueError as e:
        # Not an existing collection
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete collection: {str(e)}"
        )
