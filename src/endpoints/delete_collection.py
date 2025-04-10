"""FastAPI endpoint to delete a collection in Qdrant."""
from fastapi import APIRouter, HTTPException
from src.services.vector_store_manager import VectorStoreManager

router = APIRouter()

@router.delete("/delete_collection")
def delete_collection(collection_name: str = "esa-nasa-workshop"):
    vector_store = VectorStoreManager(embeddings_model="fake")
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
