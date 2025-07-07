from fastapi import APIRouter, HTTPException
from src.services.vector_store_manager import VectorStoreManager

router = APIRouter()


@router.get("/list-collections")
def list_collections():
    try:
        vector_store_manager = VectorStoreManager()
        collection_names: list[str] = vector_store_manager.list_collections_names()

        if not collection_names:
            raise HTTPException(status_code=404, detail="No collections found")

        return {"collections": collection_names}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
