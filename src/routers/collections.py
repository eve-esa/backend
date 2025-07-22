"""Endpoint for collection in Qdrant."""

from fastapi import APIRouter, HTTPException, status, Query
from typing import Optional
from src.schemas.collections import CollectionRequest
from src.services.collection import CollectionService

router = APIRouter()

@router.put("/collections/{collection_name}")
async def create_collection(collection_name: str, request: CollectionRequest):
    """Create a collection."""
    collection_service = CollectionService()
    
    try:
        collection = await collection_service.create_collection(request, collection_name)
        
        return {
            "collection_name": collection.name,
            "embeddings_model": collection.embeddings_model,
            "created_at": collection.created_at.isoformat() if collection.created_at else None,
            "status": "created"
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create collection '{collection_name}': {str(e)}"
        )

@router.get("/collections")
async def get_collections(
    embeddings_model: Optional[str] = Query(
        default="nasa-impact/nasa-smd-ibm-st-v2",
        description="Embeddings model to use for listing collections"
    )
):
    """Get all collections using query parameters."""
    collection_service = CollectionService()
    try:
        # Create a request object from query parameters
        request = CollectionRequest(embeddings_model=embeddings_model) if embeddings_model else None
        collection_names: list[str] = await collection_service.list_collections(request)

        if not collection_names:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No collections found")

        return {"collections": collection_names}

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Server error: {str(e)}")

@router.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str, request: Optional[CollectionRequest] = None):
    """Delete a collection. Request body is optional."""
    collection_service = CollectionService()
    try:
        await collection_service.delete_collection(collection_name, request)
        return {"message": f"Collection '{collection_name}' deleted successfully."}
    except ValueError as e:
        # Not an existing collection
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to delete collection: {str(e)}"
        )
