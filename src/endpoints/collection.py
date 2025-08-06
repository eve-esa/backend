import logging
import anyio

from src.schemas.collections import CollectionRequest, CollectionUpdate
from src.database.models.document import Document
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from src.database.models.collection import Collection
from src.database.models.user import User
from src.database.mongo_model import PaginatedResponse
from src.middlewares.auth import get_current_user
from src.core.vector_store_manager import VectorStoreManager

logger = logging.getLogger(__name__)

# Default embedding model for new collections (adjust as needed)
DEFAULT_EMBEDDINGS_MODEL = "nasa-impact/nasa-smd-ibm-st-v2"

router = APIRouter()


class Pagination(BaseModel):
    page: int = 1
    limit: int = 10


@router.get("/collections/public", response_model=PaginatedResponse[Collection])
async def list_public_collections(pagination: Pagination = Depends()):
    return await Collection.find_all_with_pagination(
        limit=pagination.limit,
        page=pagination.page,
        sort=[("timestamp", -1)],
    )


@router.get("/collections", response_model=PaginatedResponse[Collection])
async def list_collections(
    request: Pagination = Depends(), request_user: User = Depends(get_current_user)
):
    return await Collection.find_all_with_pagination(
        filter_dict={"user_id": request_user.id},
        limit=request.limit,
        page=request.page,
        sort=[("timestamp", -1)],
    )


@router.post("/collections", response_model=Collection)
async def create_collection(
    request: CollectionRequest,
    requesting_user: User = Depends(get_current_user),
):
    collection = Collection(
        name=request.name,
        user_id=requesting_user.id,
    )
    await collection.save()

    try:
        vector_store = VectorStoreManager(embeddings_model=request.embeddings_model)
        vector_store.create_collection(collection.id)
    except Exception as e:
        await collection.delete()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create vector collection: {str(e)}",
        )

    return collection


@router.patch("/collections/{collection_id}")
async def update_collection(
    request: CollectionUpdate,
    collection_id: str,
    requesting_user: User = Depends(get_current_user),
):
    try:
        collection = await Collection.find_by_id(collection_id)
        if not collection:
            raise HTTPException(status_code=404, detail="Collection not found")

        if collection.user_id != requesting_user.id:
            raise HTTPException(
                status_code=403,
                detail="You are not allowed to update this collection",
            )

        collection.name = request.name
        updated_collection = await collection.save()
        return updated_collection

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.delete("/collections/{collection_id}")
async def delete_collection(
    collection_id: str,
    requesting_user: User = Depends(get_current_user),
):
    try:
        collection = await Collection.find_by_id(collection_id)
        if not collection:
            raise HTTPException(status_code=404, detail="Collection not found")
        if collection.user_id != requesting_user.id:
            raise HTTPException(
                status_code=403,
                detail="You are not allowed to delete this collection",
            )

        await Document.delete_many({"collection_id": collection_id})

        try:
            vector_store = VectorStoreManager()
            await anyio.to_thread.run_sync(
                vector_store.delete_collection, collection_id
            )
        except Exception as e:
            logger.warning(
                f"Warning: failed to delete Qdrant collection {collection_id}: {e}"
            )

        await collection.delete()
        return {"message": "Collection deleted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
