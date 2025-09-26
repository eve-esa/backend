import logging
import anyio
import asyncio

from src.constants import DEFAULT_EMBEDDING_MODEL, WILEY_PUBLIC_COLLECTIONS
from src.schemas.common import Pagination
from src.schemas.collection import CollectionRequest, CollectionUpdate
from src.database.models.document import Document
from fastapi import APIRouter, HTTPException, Depends

from src.database.models.collection import Collection
from src.database.models.user import User
from src.database.mongo_model import PaginatedResponse, get_pagination_metadata
from src.middlewares.auth import get_current_user
from src.core.vector_store_manager import VectorStoreManager

logger = logging.getLogger(__name__)
vector_store = VectorStoreManager()

router = APIRouter()


async def _count_points_for_collection(collection_name: str) -> int:
    """Count Qdrant points for a collection name in a worker thread."""

    def _count() -> int:
        try:
            result = vector_store.client.count(
                collection_name=collection_name,
                count_filter=None,
                exact=True,
            )
            return int(getattr(result, "count", 0) or 0)
        except Exception as e:
            logger.warning(
                f"Failed to count Qdrant points for collection {collection_name}: {e}"
            )
            return 0

    return await anyio.to_thread.run_sync(_count)


async def _get_counts_for_id(collection_id: str):
    """Return (documents_count, points_count) for a single collection id."""
    documents_count_coro = Document.count_documents({"collection_id": collection_id})
    points_count_coro = _count_points_for_collection(collection_id)
    documents_count, points_count = await asyncio.gather(
        documents_count_coro, points_count_coro
    )
    return documents_count, points_count


@router.get("/collections/public", response_model=PaginatedResponse[Collection])
async def list_public_collections(pagination: Pagination = Depends()):
    public_collections, total_count = await vector_store.list_public_collections(
        page=pagination.page, limit=pagination.limit
    )

    # public_collections = WILEY_PUBLIC_COLLECTIONS + public_collections
    # total_count = total_count + len(WILEY_PUBLIC_COLLECTIONS)
    public_collections = WILEY_PUBLIC_COLLECTIONS + [
        {
            "name": "esa-data-qwen-1024",
            "description": "ESA data with Qwen-1024 for testing",
        }
    ]
    total_count = len(public_collections)
    # Pagination must be done manually since Qdrant doesn't support collection pagination
    return PaginatedResponse(
        data=[
            Collection(
                id=collection["name"],
                name=collection.get("alias") or collection["name"],
                description=collection["description"],
                user_id=None,
                embeddings_model=DEFAULT_EMBEDDING_MODEL,
            )
            for collection in public_collections
        ],
        meta=get_pagination_metadata(total_count, pagination.page, pagination.limit),
    )


@router.get("/collections", response_model=PaginatedResponse[Collection])
async def list_collections(
    request: Pagination = Depends(), request_user: User = Depends(get_current_user)
):
    return await Collection.find_all_with_pagination(
        limit=request.limit,
        page=request.page,
        filter_dict={"user_id": request_user.id},
        sort=[("timestamp", -1)],
    )


@router.get("/collections/{collection_id}")
async def get_collection(collection_id: str):
    collection = await Collection.find_by_id(collection_id)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    documents_count, points_count = await _get_counts_for_id(collection_id)

    return {
        **collection.dict(),
        "documents_count": documents_count,
        "points_count": points_count,
    }


@router.post("/collections", response_model=Collection)
async def create_collection(
    request: CollectionRequest,
    requesting_user: User = Depends(get_current_user),
):
    collection = Collection(
        name=request.name,
        user_id=requesting_user.id,
        description=request.description,
        embeddings_model=request.embeddings_model,
    )
    await collection.save()

    try:
        VectorStoreManager(embeddings_model=request.embeddings_model).create_collection(
            collection.id
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.warning(
            f"Warning: failed to create Qdrant collection {collection.id}: {e}"
        )
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

    except HTTPException as e:
        raise e
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
            await anyio.to_thread.run_sync(
                vector_store.delete_collection, collection_id
            )
        except Exception as e:
            logger.warning(
                f"Warning: failed to delete Qdrant collection {collection_id}: {e}"
            )

        await collection.delete()
        return {"message": "Collection deleted successfully"}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
