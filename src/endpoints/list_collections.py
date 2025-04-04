from fastapi import APIRouter
import requests
from src.services.vector_store_manager import VectorStoreManager
from src.config import QDRANT_URL, QDRANT_API_KEY

router = APIRouter()


@router.get("/list-collections")
def health_check():
    collection_names: list[str] = VectorStoreManager(
        QDRANT_URL, QDRANT_API_KEY
    ).list_collections_names()

    return {"collections": collection_names}
