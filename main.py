from src.services.vector_store_manager import VectorStoreManager
from src.config import QDRANT_URL, QDRANT_API_KEY

if __name__ == "__main__":
    vector_store = VectorStoreManager(
        QDRANT_URL, QDRANT_API_KEY, embeddings_model="text-embedding-3-small"
    )
    vector_store.create_collection("test_collection")
