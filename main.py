from src.services.vector_store_manager import VectorStoreManager
from dotenv import load_dotenv
import os


load_dotenv()
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

if __name__ == "__main__":
    vector_store = VectorStoreManager(
        qdrant_url, qdrant_api_key, embeddings_model="text-embedding-3-small"
    )
    vector_store.create_collection("test_collection")
