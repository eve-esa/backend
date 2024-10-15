from qdrant_client import QdrantClient
import qdrant_client
from dotenv import load_dotenv
import os
from src.services.utils import get_embeddings_model


load_dotenv()
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")


class VectorStoreManager:
    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: str,
        embeddings_model: str = "sentence-transformers/paraphrase-TinyBERT-L6-v2",
    ) -> None:
        self.client = QdrantClient(qdrant_url, api_key=qdrant_api_key)
        self.embeddings, self.embeddings_size = get_embeddings_model(
            model=embeddings_model, return_embeddings_size=True
        )
        self.collection = None

    def create_collection(self, collection_name: str) -> None:
        vectors_config = qdrant_client.http.models.VectorParams(
            size=self.embeddings_size,
            distance=qdrant_client.http.models.Distance.COSINE,
        )
        success = self.client.recreate_collection(
            collection_name=collection_name, vectors_config=vectors_config
        )
        if not success:
            raise Exception(f"Failed to create collection: {collection_name}")
        print(f"Collection '{collection_name}' created successfully.")

    def list_collections(self) -> list:
        return self.client.get_collections()


if __name__ == "__main__":
    vector_store = VectorStoreManager(
        qdrant_url, qdrant_api_key, embeddings_model="text-embedding-3-small"
    )
    vector_store.create_collection("test_collection")
