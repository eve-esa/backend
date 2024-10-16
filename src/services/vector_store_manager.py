from qdrant_client import QdrantClient
import qdrant_client
from qdrant_client.conversions import common_types as types
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

    def list_collections(self) -> types.CollectionsResponse:
        return self.client.get_collections()

    def list_collections_names(self) -> list:
        try:
            collections_list = []
            collections = self.client.get_collections()
            for collection in collections:
                for c in list(collection[1]):
                    collections_list.append(c.name)
            return collections_list
        except Exception as e:
            raise RuntimeError(f"Failed to list collections: {str(e)}") from e


if __name__ == "__main__":
    vector_store = VectorStoreManager(
        qdrant_url, qdrant_api_key, embeddings_model="text-embedding-3-small"
    )
    vector_store.create_collection("test_collection")
