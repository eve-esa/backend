import qdrant_client
from qdrant_client import QdrantClient
from qdrant_client.conversions import common_types as types
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant.qdrant import QdrantVectorStoreError
import os
from uuid import uuid4

from src.services.utils import get_embeddings_model


load_dotenv()
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")


class VectorStoreManager:
    """When initializing this class, use the same embedding function you \
        used to embed the collection you want to work with"""

    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: str,
        embeddings_model: str = "sentence-transformers/paraphrase-TinyBERT-L6-v2",
    ) -> None:
        self.client = QdrantClient(qdrant_url, api_key=qdrant_api_key)
        self.embeddings_model = embeddings_model
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

    def list_collections_names(self) -> list[str]:
        try:
            collections_list = []
            collections = self.client.get_collections()
            for collection in collections:
                for c in list(collection[1]):
                    collections_list.append(c.name)
            return collections_list
        except Exception as e:
            raise RuntimeError(f"Failed to list collections: {str(e)}") from e

    def delete_collection(self, collection_name: str) -> None:
        if collection_name not in self.list_collections_names():
            raise ValueError(f"Collection '{collection_name}' does not exist.")
        try:
            self.client.delete_collection(collection_name=collection_name)
        except Exception as e:
            raise

    def add_document_list(
        self, collection_name: str, document_list: list[Document]
    ) -> None:

        uuids = [str(uuid4()) for _ in range(len(document_list))]
        try:
            vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=collection_name,
                embedding=self.embeddings,
            )
            vector_store.add_documents(documents=document_list, ids=uuids)

        except QdrantVectorStoreError as e:
            error_message = (
                f"Embedding model mismatch or collection configuration issue: {str(e)}. "
                f"Make sure the embedding model you're using matches the one for the '{collection_name}' collection."
            )
            raise ValueError(error_message) from e

        except Exception as e:
            raise RuntimeError(
                f"An unexpected error occurred while adding documents to '{collection_name}': {str(e)}"
            ) from e


if __name__ == "__main__":
    pass
