from src.services.vector_store_manager import VectorStoreManager
from langchain_core.documents import Document

from src.config import QDRANT_URL, QDRANT_API_KEY

if __name__ == "__main__":

    # "text-embedding-3-small"
    # "sentence-transformers/paraphrase-TinyBERT-L6-v2"
    # "mistral-embed"

    vector_store = VectorStoreManager(
        QDRANT_URL, QDRANT_API_KEY, embeddings_model="text-embedding-3-small"
    )

    import os

    results = vector_store.retrieve_documents_from_query(
        query=f"empty",
        embeddings_model="mistral-embed",
        collection_name="test_llm4eo",
        score_threshold=0.7,
        k=3,
    )

    for result in results:
        print(
            f"Point ID: {result.id}, Score: {result.score}, Metadata: {result.payload['metadata']}"
        )
    if results == []:
        print("empty")
