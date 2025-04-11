import logging
from openai import Client
from collections import OrderedDict
from uuid import uuid4
from typing import Any, List

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant.qdrant import QdrantVectorStoreError

import qdrant_client
from qdrant_client import QdrantClient
from qdrant_client.conversions import common_types as types
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from src.services.utils import get_embeddings_model, runpod_api_request
from src.config import Config

from src.config import (
    OPENAI_API_KEY,
    QDRANT_URL,
    QDRANT_API_KEY

)

config = Config()

class VectorStoreManager:
    """When initializing this class, use the same embedding function you \
        used to embed the collection you want to work with"""

    def __init__(
        self,
        embeddings_model: str = "nasa-impact/nasa-smd-ibm-v0.1",  # "sentence-transformers/paraphrase-TinyBERT-L6-v2",
    ) -> None:
        self.client = QdrantClient(QDRANT_URL, api_key=QDRANT_API_KEY)
        self.embeddings_model = embeddings_model
        self.embeddings, self.embeddings_size = get_embeddings_model(
            model=embeddings_model, return_embeddings_size=True
        )
        self.collection = None
        self.openai_client = Client(api_key=OPENAI_API_KEY)


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
            vector_store : QdrantVectorStore = QdrantVectorStore(
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

    def _qdrant_filter_from_dict(self, filter: dict) -> Filter:
        if not filter:
            return None

        return Filter(
            must=[
                condition
                for key, value in filter.items()
                for condition in self._build_condition(key, value)
            ]
        )

    def _build_condition(self, key: str, value: Any) -> List[FieldCondition]:
        out = []

        if isinstance(value, dict):
            for _key, value in value.items():
                out.extend(self._build_condition(f"{key}.{_key}", value))
        elif isinstance(value, list):
            for _value in value:
                if isinstance(_value, dict):
                    out.extend(self._build_condition(f"{key}[]", _value))
                else:
                    out.extend(self._build_condition(f"{key}", _value))
        else:
            out.append(
                FieldCondition(
                    key=f"metadata.{key}",
                    match=MatchValue(value=value),
                )
            )

        return out

    def delete_docs_by_metadata_filter(self, collection_name: str, metadata=None):
        res = self.client.delete(
            collection_name=collection_name,
            points_selector=self._qdrant_filter_from_dict(metadata),
        )
        return res

    def _get_unique_source_documents(self, scored_points_list, min_docs=2):
        sorted_results = sorted(scored_points_list, key=lambda x: x.score, reverse=True)

        unique_source_items = OrderedDict()

        for item in sorted_results:
            source = item.payload["metadata"]["source"]
            if source not in unique_source_items:
                unique_source_items[source] = item
            if len(unique_source_items) >= min_docs:
                break
        return list(unique_source_items.values())

    def retrieve_documents_from_query(
        self,
        collection_name: str,
        embeddings_model: str,
        query: str,
        k: int = 5,
        score_threshold: float = 0.7,
        get_unique_docs=True,
    ):

        embeddings = get_embeddings_model(embeddings_model)
        query_vector : List[float] = [] 
        if isinstance(embeddings, HuggingFaceEmbeddings) and embeddings.model_name == "nasa-impact/nasa-smd-ibm-v0.1":
           logging.info("Using Runpod API for embedding", )
           
           #wait the function to be completed
           query_vector = runpod_api_request(
               endpoint_id= config.get_indus_embedder_id(),
               #url="https://api.runpod.ai/v2/c9zv853ctjg5ps/run",
               model=embeddings.model_name,
               user_input=query
               )
           print(query_vector)
        else:
            query_vector  =  embeddings.embed_query(query)

        if not get_unique_docs:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,  # The vector representing the query
                limit=k,
                score_threshold=score_threshold,
            )
            return results

        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,  # The vector representing the query
            limit=k * 10,
            score_threshold=score_threshold,
        )
        return self._get_unique_source_documents(results, min_docs=k)
    
    def use_rag(self, query: str) -> bool:
        prompt = f"""
        Decide whether to use RAG to answer the given query. Follow these rules:
        - Do NOT use RAG for generic, casual, or non-specific queries, such as "hi", "hello", "how are you", "what can you do", or "tell me a joke".
        - USE RAG for queries related to earth science, space science, climate, space agencies, or similar scientific topics.
        - USE RAG for specific technical or scientific questions, even if the topic is unclear (e.g., "What’s the thermal conductivity of basalt?" or "How does orbital decay work?").
        - If unsure whether RAG is needed, default to USING RAG.
        - Respond only with 'yes' or 'no'.

        Query: {query}
        """

        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
        )

        if response.choices:
            answer = response.choices[0].message.content.strip()
            if answer.lower() == "yes":
                print("I am using rag...")
                return True
            elif answer.lower() == "no":
                print("I am not using rag...")
                return False
            else:
                raise ValueError("Unexpected response from OpenAI API")
            
    

if __name__ == "__main__":
    pass
