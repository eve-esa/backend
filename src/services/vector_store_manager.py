import qdrant_client
from qdrant_client import QdrantClient
from qdrant_client.conversions import common_types as types
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant.qdrant import QdrantVectorStoreError
import os
import json
import requests
import openai
from langchain_openai import OpenAI
from uuid import uuid4
from qdrant_client.http.models import FilterSelector
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, PointsSelector
from typing import Any, List
from langchain_qdrant import QdrantVectorStore
from langchain.embeddings.openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from src.services.utils import get_embeddings_model
from langchain_qdrant import RetrievalMode


from langchain import VectorDBQA, OpenAI
from collections import OrderedDict
from src.config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    MISTRAL_API_KEY,
    OPENAI_API_KEY,
    HUGGINGFACEHUB_API_TOKEN,
)
from src.services.system_prompts import generate_prompt_1


class VectorStoreManager:
    """When initializing this class, use the same embedding function you \
        used to embed the collection you want to work with"""

    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: str,
        embeddings_model: str = "text-embedding-3-small",  # "sentence-transformers/paraphrase-TinyBERT-L6-v2",
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
        query_vector = embeddings.embed_query(query)

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

    # TODO: this should be updated to custom model
    def generate_answer(
        self, query: str, context: str, llm="llama-3.1", max_new_tokens=150
    ):
        prompt = generate_prompt_1(query=query, context=context)

        if llm == "openai":
            messages_pompt = []
            messages_pompt += [{"role": "user", "content": prompt}]

            response = openai.chat.completions.create(
                messages=messages_pompt,
                model="gpt-4",
                temperature=0.3,
                max_tokens=500,
            )
            message = response.choices[0].message.content
            return message

        elif llm == "llama-3.1":
            API_URL = (
                "https://kr3w718lc87a92az.us-east-1.aws.endpoints.huggingface.cloud"
            )
            headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}

            def query(payload):
                response = requests.post(API_URL, headers=headers, json=payload)
                return response.json()

            output = query(
                {
                    "inputs": f"{prompt}",
                }
            )
            return output

        elif llm == "eve-instruct-8B":
            context = context[: (1024 - max_new_tokens) * 4]
            prompt = generate_prompt_1(query=query, context=context)
            API_URL = (
                "https://pu3i48lp4d9ovtwg.us-east-1.aws.endpoints.huggingface.cloud"
            )
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}",
                "Content-Type": "application/json",
            }

            def query(payload):
                response = requests.post(API_URL, headers=headers, json=payload)
                return response.json()

            output = query(
                {
                    "inputs": f"{prompt}",
                    "parameters": {"max_new_tokens": max_new_tokens},
                }
            )

            # get only response from "Answer:" on
            output[0]["generated_text"] = (
                output[0]["generated_text"].split("Answer:", 1)[1].strip()
            )
            return output


if __name__ == "__main__":
    pass
