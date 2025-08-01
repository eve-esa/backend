"""
Vector Store Management for Retrieval-Augmented Generation

This module provides a complete interface for creating, managing, and querying
vector collections using Qdrant as the backend. It handles embedding generation,
document storage, and similarity search operations.
"""

import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional
from uuid import uuid4
import asyncio
import runpod

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant.qdrant import QdrantVectorStoreError

from qdrant_client import QdrantClient
from qdrant_client.conversions import common_types as types
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    VectorParams,
)

from openai import AsyncOpenAI

from src.constants import DEFAULT_EMBEDDING_MODEL
from src.utils.helpers import get_embeddings_model
from src.config import (
    Config,
    OPENAI_API_KEY,
    QDRANT_URL,
    QDRANT_API_KEY,
)

# Setup logging
logger = logging.getLogger(__name__)

# Initialize configuration
config = Config()


# Import the RunPod function from utils to avoid circular imports
from src.utils.runpod_utils import get_embedding_from_runpod


class VectorStoreManager:
    """
    Manages vector storage operations for RAG (Retrieval Augmented Generation).

    This class provides methods to create and manage document collections,
    store documents with their embeddings, and retrieve relevant documents
    based on semantic similarity to queries.

    Note: When initializing this class, use the same embedding model that
    was used to embed the collection you want to work with.
    """

    def __init__(self, embeddings_model: str = DEFAULT_EMBEDDING_MODEL) -> None:
        """
        Initialize the VectorStoreManager with the specified embeddings model.

        Args:
            embeddings_model: The name of the embeddings model to use.
                Defaults to NASA's specialized model.
        """
        self.client = QdrantClient(QDRANT_URL, api_key=QDRANT_API_KEY)
        self.embeddings_model = embeddings_model
        self.embeddings, self.embeddings_size = get_embeddings_model(
            model_name=embeddings_model, return_embeddings_size=True
        )
        # Use AsyncOpenAI for async operations
        self.openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        logger.debug(f"Initialized VectorStoreManager with model: {embeddings_model}")

    def create_collection(self, collection_name: str) -> bool:
        """
        Create a new collection in the vector store.

        Args:
            collection_name: Name of the collection to create

        Returns:
            bool: True if collection creation was successful

        Raises:
            RuntimeError: If the collection creation fails
        """
        vectors_config = VectorParams(
            size=self.embeddings_size,
            distance=Distance.COSINE,
        )

        try:
            success = self.client.recreate_collection(
                collection_name=collection_name, vectors_config=vectors_config
            )

            if not success:
                raise RuntimeError(f"Failed to create collection: {collection_name}")

            logger.info(f"Collection '{collection_name}' created successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to create collection '{collection_name}': {e}")
            raise RuntimeError(f"Failed to create collection: {str(e)}") from e

    def list_collections(self) -> types.CollectionsResponse:
        """
        Get all collections from the vector store.

        Returns:
            types.CollectionsResponse: Qdrant collections response
        """
        return self.client.get_collections()

    def list_collections_names(self) -> List[str]:
        """
        Get the names of all collections in the vector store.

        Returns:
            List[str]: List of collection names

        Raises:
            RuntimeError: If listing collections fails
        """
        try:
            collections_list = []
            collections = self.client.get_collections()

            # Fix the collection name extraction
            if hasattr(collections, "collections"):
                for collection in collections.collections:
                    collections_list.append(collection.name)
            else:
                # Handle different response format
                for collection in collections:
                    if isinstance(collection, tuple) and len(collection) > 1:
                        for c in list(collection[1]):
                            collections_list.append(c.name)
                    else:
                        collections_list.append(collection.name)

            return collections_list

        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            raise RuntimeError(f"Failed to list collections: {str(e)}") from e

    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection from the vector store.

        Args:
            collection_name: Name of the collection to delete

        Returns:
            bool: True if deletion was successful

        Raises:
            ValueError: If the collection doesn't exist
            RuntimeError: If deletion fails for other reasons
        """
        if collection_name not in self.list_collections_names():
            logger.warning(
                f"Attempted to delete non-existent collection '{collection_name}'"
            )
            raise ValueError(f"Collection '{collection_name}' does not exist")

        try:
            self.client.delete_collection(collection_name=collection_name)
            logger.info(f"Collection '{collection_name}' deleted successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to delete collection '{collection_name}': {e}")
            raise RuntimeError(f"Failed to delete collection: {str(e)}") from e

    def add_document_list(
        self, collection_name: str, document_list: List[Document]
    ) -> List[str]:
        """
        Add a list of documents to a collection.

        Args:
            collection_name: Name of the collection to add documents to
            document_list: List of documents to add

        Returns:
            List[str]: List of document IDs created

        Raises:
            ValueError: If there's a model mismatch or configuration issue
            RuntimeError: For other unexpected errors
        """
        if not document_list:
            logger.warning("Empty document list provided, nothing to add")
            return []

        uuids = [str(uuid4()) for _ in range(len(document_list))]

        try:
            vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=collection_name,
                embedding=self.embeddings,
            )

            vector_store.add_documents(documents=document_list, ids=uuids)
            logger.info(f"Added {len(document_list)} documents to '{collection_name}'")
            return uuids

        except QdrantVectorStoreError as e:
            error_message = (
                f"Embedding model mismatch or collection configuration issue: {str(e)}. "
                f"Ensure the embedding model matches the one used for '{collection_name}'."
            )
            logger.error(error_message)
            raise ValueError(error_message) from e

        except Exception as e:
            logger.error(f"Error adding documents to '{collection_name}': {e}")
            raise RuntimeError(
                f"Failed to add documents to '{collection_name}': {str(e)}"
            ) from e

    def _qdrant_filter_from_dict(
        self, filter_dict: Optional[Dict[str, Any]]
    ) -> Optional[Filter]:
        """
        Convert a Python dictionary to a Qdrant filter object.

        Args:
            filter_dict: Dictionary containing filter criteria

        Returns:
            Optional[Filter]: Qdrant filter object or None if no filter provided
        """
        if not filter_dict:
            return None

        return Filter(
            must=[
                condition
                for key, value in filter_dict.items()
                for condition in self._build_condition(key, value)
            ]
        )

    def _build_condition(self, key: str, value: Any) -> List[FieldCondition]:
        """
        Build Qdrant field conditions from keys and values.

        Handles nested structures recursively.

        Args:
            key: The metadata field key
            value: The value to match

        Returns:
            List[FieldCondition]: List of generated field conditions
        """
        conditions = []

        if isinstance(value, dict):
            for _key, _value in value.items():
                conditions.extend(self._build_condition(f"{key}.{_key}", _value))

        elif isinstance(value, list):
            for _value in value:
                if isinstance(_value, dict):
                    conditions.extend(self._build_condition(f"{key}[]", _value))
                else:
                    conditions.extend(self._build_condition(f"{key}", _value))

        else:
            conditions.append(
                FieldCondition(
                    key=f"metadata.{key}",
                    match=MatchValue(value=value),
                )
            )

        return conditions

    def delete_docs_by_metadata_filter(
        self, collection_name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Delete documents that match a metadata filter.

        Args:
            collection_name: Name of the collection
            metadata: Metadata filter to select documents for deletion

        Returns:
            Dict[str, Any]: Result of the deletion operation with deleted count

        Raises:
            RuntimeError: If deletion fails
        """
        try:
            # Get the count of documents before deletion using the count method
            filter_obj = self._qdrant_filter_from_dict(metadata)
            count_result = self.client.count(
                collection_name=collection_name,
                count_filter=filter_obj,
                exact=True,  # For an exact count
            )
            count_before = count_result.count

            # Perform the deletion
            result = self.client.delete(
                collection_name=collection_name,
                points_selector=filter_obj,
            )

            # Get the count of documents after deletion
            count_result_after = self.client.count(
                collection_name=collection_name,
                count_filter=filter_obj,
                exact=True,  # For an exact count
            )
            count_after = count_result_after.count

            # Calculate the actual number of deleted documents
            deleted_count = count_before - count_after

            logger.info(f"Deleted {deleted_count} documents from '{collection_name}'")

            # Create a result object with the deleted count
            class DeleteResult:
                def __init__(self, deleted_count):
                    self.deleted = deleted_count

            return DeleteResult(deleted_count)

        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise RuntimeError(f"Failed to delete documents: {str(e)}") from e

    def _get_unique_source_documents(
        self, scored_points_list: List[Any], min_docs: int = 2
    ) -> List[Any]:
        """
        Filter search results to keep only one document per source.

        Args:
            scored_points_list: List of search results with scores
            min_docs: Minimum number of unique documents to return

        Returns:
            List[Any]: List of unique documents ordered by relevance score
        """
        # Sort results by score (highest first)
        sorted_results = sorted(scored_points_list, key=lambda x: x.score, reverse=True)
        unique_source_items = OrderedDict()

        # Keep only one document per source, prioritizing higher scores
        for item in sorted_results:
            try:
                source = item.payload["title"]
                if source not in unique_source_items:
                    unique_source_items[source] = item
                if len(unique_source_items) >= min_docs:
                    break
            except (KeyError, TypeError) as e:
                logger.warning(f"Skipping item with malformed metadata: {e}")
                continue

        return list(unique_source_items.values())

    async def generate_query_vector(
        self, query: str, embeddings_model: str
    ) -> List[float]:
        """
        Generate an embedding vector for a query.

        Args:
            query: The query text
            embeddings_model: Model to use for embedding generation

        Returns:
            List[float]: Vector representation of the query

        Raises:
            RuntimeError: If embedding generation fails
        """
        try:
            # Special handling for NASA model which uses RunPod API
            if embeddings_model == DEFAULT_EMBEDDING_MODEL:
                logger.info("Using RunPod API for embedding generation")

                # Call the remote API to generate embeddings
                query_vector = await get_embedding_from_runpod(
                    endpoint_id=config.get_indus_embedder_id(),
                    model=embeddings_model,
                    user_input=query,
                )
                logger.debug(query_vector)
                # Validate the received vector
                if not query_vector or not isinstance(query_vector, list):
                    logger.error(f"Invalid embedding vector received: {query_vector}")
                    raise RuntimeError(
                        "Invalid embedding vector received from RunPod API"
                    )

                return query_vector

            # Standard local embedding generation for other models
            embeddings = get_embeddings_model(embeddings_model)
            if hasattr(embeddings, "embed_query_async"):
                return await embeddings.embed_query_async(query)
            else:
                return embeddings.embed_query(query)

        except Exception as e:
            logger.error(f"Failed to generate query vector: {e}")
            raise RuntimeError(f"Failed to generate embedding: {str(e)}") from e

    async def retrieve_documents_from_query(
        self,
        collection_name: str,
        query: str,
        k: int = 5,
        score_threshold: float = 0.7,
        get_unique_docs: bool = True,
        embeddings_model: Optional[str] = None,
    ) -> List[Any]:
        """
        Retrieve relevant documents for a given query.

        Args:
            collection_name: Name of the collection to search
            query: The query text
            k: Number of documents to retrieve
            score_threshold: Minimum similarity score (0-1)
            get_unique_docs: Whether to filter for unique source documents
            embeddings_model: Optional custom embedding model to use
                              (defaults to the model used at initialization)

        Returns:
            List[Any]: List of relevant documents with similarity scores

        Raises:
            RuntimeError: If retrieval fails
        """
        model = embeddings_model or self.embeddings_model

        try:
            # Generate embedding vector for the query
            query_vector = await self.generate_query_vector(query, model)

            if not get_unique_docs:
                # Simple search with limit k
                results = self.client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=k,
                    score_threshold=score_threshold,
                )
                logger.info(f"Retrieved docs: {results}")

                logger.info(
                    f"Retrieved {len(results)} documents from '{collection_name}'"
                )
                return results

            # Get more results to allow filtering for unique sources
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=k * 10,  # Get more results than needed for filtering
                score_threshold=score_threshold,
            )
            logger.info(f"Retrieved docs: {results}")

            unique_results = self._get_unique_source_documents(results, min_docs=k)
            logger.info(
                f"Retrieved {len(unique_results)} unique documents from '{collection_name}' "
                f"(filtered from {len(results)} total matches)"
            )
            return unique_results

        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            raise RuntimeError(f"Failed to retrieve documents: {str(e)}") from e

    async def use_rag(self, query: str) -> bool:
        """
        Determine if RAG should be used for a given query.

        Uses a language model to classify whether the query is appropriate
        for retrieval-augmented generation.

        Args:
            query: The user's query

        Returns:
            bool: True if RAG should be used, False otherwise

        Raises:
            RuntimeError: If determination fails
        """
        try:
            # Create a prompt to determine if RAG is appropriate
            prompt = f"""
            Decide whether to use RAG to answer the given query. Follow these rules:
            - Do NOT use RAG for generic, casual, or non-specific queries, such as "hi", 
              "hello", "how are you", "what can you do", or "tell me a joke".
            - USE RAG for queries related to earth science, space science, climate, 
              space agencies, or similar scientific topics.
            - USE RAG for specific technical or scientific questions, even if the topic is unclear
              (e.g., "What's the thermal conductivity of basalt?" or "How does orbital decay work?").
            - If unsure whether RAG is needed, default to USING RAG.
            - Respond only with 'yes' or 'no'.

            Query: {query}
            """

            # Call OpenAI to decide
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0,
            )

            if response.choices:
                answer = response.choices[0].message.content.strip().lower()

                if answer == "yes":
                    logger.info(f"Using RAG for query: '{query}'")
                    return True

                elif answer == "no":
                    logger.info(f"Not using RAG for query: '{query}'")
                    return False

                else:
                    logger.warning(f"Unexpected RAG determination response: '{answer}'")
                    # Default to using RAG when response is unclear
                    return True

            else:
                logger.warning(
                    "Empty response from language model for RAG determination"
                )
                # Default to using RAG when there's no response
                return True

        except Exception as e:
            logger.error(f"Failed to determine if RAG should be used: {e}")
            # In case of errors, default to using RAG to be safe
            return True

    def sync_retrieve_documents_from_query(
        self,
        collection_name: str,
        query: str,
        k: int = 5,
        score_threshold: float = 0.7,
        get_unique_docs: bool = True,
        embeddings_model: Optional[str] = None,
    ) -> List[Any]:
        """
        Synchronous wrapper for retrieve_documents_from_query.

        This method allows calling the async retrieval method from sync contexts.
        """
        return asyncio.run(
            self.retrieve_documents_from_query(
                collection_name=collection_name,
                query=query,
                k=k,
                score_threshold=score_threshold,
                get_unique_docs=get_unique_docs,
                embeddings_model=embeddings_model,
            )
        )

    def sync_use_rag(self, query: str) -> bool:
        """
        Synchronous wrapper for use_rag.

        This method allows calling the async RAG determination from sync contexts.
        """
        return asyncio.run(self.use_rag(query))


if __name__ == "__main__":
    # Configure logging when run as a script
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
