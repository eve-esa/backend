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

from langchain_core.documents import Document
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

from src.core.llm_manager import LLMManager

from src.constants import DEFAULT_EMBEDDING_MODEL, PUBLIC_COLLECTIONS
from src.utils.helpers import get_embeddings_model
from src.config import (
    Config,
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
        # Initialize Qdrant client with timeout configuration
        self.client = QdrantClient(
            QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=120.0,  # 2 minutes timeout for operations
        )
        self.embeddings_model = embeddings_model
        self.embeddings, self.embeddings_size = get_embeddings_model(
            model_name=embeddings_model, return_embeddings_size=True
        )
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

    def list_public_collections(
        self, page: int = 1, limit: int = 10
    ) -> List[Dict[str, str]]:
        """
        Get all public collections from the vector store.

        Returns:
            Dict[str, str]: Dictionary of collection names and descriptions
        """
        collections = self.client.get_collections()
        start = (page - 1) * limit
        end = start + limit
        return [
            {
                "name": collection.name,
                "description": PUBLIC_COLLECTIONS[collection.name],
            }
            for collection in collections.collections
            if collection.name in PUBLIC_COLLECTIONS.keys()
        ][start:end]

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
            collection_name: Name of the collection
            document_list: List of documents to add

        Returns:
            List[str]: List of UUIDs for the added documents

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

            # Use batch writing to prevent timeout issues
            # Batch size 32 matches RunPod Infinity Embedding BATCH_SIZES=32 configuration
            self._add_documents_in_batches(
                vector_store, document_list, uuids, batch_size=32
            )
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

    def _add_documents_in_batches(
        self,
        vector_store: QdrantVectorStore,
        documents: List[Document],
        uuids: List[str],
        batch_size: int = 32,
    ) -> None:
        """
        Add documents to Qdrant in batches to prevent timeout issues.

        Args:
            vector_store: The Qdrant vector store instance
            documents: List of documents to add
            uuids: List of UUIDs for the documents
            batch_size: Number of documents to process in each batch
        """
        import time

        total_documents = len(documents)
        logger.info(f"Adding {total_documents} documents in batches of {batch_size}")

        for i in range(0, total_documents, batch_size):
            batch_end = min(i + batch_size, total_documents)
            batch_documents = documents[i:batch_end]
            batch_uuids = uuids[i:batch_end]

            batch_num = (i // batch_size) + 1
            total_batches = (total_documents + batch_size - 1) // batch_size

            logger.info(
                f"Processing batch {batch_num}/{total_batches} ({len(batch_documents)} documents)"
            )

            try:
                # Use direct Qdrant client operations instead of LangChain wrapper
                self._add_documents_directly(
                    batch_documents, batch_uuids, vector_store.collection_name
                )
                logger.info(f"Successfully added batch {batch_num}/{total_batches}")

                # Add a small delay between batches to prevent overwhelming Qdrant
                if batch_num < total_batches:
                    time.sleep(0.5)

            except Exception as e:
                logger.error(
                    f"Failed to add batch {batch_num}/{total_batches}: {str(e)}"
                )
                raise RuntimeError(f"Failed to add batch {batch_num}: {str(e)}") from e

    def _add_documents_directly(
        self, documents: List[Document], uuids: List[str], collection_name: str
    ) -> None:
        """
        Add documents directly to Qdrant using the client API to avoid LangChain wrapper issues.

        Args:
            documents: List of documents to add
            uuids: List of UUIDs for the documents
            collection_name: Name of the collection
        """
        from qdrant_client.models import PointStruct

        # Extract texts and metadata
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Generate embeddings for this batch
        embeddings = self.embeddings.embed_documents(texts)

        # Create points for Qdrant
        points = []
        for i, (text, metadata, embedding, uuid) in enumerate(
            zip(texts, metadatas, embeddings, uuids)
        ):
            point = PointStruct(
                id=uuid, vector=embedding, payload={"text": text, "metadata": metadata}
            )
            points.append(point)

        # Upsert points to Qdrant
        self.client.upsert(collection_name=collection_name, points=points, wait=True)

    def _qdrant_filter_from_dict(
        self, filter_dict: Optional[Dict[str, Any]]
    ) -> Optional[Filter]:
        """
        Convert a simple dictionary (key -> value) to a Qdrant filter object,
        applying conditions against payload.metadata.* keys by default.

        This is kept for backwards compatibility where callers pass a flat dict.
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

    def _search_across_collections(
        self,
        collection_names: List[str],
        query_vector: List[float],
        score_threshold: float,
        query_filter: Optional[Filter],
        limit_per_collection: int,
    ) -> List[Any]:
        """
        Perform a search against multiple collections and aggregate results.

        Args:
            collection_names: Names of collections to query
            query_vector: Embedded query vector
            score_threshold: Minimum similarity score
            query_filter: Optional Qdrant filter to apply
            limit_per_collection: Max results to fetch per collection

        Returns:
            Aggregated list of search results from all collections
        """
        aggregated_results: List[Any] = []
        for collection_name in collection_names:
            try:
                results = self.client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit_per_collection,
                    score_threshold=score_threshold,
                    query_filter=query_filter,
                )
                aggregated_results.extend(results)
            except Exception as e:
                logger.warning(f"Failed to search collection '{collection_name}': {e}")
                continue

        logger.info(
            f"Retrieved {len(aggregated_results)} total documents from {len(collection_names)} collections"
        )
        return aggregated_results

    @staticmethod
    def _sort_by_score_desc(results: List[Any]) -> List[Any]:
        """
        Sort a list of scored results in descending order of score.

        Args:
            results: List of results with a 'score' attribute

        Returns:
            Sorted list by score (highest first)
        """
        try:
            return sorted(results, key=lambda x: x.score, reverse=True)
        except Exception:
            # If objects don't have score attribute, return as-is
            return results

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
                    key=f"{key}",
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
                    use_retries=False,
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
        collection_names: List[str],
        query: str,
        k: int = 5,
        score_threshold: float = 0.7,
        embeddings_model: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """
        Retrieve relevant documents for a given query from multiple collections.

        Args:
            collection_names: List of names of the collections to search
            query: The query text
            year: List with two values [start_year, end_year] to filter by publication year.
            keywords: List of keywords to filter by title.
            k: Number of documents to retrieve
            score_threshold: Minimum similarity score (0-1)
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
            query_filter = Filter(**filters) if filters else None

            # Get more results to allow filtering for documents that match the query
            all_results = self._search_across_collections(
                collection_names=collection_names,
                query_vector=query_vector,
                score_threshold=score_threshold,
                query_filter=query_filter,
                limit_per_collection=k
                * 10,  # Get more per-collection results for filtering
            )

            logger.info(
                f"Retrieved {len(all_results)} documents from {len(collection_names)} collections "
                f"(filtered from {len(all_results)} total matches)"
            )
            return all_results

        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            raise RuntimeError(f"Failed to retrieve documents: {str(e)}") from e

    # RAG decision moved to LLMManager.should_use_rag

    async def retrieve_documents_with_latencies(
        self,
        collection_names: List[str],
        query: str,
        k: int = 5,
        score_threshold: float = 0.7,
        embeddings_model: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> tuple[List[Any], Dict[str, Optional[float]]]:
        """
        Retrieve relevant documents and measure query embedding and Qdrant retrieval latencies.

        Returns a tuple of (results, latencies)
        where latencies contains keys: "query_embedding_latency", "qdrant_retrieval_latency".
        """
        import time

        model = embeddings_model or self.embeddings_model

        embedding_latency: Optional[float] = None
        retrieval_latency: Optional[float] = None

        try:
            # Generate embedding vector for the query
            t0 = time.perf_counter()
            query_vector = await self.generate_query_vector(query, model)
            embedding_latency = time.perf_counter() - t0

            query_filter = Filter(**filters) if filters else None

            # Search across collections
            t1 = time.perf_counter()
            all_results = self._search_across_collections(
                collection_names=collection_names,
                query_vector=query_vector,
                score_threshold=score_threshold,
                query_filter=query_filter,
                limit_per_collection=k * 10,
            )
            retrieval_latency = time.perf_counter() - t1

            logger.info(
                f"Retrieved {len(all_results)} documents from {len(collection_names)} collections "
                f"(filtered from {len(all_results)} total matches)"
            )

            latencies: Dict[str, Optional[float]] = {
                "query_embedding_latency": embedding_latency,
                "qdrant_retrieval_latency": retrieval_latency,
            }
            return all_results, latencies

        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            raise RuntimeError(f"Failed to retrieve documents: {str(e)}") from e

    def sync_retrieve_documents_from_query(
        self,
        collection_name: str,
        query: str,
        k: int = 5,
        score_threshold: float = 0.7,
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
                embeddings_model=embeddings_model,
            )
        )

    # RAG decision moved to LLMManager.should_use_rag


if __name__ == "__main__":
    # Configure logging when run as a script
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
