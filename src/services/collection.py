"""
Collection Service for managing Qdrant collections.

This module provides a service layer for creating, managing, and interacting
with collections in the Qdrant vector store.
"""

import logging
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

from src.core.vector_store_manager import VectorStoreManager
from schemas.collection import CollectionRequest
from src.constants import DEFAULT_EMBEDDING_MODEL

logger = logging.getLogger(__name__)


@dataclass
class Collection:
    """Represents a collection in the vector store."""

    name: str
    embeddings_model: str
    created_at: Optional[datetime] = None


class CollectionService:
    """
    Service for managing collections in the vector store.

    This class provides methods to create, list, and manage collections
    using the VectorStoreManager.
    """

    def __init__(self):
        """Initialize the collection service."""
        self.vector_store_manager = None

    def _get_vector_store_manager(
        self, embeddings_model: str = DEFAULT_EMBEDDING_MODEL
    ) -> VectorStoreManager:
        """
        Get or create a VectorStoreManager instance.

        Args:
            embeddings_model: The embeddings model to use

        Returns:
            VectorStoreManager: The vector store manager instance
        """
        if (
            self.vector_store_manager is None
            or self.vector_store_manager.embeddings_model != embeddings_model
        ):
            self.vector_store_manager = VectorStoreManager(
                embeddings_model=embeddings_model
            )
        return self.vector_store_manager

    async def create_collection(
        self, request: CollectionRequest, collection_name: str
    ) -> Collection:
        """
        Create a new collection in the vector store.

        Args:
            request: The collection request containing configuration
            collection_name: Name of the collection to create

        Returns:
            Collection: The created collection object

        Raises:
            ValueError: If collection already exists or creation fails
            Exception: For other unexpected errors
        """
        try:
            # Get vector store manager with the specified embeddings model
            vector_store_manager = self._get_vector_store_manager(
                request.embeddings_model
            )

            # Check if collection already exists
            existing_collections = vector_store_manager.list_collections_names()
            if collection_name in existing_collections:
                logger.info(f"Collection '{collection_name}' already exists")
                raise ValueError(f"Collection '{collection_name}' already exists")

            # Create the collection
            success = vector_store_manager.create_collection(collection_name)

            if not success:
                raise ValueError(f"Failed to create collection '{collection_name}'")

            logger.info(
                f"Collection '{collection_name}' created successfully with model '{request.embeddings_model}'"
            )

            return Collection(
                name=collection_name,
                embeddings_model=request.embeddings_model,
                created_at=datetime.now(),
            )

        except ValueError as e:
            logger.error(
                f"ValueError creating collection '{collection_name}': {str(e)}"
            )
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error creating collection '{collection_name}': {str(e)}"
            )
            raise Exception(f"Failed to create collection: {str(e)}")

    async def list_collections(self, embeddings_model: str) -> list[str]:
        """
        List all collections in the vector store.

        Args:
            embeddings_model: The embeddings model to use

        Returns:
            list[str]: List of collection names

        Raises:
            Exception: If listing collections fails
        """
        try:
            vector_store_manager = self._get_vector_store_manager(embeddings_model)
            return vector_store_manager.list_collections_names()
        except Exception as e:
            logger.error(f"Error listing collections: {str(e)}")
            raise Exception(f"Failed to list collections: {str(e)}")

    async def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection from the vector store.

        Args:
            collection_name: Name of the collection to delete
            request: Optional collection request with embeddings model configuration

        Returns:
            bool: True if deletion was successful

        Raises:
            ValueError: If the collection doesn't exist
            Exception: If deletion fails for other reasons
        """
        try:
            vector_store_manager = self._get_vector_store_manager()
            return vector_store_manager.delete_collection(collection_name)
        except ValueError as e:
            logger.error(
                f"ValueError deleting collection '{collection_name}': {str(e)}"
            )
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error deleting collection '{collection_name}': {str(e)}"
            )
            raise Exception(f"Failed to delete collection: {str(e)}")

    async def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists.

        Args:
            collection_name: Name of the collection to check

        Returns:
            bool: True if collection exists, False otherwise
        """
        try:
            collections = await self.list_collections()
            return collection_name in collections
        except Exception as e:
            logger.error(
                f"Error checking if collection '{collection_name}' exists: {str(e)}"
            )
            return False
