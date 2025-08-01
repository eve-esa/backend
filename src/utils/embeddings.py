"""
Embeddings utilities and custom embedding classes.

This module provides specialized embedding classes and related utilities
for working with various embedding models, particularly those requiring
remote API calls rather than local model loading.
"""

import asyncio
import logging
from typing import List
from langchain_core.embeddings import Embeddings
from src.config import Config

# Configure logging
logger = logging.getLogger(__name__)

# Load configuration
config = Config()


# Import the consolidated function from runpod_utils
from src.utils.runpod_utils import get_embedding_from_runpod


class RunPodEmbeddings(Embeddings):
    """
    Custom embeddings class that uses RunPod API for remote models.

    This prevents models from being downloaded locally and instead
    defers embedding generation to the RunPod API.
    """

    def __init__(self, model_name: str, embedding_size: int = 768):
        """Initialize the RunPod embeddings proxy."""
        self.model_name = model_name
        self.embedding_size = embedding_size
        # Get the RunPod endpoint ID from configuration
        self.endpoint_id = config.get_indus_embedder_id()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text using the RunPod API."""
        import asyncio
        import concurrent.futures

        def run_async_embed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._embed(text))
            finally:
                loop.close()

        # Use ThreadPoolExecutor to run async code in sync context
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_async_embed)
            return future.result()

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of documents using the RunPod API."""
        if not documents:
            return []

        # For single document, use the regular method
        if len(documents) == 1:
            return [self.embed_query(documents[0])]

        # For multiple documents, process them in batches
        embeddings = []
        for text in documents:
            embedding = self.embed_query(text)
            embeddings.append(embedding)

        return embeddings

    async def embed_query_async(self, text: str) -> List[float]:
        """Async version of embed_query for use in async contexts."""
        return await self._embed(text)

    async def embed_documents_async(self, documents: List[str]) -> List[List[float]]:
        """Async version of embed_documents for use in async contexts."""
        if not documents:
            return []

        # For single document, use the regular method
        if len(documents) == 1:
            return [await self._embed(documents[0])]

        # For multiple documents, process them in batches
        embeddings = []
        for text in documents:
            embedding = await self._embed(text)
            embeddings.append(embedding)

        return embeddings

    async def _embed(self, text: str) -> List[float]:
        """Embed a single document using the RunPod API."""
        try:
            # Use the consolidated function from vector_store_manager
            embedding = await get_embedding_from_runpod(
                endpoint_id=self.endpoint_id, model=self.model_name, user_input=text
            )
            return embedding
        except Exception as e:
            raise Exception(f"RunPod API request failed: {str(e)}")
