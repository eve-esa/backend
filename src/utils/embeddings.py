"""
Embeddings utilities and custom embedding classes.

This module provides specialized embedding classes and related utilities
for working with various embedding models, particularly those requiring
remote API calls rather than local model loading.
"""

import asyncio
import logging
import runpod
from typing import List
from langchain_core.embeddings import Embeddings
from src.config import RUNPOD_API_KEY, Config

# Configure logging
logger = logging.getLogger(__name__)

# Configure RunPod API key
runpod.api_key = RUNPOD_API_KEY

# Load configuration
config = Config()


async def runpod_api_request(
    endpoint_id: str, model: str, user_input: str, timeout: int = 60
) -> List[float]:
    """
    Get embeddings as a vector using the RunPod API.

    Args:
        endpoint_id: The RunPod endpoint ID
        model: The model to use for embedding
        user_input: The text to embed
        timeout: Maximum time to wait for the result in seconds

    Returns:
        List[float]: The embedding vector

    Raises:
        ValueError: If the returned embedding is invalid
        RuntimeError: If the API call fails
    """
    # Prepare the input payload
    payload = {"input": {"model": model, "input": user_input}}

    try:
        # Create an endpoint instance
        endpoint = runpod.Endpoint(endpoint_id)

        # Submit the job
        run_request = endpoint.run(payload)
        job_id = run_request.job_id
        logger.info(f"RunPod job submitted: {job_id}")

        # Poll for the output with a timeout
        result = run_request.output(timeout=timeout)

        # Validate the result
        if not result or "data" not in result:
            logger.error(f"Invalid response from RunPod: {result}")
            raise ValueError(f"Invalid response from RunPod: {result}")

        # Extract and validate the embedding
        embedding = result["data"][0]["embedding"]

        if not isinstance(embedding, list) or not all(
            isinstance(x, (int, float)) for x in embedding
        ):
            logger.error(f"Invalid embedding format: {type(embedding)}")
            raise ValueError("Embedding is not a valid list of numbers")

        logger.info(f"RunPod job completed successfully: {job_id}")
        return embedding

    except asyncio.TimeoutError:
        logger.error(f"RunPod request timed out after {timeout} seconds")
        raise RuntimeError(f"RunPod API request timed out after {timeout} seconds")

    except Exception as e:
        logger.error(f"RunPod API error: {str(e)}")
        raise RuntimeError(f"RunPod API request failed: {str(e)}") from e


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
            # Use the runpod_api_request function from utils
            embedding = await runpod_api_request(
                endpoint_id=self.endpoint_id, model=self.model_name, user_input=text
            )
            return embedding
        except Exception as e:
            raise Exception(f"RunPod API request failed: {str(e)}")
