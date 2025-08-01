"""
RunPod API utilities for embedding generation.

This module provides functions for interacting with RunPod API
to generate embeddings without creating circular imports.
"""

import asyncio
import logging
import runpod
from typing import List

# Configure logging
logger = logging.getLogger(__name__)


async def get_embedding_from_runpod(
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
