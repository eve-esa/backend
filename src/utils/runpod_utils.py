"""
RunPod API utilities for embedding generation.

This module provides functions for interacting with RunPod API
to generate embeddings without creating circular imports.
"""

import asyncio
import logging
import runpod
from typing import List, Optional
import time

# Configure logging
logger = logging.getLogger(__name__)


async def get_embedding_from_runpod(
    endpoint_id: str,
    model: str,
    user_input: str,
    timeout: int = 60,
    max_retries: int = 3,
) -> List[float]:
    """
    Get embeddings as a vector using the RunPod API.

    Args:
        endpoint_id: The RunPod endpoint ID
        model: The model to use for embedding
        user_input: The text to embed
        timeout: Maximum time to wait for the result in seconds
        max_retries: Maximum number of retry attempts

    Returns:
        List[float]: The embedding vector

    Raises:
        ValueError: If the returned embedding is invalid
        RuntimeError: If the API call fails
    """
    # Prepare the input payload
    # Note: Never log the actual text content for privacy/security reasons
    payload = {"input": {"model": model, "input": user_input}}

    for attempt in range(max_retries):
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
            logger.error(
                f"RunPod request timed out after {timeout} seconds (attempt {attempt + 1}/{max_retries})"
            )
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"RunPod API request timed out after {timeout} seconds"
                )
            time.sleep(2**attempt)  # Exponential backoff

        except Exception as e:
            logger.error(
                f"RunPod API error (attempt {attempt + 1}/{max_retries}): {str(e)}"
            )
            if attempt == max_retries - 1:
                raise RuntimeError(f"RunPod API request failed: {str(e)}") from e
            time.sleep(2**attempt)  # Exponential backoff

    raise RuntimeError("All retry attempts failed")


async def get_batch_embeddings_from_runpod(
    endpoint_id: str,
    model: str,
    texts: List[str],
    timeout: int = 120,
    max_retries: int = 3,
) -> List[List[float]]:
    """
    Get embeddings for multiple texts using the RunPod Infinity Embedding API.

    Based on the Infinity Embedding worker documentation, this function uses
    the correct batch format for processing multiple texts efficiently.

    Args:
        endpoint_id: The RunPod endpoint ID
        model: The model to use for embedding
        texts: List of texts to embed
        timeout: Maximum time to wait for the result in seconds
        max_retries: Maximum number of retry attempts

    Returns:
        List[List[float]]: List of embedding vectors

    Raises:
        ValueError: If the returned embeddings are invalid
        RuntimeError: If the API call fails after all retries
    """
    if not texts:
        return []

    # Prepare the input payload for batch processing
    # Note: Never log the actual text content for privacy/security reasons
    payload = {
        "input": {"model": model, "input": texts}
    }  # Send all texts as a single list

    for attempt in range(max_retries):
        try:
            # Create an endpoint instance
            endpoint = runpod.Endpoint(endpoint_id)

            # Submit the job
            run_request = endpoint.run(payload)
            job_id = run_request.job_id
            logger.info(f"RunPod batch job submitted: {job_id} for {len(texts)} texts")

            # Poll for the output with a timeout
            result = run_request.output(timeout=timeout)

            # Validate the result
            if not result or "data" not in result:
                logger.error(f"Invalid response from RunPod: {result}")
                raise ValueError(f"Invalid response from RunPod: {result}")

            # Extract and validate the embeddings
            # The Infinity Embedding API returns data in a specific format
            embeddings_data = result["data"]

            if not isinstance(embeddings_data, list):
                logger.error(f"Invalid embeddings data format: {type(embeddings_data)}")
                raise ValueError("Embeddings data is not a valid list")

            # Extract embeddings from the response format
            embeddings = []
            for i, item in enumerate(embeddings_data):
                if isinstance(item, dict) and "embedding" in item:
                    embedding = item["embedding"]
                    if isinstance(embedding, list) and all(
                        isinstance(x, (int, float)) for x in embedding
                    ):
                        embeddings.append(embedding)
                    else:
                        logger.error(
                            f"Invalid embedding format at index {i}: {type(embedding)}"
                        )
                        raise ValueError(
                            f"Embedding at index {i} is not a valid list of numbers"
                        )
                else:
                    logger.error(
                        f"Invalid embedding item format at index {i}: {type(item)}"
                    )
                    raise ValueError(
                        f"Embedding item at index {i} does not contain 'embedding' field"
                    )

            logger.info(f"RunPod batch job completed successfully: {job_id}")
            return embeddings

        except asyncio.TimeoutError:
            logger.error(
                f"RunPod batch request timed out after {timeout} seconds (attempt {attempt + 1}/{max_retries})"
            )
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"RunPod API request timed out after {timeout} seconds"
                )
            time.sleep(2**attempt)  # Exponential backoff

        except Exception as e:
            logger.error(
                f"RunPod API error (attempt {attempt + 1}/{max_retries}): {str(e)}"
            )
            if attempt == max_retries - 1:
                raise RuntimeError(f"RunPod API request failed: {str(e)}") from e
            time.sleep(2**attempt)  # Exponential backoff

    raise RuntimeError("All retry attempts failed")
