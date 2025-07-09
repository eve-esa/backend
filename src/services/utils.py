"""
Utility functions for file handling and embeddings generation.

This module provides utility functions for handling file uploads,
getting embedding models based on model names, and making API requests
to RunPod for embeddings generation.
"""

import asyncio
import hashlib
import logging
import tempfile
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import runpod
from fastapi import UploadFile

from langchain_core.embeddings import Embeddings, FakeEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from langchain_openai import OpenAIEmbeddings

from src.services.embeddings import RunPodEmbeddings, NASA_MODEL
from src.config import MISTRAL_API_KEY, OPENAI_API_KEY, RUNPOD_API_KEY


# Configure logging
logger = logging.getLogger(__name__)

# Configure RunPod API key
runpod.api_key = RUNPOD_API_KEY

# Constants
NASA_MODEL = "nasa-impact/nasa-smd-ibm-st-v2"


class EmbeddingModelType(Enum):
    """Supported embedding model types."""

    FAKE = "fake"
    MISTRAL = "mistral-embed"
    OPENAI_ADA = "text-embedding-ada-002"
    OPENAI_SMALL = "text-embedding-3-small"
    OPENAI_LARGE = "text-embedding-3-large"
    NASA = NASA_MODEL


# OpenAI embedding model dimensions
OPENAI_EMBEDDING_DIMENSIONS = {
    EmbeddingModelType.OPENAI_ADA.value: 1536,
    EmbeddingModelType.OPENAI_SMALL.value: 1536,
    EmbeddingModelType.OPENAI_LARGE.value: 3072,
}


async def save_upload_file_to_temp(upload_file: UploadFile) -> str:
    """
    Save an uploaded file to a temporary file.

    Args:
        upload_file: The uploaded file object from FastAPI

    Returns:
        str: The path to the temporary file

    Raises:
        IOError: If there's an error reading or writing the file
    """
    try:
        # Use suffix based on original file extension if possible
        original_filename = upload_file.filename or ""
        suffix = (
            f".{original_filename.split('.')[-1]}"
            if "." in original_filename
            else ".pdf"
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            contents = await upload_file.read()
            temp_file.write(contents)
            temp_file.flush()

            file_path = temp_file.name
            logger.info(f"Saved uploaded file to temporary path: {file_path}")
            return file_path

    except Exception as e:
        logger.error(f"Failed to save uploaded file: {str(e)}")
        raise IOError(f"Failed to save uploaded file: {str(e)}") from e


def get_embeddings_model(
    model_name: str, return_embeddings_size: bool = False
) -> Union[Embeddings, Tuple[Embeddings, int]]:
    """
    Get an embeddings model based on the model name.

    Args:
        model: Name of the embedding model to use
        return_embeddings_size: Whether to also return the embedding dimension

    Returns:
        Union[Embeddings, Tuple[Embeddings, int]]:
            The embeddings model and optionally its dimension

    Raises:
        ValueError: If an unsupported model is specified
    """
    # Handle NASA model specially to prevent local loading
    model = model_name
    if model == NASA_MODEL:
        logger.info(f"Using RunPod proxy for NASA embedding model: {model}")
        embeddings = RunPodEmbeddings(model_name=model, embedding_size=768)
        embeddings_size = embeddings.embedding_size

    # Handle fake embeddings (for testing)
    elif model == EmbeddingModelType.FAKE.value:
        logger.info("Using fake embeddings for testing")
        embeddings = FakeEmbeddings(size=4096)
        embeddings_size = 4096

    # Handle Mistral embeddings
    elif model == EmbeddingModelType.MISTRAL.value:
        logger.info("Using Mistral embeddings")
        embeddings = MistralAIEmbeddings(model=model, api_key=MISTRAL_API_KEY)
        embeddings_size = 1024

    # Handle OpenAI embeddings
    elif model in OPENAI_EMBEDDING_DIMENSIONS:
        logger.info(f"Using OpenAI embeddings model: {model}")
        embeddings = OpenAIEmbeddings(model=model, api_key=OPENAI_API_KEY)
        embeddings_size = OPENAI_EMBEDDING_DIMENSIONS[model]

    # Handle Hugging Face embeddings (default case)
    else:
        logger.info(f"Using Hugging Face embeddings model: {model}")

        # TODO - do not load the model into memory get it from the config file
        # embeddings = HuggingFaceEmbeddings(model_name=model)

        try:
            # Get embedding size from the model
            # embeddings_size = embeddings.client.get_sentence_embedding_dimension()
            pass
        except AttributeError:
            # Fallback if client structure is different
            try:
                embeddings_size = embeddings._client[1].word_embedding_dimension
            except (AttributeError, IndexError):
                logger.warning(
                    f"Could not determine embedding size for {model}, using default"
                )
                embeddings_size = 768

    if return_embeddings_size:
        return embeddings, embeddings_size
    return embeddings


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


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()
