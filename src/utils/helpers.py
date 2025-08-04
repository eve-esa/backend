"""
Utility functions for file handling and embeddings generation.

This module provides utility functions for handling file uploads,
getting embedding models based on model names, and making API requests
to RunPod for embeddings generation.
"""

import asyncio
import logging
import tempfile
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import runpod
from fastapi import UploadFile

from langchain_core.embeddings import Embeddings, FakeEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.utils.embeddings import RunPodEmbeddings
from src.config import MISTRAL_API_KEY, OPENAI_API_KEY
from src.constants import DEFAULT_EMBEDDING_MODEL

# Configure logging
logger = logging.getLogger(__name__)


class EmbeddingModelType(Enum):
    """Supported embedding model types."""

    FAKE = "fake"
    MISTRAL = "mistral-embed"
    OPENAI_ADA = "text-embedding-ada-002"
    OPENAI_SMALL = "text-embedding-3-small"
    OPENAI_LARGE = "text-embedding-3-large"
    NASA = DEFAULT_EMBEDDING_MODEL

    # Popular Hugging Face embedding models
    ALL_MINILM_L6_V2 = "sentence-transformers/all-MiniLM-L6-v2"
    ALL_MPNET_BASE_V2 = "sentence-transformers/all-mpnet-base-v2"
    E5_BASE_V2 = "intfloat/e5-base-v2"
    BGE_SMALL_EN_V1_5 = "BAAI/bge-small-en-v1.5"
    BGE_BASE_EN_V1_5 = "BAAI/bge-base-en-v1.5"


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
    if model == DEFAULT_EMBEDDING_MODEL:
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

        try:
            # Load Hugging Face embeddings model
            embeddings = HuggingFaceEmbeddings(
                model_name=model,
                model_kwargs={"device": "cpu"},  # Use CPU by default
                encode_kwargs={"normalize_embeddings": True},
            )

            # Get embedding size from the model
            try:
                # Try to get the embedding dimension from the model
                embeddings_size = embeddings.client.get_sentence_embedding_dimension()
            except AttributeError:
                # Fallback if client structure is different
                try:
                    embeddings_size = embeddings._client[1].word_embedding_dimension
                except (AttributeError, IndexError):
                    logger.warning(
                        f"Could not determine embedding size for {model}, using default"
                    )
                    embeddings_size = 768

        except Exception as e:
            logger.warning(f"Failed to load Hugging Face model {model}: {str(e)}")
            logger.info(f"Falling back to RunPod embeddings for model: {model}")
            embeddings = RunPodEmbeddings(model_name=model, embedding_size=768)
            embeddings_size = 768

    if return_embeddings_size:
        return embeddings, embeddings_size
    return embeddings
