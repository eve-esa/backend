"""
Utility functions for file handling and embeddings generation.

This module provides utility functions for handling file uploads,
getting embedding models based on model names, and making API requests
to RunPod for embeddings generation.
"""

import logging
import tempfile
from enum import Enum
from typing import Any, Optional, Tuple, Union, List, Dict

from fastapi import UploadFile

from langchain_core.embeddings import Embeddings, FakeEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import (
    HuggingFaceEmbeddings,
    DeepInfraEmbeddings,
)
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
import tiktoken

from src.utils.embeddings import RunPodEmbeddings
from src.config import (
    INFERENCE_API_KEY,
    MISTRAL_API_KEY,
    OPENAI_API_KEY,
    MONGO_HOST,
    MONGO_PORT,
    MONGO_USERNAME,
    MONGO_PASSWORD,
    MONGO_DATABASE,
    MONGO_PARAMS,
    DEEPINFRA_API_TOKEN,
)
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
    QWEN_3_4B = "Qwen/Qwen3-Embedding-4B"
    QWEN_3_4B_INFERENCE = "qwen/qwen3-embedding-4b"
    NASA = "nasa-impact/nasa-smd-ibm-st-v2"

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
        logger.info(f"Using DeepInfra for Qwen 3.4B embedding model: {model}")
        api_token = DEEPINFRA_API_TOKEN
        if not api_token:
            logger.warning("DEEPINFRA_API_TOKEN environment variable not set")
            return None

        embeddings = DeepInfraEmbeddings(
            model_id=model,
            deepinfra_api_token=api_token,
        )
        embeddings_size = 2560

    elif model == EmbeddingModelType.QWEN_3_4B_INFERENCE.value:
        logger.info(f"Using Inference for Qwen 3.4B embedding model: {model}")
        api_token = INFERENCE_API_KEY
        if not api_token:
            logger.warning("INFERENCE_API_KEY environment variable not set")
            return None
        embeddings = OpenAIEmbeddings(
            base_url="https://api.inference.net/v1",
            model=model,
            api_key=api_token,
        )
        embeddings_size = 2560

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

    elif model == EmbeddingModelType.NASA.value:
        logger.info(f"Using RunPod proxy for NASA embedding model: {model}")
        embeddings = RunPodEmbeddings(model_name=model, embedding_size=768)
        embeddings_size = embeddings.embedding_size

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


def _field(obj: Any, key: str, default: Any = None) -> Any:
    """Return value for key from dict-like or attribute-like object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _to_int(value: Any) -> Any:
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def _to_float(value: Any) -> Any:
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


def extract_year_range_from_filters(filters: Any) -> Optional[List[int]]:
    """Extract [start_year, end_year] from request.filters structure.

    Expected shape:
      {
        "must": [
          {"key": "year", "range": {"gte": <start>, "lte": <end>}},
          ...
        ]
      }
    Returns None if not found or values are invalid.
    """
    try:
        if not isinstance(filters, dict):
            return None
        conditions = filters.get("must") or []
        if not isinstance(conditions, list):
            return None
        for cond in conditions:
            if not isinstance(cond, dict):
                continue
            if cond.get("key") != "year":
                continue
            rng = cond.get("range") or {}
            if not isinstance(rng, dict):
                continue
            start = _to_int(rng.get("gte"))
            end = _to_int(rng.get("lte"))
            if start is None and end is None:
                return None
            if start is not None and end is not None:
                return [start, end]
            if start is not None:
                return [start, start]
            if end is not None:
                return [end, end]
        return None
    except Exception:
        return None


def extract_document_data(result: Any) -> Dict[str, Any]:
    result_id = _field(result, "id")
    result_version = _to_int(_field(result, "version"))
    result_score = _to_float(_field(result, "score") or _field(result, "distance"))
    result_payload = (
        _field(result, "payload", {}) or _field(result, "document", {}) or {}
    )
    collection_name = _field(result, "collection_name")
    if not collection_name and isinstance(result_payload, dict):
        collection_name = result_payload.get("collection_name") or (
            result_payload.get("metadata") or {}
        ).get("collection_name")
    # if result_payload has key "content" and doesn't have key "text", set "text" with "content"
    if "content" in result_payload and "text" not in result_payload:
        result_payload["text"] = result_payload["content"]
    result_text = _field(result, "text", "") or ""
    result_metadata = _field(result, "metadata", {}) or {}

    # Fallbacks from payload
    if not result_text and isinstance(result_payload, dict):
        result_text = result_payload.get("text", "") or ""
    if not result_metadata and isinstance(result_payload, dict):
        result_metadata = result_payload.get("metadata", {}) or {}

    return {
        "id": str(result_id) if result_id is not None else None,
        "version": result_version,
        "score": result_score,
        "collection_name": collection_name,
        "payload": result_payload,
        "text": result_text,
        "metadata": result_metadata,
    }


def str_token_counter(text: str) -> int:
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4


def tiktoken_counter(messages: List[BaseMessage]) -> int:
    """Custom token counter for messages using tiktoken."""
    num_tokens = 3  # every reply is primed with <|start|>assistant<|message|>
    tokens_per_message = 3
    tokens_per_name = 1
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, ToolMessage):
            role = "tool"
        elif isinstance(msg, SystemMessage):
            role = "system"
        else:
            raise ValueError(f"Unsupported messages type {msg.__class__}")
        num_tokens += (
            tokens_per_message
            + str_token_counter(role)
            + str_token_counter(msg.content)
        )
        if hasattr(msg, "name") and msg.name:
            num_tokens += tokens_per_name + str_token_counter(msg.name)
    return num_tokens


def trim_text_to_token_limit(text: str, max_tokens: int) -> str:
    """Trim text to be at most max_tokens using tiktoken, with a char fallback."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        toks = enc.encode(text)
        if len(toks) <= max_tokens:
            return text
        return enc.decode(toks[:max_tokens])
    except Exception:
        est_chars = max(0, max_tokens * 4)
        return text[:est_chars]


def get_mongodb_uri() -> str:
    """Build MongoDB URI from environment, appending MONGO_PARAMS if provided."""
    params = MONGO_PARAMS or ""
    if params and not params.startswith("?"):
        params = f"?{params}"
    return f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/{MONGO_DATABASE}{params}"
