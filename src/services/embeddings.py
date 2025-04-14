"""
Embeddings utilities and custom embedding classes.

This module provides specialized embedding classes and related utilities
for working with various embedding models, particularly those requiring
remote API calls rather than local model loading.
"""
from typing import List
from langchain_core.embeddings import Embeddings

# Constants
NASA_MODEL = "nasa-impact/nasa-smd-ibm-v0.1"

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
    
    def embed_query(self, text: str) -> List[float]:
        """Placeholder that should never be called directly."""
        raise NotImplementedError(
            "This model requires using the RunPod API. Use runpod_api_request instead."
        )
        
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Placeholder that should never be called directly."""
        raise NotImplementedError(
            "This model requires using the RunPod API. Use runpod_api_request instead."
        )