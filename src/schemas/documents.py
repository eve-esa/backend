from typing import List, Optional
from pydantic import BaseModel

# Constants
DEFAULT_QUERY = "What is ESA?"
DEFAULT_COLLECTION = "esa-nasa-workshop"
DEFAULT_EMBEDDING_MODEL = "nasa-impact/nasa-smd-ibm-st-v2"
DEFAULT_SCORE_THRESHOLD = 0.7
DEFAULT_K = 3
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 0

class RetrieveRequest(BaseModel):
    query: str = DEFAULT_QUERY
    embeddings_model: str = DEFAULT_EMBEDDING_MODEL
    score_threshold: float = DEFAULT_SCORE_THRESHOLD
    k: int = DEFAULT_K

class DeleteRequest(BaseModel):
    embeddings_model: str = DEFAULT_EMBEDDING_MODEL
    document_list: List[str] = []

class AddDocumentRequest(BaseModel):
    embeddings_model: str = DEFAULT_EMBEDDING_MODEL
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    metadata_urls: Optional[List[str]] = None
    metadata_names: Optional[List[str]] = None

class UpdateDocumentRequest(BaseModel):
    embeddings_model: str = DEFAULT_EMBEDDING_MODEL
    source_name: str
    new_metadata: Optional[dict] = None