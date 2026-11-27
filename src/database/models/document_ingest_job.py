from datetime import datetime
from typing import ClassVar, List, Optional

from pydantic import BaseModel, Field

from src.database.mongo_model import MongoModel

JOB_QUEUED = "queued"
JOB_RUNNING = "running"
JOB_DONE = "done"
JOB_FAILED = "failed"
# The files are with document-parser; results arrive on parse-results.
JOB_PARSING = "parsing"

PARSE_QUEUED = "queued"
PARSE_PROCESSING = "processing"
PARSE_DONE = "done"
PARSE_FAILED = "failed"


class IngestJobFile(BaseModel):
    document_id: str
    filename: str
    storage_key: str
    source_url: str = ""


class DocumentIngestJob(MongoModel):
    """One upload request waiting for, or going through, parsing and embedding."""

    user_id: str
    collection_id: str
    files: List[IngestJobFile]
    embeddings_model: str
    chunk_size: int
    chunk_overlap: int
    max_chunks: Optional[int] = None

    status: str = JOB_QUEUED
    attempts: int = 0
    available_at: datetime = Field(..., description="Not claimed before this time")
    lease_expires_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    # Stable and safe to show: never an exception's text.
    error_code: Optional[str] = None
    chunk_counts: dict = Field(default_factory=dict)
    # document_id -> queued / processing / done / failed, and the error
    # code of each failed one.
    parse_state: dict = Field(default_factory=dict)
    parse_errors: dict = Field(default_factory=dict)

    collection_name: ClassVar[str] = "document_ingest_jobs"
