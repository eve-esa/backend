from enum import Enum
from typing import Optional, Any, Dict, List
from pydantic import BaseModel, Field


class FeedbackEnum(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class MessageUpdate(BaseModel):
    was_copied: Optional[bool] = None
    feedback: Optional[FeedbackEnum] = None
    feedback_reason: Optional[str] = None


# -------- Response models for create_message endpoint --------


class DocumentReference(BaseModel):
    id: Optional[str] = Field(default=None, description="Document ID if available")
    version: Optional[int] = Field(
        default=None, description="Document version if provided"
    )
    score: Optional[float] = Field(
        default=None, description="Similarity or relevance score if present"
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict, description="Raw payload from vector store or MCP"
    )
    text: str = Field(
        default="", description="Extracted text content for this document"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Metadata associated with the document"
    )


class HallucinationLatencies(BaseModel):
    detection_latency: Optional[float] = None
    span_reprompting_latency: Optional[float] = None
    query_rewriting_latency: Optional[float] = None
    regeneration_latency: Optional[float] = None
    overall_latency: Optional[float] = None


class Latencies(BaseModel):
    # Vector store latencies
    query_embedding_latency: Optional[float] = Field(
        default=None, description="Time to embed the query"
    )
    qdrant_retrieval_latency: Optional[float] = Field(
        default=None, description="Time to retrieve documents from Qdrant"
    )

    # MCP latencies
    mcp_retrieval_latency: Optional[float] = Field(
        default=None, description="Time to retrieve documents via MCP"
    )

    reranking_latency: Optional[float] = Field(
        default=None, description="Time to re-rank MCP documents"
    )

    # Generation and total
    generation_latency: Optional[float] = Field(
        default=None, description="Time taken by the LLM to generate the answer"
    )
    hallucination_latency: Optional[HallucinationLatencies] = Field(
        default=None, description="Detailed latencies within hallucination loop"
    )
    total_latency: Optional[float] = Field(
        default=None, description="End-to-end latency for the request"
    )


class ResponseMetadata(BaseModel):
    latencies: Latencies


# Import detailed schemas from hallucination pipeline for loop_result typing
try:
    from src.hallucination_pipeline.schemas import (
        generation_schema,
        hallucination_schema,
        rewrite_schema,
        self_reflect_schema,
        ranking_schema,
    )
except Exception:  # If module is unavailable at import time, use loose typing fallbacks
    generation_schema = Dict[str, Any]
    hallucination_schema = Dict[str, Any]
    rewrite_schema = Dict[str, Any]
    self_reflect_schema = Dict[str, Any]
    ranking_schema = Dict[str, Any]


class LoopResult(BaseModel):
    final_answer: Optional[str] = None
    generation_response: Optional[generation_schema] = None
    hallucination_response: Optional[hallucination_schema] = None
    rewrite_response: Optional[rewrite_schema] = None
    reflected_response: Optional[self_reflect_schema] = None
    ranked_output: Optional[ranking_schema] = None
    docs: Optional[str] = None


class CreateMessageResponse(BaseModel):
    id: str
    query: str
    answer: str
    documents: List[DocumentReference]
    use_rag: bool
    conversation_id: str
    loop_result: Optional[LoopResult] = Field(
        default=None, description="Hallucination loop output when enabled"
    )
    metadata: ResponseMetadata
