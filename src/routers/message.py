import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.config import IS_PROD, MODEL_TIMEOUT
from src.constants import (
    PUBLIC_COLLECTIONS,
    STAGING_PUBLIC_COLLECTIONS,
    WILEY_PUBLIC_COLLECTIONS,
)
from src.core.llm_manager import LLMType
from src.database.models.co2eq_comparison import CO2EQComparison
from src.database.models.collection import Collection as CollectionModel
from src.database.models.conversation import Conversation
from src.database.models.mcp_server import MCPServer
from src.database.models.message import Message
from src.database.models.user import User
from src.middlewares.auth import get_current_user
from src.schemas.co2 import CO2EquivalenceComparison, CO2EquivalenceResult
from src.schemas.generation_request import GenerationRequest
from src.schemas.message import CreateMessageResponse, MessageUpdate
from src.services.cancel_manager import get_cancel_manager
from src.services.generate_answer import (
    generate_answer,
    get_shared_llm_manager,
    maybe_rollup_and_trim_history,
    run_generation_to_bus,
    setup_rag_and_context,
    should_use_rag,
)
from src.services.generate_answer_agentic import (
    generate_answer_agentic,
    run_agentic_generation_to_bus,
)
from src.services.hallucination_detector import HallucinationDetector
from src.services.stream_bus import get_stream_bus
from src.services.llm_inference import invoke_llm_and_consume_tokens
from src.services.token_rate_limiter import (
    consume_tokens_for_user,
    count_tokens_for_texts,
    enforce_token_budget_or_raise,
)
from src.utils.error_logger import (
    Component,
    PipelineStage,
    get_error_logger,
    set_conversation_context,
    set_message_context,
    set_user_context,
)
from src.utils.helpers import (
    build_context,
    extract_document_data,
    extract_year_range_from_filters,
    get_co2_usage_kg,
    pluralize,
)

logger = logging.getLogger(__name__)

router = APIRouter()


async def get_lower_bound(usage_kg: float) -> Optional[CO2EquivalenceComparison]:
    """Find the largest co2eq_kg <= usage_kg."""
    try:
        col = CO2EQComparison.get_collection()

        query = {"enabled": True, "co2eq_kg": {"$lte": usage_kg}}
        doc = await col.find_one(query, sort=[("co2eq_kg", -1)], projection={"_id": 0})

        if doc:
            return CO2EquivalenceComparison(**doc)
        return None
    except Exception as e:
        logger.error(f"Error in get_lower_bound: {str(e)}", exc_info=True)
        return None


async def closest_with_direction(usage_kg: float) -> Optional[CO2EquivalenceComparison]:
    """
    Returns either lower bound or upper bound if lower is missing (usage smaller than min >0).
    """
    try:
        lower = await get_lower_bound(usage_kg)
        if lower and lower.co2eq_kg > 0:
            return lower

        col = CO2EQComparison.get_collection()
        upper_doc = await col.find_one(
            {"enabled": True, "co2eq_kg": {"$gt": usage_kg}},
            sort=[("co2eq_kg", 1)],
            projection={"_id": 0},
        )
        if upper_doc:
            return CO2EquivalenceComparison(**upper_doc)
        return None
    except Exception as e:
        logger.error(f"Error in closest_with_direction: {str(e)}", exc_info=True)
        return None


async def build_equivalence_sentence(usage_kg: float) -> CO2EquivalenceResult:
    """Build an equivalence sentence for CO2 usage."""
    doc = await closest_with_direction(usage_kg)
    if not doc:
        raise RuntimeError("No comparison items in DB")
    base = doc.co2eq_kg
    title = doc.title

    if base == 0:
        text = ""
        return CO2EquivalenceResult(
            title=title, co2eq_kg=base, equivalent_count=None, text=text
        )

    count = max(1, int(round(usage_kg / base)))
    unit = pluralize(count, doc.unit_singular, doc.unit_plural)

    text = f"This is equivalent to: {count} {unit}"

    return CO2EquivalenceResult(
        title=title, co2eq_kg=base, equivalent_count=count, text=text
    )


class SourceLogsRequest(BaseModel):
    source_id: str = Field(default=None, description="Source ID")
    source_url: str = Field(default=None, description="Source URL")
    source_title: str = Field(default=None, description="Source title")
    source_collection_name: str = Field(
        default=None, description="Source collection name"
    )


class HallucinationDetectResponse(BaseModel):
    label: int
    reason: str
    original_question: str
    rewritten_question: Optional[str] = None
    final_answer: Optional[str] = None
    latencies: Optional[Dict[str, Optional[float]]] = None


class GenerateLLMRequest(BaseModel):
    """Request body for LLM-only generation (EVE-Instruct v5, no RAG, no conversation)."""

    query: str = Field(..., description="User prompt to send to the LLM")


@router.get("/conversations/messages/average-latencies")
async def get_average_latencies(
    start_date: datetime | None = None, end_date: datetime | None = None
) -> dict:
    """
    Return average latencies aggregated across all messages.

    Optionally filters the aggregation by a timestamp window.

    Args:
        start_date (datetime | None): Optional start of the time window (inclusive).
        end_date (datetime | None): Optional end of the time window (inclusive).

    Returns:
        Mapping of latency metric name to average value.

    Raises:
        HTTPException: 500 for server errors during aggregation.
    """
    try:
        messages_col = Message.get_collection()
        pipeline = []
        if start_date is not None or end_date is not None:
            time_filter = {}
            if start_date is not None:
                time_filter["$gte"] = start_date
            if end_date is not None:
                time_filter["$lte"] = end_date
            pipeline.append({"$match": {"timestamp": time_filter}})
        pipeline.append(
            {
                "$group": {
                    "_id": None,
                    "rag_decision_latency": {
                        "$avg": "$metadata.latencies.rag_decision_latency"
                    },
                    "query_embedding_latency": {
                        "$avg": "$metadata.latencies.query_embedding_latency"
                    },
                    "qdrant_retrieval_latency": {
                        "$avg": "$metadata.latencies.qdrant_retrieval_latency"
                    },
                    "mcp_retrieval_latency": {
                        "$avg": "$metadata.latencies.mcp_retrieval_latency"
                    },
                    "reranking_latency": {
                        "$avg": "$metadata.latencies.reranking_latency"
                    },
                    "first_token_latency": {
                        "$avg": "$metadata.latencies.first_token_latency"
                    },
                    "mistral_first_token_latency": {
                        "$avg": "$metadata.latencies.mistral_first_token_latency"
                    },
                    "base_generation_latency": {
                        "$avg": "$metadata.latencies.base_generation_latency"
                    },
                }
            }
        )
        cursor = messages_col.aggregate(pipeline, allowDiskUse=True)
        results = await cursor.to_list(length=1)
        return results[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.get("/conversations/messages/me/stats")
async def get_my_message_stats(
    requesting_user: User = Depends(get_current_user),
) -> dict:
    """
    Return counts and character totals for the current user's messages.

    Aggregates across all messages belonging to conversations owned by the user.

    Args:
        requesting_user (User): Authenticated user injected by dependency.

    Returns:
        Aggregated stats including counts and character sums.

    Raises:
        HTTPException: 500 for server errors during aggregation.
    """
    try:
        messages_col = Message.get_collection()

        # Fetch conversation IDs owned by the current user (avoid $lookup pipelines unsupported by DocumentDB)
        user_conversations = await Conversation.find_all(
            filter_dict={"user_id": requesting_user.id}
        )
        conversation_ids = [c.id for c in user_conversations if getattr(c, "id", None)]

        if not conversation_ids:
            return {
                "message_count": 0,
                "input_characters": 0,
                "output_characters": 0,
                "total_characters": 0,
                "co2eq_kg": 0.0,
                "text": "",
            }

        pipeline = [
            {"$match": {"conversation_id": {"$in": conversation_ids}}},
            {
                "$group": {
                    "_id": None,
                    "message_count": {"$sum": 1},
                    "input_characters": {
                        "$sum": {
                            "$strLenCP": {
                                "$ifNull": [
                                    "$metadata.prompts.generation_prompt",
                                    "",
                                ]
                            }
                        }
                    },
                    "output_characters": {
                        "$sum": {"$strLenCP": {"$ifNull": ["$output", ""]}}
                    },
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "message_count": 1,
                    "input_characters": 1,
                    "output_characters": 1,
                    "total_characters": {
                        "$add": ["$input_characters", "$output_characters"],
                    },
                }
            },
        ]

        cursor = messages_col.aggregate(pipeline, allowDiskUse=True)
        results = await cursor.to_list(length=1)

        if results:
            stats = results[0]
        else:
            stats = {
                "message_count": 0,
                "input_characters": 0,
                "output_characters": 0,
                "total_characters": 0,
            }

        total_chars = stats.get("total_characters", 0)
        usage_kg = get_co2_usage_kg(total_chars=total_chars)

        # Get CO2 equivalence data
        text = ""
        try:
            equivalence_data = await build_equivalence_sentence(usage_kg)
            text = equivalence_data.text
        except Exception as e:
            logger.error(f"Failed to get CO2 equivalence data: {str(e)}", exc_info=True)

        # Add CO2 data to response
        stats["co2eq_kg"] = usage_kg
        stats["text"] = text

        return stats
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.post(
    "/conversations/{conversation_id}/messages", response_model=CreateMessageResponse
)
async def create_message(
    request: GenerationRequest,
    conversation_id: str,
    background_tasks: BackgroundTasks,
    requesting_user: User = Depends(get_current_user),
) -> CreateMessageResponse:
    """
    Create a new message in a conversation and generate an answer.

    Validates conversation ownership, normalizes requested public collections, persists a placeholder `Message`, runs generation, updates the message with answer and retrieval metadata, and schedules rollup/trimming of history.

    Args:
        request (GenerationRequest): Generation parameters including query, collections, and model settings.
        conversation_id (str): Target conversation identifier.
        background_tasks (BackgroundTasks): Background task runner used to schedule rollups.
        requesting_user (User): Authenticated user injected by dependency.

    Returns:
        Message id, query, answer, documents, flags, and metadata.

    Raises:
        HTTPException: 404 if conversation is not found; 403 if ownership/collections invalid; 500 for server errors.
    """
    set_user_context(requesting_user.id)
    set_conversation_context(conversation_id)

    message = None
    try:
        conversation = await Conversation.find_by_id(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        if conversation.user_id != requesting_user.id:
            raise HTTPException(
                status_code=403,
                detail="You are not allowed to add a message to this conversation",
            )
        await enforce_token_budget_or_raise(requesting_user)
        original_query = request.query

        # Normalize and validate requested public collections against allowed lists
        allowed_source = PUBLIC_COLLECTIONS if IS_PROD else STAGING_PUBLIC_COLLECTIONS
        try:
            allowed_names = {
                item.get("name")
                for item in (allowed_source + WILEY_PUBLIC_COLLECTIONS)
                if isinstance(item, dict) and item.get("name")
            }
        except Exception:
            allowed_names = set()

        public_collections = [
            n for n in request.public_collections if n in allowed_names
        ]
        request.public_collections = public_collections

        # lookup query to check if some of the collection ids from other users are in the request.collection_ids
        other_users_collections = await CollectionModel.find_all(
            filter_dict={
                "id": {"$in": request.public_collections},
                "user_id": {"$ne": requesting_user.id},
            }
        )

        if len(other_users_collections) > 0:
            raise HTTPException(
                status_code=403,
                detail="You are not allowed to use collections from other users",
            )

        request.collection_ids = request.collection_ids + request.public_collections

        # All user collections are used by default
        user_collections = await CollectionModel.find_all(
            filter_dict={"user_id": requesting_user.id}
        )

        request.private_collections_map = {c.id: c.name for c in user_collections}
        if len(user_collections) > 0:
            request.collection_ids = request.collection_ids + [
                c.id for c in user_collections
            ]
        # remove "Wiley AI Gateway" from collection_ids
        request.collection_ids = [
            c for c in request.collection_ids if c != "Wiley AI Gateway"
        ]
        logger.info(f"Collection IDs: {request.collection_ids}")

        # Extract year range from filters for MCP usage
        try:
            request.year = extract_year_range_from_filters(request.filters)
        except Exception:
            request.year = None

        message = await Message.create(
            conversation_id=conversation_id,
            input=request.query,
            output="",
            documents=[],
            use_rag=False,
            request_input=request,
            metadata={},
        )

        set_message_context(message.id)

        (
            answer,
            results,
            is_rag,
            latencies,
            prompts,
            retrieved_docs,
        ) = await generate_answer(
            request, conversation_id=conversation_id, user_id=requesting_user.id
        )

        documents_data = []
        if results:
            documents_data = [extract_document_data(result) for result in results]

        message.output = answer
        message.documents = documents_data
        message.use_rag = is_rag
        existing_metadata = dict(getattr(message, "metadata", {}) or {})
        existing_metadata.update(
            {
                "latencies": latencies,
                "prompts": prompts,
                "retrieved_docs": retrieved_docs,
            }
        )
        message.metadata = existing_metadata
        await message.save()
        await consume_tokens_for_user(
            requesting_user, count_tokens_for_texts(original_query, answer)
        )

        # Schedule rollup as background task to avoid blocking response
        background_tasks.add_task(maybe_rollup_and_trim_history, conversation_id)

        return {
            "id": message.id,
            "query": request.query,
            "answer": answer,
            "documents": documents_data,
            "use_rag": is_rag,
            "conversation_id": conversation_id,
            "collection_ids": request.collection_ids,
            "metadata": {
                "latencies": latencies,
            },
        }
    except HTTPException as http_exc:
        if message:
            try:
                existing_metadata = dict(getattr(message, "metadata", {}) or {})
                existing_metadata["error"] = str(getattr(http_exc, "detail", http_exc))
                message.metadata = existing_metadata
                await message.save()
            except Exception:
                pass
        raise
    except Exception as e:
        if message:
            try:
                existing_metadata = dict(getattr(message, "metadata", {}) or {})
                existing_metadata["error"] = str(e)
                message.metadata = existing_metadata
                await message.save()
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.post("/conversations/{conversation_id}/messages/{message_id}/retry")
async def retry(
    conversation_id: str,
    message_id: str,
    background_tasks: BackgroundTasks,
    requesting_user: User = Depends(get_current_user),
) -> dict:
    """
    Retry generation for an existing message.

    Re-validates conversation ownership and message relationship, reuses the original `request_input` stored on the message, regenerates the answer, and updates message content, documents, and metadata.

    Args:
        conversation_id (str): Conversation identifier.
        message_id (str): Message identifier to retry.
        background_tasks (BackgroundTasks): Background task runner used to schedule rollups.
        requesting_user (User): Authenticated user injected by dependency.

    Returns:
        Response payload mirroring create_message with updated answer and metadata.

    Raises:
        HTTPException: 404 if conversation/message not found; 403 if ownership invalid; 400 if message cannot be retried; 500 for server errors.
    """
    try:
        conversation = await Conversation.find_by_id(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        if conversation.user_id != requesting_user.id:
            raise HTTPException(
                status_code=403,
                detail="You are not allowed to add a message to this conversation",
            )

        message = await Message.find_by_id(message_id)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")

        if message.conversation_id != conversation_id:
            raise HTTPException(
                status_code=404, detail="Message not found in this conversation"
            )

        if not message.request_input:
            raise HTTPException(
                status_code=400,
                detail="This message cannot be retried",
            )
        await enforce_token_budget_or_raise(requesting_user)

        (
            answer,
            results,
            is_rag,
            latencies,
            prompts,
            retrieved_docs,
        ) = await generate_answer(
            message.request_input, conversation_id=conversation_id
        )

        documents_data = []
        if results:
            documents_data = [extract_document_data(result) for result in results]

        message.output = answer
        message.documents = documents_data
        message.use_rag = is_rag
        existing_metadata = dict(getattr(message, "metadata", {}) or {})
        existing_metadata.update(
            {
                "latencies": latencies,
                "prompts": prompts,
                "retrieved_docs": retrieved_docs,
            }
        )
        message.metadata = existing_metadata
        await message.save()
        await consume_tokens_for_user(
            requesting_user, count_tokens_for_texts(message.input, answer)
        )

        # Schedule rollup as background task to avoid blocking response
        background_tasks.add_task(maybe_rollup_and_trim_history, conversation_id)

        return {
            "id": message.id,
            "query": message.input,
            "answer": answer,
            "documents": documents_data,
            "use_rag": is_rag,
            "conversation_id": conversation_id,
            "collection_ids": message.request_input.collection_ids,
            "metadata": {
                "latencies": latencies,
                "prompts": prompts,
                "retrieved_docs": retrieved_docs,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.patch("/conversations/{conversation_id}/messages/{message_id}")
async def update_message(
    conversation_id: str,
    message_id: str,
    request: MessageUpdate,
    requesting_user: User = Depends(get_current_user),
) -> dict:
    """
    Update message feedback and related annotations.

    Supports updating fields such as `feedback`, `feedback_reason`, `was_copied`, and hallucination feedback metadata on the target message.

    Args:
        conversation_id (str): Conversation identifier.
        message_id (str): Message identifier to update.
        request (MessageUpdate): Partial update payload for feedback fields.
        requesting_user (User): Authenticated user injected by dependency.

    Returns:
        Success message upon update.

    Raises:
        HTTPException: 404 if conversation/message not found or mismatched; 403 if ownership invalid; 500 for server errors.
    """
    try:
        conversation = await Conversation.find_by_id(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        message = await Message.find_by_id(message_id)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")

        if message.conversation_id != conversation_id:
            raise HTTPException(
                status_code=404, detail="Message not found in this conversation"
            )

        if conversation.user_id != requesting_user.id:
            raise HTTPException(
                status_code=403,
                detail="You are not allowed to update feedback for this message",
            )

        if request.feedback is not None:
            message.feedback = request.feedback.value

        if request.was_copied is not None:
            message.was_copied = request.was_copied

        if request.feedback_reason is not None:
            message.feedback_reason = request.feedback_reason

        if request.hallucination_feedback is not None:
            if message.hallucination is None:
                message.hallucination = {}
            message.hallucination["feedback"] = request.hallucination_feedback.value

        if request.hallucination_feedback_reason is not None:
            if message.hallucination is None:
                message.hallucination = {}
            message.hallucination["feedback_reason"] = (
                request.hallucination_feedback_reason
            )

        if request.hallucination_was_copied is not None:
            if message.hallucination is None:
                message.hallucination = {}
            message.hallucination["was_copied"] = request.hallucination_was_copied

        await message.save()

        return {"message": "Feedback updated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.post(
    "/conversations/{conversation_id}/stream_messages",
    response_class=StreamingResponse,
)
async def create_message_stream(
    request: GenerationRequest,
    conversation_id: str,
    background_tasks: BackgroundTasks,
    requesting_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """
    Create a new message and stream generation via Server-Sent Events (SSE).

    Sets up a per-message stream bus and runs generation in a decoupled task. Yields SSE-formatted chunks including status updates, tokens, and final payloads.

    Args:
        request (GenerationRequest): Generation parameters including query, collections, and model settings.
        conversation_id (str): Target conversation identifier.
        background_tasks (BackgroundTasks): Background task runner used to schedule rollups.
        requesting_user (User): Authenticated user injected by dependency.

    Returns:
        SSE stream for the generation lifecycle.

    Raises:
        HTTPException: 404 if conversation is not found; 403 if ownership/collections invalid; 500 for server errors.
    """
    set_user_context(requesting_user.id)
    set_conversation_context(conversation_id)

    message = None
    try:
        conversation = await Conversation.find_by_id(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        if conversation.user_id != requesting_user.id:
            raise HTTPException(
                status_code=403,
                detail="You are not allowed to add a message to this conversation",
            )
        await enforce_token_budget_or_raise(requesting_user)

        # Normalize and validate requested public collections against allowed lists
        allowed_source = PUBLIC_COLLECTIONS if IS_PROD else STAGING_PUBLIC_COLLECTIONS
        try:
            allowed_names = {
                item.get("name")
                for item in (allowed_source + WILEY_PUBLIC_COLLECTIONS)
                if isinstance(item, dict) and item.get("name")
            }
        except Exception:
            allowed_names = set()

        public_collections = [
            n for n in request.public_collections if n in allowed_names
        ]
        request.public_collections = public_collections

        # lookup query to check if some of the collection ids from other users are in the request.collection_ids
        other_users_collections = await CollectionModel.find_all(
            filter_dict={
                "id": {"$in": request.public_collections},
                "user_id": {"$ne": requesting_user.id},
            }
        )

        if len(other_users_collections) > 0:
            raise HTTPException(
                status_code=403,
                detail="You are not allowed to use collections from other users",
            )

        request.collection_ids = request.collection_ids + request.public_collections

        # All user collections are used by default
        user_collections = await CollectionModel.find_all(
            filter_dict={"user_id": requesting_user.id}
        )

        request.private_collections_map = {c.id: c.name for c in user_collections}
        if len(user_collections) > 0:
            request.collection_ids = request.collection_ids + [
                c.id for c in user_collections
            ]
        # remove "Wiley AI Gateway" from collection_ids
        request.collection_ids = [
            c for c in request.collection_ids if c != "Wiley AI Gateway"
        ]
        logger.info(f"Collection IDs: {request.collection_ids}")

        # Extract year range from filters for MCP usage
        try:
            request.year = extract_year_range_from_filters(request.filters)
        except Exception:
            request.year = None

        message = await Message.create(
            conversation_id=conversation_id,
            input=request.query,
            output="",
            documents=[],
            use_rag=False,
            request_input=request,
            metadata={},
        )

        set_message_context(message.id)

        # Start decoupled background job that publishes to bus
        cancel_mgr = get_cancel_manager()
        cancel_event = cancel_mgr.create(message.id)
        cancel_mgr.link_conversation(conversation_id, message.id)
        gen_task = asyncio.create_task(
            run_generation_to_bus(
                request=request,
                conversation_id=conversation_id,
                message_id=message.id,
                background_tasks=background_tasks,
                cancel_event=cancel_event,
                user_id=requesting_user.id,
            )
        )
        cancel_mgr.set_task(message.id, gen_task)

        bus = get_stream_bus()

        async def _gen():
            # Optional catch-up from currently saved output (usually empty right after create)
            try:
                if message.output:
                    yield f"data: {json.dumps({'type': 'partial', 'content': message.output})}\n\n"
            except Exception:
                pass
            async for data in bus.subscribe(message.id):
                yield data

        response = StreamingResponse(_gen(), media_type="text/event-stream")
        # Set SSE-friendly headers to prevent proxy/client reconnect loops
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Connection"] = "keep-alive"
        response.headers["X-Accel-Buffering"] = "no"  # Nginx buffering off if present
        return response
    except HTTPException as http_exc:
        if message:
            error_logger = get_error_logger()
            await error_logger.log_error_sync(
                error=http_exc,
                component=Component.ROUTER,
                pipeline_stage=PipelineStage.ROUTER,
                description="HTTPException in create_message_stream",
                error_type=type(http_exc).__name__,
            )
        raise http_exc
    except Exception as e:
        if message:
            error_logger = get_error_logger()
            await error_logger.log_error_sync(
                error=e,
                component=Component.ROUTER,
                pipeline_stage=PipelineStage.ROUTER,
                description="Exception in create_message_stream",
                error_type=type(e).__name__,
            )
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.post("/conversations/{conversation_id}/stop")
async def stop_conversation(
    conversation_id: str,
    requesting_user: User = Depends(get_current_user),
) -> dict:
    """
    Signal cancellation for the active generation within a conversation.

    Uses the cancel manager to locate the in-flight message/task and requests cooperative cancellation, also notifying downstream subscribers via the stream bus.

    Args:
        conversation_id (str): Conversation identifier to stop generation for.
        requesting_user (User): Authenticated user injected by dependency.

    Returns:
        Status payload indicating stop state or absence of active generation.

    Raises:
        HTTPException: 404 if conversation is not found; 403 if ownership invalid; 500 for server errors.
    """
    try:
        logger.info(
            "generation.stop.requested user_id=%s conversation_id=%s",
            requesting_user.id,
            conversation_id,
        )
        conversation = await Conversation.find_by_id(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        if conversation.user_id != requesting_user.id:
            raise HTTPException(
                status_code=403,
                detail="You are not allowed to access this conversation",
            )

        cancel_mgr = get_cancel_manager()
        # Prefer async lookup to support Redis-backed mapping across workers
        try:
            message_id = await cancel_mgr.get_message_for_conversation_async(
                conversation_id
            )  # type: ignore
        except Exception:
            message_id = cancel_mgr.get_message_for_conversation(conversation_id)
        if not message_id:
            # Nothing active to stop; respond success for idempotency
            logger.info(
                "generation.stop.no_active user_id=%s conversation_id=%s",
                requesting_user.id,
                conversation_id,
            )
            return {"status": "no_active_generation"}

        cancel_mgr.cancel(message_id)
        try:
            bus = get_stream_bus()
            await bus.publish(
                message_id, f"data: {json.dumps({'type': 'stopped'})}\n\n"
            )
            await bus.close(message_id)
            logger.info(
                "generation.stop.signaled user_id=%s conversation_id=%s message_id=%s",
                requesting_user.id,
                conversation_id,
                message_id,
            )
        except Exception as e:
            logger.warning(
                "generation.stop.signal_failed conversation_id=%s message_id=%s err=%s",
                conversation_id,
                message_id,
                str(e),
            )
        return {"status": "stopping", "message_id": message_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.post("/conversations/{conversation_id}/messages/{message_id}/source_logs")
async def get_source_logs(
    conversation_id: str,
    message_id: str,
    request: SourceLogsRequest,
    requesting_user: User = Depends(get_current_user),
) -> dict:
    """
    Append a source log entry to a message's metadata.

    Stores user-attributed source inspection information such as id, url, title, and collection name, with a server-side timestamp.

    Args:
        conversation_id (str): Conversation identifier.
        message_id (str): Message identifier.
        request (SourceLogsRequest): Source log details to append.
        requesting_user (User): Authenticated user injected by dependency.

    Returns:
        Confirmation message upon successful append.

    Raises:
        HTTPException: 404 if conversation/message not found or mismatched; 500 for server errors.
    """
    try:
        conversation = await Conversation.find_by_id(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        message = await Message.find_by_id(message_id)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")

        if message.conversation_id != conversation_id:
            raise HTTPException(
                status_code=404, detail="Message not found in this conversation"
            )

        # store source logs as an array and append each new entry
        existing_metadata = dict(getattr(message, "metadata", {}) or {})
        source_logs = list(existing_metadata.get("source_logs") or [])
        source_logs.append(
            {
                **request.model_dump(),
                "timestamp": datetime.now().isoformat(),
                "user_id": requesting_user.id,
            }
        )
        existing_metadata["source_logs"] = source_logs
        message.metadata = existing_metadata
        await message.save()
        return {"message": "Source logs stored successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.post(
    "/conversations/{conversation_id}/messages/{message_id}/hallucination",
    response_model=HallucinationDetectResponse,
)
async def hallucination_detect(
    conversation_id: str,
    message_id: str,
    requesting_user: User = Depends(get_current_user),
) -> HallucinationDetectResponse:
    """
    Detect and persist hallucination analysis for a message.

    Runs a multi-step pipeline (detect, optionally rewrite, retrieve, answer) and stores the result and latency breakdown on the message metadata.

    Args:
        conversation_id (str): Conversation identifier.
        message_id (str): Message identifier to analyze.
        requesting_user (User): Authenticated user injected by dependency.

    Returns:
        Structured hallucination analysis with optional final answer.

    Raises:
        HTTPException: 404 if conversation/message not found or mismatched; 403 if ownership invalid; 500 for server errors.
    """
    # Validate conversation ownership and message relationship
    conversation = await Conversation.find_by_id(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if conversation.user_id != requesting_user.id:
        raise HTTPException(
            status_code=403,
            detail="You are not allowed to access this conversation",
        )

    message = await Message.find_by_id(message_id)
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    if message.conversation_id != conversation_id:
        raise HTTPException(
            status_code=400, detail="Message does not belong to this conversation"
        )

    try:
        total_start = time.perf_counter()
        detector = HallucinationDetector()

        (
            label,
            reason,
            _orig_q,
            rewritten_question,
            final_answer,
            latencies,
        ) = await detector.run(
            query=message.input,
            model_response=message.output,
            docs=build_context(message.documents),
            llm_type=message.request_input.llm_type,
        )
        total_latency = time.perf_counter() - total_start

        # Persist hallucination result to Message
        try:
            existing_metadata = dict(getattr(message, "metadata", {}) or {})
            hallucination_payload = {
                "label": label,
                "reason": reason,
                "rewritten_question": rewritten_question,
                "final_answer": final_answer,
                "latencies": {
                    "detect": latencies.get("detect") if latencies else None,
                    "rewrite": latencies.get("rewrite") if latencies else None,
                    "final_answer": (
                        latencies.get("final_answer") if latencies else None
                    ),
                    "total": total_latency,
                },
            }
            existing_metadata["hallucination"] = hallucination_payload
            message.metadata = existing_metadata
            await message.save()
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to update Message with hallucination result: {e}")

        return HallucinationDetectResponse(
            label=label,
            reason=reason,
            original_question=message.input,
            rewritten_question=rewritten_question,
            final_answer=final_answer,
            latencies=latencies,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hallucination detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.post(
    "/conversations/{conversation_id}/messages/{message_id}/stream-hallucination",
    response_class=StreamingResponse,
)
async def stream_hallucination(
    conversation_id: str,
    message_id: str,
    requesting_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """
    Stream hallucination handling result as Server-Sent Events (SSE).

    Streams structured events for detection, optional rewriting, retrieval, and answer generation steps.

    - If label == 0 (factual), emits a final event with the reason.
    - If label == 1 (hallucination), streams tokens for the final answer and then a final event.

    Args:
        conversation_id (str): Conversation identifier.
        message_id (str): Message identifier to analyze.
        requesting_user (User): Authenticated user injected by dependency.

    Returns:
        SSE events for the detection workflow.

    Raises:
        HTTPException: 404 if conversation/message not found or mismatched; 403 if access is forbidden; 500 for streaming errors.
    """
    # Validate conversation ownership and message relationship
    conversation = await Conversation.find_by_id(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if conversation.user_id != requesting_user.id:
        raise HTTPException(
            status_code=403,
            detail="You are not allowed to access this conversation",
        )

    message = await Message.find_by_id(message_id)
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    if message.conversation_id != conversation_id:
        raise HTTPException(
            status_code=400, detail="Message does not belong to this conversation"
        )
    await enforce_token_budget_or_raise(requesting_user)

    async def _generator():
        import json
        import time

        from src.utils.template_loader import get_template

        total_start = time.perf_counter()
        detector = HallucinationDetector()
        try:
            # Emit an initial status event early to start the stream promptly
            yield f"data: {json.dumps({'type': 'status', 'content': 'hallucination detection started...'})}\n\n"
            # Step 1: Detect
            t0 = time.perf_counter()
            label, reason = await detector.detect(
                query=message.input,
                model_response=message.output,
                docs=build_context(message.documents),
                llm_type=message.request_input.llm_type,
            )
            detect_latency = time.perf_counter() - t0

            yield f"data: {json.dumps({'type': 'label', 'content': label})}\n\n"
            yield f"data: {json.dumps({'type': 'reason', 'content': reason})}\n\n"

            # If factual (label == 0), emit reason and finish
            if label == 0:
                total_latency = time.perf_counter() - total_start
                latencies = {
                    "detect": detect_latency,
                    "rewrite": None,
                    "final_answer": None,
                    "total": total_latency,
                }
                # Persist to message metadata
                try:
                    message.hallucination = {
                        "label": label,
                        "reason": reason,
                        "rewritten_question": None,
                        "final_answer": None,
                        "latencies": latencies,
                    }
                    await message.save()
                except Exception:
                    pass
                try:
                    await consume_tokens_for_user(
                        requesting_user,
                        count_tokens_for_texts(message.input, reason),
                    )
                except Exception as consume_error:
                    logger.warning(
                        "Failed to apply token usage for streamed hallucination: %s",
                        consume_error,
                    )

                final_payload = {
                    "type": "final",
                    "label": label,
                    "reason": reason,
                    "rewritten_question": None,
                    "answer": None,
                    "latencies": latencies,
                    "top_k_retrieved_docs": None,
                }
                yield f"data: {json.dumps(final_payload)}\n\n"
                return

            # Step 2: Rewrite (for hallucination)
            # Transparency: emit rewriting step
            yield f"data: {json.dumps({'type': 'status', 'content': 'Rewriting query...'})}\n\n"
            t1 = time.perf_counter()
            _orig_q, rewritten_question = await detector.rewrite_query(
                query=message.input,
                answer=message.output,
                reason=reason,
                llm_type=message.request_input.llm_type,
            )
            rewrite_latency = time.perf_counter() - t1
            yield f"data: {json.dumps({'type': 'rewritten_question', 'content': rewritten_question})}\n\n"

            # Step 3: Retrieve docs for rewritten_question (Qdrant + Wiley MCP)
            # Transparency: emit retrieving step
            yield f"data: {json.dumps({'type': 'status', 'content': 'Retrieving relevant documents...'})}\n\n"
            # Build a new GenerationRequest based on original, overriding the query
            req_in = message.request_input or GenerationRequest(query=message.input)
            rewritten_request = GenerationRequest(
                query=rewritten_question or message.input,
                year=getattr(req_in, "year", None),
                filters=getattr(req_in, "filters", None),
                llm_type=getattr(req_in, "llm_type", None),
                embeddings_model=getattr(req_in, "embeddings_model", None),
                k=getattr(req_in, "k", 5),
                temperature=getattr(req_in, "temperature", 0.3),
                score_threshold=getattr(req_in, "score_threshold", 0.7),
                max_new_tokens=getattr(req_in, "max_new_tokens", 1024),
                public_collections=list(
                    getattr(req_in, "public_collections", []) or []
                ),
            )
            try:
                rewritten_request.collection_ids = list(
                    getattr(req_in, "collection_ids", []) or []
                )
            except Exception:
                pass
            try:
                rewritten_request.private_collections_map = dict(
                    getattr(req_in, "private_collections_map", {}) or {}
                )
            except Exception:
                pass

            context = ""
            retrieved_docs = []
            rag_latencies = {}
            try:
                (
                    context,
                    results,
                    rag_latencies,
                    retrieved_docs,
                ) = await setup_rag_and_context(rewritten_request)
            except Exception as e:
                # Soft-fail RAG retrieval; proceed without new docs
                rag_latencies = {"rag_error": str(e)}
                retrieved_docs = []
                context = ""

            # Step 4: Stream final answer from LLM using the hallucination answer template
            template = get_template(
                "llm_answer_template", filename="hallucination_detector.yaml"
            )
            # Inject retrieved context for better grounding
            prompt = template.format(query=rewritten_question or message.input)
            if context:
                prompt = f"{prompt}\n\nContext:\n{context}"

            yield f"data: {json.dumps({'type': 'status', 'content': 'Generating answer...'})}\n\n"
            final_answer_chunks = []
            t2 = time.perf_counter()
            # Try primary provider streaming first with a first-token timeout, then fallback to Mistral
            llm = detector.llm_manager.get_client_for_model(
                message.request_input.llm_type
            )
            used_stream = False
            try:
                astream = llm.astream(prompt)
                # Enforce first token timeout similar to generate_answer
                llm_instruct_timeout = MODEL_TIMEOUT
                async with asyncio.timeout(llm_instruct_timeout):
                    first = await astream.__anext__()
                    first_text = getattr(first, "content", None)
                    if first_text:
                        final_answer_chunks.append(first_text)
                        yield f"data: {json.dumps({'type': 'token', 'content': first_text})}\n\n"
                # Continue without timeout
                async for token in astream:
                    text = getattr(token, "content", None)
                    if not text:
                        continue
                    final_answer_chunks.append(text)
                    yield f"data: {json.dumps({'type': 'token', 'content': text})}\n\n"
                used_stream = True
            except Exception:
                used_stream = False

            # Fallback to Mistral streaming if needed
            if not used_stream:
                logger.info("Hallucination Falling back to Mistral streaming")
                async for (
                    token,
                    _prompt,
                ) in detector.llm_manager.generate_answer_mistral_stream(
                    query=rewritten_question or message.input,
                    context=context or "",
                    temperature=getattr(message.request_input, "temperature", 0.3),
                    conversation_context="",
                ):
                    if not token:
                        continue
                    final_answer_chunks.append(str(token))
                    yield f"data: {json.dumps({'type': 'token', 'content': str(token)})}\n\n"

            final_latency = time.perf_counter() - t2

            final_answer = "".join(final_answer_chunks)
            try:
                await consume_tokens_for_user(
                    requesting_user,
                    count_tokens_for_texts(message.input, final_answer),
                )
            except Exception as consume_error:
                logger.warning(
                    "Failed to apply token usage for streamed hallucination: %s",
                    consume_error,
                )
            total_latency = time.perf_counter() - total_start
            latencies = {
                "detect": detect_latency,
                "rewrite": rewrite_latency,
                **(rag_latencies or {}),
                "final_answer": final_latency,
                "total": total_latency,
            }

            # Persist to message.hallucination
            try:
                message.hallucination = {
                    "label": label,
                    "reason": reason,
                    "rewritten_question": rewritten_question,
                    "final_answer": final_answer,
                    "latencies": latencies,
                    "top_k_retrieved_docs": results,
                    "retrieved_docs": retrieved_docs,
                }
                await message.save()
            except Exception:
                pass

            final_payload = {
                "type": "final",
                "label": label,
                "reason": reason,
                "rewritten_question": rewritten_question,
                "answer": final_answer,
                "latencies": latencies,
                "top_k_retrieved_docs": results,
            }
            yield f"data: {json.dumps(final_payload)}\n\n"
        except Exception as e:
            # Persist error and stream error event
            try:
                existing_metadata = dict(getattr(message, "metadata", {}) or {})
                existing_metadata["error"] = str(e)
                message.metadata = existing_metadata
                await message.save()
            except Exception:
                pass
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    response = StreamingResponse(_generator(), media_type="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"
    response.headers["X-Accel-Buffering"] = "no"
    return response


@router.post("/generate-llm")
async def generate_llm(
    request: GenerateLLMRequest,
    requesting_user: User = Depends(get_current_user),
) -> dict:
    """
    Call EVE-Instruct (v5) (Main model) with a single query. No RAG, no conversation context.

    Body: query. Returns the model reply only.
    """
    try:
        content = await invoke_llm_and_consume_tokens(
            user=requesting_user,
            model=LLMType.Main.value,
            messages=[HumanMessage(content=request.query)],
            prompt_for_token_count=request.query,
        )
        return {"answer": content}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("generate_llm failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate")
async def generate(
    request: GenerationRequest,
    requesting_user: User = Depends(get_current_user),
) -> dict:
    """
    Run a one-off generation (testing only) and return the full answer and metadata.

    Normalizes and validates requested public collections against allowed lists,
    ensures the user does not reference other users' collections, merges the user's
    collections and public collections (excluding "Wiley AI Gateway"), extracts year
    range from filters, then runs the full generation pipeline via generate_answer
    and returns the answer, documents, RAG flag, latencies, prompts, and retrieved docs.

    Args:
        request (GenerationRequest): Generation parameters including query, collections,
            and model settings.
        requesting_user (User): Authenticated user injected by dependency.

    Returns:
        Dictionary containing:
            - answer: Generated answer text.
            - documents: Extracted document data from retrieval results.
            - use_rag: Whether RAG was used for this generation.
            - latencies: Timing information for pipeline steps.
            - prompts: Prompt data from generation.
            - retrieved_docs: Raw retrieved documents from RAG.

    Raises:
        HTTPException: 403 if the request references collections owned by other users.
        HTTPException: 500 for server errors during generation.
    """
    message = None
    try:
        await enforce_token_budget_or_raise(requesting_user)
        original_query = request.query
        # Normalize and validate requested public collections against allowed lists
        allowed_source = PUBLIC_COLLECTIONS if IS_PROD else STAGING_PUBLIC_COLLECTIONS
        try:
            allowed_names = {
                item.get("name")
                for item in (allowed_source + WILEY_PUBLIC_COLLECTIONS)
                if isinstance(item, dict) and item.get("name")
            }
        except Exception:
            allowed_names = set()

        public_collections = [
            n for n in request.public_collections if n in allowed_names
        ]
        request.public_collections = public_collections

        # lookup query to check if some of the collection ids from other users are in the request.collection_ids
        other_users_collections = await CollectionModel.find_all(
            filter_dict={
                "id": {"$in": request.public_collections},
                "user_id": {"$ne": requesting_user.id},
            }
        )

        if len(other_users_collections) > 0:
            raise HTTPException(
                status_code=403,
                detail="You are not allowed to use collections from other users",
            )

        request.collection_ids = request.collection_ids + request.public_collections

        # All user collections are used by default
        user_collections = await CollectionModel.find_all(
            filter_dict={"user_id": requesting_user.id}
        )

        request.private_collections_map = {c.id: c.name for c in user_collections}
        if len(user_collections) > 0:
            request.collection_ids = request.collection_ids + [
                c.id for c in user_collections
            ]
        # remove "Wiley AI Gateway" from collection_ids
        request.collection_ids = [
            c for c in request.collection_ids if c != "Wiley AI Gateway"
        ]
        logger.info(f"Collection IDs: {request.collection_ids}")

        # Extract year range from filters for MCP usage
        try:
            request.year = extract_year_range_from_filters(request.filters)
        except Exception:
            request.year = None

        (
            answer,
            results,
            is_rag,
            latencies,
            prompts,
            retrieved_docs,
        ) = await generate_answer(request)
        (
            answer,
            results,
            is_rag,
            latencies,
            prompts,
            retrieved_docs,
        ) = await generate_answer(request)
        await consume_tokens_for_user(
            requesting_user, count_tokens_for_texts(original_query, answer)
        )

        documents_data = []
        if results:
            documents_data = [extract_document_data(result) for result in results]

        return {
            "answer": answer,
            "documents": documents_data,
            "use_rag": is_rag,
            "latencies": latencies,
            "prompts": prompts,
            "retrieved_docs": retrieved_docs,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.post("/retrieve")
async def retrieve(
    request: GenerationRequest, requesting_user: User = Depends(get_current_user)
) -> dict:
    """
    Run the entire retrieval pipeline and return all documents.

    Runs the requery/rewrite step (same as generate_answer) to refine the query
    for retrieval, then executes the RAG retrieval pipeline using
    setup_rag_and_context and returns all retrieved documents.

    Args:
        request (GenerationRequest): Generation parameters including query, collections, and model settings.
        requesting_user (User): Authenticated user injected by dependency.

    Returns:
        Dictionary containing:
            - retrieved_docs: All formatted documents from the retrieval pipeline
            - latencies: Timing information (includes rewrite and retrieval operations)
            - original_query: The query as sent in the request
            - requery: The rewritten query used for retrieval (or original if rewrite skipped/failed)
    """
    try:
        await enforce_token_budget_or_raise(requesting_user)
        allowed_source = PUBLIC_COLLECTIONS if IS_PROD else STAGING_PUBLIC_COLLECTIONS
        try:
            allowed_names = {
                item.get("name")
                for item in (allowed_source + WILEY_PUBLIC_COLLECTIONS)
                if isinstance(item, dict) and item.get("name")
            }
        except Exception:
            allowed_names = set()

        public_collections = [
            n for n in request.public_collections if n in allowed_names
        ]
        request.public_collections = public_collections

        other_users_collections = await CollectionModel.find_all(
            filter_dict={
                "id": {"$in": request.public_collections},
                "user_id": {"$ne": requesting_user.id},
            }
        )

        if len(other_users_collections) > 0:
            raise HTTPException(
                status_code=403,
                detail="You are not allowed to use collections from other users",
            )

        request.collection_ids = request.collection_ids + request.public_collections

        user_collections = await CollectionModel.find_all(
            filter_dict={"user_id": requesting_user.id}
        )

        request.private_collections_map = {c.id: c.name for c in user_collections}
        if len(user_collections) > 0:
            request.collection_ids = request.collection_ids + [
                c.id for c in user_collections
            ]
        request.collection_ids = [
            c for c in request.collection_ids if c != "Wiley AI Gateway"
        ]
        logger.info(f"Collection IDs: {request.collection_ids}")

        try:
            request.year = extract_year_range_from_filters(request.filters)
        except Exception:
            request.year = None

        original_query = request.query
        rewrite_latency = None
        requery = None
        try:
            t_rewrite = time.perf_counter()
            llm_manager = get_shared_llm_manager()
            rag_decision_result, _rag_prompt, _ = await should_use_rag(
                llm_manager,
                request.query,
                conversation="",
                llm_type=request.llm_type,
            )
            rewrite_latency = time.perf_counter() - t_rewrite
            if rag_decision_result and getattr(rag_decision_result, "requery", None):
                requery = rag_decision_result.requery
                request.query = requery
        except Exception as e:
            logger.warning(
                f"Requery/rewrite failed in /retrieve, using original query: {e}"
            )
            requery = None

        _context, _results, latencies, formated_results = await setup_rag_and_context(
            request
        )

        if rewrite_latency is not None:
            latencies = dict(latencies) if latencies else {}
            latencies["rewrite"] = rewrite_latency

        token_usage_inputs = [original_query]
        if requery and requery != original_query:
            token_usage_inputs.append(requery)
        await consume_tokens_for_user(
            requesting_user, count_tokens_for_texts(*token_usage_inputs)
        )

        return {
            "retrieved_docs": formated_results,
            "latencies": latencies,
            "original_query": original_query,
            "requery": requery or original_query,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


# ─── Agentic endpoints ────────────────────────────────────────────────────────


async def _prepare_agentic_request(
    request: GenerationRequest,
    requesting_user: User,
) -> GenerationRequest:
    """Normalise collections and resolve MCP servers on the request."""
    allowed_source = PUBLIC_COLLECTIONS if IS_PROD else STAGING_PUBLIC_COLLECTIONS
    try:
        allowed_names = {
            item.get("name")
            for item in (allowed_source + WILEY_PUBLIC_COLLECTIONS)
            if isinstance(item, dict) and item.get("name")
        }
    except Exception:
        allowed_names = set()

    request.public_collections = [
        n for n in request.public_collections if n in allowed_names
    ]
    request.collection_ids = request.collection_ids + request.public_collections
    request.collection_ids = [
        c for c in request.collection_ids if c != "Wiley AI Gateway"
    ]

    try:
        request.year = extract_year_range_from_filters(request.filters)
    except Exception:
        request.year = None

    # Resolve MCP servers by name from MongoDB.
    if request.public_mcp_servers:
        mcp_docs = await MCPServer.find_all(
            filter_dict={
                "name": {"$in": request.public_mcp_servers},
                "enabled": True,
            }
        )
        found_names = {s.name for s in mcp_docs}
        missing = set[str](request.public_mcp_servers) - found_names
        if missing:
            logger.warning("Requested MCP servers not found or disabled: %s", missing)
        request.mcp_server_configs = list(mcp_docs)
        logger.info(
            "Resolved %d MCP server(s): %s",
            len(mcp_docs),
            [s.name for s in mcp_docs],
        )

    return request


@router.post(
    "/conversations/{conversation_id}/generate-agentic",
    response_model=CreateMessageResponse,
)
async def create_agentic_message(
    request: GenerationRequest,
    conversation_id: str,
    http_request: Request,
    background_tasks: BackgroundTasks,
    requesting_user: User = Depends(get_current_user),
) -> CreateMessageResponse:
    """
    Create a new message using the fully agentic LangGraph pipeline.

    The agent autonomously decides when to search the knowledge base or the
    Wiley AI Gateway (via tool calls) and loops until it produces a final answer.

    ``public_mcp_servers`` controls which MCP servers are available for this
    request. Pass server names in ``request.public_mcp_servers``; only enabled
    servers found in the MCP server store are resolved and attached to the
    request as tool configs before generation starts.

    Args:
        request (GenerationRequest): Generation parameters including query, collections, and model settings.
        conversation_id (str): Target conversation identifier.
        background_tasks (BackgroundTasks): Background task runner used to schedule rollups.
        requesting_user (User): Authenticated user injected by dependency.

    Returns:
        Message id, query, answer, documents, flags, and metadata.
        ``trace`` contains the agentic execution steps captured during generation.

    Raises:
        HTTPException: 404 if conversation is not found; 403 if ownership invalid; 500 for server errors.
    """
    set_user_context(requesting_user.id)
    set_conversation_context(conversation_id)

    message = None
    try:
        conversation = await Conversation.find_by_id(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        if conversation.user_id != requesting_user.id:
            raise HTTPException(
                status_code=403,
                detail="You are not allowed to add a message to this conversation",
            )

        other_users_collections = await CollectionModel.find_all(
            filter_dict={
                "id": {"$in": request.public_collections},
                "user_id": {"$ne": requesting_user.id},
            }
        )
        if other_users_collections:
            raise HTTPException(
                status_code=403,
                detail="You are not allowed to use collections from other users",
            )

        user_collections = await CollectionModel.find_all(
            filter_dict={"user_id": requesting_user.id}
        )
        request.private_collections_map = {c.id: c.name for c in user_collections}
        if user_collections:
            request.collection_ids = request.collection_ids + [
                c.id for c in user_collections
            ]

        request = await _prepare_agentic_request(request, requesting_user)
        auth_header = http_request.headers.get("Authorization") or ""
        if auth_header.startswith("Bearer "):
            request.mcp_proxy_bearer_token = auth_header[7:]
        logger.info("Agentic collection IDs: %s", request.collection_ids)

        message = await Message.create(
            conversation_id=conversation_id,
            input=request.query,
            output="",
            documents=[],
            use_rag=False,
            request_input=request,
            metadata={},
        )
        set_message_context(message.id)

        (
            answer,
            tool_results,
            use_rag,
            latencies,
            prompts,
            trace_entries,
        ) = await generate_answer_agentic(
            request, conversation_id=conversation_id, user_id=requesting_user.id
        )

        message.output = answer
        message.documents = tool_results
        message.use_rag = use_rag
        message.trace = trace_entries if trace_entries else None
        existing_metadata = dict(getattr(message, "metadata", {}) or {})
        existing_metadata.update({"latencies": latencies, "prompts": prompts})
        message.metadata = existing_metadata
        await message.save()
        await consume_tokens_for_user(
            requesting_user, count_tokens_for_texts(request.query, answer)
        )

        background_tasks.add_task(maybe_rollup_and_trim_history, conversation_id)

        return {
            "id": message.id,
            "query": request.query,
            "answer": answer,
            "documents": tool_results,
            "use_rag": use_rag,
            "conversation_id": conversation_id,
            "collection_ids": request.collection_ids,
            "trace": trace_entries if trace_entries else None,
            "metadata": {"latencies": latencies},
        }
    except HTTPException as http_exc:
        if message:
            try:
                existing_metadata = dict(getattr(message, "metadata", {}) or {})
                existing_metadata["error"] = str(getattr(http_exc, "detail", http_exc))
                message.metadata = existing_metadata
                await message.save()
            except Exception:
                pass
        raise
    except Exception as e:
        if message:
            try:
                existing_metadata = dict(getattr(message, "metadata", {}) or {})
                existing_metadata["error"] = str(e)
                message.metadata = existing_metadata
                await message.save()
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.post(
    "/conversations/{conversation_id}/stream-generate-agentic",
    response_class=StreamingResponse,
)
async def create_agentic_message_stream(
    request: GenerationRequest,
    conversation_id: str,
    http_request: Request,
    background_tasks: BackgroundTasks,
    requesting_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """
    Create a new message using the agentic pipeline and stream the result via SSE.

    ``public_mcp_servers`` controls which MCP servers are available for this
    request. Pass server names in ``request.public_mcp_servers``; only enabled
    servers found in the MCP server store are resolved and attached to the
    request as tool configs before streaming generation starts.

    Streams structured events:
    - ``tool_call``   — agent is searching (with the query used)
    - ``tool_result`` — tool returned a preview
    - ``token``       — LLM final-answer token
    - ``final``       — complete answer + latencies
    - ``stopped``     — generation cancelled by client
    - ``error``       — unhandled exception

    Args:
        request (GenerationRequest): Generation parameters including query, collections, and model settings.
        conversation_id (str): Target conversation identifier.
        background_tasks (BackgroundTasks): Background task runner used to schedule rollups.
        requesting_user (User): Authenticated user injected by dependency.

    Returns:
        SSE stream for the agentic generation lifecycle.
        The ``final`` event includes ``trace`` with the agentic execution steps captured during generation.

    Raises:
        HTTPException: 404 if conversation is not found; 403 if ownership invalid; 500 for server errors.
    """
    set_user_context(requesting_user.id)
    set_conversation_context(conversation_id)

    message = None
    try:
        conversation = await Conversation.find_by_id(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        if conversation.user_id != requesting_user.id:
            raise HTTPException(
                status_code=403,
                detail="You are not allowed to add a message to this conversation",
            )

        other_users_collections = await CollectionModel.find_all(
            filter_dict={
                "id": {"$in": request.public_collections},
                "user_id": {"$ne": requesting_user.id},
            }
        )
        if other_users_collections:
            raise HTTPException(
                status_code=403,
                detail="You are not allowed to use collections from other users",
            )

        user_collections = await CollectionModel.find_all(
            filter_dict={"user_id": requesting_user.id}
        )
        request.private_collections_map = {c.id: c.name for c in user_collections}
        if user_collections:
            request.collection_ids = request.collection_ids + [
                c.id for c in user_collections
            ]

        request = await _prepare_agentic_request(request, requesting_user)
        auth_header = http_request.headers.get("Authorization") or ""
        if auth_header.startswith("Bearer "):
            request.mcp_proxy_bearer_token = auth_header[7:]
        logger.info("Agentic stream collection IDs: %s", request.collection_ids)

        message = await Message.create(
            conversation_id=conversation_id,
            input=request.query,
            output="",
            documents=[],
            use_rag=False,
            request_input=request,
            metadata={},
        )
        set_message_context(message.id)

        cancel_mgr = get_cancel_manager()
        cancel_event = cancel_mgr.create(message.id)
        cancel_mgr.link_conversation(conversation_id, message.id)
        gen_task = asyncio.create_task(
            run_agentic_generation_to_bus(
                request=request,
                conversation_id=conversation_id,
                message_id=message.id,
                background_tasks=background_tasks,
                cancel_event=cancel_event,
                user_id=requesting_user.id,
            )
        )
        cancel_mgr.set_task(message.id, gen_task)

        bus = get_stream_bus()

        async def _gen():
            try:
                if message.output:
                    yield f"data: {json.dumps({'type': 'partial', 'content': message.output})}\n\n"
            except Exception:
                pass
            async for data in bus.subscribe(message.id):
                yield data

        response = StreamingResponse(_gen(), media_type="text/event-stream")
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Connection"] = "keep-alive"
        response.headers["X-Accel-Buffering"] = "no"
        return response

    except HTTPException as http_exc:
        if message:
            error_logger = get_error_logger()
            await error_logger.log_error_sync(
                error=http_exc,
                component=Component.ROUTER,
                pipeline_stage=PipelineStage.ROUTER,
                description="HTTPException in create_agentic_message_stream",
                error_type=type(http_exc).__name__,
            )
        raise http_exc
    except Exception as e:
        if message:
            error_logger = get_error_logger()
            await error_logger.log_error_sync(
                error=e,
                component=Component.ROUTER,
                pipeline_stage=PipelineStage.ROUTER,
                description="Exception in create_agentic_message_stream",
                error_type=type(e).__name__,
            )
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
