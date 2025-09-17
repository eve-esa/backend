from src.core.vector_store_manager import VectorStoreManager
from src.schemas.message import MessageUpdate, CreateMessageResponse
from src.services.generate_answer import (
    GenerationRequest,
    generate_answer,
    maybe_rollup_and_trim_history,
)
from src.database.models.conversation import Conversation
from src.database.models.message import Message
from src.database.models.collection import Collection as CollectionModel
from fastapi import APIRouter, HTTPException, Depends
from src.database.models.user import User
from src.middlewares.auth import get_current_user
from typing import Dict, Any, Optional, List

router = APIRouter()


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


def _extract_document_data(result: Any) -> Dict[str, Any]:
    result_id = _field(result, "id")
    result_version = _to_int(_field(result, "version"))
    result_score = _to_float(
        _field(result, "score") or _field(result, "relevance_score")
    )
    result_payload = (
        _field(result, "payload", {}) or _field(result, "document", {}) or {}
    )
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
        "payload": result_payload,
        "text": result_text,
        "metadata": result_metadata,
    }


def _extract_year_range_from_filters(filters: Any) -> Optional[List[int]]:
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


@router.post(
    "/conversations/{conversation_id}/messages", response_model=CreateMessageResponse
)
async def create_message(
    request: GenerationRequest,
    conversation_id: str,
    requesting_user: User = Depends(get_current_user),
):
    try:
        conversation = await Conversation.find_by_id(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        if conversation.user_id != requesting_user.id:
            raise HTTPException(
                status_code=403,
                detail="You are not allowed to add a message to this conversation",
            )

        # All public collections are used by default
        public_collections, _ = await VectorStoreManager().list_public_collections()
        if len(public_collections) > 0:
            request.collection_ids = [c["name"] for c in public_collections]

        # All user collections are used by default
        user_collections = await CollectionModel.find_all(
            filter_dict={"user_id": requesting_user.id}
        )

        if len(user_collections) > 0:
            request.collection_ids = request.collection_ids + [
                c.id for c in user_collections
            ]

        # Extract year range from filters for MCP usage
        try:
            request.year = _extract_year_range_from_filters(request.filters)
        except Exception:
            request.year = None

        answer, results, is_rag, loop_result, latencies = await generate_answer(
            request, conversation_id=conversation_id
        )

        documents_data = []
        if results:
            documents_data = [_extract_document_data(result) for result in results]

        message = await Message.create(
            conversation_id=conversation_id,
            input=request.query,
            output=answer,
            documents=documents_data,
            use_rag=is_rag,
            metadata={
                "latencies": latencies,
            },
        )

        # Every-N-turn rolling summary and memory reset
        try:
            await maybe_rollup_and_trim_history(conversation_id)
        except Exception:
            pass

        return {
            "id": message.id,
            "query": request.query,
            "answer": answer,
            "documents": documents_data,
            "use_rag": is_rag,
            "conversation_id": conversation_id,
            "loop_result": loop_result,
            "metadata": {
                "latencies": latencies,
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
):
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

        await message.save()

        return {"message": "Feedback updated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
