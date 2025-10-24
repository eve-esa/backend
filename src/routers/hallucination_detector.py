import logging
import time
from typing import List, Optional, Dict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.middlewares.auth import get_current_user
from src.database.models.user import User
from src.database.models.conversation import Conversation
from src.services.hallucination_detector import HallucinationDetector
from src.database.models.message import Message
from src.constants import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_K,
    DEFAULT_SCORE_THRESHOLD,
)


logger = logging.getLogger(__name__)
router = APIRouter()


class HallucinationDetectRequest(BaseModel):
    conversation_id: str = Field(..., description="Conversation ID for the message")
    message_id: str = Field(
        ..., description="Message ID to attach hallucination result to"
    )
    llm_type: Optional[str] = Field(
        default=None,
        description="LLM type to use. Options: 'runpod' or 'mistral'. Defaults to env behavior.",
    )


class HallucinationDetectResponse(BaseModel):
    label: int
    reason: str
    original_question: str
    rewritten_question: Optional[str] = None
    final_answer: Optional[str] = None
    latencies: Optional[Dict[str, Optional[float]]] = None


@router.post("/hallucination", response_model=HallucinationDetectResponse)
async def hallucination_detect(
    request: HallucinationDetectRequest,
    requesting_user: User = Depends(get_current_user),
):
    # Validate conversation ownership and message relationship
    conversation = await Conversation.find_by_id(request.conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if conversation.user_id != requesting_user.id:
        raise HTTPException(
            status_code=403,
            detail="You are not allowed to access this conversation",
        )

    message = await Message.find_by_id(request.message_id)
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    if message.conversation_id != request.conversation_id:
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
            collection_names=message.request_input.collection_ids,
            k=message.request_input.k,
            score_threshold=message.request_input.score_threshold,
            llm_type=request.llm_type,
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
            original_question=request.query,
            rewritten_question=rewritten_question,
            final_answer=final_answer,
            latencies=latencies,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hallucination detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
