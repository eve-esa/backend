import logging
import time
from typing import Optional, Dict

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.middlewares.auth import get_current_user
from src.database.models.user import User
from src.database.models.conversation import Conversation
from src.services.hallucination_detector import HallucinationDetector
from src.database.models.message import Message
from src.utils.helpers import build_context

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
            docs=build_context(message.documents),
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
    "/stream-hallucination",
    response_class=StreamingResponse,
)
async def stream_hallucination(
    request: HallucinationDetectRequest,
    requesting_user: User = Depends(get_current_user),
):
    """Stream hallucination handling result.

    - If label == 0 (factual), stream a single final event with the reason.
    - If label == 1 (hallucination), stream tokens for the final answer and then a final event.
    """
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

    async def _generator():
        import json
        import time
        from src.utils.template_loader import get_template

        total_start = time.perf_counter()
        detector = HallucinationDetector()
        try:
            # Step 1: Detect
            t0 = time.perf_counter()
            label, reason = await detector.detect(
                query=message.input,
                model_response=message.output,
                docs=build_context(message.documents),
                llm_type=request.llm_type,
            )
            detect_latency = time.perf_counter() - t0

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
                    existing_metadata = dict(getattr(message, "metadata", {}) or {})
                    existing_metadata["hallucination"] = {
                        "label": label,
                        "reason": reason,
                        "rewritten_question": None,
                        "final_answer": None,
                        "latencies": latencies,
                    }
                    message.metadata = existing_metadata
                    await message.save()
                except Exception:
                    pass

                final_payload = {
                    "type": "final",
                    "answer": reason,
                    "latencies": latencies,
                }
                yield f"data: {json.dumps(final_payload)}\n\n"
                return

            # Step 2: Rewrite (for hallucination)
            t1 = time.perf_counter()
            _orig_q, rewritten_question = await detector.rewrite_query(
                query=message.input,
                answer=message.output,
                reason=reason,
                llm_type=request.llm_type,
            )
            rewrite_latency = time.perf_counter() - t1

            # Step 3: Stream final answer from LLM using the hallucination answer template
            template = get_template(
                "llm_answer_template", filename="hallucination_detector.yaml"
            )
            prompt = template.format(query=rewritten_question or message.input)

            final_answer_chunks = []
            t2 = time.perf_counter()
            # Use mistral streaming path (follows existing generate_answer streaming behavior)
            llm = detector.llm_manager.get_client_for_model(
                message.request_input.llm_type
            )
            async for token in llm.astream(prompt):
                final_answer_chunks.append(str(token))
                yield f"data: {json.dumps({'type':'token','content':str(token)})}\n\n"
            final_latency = time.perf_counter() - t2

            final_answer = "".join(final_answer_chunks)
            total_latency = time.perf_counter() - total_start
            latencies = {
                "detect": detect_latency,
                "rewrite": rewrite_latency,
                "final_answer": final_latency,
                "total": total_latency,
            }

            # Persist to message metadata
            try:
                existing_metadata = dict(getattr(message, "metadata", {}) or {})
                existing_metadata["hallucination"] = {
                    "label": label,
                    "reason": reason,
                    "rewritten_question": rewritten_question,
                    "final_answer": final_answer,
                    "latencies": latencies,
                }
                message.metadata = existing_metadata
                await message.save()
            except Exception:
                pass

            final_payload = {
                "type": "final",
                "answer": final_answer,
                "latencies": latencies,
            }
            yield f"data: {json.dumps(final_payload)}\n\n"
        except Exception as e:
            # Persist error and stream error event
            try:
                message.metadata = {"error": str(e)}
                await message.save()
            except Exception:
                pass
            yield f"data: {json.dumps({'type':'error','message':str(e)})}\n\n"

    response = StreamingResponse(_generator(), media_type="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"
    response.headers["X-Accel-Buffering"] = "no"
    return response
