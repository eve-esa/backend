import logging
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.core.llm_manager import LLMType
from src.database.models.user import User
from src.middlewares.auth import get_current_user
from src.services.llm_inference import (
    invoke_llm_and_consume_tokens,
    invoke_llm_stream_and_consume_tokens,
)
from src.services.token_rate_limiter import count_tokens_for_texts

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1")


class OpenAITextPart(BaseModel):
    type: str
    text: Optional[str] = None


class OpenAIMessage(BaseModel):
    role: str
    content: Union[str, List[OpenAITextPart]]


class ChatCompletionsRequest(BaseModel):
    model: str = Field(default=LLMType.Main.value)
    messages: List[OpenAIMessage]
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    stream: bool = False
    user: Optional[str] = None


def _extract_text_content(content: Union[str, List[OpenAITextPart]]) -> str:
    if isinstance(content, str):
        return content

    parts: List[str] = []
    for item in content:
        if item.type == "text" and item.text:
            parts.append(item.text)
    return "\n".join(parts)


def _to_langchain_messages(messages: List[OpenAIMessage]) -> List[Any]:
    converted: List[Any] = []
    for msg in messages:
        text_content = _extract_text_content(msg.content)
        if msg.role == "system":
            converted.append(SystemMessage(content=text_content))
        elif msg.role == "assistant":
            converted.append(AIMessage(content=text_content))
        else:
            converted.append(HumanMessage(content=text_content))
    return converted


@router.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionsRequest,
    requesting_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    OpenAI-compatible chat completions endpoint.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    started_at = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"

    try:
        prompt_text = "\n".join(
            _extract_text_content(message.content) for message in request.messages
        )
        if request.stream:
            async def _event_stream():
                # Match OpenAI semantics: initial role delta, token deltas, done marker.
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": started_at,
                            "model": request.model,
                            "choices": [
                                {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                            ],
                        }
                    )
                    + "\n\n"
                )

                async for token in invoke_llm_stream_and_consume_tokens(
                    user=requesting_user,
                    model=request.model,
                    messages=_to_langchain_messages(request.messages),
                    prompt_for_token_count=prompt_text,
                ):
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": started_at,
                                "model": request.model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": token},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )
                        + "\n\n"
                    )

                yield (
                    "data: "
                    + json.dumps(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": started_at,
                            "model": request.model,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        }
                    )
                    + "\n\n"
                )
                yield "data: [DONE]\n\n"

            response = StreamingResponse(_event_stream(), media_type="text/event-stream")
            response.headers["Cache-Control"] = "no-cache"
            response.headers["Connection"] = "keep-alive"
            response.headers["X-Accel-Buffering"] = "no"
            return response

        answer = await invoke_llm_and_consume_tokens(
            user=requesting_user,
            model=request.model,
            messages=_to_langchain_messages(request.messages),
            prompt_for_token_count=prompt_text,
        )
        usage = count_tokens_for_texts(prompt_text, answer)

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": started_at,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": usage,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("chat_completions failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
