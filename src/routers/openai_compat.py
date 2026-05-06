import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.core.llm_manager import LLMType
from src.database.models.user import User
from src.middlewares.auth import get_current_user
from src.services.llm_inference import invoke_llm_and_consume_tokens
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
    OpenAI-compatible chat completions endpoint (non-streaming).
    """
    if request.stream:
        raise HTTPException(
            status_code=400,
            detail="Streaming is not supported on this endpoint yet. Use stream=false.",
        )

    if not request.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    started_at = int(time.time())

    try:
        prompt_text = "\n".join(
            _extract_text_content(message.content) for message in request.messages
        )
        answer = await invoke_llm_and_consume_tokens(
            user=requesting_user,
            model=request.model,
            messages=_to_langchain_messages(request.messages),
            prompt_for_token_count=prompt_text,
        )
        usage = count_tokens_for_texts(prompt_text, answer)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
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
