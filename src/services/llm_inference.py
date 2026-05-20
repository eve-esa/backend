from typing import Any, AsyncGenerator, Sequence

from src.database.models.user import User
from src.services.generate_answer import get_shared_llm_manager
from src.services.token_rate_limiter import (
    consume_tokens_for_user,
    count_tokens_for_texts,
    enforce_token_budget_or_raise,
)


async def invoke_llm_and_consume_tokens(
    *,
    user: User,
    model: str,
    messages: Sequence[Any],
    prompt_for_token_count: str,
) -> str:
    """
    Run a plain LLM invoke with existing auth/rate-limit semantics.
    """
    await enforce_token_budget_or_raise(user)
    llm_manager = get_shared_llm_manager()
    llm = llm_manager.get_client_for_model(model)
    response = await llm.ainvoke(list(messages))
    answer = str(getattr(response, "content", "") or "")
    await consume_tokens_for_user(
        user, count_tokens_for_texts(prompt_for_token_count, answer)
    )
    return answer


async def invoke_llm_stream_and_consume_tokens(
    *,
    user: User,
    model: str,
    messages: Sequence[Any],
    prompt_for_token_count: str,
) -> AsyncGenerator[str, None]:
    """
    Run a plain LLM stream with existing auth/rate-limit semantics.
    """
    await enforce_token_budget_or_raise(user)
    llm_manager = get_shared_llm_manager()
    llm = llm_manager.get_client_for_model(model)

    collected_chunks: list[str] = []
    async for chunk in llm.astream(list(messages)):
        content = getattr(chunk, "content", None)
        if isinstance(content, str):
            text = content
        elif isinstance(chunk, dict):
            text = str(chunk.get("content", "") or "")
        else:
            text = str(content or "")

        if text:
            collected_chunks.append(text)
            yield text

    await consume_tokens_for_user(
        user,
        count_tokens_for_texts(prompt_for_token_count, "".join(collected_chunks)),
    )
