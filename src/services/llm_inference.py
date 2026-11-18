from typing import Any, Sequence

from src.core.llm_health import EndpointRateLimited
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
    try:
        response = await llm.ainvoke(list(messages))
    except EndpointRateLimited:
        # A rate limit window refused the call, or the endpoint answered 429
        # and opened one: answer from the fallback model, as the chain does,
        # instead of failing the request.
        response = await llm_manager.get_fallback_llm().ainvoke(list(messages))
    answer = str(getattr(response, "content", "") or "")
    await consume_tokens_for_user(
        user, count_tokens_for_texts(prompt_for_token_count, answer)
    )
    return answer
