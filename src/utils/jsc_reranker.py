import logging
import time
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# Blablador rate-limits per key (shared by every environment). After a 429 the
# reranker is skipped for the Retry-After window, capped, so a throttled JSC
# does not cost every answer a round trip before the fallback.
RETRY_AFTER_CAP_S = 30.0
RETRY_AFTER_DEFAULT_S = 10.0
_backoff_until = 0.0


class JSCRateLimited(httpx.HTTPStatusError):
    """JSC answered 429; the reranker is in its backoff window."""


def backoff_remaining() -> float:
    """Seconds left in the current 429 window, 0 when JSC may be called."""
    return max(0.0, _backoff_until - time.monotonic())


def _retry_after_seconds(value: Optional[str]) -> float:
    if not value:
        return RETRY_AFTER_DEFAULT_S
    value = value.strip()
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = parsedate_to_datetime(value).timestamp() - time.time()
        except (TypeError, ValueError, IndexError, OverflowError):
            return RETRY_AFTER_DEFAULT_S
    return min(max(seconds, 0.0), RETRY_AFTER_CAP_S)


def note_rate_limited(retry_after: Optional[str]) -> float:
    """Open (or keep) the backoff window; log once when a window opens."""
    global _backoff_until
    window = _retry_after_seconds(retry_after)
    if backoff_remaining() <= 0:
        logger.warning("JSC reranker rate limited (429), skipped for %.0f seconds", window)
    _backoff_until = max(_backoff_until, time.monotonic() + window)
    return window


def reset_backoff() -> None:
    global _backoff_until
    _backoff_until = 0.0


class JSCReranker:
    """JSC Blablador reranker (Cohere-style ``/rerank``) over a shared async client."""

    def __init__(
        self,
        api_token: str,
        base_url: str,
        model_name: str = "alias-qwen3-4b-reranking",
    ):
        self.api_token = api_token
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    async def rerank(
        self, client: httpx.AsyncClient, queries: List[str], documents: List[str]
    ) -> List[Dict[str, Any]]:
        response = await client.post(
            f"{self.base_url}/rerank",
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_name,
                "query": queries[0],
                "documents": documents,
            },
        )
        if response.status_code == 429:
            note_rate_limited(response.headers.get("Retry-After"))
            raise JSCRateLimited(
                "JSC reranker rate limited (429)",
                request=response.request,
                response=response,
            )
        response.raise_for_status()
        # A non-JSON body (Blablador answers throttling with 200 text/plain)
        # raises ValueError here, which the caller treats as a failed provider.
        data = response.json()
        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list):
            raise ValueError("JSC reranker response did not include results")

        scores_with_indices = [
            {
                "index": item["index"],
                "reranking_score": item["relevance_score"],
            }
            for item in results
        ]
        scores_with_indices.sort(key=lambda x: x["reranking_score"], reverse=True)
        return scores_with_indices
