"""Ordered rerank providers for the classic answer path.

Every provider call is awaited on the event loop through a shared
``httpx.AsyncClient`` and bounded by ``asyncio.wait_for``, so a slow or hung
reranker costs the request that waits for it and nothing else: other SSE
streams and ``/health`` on the same worker keep running.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import httpx

from src.config import (
    DEEPINFRA_API_TOKEN,
    EVE_JSC_BASE_URL,
    JSC_RERANKER_API_KEY,
    JSC_RERANKER_MODEL_NAME,
    RERANK_PROVIDER_ORDER,
)
from src.utils.deepinfra_reranker import DeepInfraReranker
from src.utils.error_logger import Component, PipelineStage, get_error_logger
from src.utils.jsc_reranker import JSCReranker

logger = logging.getLogger(__name__)

RerankResult = List[Dict[str, Any]]
RerankCall = Callable[[str, List[str]], Awaitable[RerankResult]]

# Per provider: connect 3 s, read 10 s. asyncio.wait_for bounds the whole call.
RERANK_HTTP_TIMEOUT = httpx.Timeout(10.0, connect=3.0)
RERANK_CALL_TIMEOUT_S = 10.0


class _LoopBoundClient:
    """One ``httpx.AsyncClient`` per provider, rebuilt if the event loop changes.

    A worker runs one loop for its whole life, so in production this is one
    pooled client per provider. pytest-asyncio uses a fresh loop per test, and a
    client created on a closed loop fails with ``Event loop is closed``.
    """

    def __init__(self) -> None:
        self._client: Optional[httpx.AsyncClient] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def get(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        if self._client is None or self._loop is not loop or self._loop.is_closed():
            self._client = httpx.AsyncClient(timeout=RERANK_HTTP_TIMEOUT)
            self._loop = loop
        return self._client

    async def aclose(self) -> None:
        client, self._client, self._loop = self._client, None, None
        if client is not None:
            await client.aclose()


_jsc_client = _LoopBoundClient()
_deepinfra_client = _LoopBoundClient()


@dataclass(frozen=True)
class RerankProvider:
    name: str
    label: str
    # Why the provider is unusable (the log line), or None when configured.
    missing_config: Optional[str]
    call: RerankCall = field(compare=False)


def _jsc_provider() -> RerankProvider:
    missing = None
    if not JSC_RERANKER_API_KEY:
        missing = "JSC_RERANKER_API_KEY/EVE_JSC_API_KEY environment variable not set"
    elif not EVE_JSC_BASE_URL:
        missing = "EVE_JSC_BASE_URL environment variable not set"

    async def call(query: str, documents: List[str]) -> RerankResult:
        reranker = JSCReranker(
            api_token=JSC_RERANKER_API_KEY,
            base_url=EVE_JSC_BASE_URL,
            model_name=JSC_RERANKER_MODEL_NAME,
        )
        return await reranker.rerank(_jsc_client.get(), [query], documents)

    return RerankProvider("jsc", "JSC", missing, call)


def _deepinfra_provider() -> RerankProvider:
    missing = (
        None if DEEPINFRA_API_TOKEN else "DEEPINFRA_API_TOKEN environment variable not set"
    )

    async def call(query: str, documents: List[str]) -> RerankResult:
        reranker = DeepInfraReranker(DEEPINFRA_API_TOKEN)
        return await reranker.rerank(_deepinfra_client.get(), [query], documents)

    return RerankProvider("deepinfra", "DeepInfra", missing, call)


PROVIDER_FACTORIES: Dict[str, Callable[[], RerankProvider]] = {
    "jsc": _jsc_provider,
    "deepinfra": _deepinfra_provider,
}


@lru_cache(maxsize=8)
def parse_provider_order(order: str) -> Tuple[str, ...]:
    """Known provider names in order, duplicates dropped, unknown names skipped.

    Cached per order string so an unknown name is warned about once per worker,
    not once per answer.
    """
    names: List[str] = []
    for raw in order.split(","):
        name = raw.strip().lower()
        if not name or name in names:
            continue
        if name not in PROVIDER_FACTORIES:
            logger.warning("RERANK_PROVIDER_ORDER: unknown reranker %r skipped", name)
            continue
        names.append(name)
    return tuple(names)


def configured_providers(order: Optional[str] = None) -> List[RerankProvider]:
    return [
        PROVIDER_FACTORIES[name]()
        for name in parse_provider_order(
            RERANK_PROVIDER_ORDER if order is None else order
        )
    ]


async def rerank_candidates(
    candidate_texts: List[str],
    query: str,
    *,
    providers: Optional[List[RerankProvider]] = None,
    timeout: float = RERANK_CALL_TIMEOUT_S,
) -> RerankResult:
    """Rerank with the first provider that answers in time.

    Returns ``[{"index": i, "reranking_score": s}, ...]`` sorted by score, or
    ``[]`` when there is nothing to rerank or every provider failed.
    Cancellation (Stop) propagates: only timeouts and provider errors are
    caught.
    """
    if not candidate_texts:
        return []

    error_logger = get_error_logger()
    providers = configured_providers() if providers is None else providers
    for position, provider in enumerate(providers):
        if provider.missing_config:
            logger.warning(provider.missing_config)
            continue

        # The first provider in the order keeps today's wording ("JSC reranker
        # failed"); any later one is a fallback ("DeepInfra reranker fallback
        # failed"), so the stored error descriptions do not change meaning.
        what = f"{provider.label} reranker" + (" fallback" if position else "")
        logger.info("Using %s", what)
        try:
            return await asyncio.wait_for(
                provider.call(query, candidate_texts), timeout=timeout
            )
        except (asyncio.TimeoutError, httpx.TimeoutException) as e:
            logger.warning("%s timed out after %s seconds", what, timeout)
            await error_logger.log_error(
                error=e,
                component=Component.RE_RANKER,
                pipeline_stage=PipelineStage.RETRIEVAL,
                description=f"{what} timed out",
                error_type=type(e).__name__,
            )
        except Exception as e:
            logger.warning("%s failed", what, exc_info=True)
            await error_logger.log_error(
                error=e,
                component=Component.RE_RANKER,
                pipeline_stage=PipelineStage.RETRIEVAL,
                description=f"{what} failed",
                error_type=type(e).__name__,
            )

    return []


async def aclose_rerank_clients() -> None:
    await _jsc_client.aclose()
    await _deepinfra_client.aclose()
