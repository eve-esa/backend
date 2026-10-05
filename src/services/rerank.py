"""Ordered rerank providers for the classic answer path.

Every provider call is awaited on the event loop through a shared
``httpx.AsyncClient`` and bounded by one deadline for the whole provider loop,
so a slow or hung reranker costs the request that waits for it and nothing
else: other SSE streams and ``/health`` on the same worker keep running.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

import httpx

from src.config import (
    DEEPINFRA_API_TOKEN,
    EVE_JSC_BASE_URL,
    JSC_RERANKER_API_KEY,
    JSC_RERANKER_MODEL_NAME,
    RERANK_PROVIDER_ORDER,
)
from src.utils import jsc_reranker
from src.utils.deepinfra_reranker import DeepInfraReranker
from src.utils.error_logger import Component, PipelineStage, get_error_logger
from src.utils.jsc_reranker import JSCReranker

logger = logging.getLogger(__name__)

RerankResult = List[Dict[str, Any]]
RerankCall = Callable[[str, List[str]], Awaitable[RerankResult]]

# One deadline for the whole provider loop. Each attempt is also capped below
# the total, so a hung first provider still leaves the fallback some budget.
RERANK_TOTAL_BUDGET_S = 10.0
RERANK_ATTEMPT_TIMEOUT_S = 6.0
# Waiting for a free slot on a saturated provider: past this, try the next one.
RERANK_ACQUIRE_TIMEOUT_S = 2.0

RERANK_HTTP_TIMEOUT = httpx.Timeout(10.0, connect=3.0)
RERANK_HTTP_LIMITS = httpx.Limits(max_connections=64, max_keepalive_connections=32)

# Stale clients being closed in the background (keeps the tasks referenced).
_closing: Set["asyncio.Task[None]"] = set()


async def _close_quietly(client: httpx.AsyncClient) -> None:
    try:
        await client.aclose()
    except Exception:
        logger.debug("Closing a stale rerank client failed", exc_info=True)


class _LoopBoundClient:
    """One ``httpx.AsyncClient`` and one concurrency semaphore per provider.

    Both are bound to the event loop that created them. A worker runs one loop
    for its whole life, so in production this is one pooled client per
    provider. pytest-asyncio uses a fresh loop per test: there the stale client
    is closed in the background and a new pair is built.
    """

    def __init__(
        self, concurrency: int, transport: Optional[httpx.AsyncBaseTransport] = None
    ) -> None:
        self._concurrency = concurrency
        self._transport = transport
        self._client: Optional[httpx.AsyncClient] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _ensure(self) -> None:
        loop = asyncio.get_running_loop()
        if self._client is not None and self._loop is loop and not loop.is_closed():
            return
        stale = self._client
        if stale is not None:
            task = loop.create_task(_close_quietly(stale))
            _closing.add(task)
            task.add_done_callback(_closing.discard)
        self._client = httpx.AsyncClient(
            timeout=RERANK_HTTP_TIMEOUT,
            limits=RERANK_HTTP_LIMITS,
            transport=self._transport,
        )
        self._semaphore = asyncio.Semaphore(self._concurrency)
        self._loop = loop

    def get(self) -> httpx.AsyncClient:
        self._ensure()
        return self._client

    def semaphore(self) -> asyncio.Semaphore:
        self._ensure()
        return self._semaphore

    async def aclose(self) -> None:
        client, self._client, self._semaphore, self._loop = self._client, None, None, None
        if client is not None:
            await client.aclose()


_jsc_client = _LoopBoundClient(concurrency=16)
_deepinfra_client = _LoopBoundClient(concurrency=32)


@dataclass(frozen=True)
class RerankProvider:
    name: str
    label: str
    # Why the provider is unusable (the log line), or None when configured.
    missing_config: Optional[str]
    call: RerankCall = field(compare=False)
    # Concurrency slots; None means unbounded (test doubles).
    semaphore: Optional[Callable[[], asyncio.Semaphore]] = field(default=None, compare=False)
    # Seconds the provider asked to be left alone (JSC 429 window), 0 if none.
    backoff: Callable[[], float] = field(default=lambda: 0.0, compare=False)


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

    return RerankProvider(
        "jsc",
        "JSC",
        missing,
        call,
        semaphore=lambda: _jsc_client.semaphore(),
        backoff=jsc_reranker.backoff_remaining,
    )


def _deepinfra_provider() -> RerankProvider:
    missing = (
        None if DEEPINFRA_API_TOKEN else "DEEPINFRA_API_TOKEN environment variable not set"
    )

    async def call(query: str, documents: List[str]) -> RerankResult:
        reranker = DeepInfraReranker(DEEPINFRA_API_TOKEN)
        return await reranker.rerank(_deepinfra_client.get(), [query], documents)

    return RerankProvider(
        "deepinfra",
        "DeepInfra",
        missing,
        call,
        semaphore=lambda: _deepinfra_client.semaphore(),
    )


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


def error_type_name(error: BaseException) -> str:
    """The ``error_type`` stored on RE_RANKER rows, as the requests-based code wrote it.

    Timeouts were ``TimeoutError`` (concurrent.futures), HTTP and transport
    failures ``RequestException``, bad JSON ``ValueError``. Keeping the names
    keeps dashboards on those rows stable.
    """
    if isinstance(error, (asyncio.TimeoutError, httpx.TimeoutException)):
        return "TimeoutError"
    if isinstance(error, httpx.HTTPError):
        return "RequestException"
    if isinstance(error, json.JSONDecodeError):
        return "ValueError"
    return type(error).__name__


async def rerank_candidates(
    candidate_texts: List[str],
    query: str,
    *,
    providers: Optional[List[RerankProvider]] = None,
    timeout: float = RERANK_TOTAL_BUDGET_S,
    attempt_timeout: float = RERANK_ATTEMPT_TIMEOUT_S,
    acquire_timeout: float = RERANK_ACQUIRE_TIMEOUT_S,
) -> RerankResult:
    """Rerank with the first provider that answers before the deadline.

    ``timeout`` is the budget for the whole loop; each attempt gets at most
    ``attempt_timeout`` of what is left. Returns
    ``[{"index": i, "reranking_score": s}, ...]`` sorted by score. When no
    provider answers (all failed, all skipped, or the budget ran out) it
    returns every candidate in retrieval order with a ``None`` score, so the
    answer keeps its sources; the caller deduplicates and cuts to top_k.
    ``[]`` only when there is nothing to rerank. Cancellation (Stop)
    propagates: only timeouts and provider errors are caught.
    """
    if not candidate_texts:
        return []

    error_logger = get_error_logger()
    providers = configured_providers() if providers is None else providers
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    attempted = False
    reason = "no_provider_available"

    for position, provider in enumerate(providers):
        if provider.missing_config:
            logger.warning(provider.missing_config)
            continue
        if provider.backoff() > 0:
            # Logged once when the window opened (jsc_reranker.note_rate_limited).
            logger.debug("%s reranker in backoff, skipped", provider.label)
            continue

        # The first provider in the order is "<label> reranker", any later one
        # "<label> reranker fallback" (with the default order: "DeepInfra
        # reranker failed", "JSC reranker fallback failed"). The back office
        # stats classify these rows by provider name, not by position.
        what = f"{provider.label} reranker" + (" fallback" if position else "")
        remaining = deadline - loop.time()
        if remaining <= 0:
            logger.warning("Rerank budget of %s seconds spent, %s not tried", timeout, what)
            reason = "budget_spent"
            break

        semaphore = provider.semaphore() if provider.semaphore else None
        if semaphore is not None:
            try:
                async with asyncio.timeout(min(acquire_timeout, remaining)):
                    await semaphore.acquire()
            except TimeoutError:
                logger.warning("%s busy, no free slot, skipped", what)
                continue

        logger.info("Using %s", what)
        attempted = True
        budget = min(attempt_timeout, max(deadline - loop.time(), 0.0))
        try:
            async with asyncio.timeout(budget):
                return await provider.call(query, candidate_texts)
        except (TimeoutError, httpx.TimeoutException) as e:
            logger.warning("%s timed out after %.1f seconds", what, budget)
            await error_logger.log_error(
                error=e,
                component=Component.RE_RANKER,
                pipeline_stage=PipelineStage.RETRIEVAL,
                description=f"{what} timed out",
                error_type=error_type_name(e),
            )
        except Exception as e:
            logger.warning("%s failed", what, exc_info=True)
            await error_logger.log_error(
                error=e,
                component=Component.RE_RANKER,
                pipeline_stage=PipelineStage.RETRIEVAL,
                description=f"{what} failed",
                error_type=error_type_name(e),
            )
        finally:
            if semaphore is not None:
                semaphore.release()

    if attempted and reason != "budget_spent":
        reason = "all_providers_failed"
    return retrieval_order_fallback(candidate_texts, reason)


def retrieval_order_fallback(candidate_texts: List[str], reason: str) -> RerankResult:
    """Candidates in retrieval order, unscored, after one WARNING with the reason."""
    logger.warning(
        "rerank.skipped reason=%s candidates=%d", reason, len(candidate_texts)
    )
    return [{"index": i, "reranking_score": None} for i in range(len(candidate_texts))]


async def aclose_rerank_clients() -> None:
    await _jsc_client.aclose()
    await _deepinfra_client.aclose()
