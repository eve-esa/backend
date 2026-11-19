"""Context window of a model on an endpoint, looked up once and stored.

Order: the process cache, then Mongo, then a catalog override, then
``GET /models``. A lookup that finds nothing is remembered for ten minutes.
A stored window older than seven days is still used, and refreshed in the
background. The API key is never logged.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from urllib.parse import quote, urlparse

import httpx

from src.config import AGENT_CONTEXT_WINDOW_TOKENS

logger = logging.getLogger(__name__)

WINDOW_FIELDS = (
    "max_model_len",
    "max_context_length",
    "context_length",
    "max_input_tokens",
)
FETCH_TIMEOUT_S = 3.0
MISS_TTL = timedelta(minutes=10)
REFRESH_AFTER = timedelta(days=7)

_memory: dict[tuple[str, str], tuple[int, datetime]] = {}
_misses: dict[tuple[str, str], datetime] = {}
_refreshing: set[tuple[str, str]] = set()


def clear_window_cache_for_tests() -> None:
    _memory.clear()
    _misses.clear()
    _refreshing.clear()


def window_from_payload(payload: Any) -> Optional[int]:
    """First positive window field of one ``/models`` object."""
    if not isinstance(payload, dict):
        return None
    for field in WINDOW_FIELDS:
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        if value > 0:
            return value
    return None


def _key(base_url: str, model_name: str) -> tuple[str, str]:
    return base_url.rstrip("/"), model_name


def _aware(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment


def _fresh(fetched_at: datetime, now: datetime) -> bool:
    return now - _aware(fetched_at) < REFRESH_AFTER


def _host(base_url: str) -> str:
    return urlparse(base_url).netloc or base_url


async def _read_stored(base_url: str, model_name: str):
    from src.database.models.model_context_window import ModelContextWindow

    return await ModelContextWindow.find_one(
        {"base_url": base_url, "model_name": model_name}
    )


async def _write_stored(
    base_url: str, model_name: str, window: int, source: str, fetched_at: datetime
) -> None:
    from src.database.models.model_context_window import ModelContextWindow

    row = await _read_stored(base_url, model_name)
    if row is None:
        row = ModelContextWindow(
            base_url=base_url,
            model_name=model_name,
            context_window=window,
            source=source,
            fetched_at=fetched_at,
        )
    else:
        row.context_window = window
        row.source = source
        row.fetched_at = fetched_at
    await row.save()


async def _catalog_window(base_url: str, model_name: str) -> Optional[int]:
    from src.database.models.catalog_provider import CatalogProviderDoc

    docs = await CatalogProviderDoc.find_all(filter_dict={"base_url": base_url})
    for doc in docs:
        for model in doc.models:
            if model.model_name == model_name and model.context_window:
                return int(model.context_window)
    return None


async def _fetch_window(
    base_url: str, model_name: str, api_key: str
) -> Optional[int]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    one = f"{base_url}/models/{quote(model_name, safe='')}"
    listing = f"{base_url}/models"
    async with httpx.AsyncClient(timeout=FETCH_TIMEOUT_S) as client:
        response = await client.get(one, headers=headers)
        if response.status_code == 200:
            found = window_from_payload(response.json())
            if found is not None:
                return found
        response = await client.get(listing, headers=headers)
        response.raise_for_status()
        for item in response.json().get("data") or []:
            if isinstance(item, dict) and item.get("id") == model_name:
                return window_from_payload(item)
    return None


def _remember(key: tuple[str, str], window: int, fetched_at: datetime) -> None:
    _memory[key] = (window, fetched_at)
    _misses.pop(key, None)


def _schedule_refresh(
    base_url: str, model_name: str, api_key: str
) -> None:
    key = (base_url, model_name)
    if key in _refreshing:
        return
    _refreshing.add(key)

    async def _run() -> None:
        try:
            await _refresh(base_url, model_name, api_key)
        finally:
            _refreshing.discard(key)

    asyncio.create_task(_run())


async def _refresh(base_url: str, model_name: str, api_key: str) -> None:
    now = datetime.now(timezone.utc)
    try:
        override = await _catalog_window(base_url, model_name)
        if override:
            await _write_stored(base_url, model_name, override, "catalog", now)
            _remember((base_url, model_name), override, now)
            return
        found = await _fetch_window(base_url, model_name, api_key)
    except Exception:
        logger.warning(
            "context window refresh failed model=%s host=%s",
            model_name,
            _host(base_url),
        )
        return
    if found is None:
        return
    await _write_stored(base_url, model_name, found, "endpoint", now)
    _remember((base_url, model_name), found, now)


async def window_for(base_url: str, model_name: str, api_key: str = "") -> int:
    """Token window of *model_name* at *base_url*.

    Falls back to ``AGENT_CONTEXT_WINDOW_TOKENS`` when nobody reports one.
    """
    key = _key(base_url, model_name)
    base_url, model_name = key
    now = datetime.now(timezone.utc)

    cached = _memory.get(key)
    if cached is not None:
        window, fetched_at = cached
        if not _fresh(fetched_at, now):
            _schedule_refresh(base_url, model_name, api_key)
        return window

    try:
        stored = await _read_stored(base_url, model_name)
    except Exception:
        logger.warning(
            "context window store unreadable model=%s host=%s",
            model_name,
            _host(base_url),
        )
        stored = None
    if stored is not None:
        _remember(key, stored.context_window, _aware(stored.fetched_at))
        if not _fresh(stored.fetched_at, now):
            _schedule_refresh(base_url, model_name, api_key)
        return stored.context_window

    missed = _misses.get(key)
    if missed is not None and now - _aware(missed) < MISS_TTL:
        return AGENT_CONTEXT_WINDOW_TOKENS

    try:
        override = await _catalog_window(base_url, model_name)
    except Exception:
        logger.warning(
            "context window catalog unreadable model=%s host=%s",
            model_name,
            _host(base_url),
        )
        override = None
    if override:
        await _store(base_url, model_name, override, "catalog", now)
        return override

    try:
        found = await _fetch_window(base_url, model_name, api_key)
    except Exception:
        logger.warning(
            "context window lookup failed model=%s host=%s",
            model_name,
            _host(base_url),
        )
        _misses[key] = now
        return AGENT_CONTEXT_WINDOW_TOKENS
    if found is None:
        _misses[key] = now
        return AGENT_CONTEXT_WINDOW_TOKENS
    await _store(base_url, model_name, found, "endpoint", now)
    return found


async def _store(
    base_url: str, model_name: str, window: int, source: str, fetched_at: datetime
) -> None:
    try:
        await _write_stored(base_url, model_name, window, source, fetched_at)
    except Exception:
        logger.warning(
            "context window store failed model=%s host=%s",
            model_name,
            _host(base_url),
        )
    _remember((base_url, model_name), window, fetched_at)


def schedule_window_lookup(base_url: str, model_name: str, api_key: str) -> None:
    """Warm the cache. A failure stays in the task and never reaches the caller.

    Under pytest this does nothing unless ``MODEL_LIMITS_LIVE`` is set, so
    creating a custom model in a test does not call the provider.
    """
    import os

    if os.getenv("PYTEST_CURRENT_TEST") and not os.getenv("MODEL_LIMITS_LIVE"):
        return

    async def _run() -> None:
        try:
            await window_for(base_url, model_name, api_key)
        except Exception:
            logger.warning(
                "context window warm-up failed model=%s host=%s",
                model_name,
                _host(base_url),
            )

    try:
        asyncio.get_running_loop().create_task(_run())
    except RuntimeError:
        return
