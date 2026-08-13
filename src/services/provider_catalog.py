"""Provider and platform model catalog loaded from MongoDB.

The catalog is seeded from ``provider_models.yaml`` on first startup when the
collections are empty. Back office can add or update providers/models in Mongo
without redeploying; changes are picked up after the in-memory cache TTL.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import List
from urllib.parse import urlparse

import yaml
from fastapi import HTTPException
from pydantic import BaseModel, Field

from src.config import EVE_JSC_BASE_URL, FEATURE_JSC_MODEL, IS_PROD
from src.database.models.catalog_platform_model import CatalogPlatformModelDoc
from src.database.models.catalog_provider import (
    CatalogProviderDoc,
    CatalogProviderModelEntry,
)
from src.schemas.custom_model import (
    PlatformModel,
    ProviderCatalogModelPublic,
    ProviderCatalogPublic,
)

logger = logging.getLogger(__name__)

PROVIDER_MODELS_PATH = os.getenv("PROVIDER_MODELS_PATH", "provider_models.yaml")
PROVIDER_CATALOG_CACHE_TTL_SECONDS = float(
    os.getenv("PROVIDER_CATALOG_CACHE_TTL_SECONDS", "60")
)

JSC_PLATFORM_MODEL_ID = "eve-instruct-jsc"
JSC_LLM_TYPE = "eve_jsc"


class CatalogModel(BaseModel):
    id: str
    display_name: str
    model_name: str


class CatalogProvider(BaseModel):
    id: str
    display_name: str
    base_url: str
    models: List[CatalogModel] = Field(default_factory=list)


class CatalogModelEntry(BaseModel):
    provider: CatalogProvider
    model: CatalogModel


class CatalogPlatformModel(BaseModel):
    id: str
    llm_type: str
    display_name: str
    description: str | None = None


class LoadedCatalog(BaseModel):
    platform: tuple[CatalogPlatformModel, ...]
    providers: tuple[CatalogProvider, ...]


_cache: LoadedCatalog | None = None
_cache_expires_at: float = 0.0
_cache_lock = asyncio.Lock()


def _validate_provider_base_url(base_url: str, *, provider_id: str) -> str:
    normalized = base_url.strip().rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            f"provider catalog: provider {provider_id!r} has invalid base_url"
        )
    if IS_PROD and parsed.scheme != "https":
        raise ValueError(
            f"provider catalog: provider {provider_id!r} must use HTTPS in production"
        )
    return normalized


def _read_yaml_seed() -> dict:
    with open(PROVIDER_MODELS_PATH, encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


async def ensure_provider_catalog_seeded() -> None:
    """Seed Mongo catalog collections from YAML when empty (first deploy/dev)."""
    platform_count = await CatalogPlatformModelDoc.count_documents({})
    provider_count = await CatalogProviderDoc.count_documents({})
    if platform_count > 0 and provider_count > 0:
        return

    raw = _read_yaml_seed()
    if platform_count == 0:
        for index, item in enumerate(raw.get("platform", [])):
            doc = CatalogPlatformModelDoc(
                catalog_id=item["id"],
                llm_type=item["llm_type"],
                display_name=item["display_name"],
                description=item.get("description"),
                enabled=True,
                sort_order=index,
            )
            try:
                await doc.save()
            except ValueError:
                logger.info(
                    "Platform model %r already seeded by another worker, skipping",
                    item["id"],
                )
        logger.info(
            "Seeded %d platform model(s) into catalog_platform_models",
            len(raw.get("platform", [])),
        )

    if provider_count == 0:
        for index, item in enumerate(raw.get("providers", [])):
            provider_id = item["id"]
            base_url = _validate_provider_base_url(
                item["base_url"], provider_id=provider_id
            )
            models = [
                CatalogProviderModelEntry(
                    catalog_model_id=model["id"],
                    display_name=model["display_name"],
                    model_name=model["model_name"],
                    enabled=True,
                )
                for model in item.get("models", [])
            ]
            doc = CatalogProviderDoc(
                catalog_id=provider_id,
                display_name=item["display_name"],
                base_url=base_url,
                models=models,
                enabled=True,
                sort_order=index,
            )
            try:
                await doc.save()
            except ValueError:
                logger.info(
                    "Provider %r already seeded by another worker, skipping",
                    provider_id,
                )
        logger.info(
            "Seeded %d provider(s) into catalog_providers",
            len(raw.get("providers", [])),
        )

    invalidate_catalog_cache()


def invalidate_catalog_cache() -> None:
    global _cache, _cache_expires_at
    _cache = None
    _cache_expires_at = 0.0


def clear_provider_catalog_cache_for_tests() -> None:
    # The lock is replaced, not only the cache: /models loads the catalog through
    # asyncio.gather, and a contended acquire binds the lock to the running event
    # loop. pytest-asyncio gives every test a fresh loop, so a binding left behind
    # by one test makes the next contended acquire raise "bound to a different
    # event loop".
    global _cache_lock
    invalidate_catalog_cache()
    _cache_lock = asyncio.Lock()


async def _fetch_catalog_from_mongo() -> LoadedCatalog:
    platform_docs = await CatalogPlatformModelDoc.find_all(
        filter_dict={"enabled": True},
        sort=[("sort_order", 1), ("catalog_id", 1)],
    )
    provider_docs = await CatalogProviderDoc.find_all(
        filter_dict={"enabled": True},
        sort=[("sort_order", 1), ("catalog_id", 1)],
    )

    platform = tuple(
        CatalogPlatformModel(
            id=doc.catalog_id,
            llm_type=doc.llm_type,
            display_name=doc.display_name,
            description=doc.description,
        )
        for doc in platform_docs
    )

    providers: list[CatalogProvider] = []
    for doc in provider_docs:
        enabled_models = [model for model in doc.models if model.enabled]
        providers.append(
            CatalogProvider(
                id=doc.catalog_id,
                display_name=doc.display_name,
                base_url=doc.base_url,
                models=[
                    CatalogModel(
                        id=model.catalog_model_id,
                        display_name=model.display_name,
                        model_name=model.model_name,
                    )
                    for model in enabled_models
                ],
            )
        )

    return LoadedCatalog(platform=platform, providers=tuple(providers))


async def _load_catalog() -> LoadedCatalog:
    global _cache, _cache_expires_at

    now = time.monotonic()
    if _cache is not None and now < _cache_expires_at:
        return _cache

    async with _cache_lock:
        now = time.monotonic()
        if _cache is not None and now < _cache_expires_at:
            return _cache

        catalog = await _fetch_catalog_from_mongo()
        _cache = catalog
        _cache_expires_at = now + PROVIDER_CATALOG_CACHE_TTL_SECONDS
        return catalog


def _should_inject_jsc(catalog: LoadedCatalog) -> bool:
    """Whether to prepend the JSC platform model at serve time.

    Injected here rather than seeded: the flag must be able to make the entry
    disappear, and a Mongo document would survive the flag being turned off.
    A blank base URL wins over the flag. If the back office ever inserts a JSC
    document into Mongo, that document wins and this injection steps aside.
    """
    if not (FEATURE_JSC_MODEL and EVE_JSC_BASE_URL):
        return False
    return not any(
        model.id == JSC_PLATFORM_MODEL_ID or model.llm_type == JSC_LLM_TYPE
        for model in catalog.platform
    )


def _transparent_jsc_default() -> bool:
    """Whether the default EVE model should transparently answer via JSC.

    On environments where JSC is configured but its picker entry is hidden
    (``FEATURE_JSC_MODEL`` off = staging/prod), the single "EVE-Instruct" pick
    IS the JSC endpoint: it maps to ``eve_jsc``, whose chain leads with JSC and
    fails over to RunPod/Mistral. The user sees one "EVE-Instruct" entry and
    never RunPod's cold start. When the flag is on (dev) the picker instead
    shows a separate "EVE-Instruct (JSC)" entry and the default stays ``main``.
    """
    return bool(EVE_JSC_BASE_URL and not FEATURE_JSC_MODEL)


async def list_platform_models() -> List[PlatformModel]:
    catalog = await _load_catalog()
    transparent_jsc = _transparent_jsc_default()
    models = [
        PlatformModel(
            id=model.id,
            llm_type=(
                JSC_LLM_TYPE
                if transparent_jsc and model.llm_type == "main"
                else model.llm_type
            ),
            display_name=model.display_name,
            description=model.description,
        )
        for model in catalog.platform
    ]
    if _should_inject_jsc(catalog):
        models.insert(
            0,
            PlatformModel(
                id=JSC_PLATFORM_MODEL_ID,
                llm_type=JSC_LLM_TYPE,
                display_name="EVE-Instruct (JSC)",
                description=(
                    "EVE instruction-tuned model served by the "
                    "Jülich Supercomputing Centre"
                ),
            ),
        )
    return models


async def list_provider_catalog() -> List[ProviderCatalogPublic]:
    catalog = await _load_catalog()
    return [
        ProviderCatalogPublic(
            id=provider.id,
            display_name=provider.display_name,
            models=[
                ProviderCatalogModelPublic(
                    id=model.id,
                    display_name=model.display_name,
                    model_name=model.model_name,
                )
                for model in provider.models
            ],
        )
        for provider in catalog.providers
    ]


async def resolve_catalog_entry(
    provider_id: str, catalog_model_id: str
) -> CatalogModelEntry:
    catalog = await _load_catalog()
    for provider in catalog.providers:
        if provider.id != provider_id:
            continue
        for model in provider.models:
            if model.id == catalog_model_id:
                return CatalogModelEntry(provider=provider, model=model)
        raise HTTPException(
            status_code=422,
            detail=f"Unknown model {catalog_model_id!r} for provider {provider_id!r}",
        )
    raise HTTPException(status_code=422, detail=f"Unknown provider {provider_id!r}")
