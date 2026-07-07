"""Fixed provider/model catalog for user-owned custom models."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List
from urllib.parse import urlparse

import yaml
from fastapi import HTTPException
from pydantic import BaseModel, Field

from src.config import IS_PROD
from src.schemas.custom_model import ProviderCatalogModelPublic, ProviderCatalogPublic

PROVIDER_MODELS_PATH = os.getenv("PROVIDER_MODELS_PATH", "provider_models.yaml")


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


def _validate_provider_base_url(base_url: str, *, provider_id: str) -> str:
    normalized = base_url.strip().rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            f"provider_models.yaml: provider {provider_id!r} has invalid base_url"
        )
    if IS_PROD and parsed.scheme != "https":
        raise ValueError(
            f"provider_models.yaml: provider {provider_id!r} must use HTTPS in production"
        )
    return normalized


@lru_cache(maxsize=1)
def _load_catalog() -> tuple[CatalogProvider, ...]:
    with open(PROVIDER_MODELS_PATH, encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    providers: list[CatalogProvider] = []
    seen_provider_ids: set[str] = set()

    for item in raw.get("providers", []):
        provider = CatalogProvider.model_validate(item)
        if provider.id in seen_provider_ids:
            raise ValueError(
                f"provider_models.yaml: duplicate provider id {provider.id!r}"
            )
        seen_provider_ids.add(provider.id)

        base_url = _validate_provider_base_url(provider.base_url, provider_id=provider.id)
        seen_model_ids: set[str] = set()
        models: list[CatalogModel] = []
        for model in provider.models:
            if model.id in seen_model_ids:
                raise ValueError(
                    f"provider_models.yaml: duplicate model id {model.id!r} "
                    f"for provider {provider.id!r}"
                )
            seen_model_ids.add(model.id)
            models.append(model)

        providers.append(
            provider.model_copy(update={"base_url": base_url, "models": models})
        )

    return tuple(providers)


def list_provider_catalog() -> List[ProviderCatalogPublic]:
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
        for provider in _load_catalog()
    ]


def resolve_catalog_entry(provider_id: str, catalog_model_id: str) -> CatalogModelEntry:
    for provider in _load_catalog():
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


def clear_provider_catalog_cache_for_tests() -> None:
    _load_catalog.cache_clear()
