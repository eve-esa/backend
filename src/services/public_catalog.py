"""Public collection catalog loaded from MongoDB.

Seeded from ``src.constants`` on first startup when ``public_catalog`` is empty.
Back office (or Compass) can add or update rows without redeploying; changes
are picked up after the in-memory cache TTL.

See ``src.services.provider_catalog`` for the same seed-if-empty + TTL pattern.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Iterable, Optional, Sequence

from pydantic import BaseModel

from src.constants import (
    EVE_PUBLIC_COLLECTION_NAME_PROD,
    PUBLIC_COLLECTIONS,
    WILEY_PUBLIC_COLLECTIONS,
)
from src.database.models.public_catalog import PublicCatalogDoc, PublicCatalogKind

logger = logging.getLogger(__name__)

PUBLIC_CATALOG_CACHE_TTL_SECONDS = float(
    os.getenv("PUBLIC_CATALOG_CACHE_TTL_SECONDS", "60")
)

WILEY_KIND: PublicCatalogKind = "wiley"
SATCOM_KIND: PublicCatalogKind = "satcom"
QDRANT_KIND: PublicCatalogKind = "qdrant"


class PublicCatalogRow(BaseModel):
    """In-memory public catalog row (Mongo document or seed snapshot)."""

    name: str
    alias: Optional[str] = None
    description: Optional[str] = None
    enabled: bool = True
    sort_order: int = 0
    visible_in_prod: bool = True
    visible_in_non_prod: bool = True
    kind: PublicCatalogKind = "qdrant"
    applies_eve_filters: bool = False


_cache: tuple[PublicCatalogRow, ...] | None = None
_cache_expires_at: float = 0.0
_cache_lock = asyncio.Lock()


def default_public_catalog_seed() -> list[PublicCatalogRow]:
    """Catalog rows matching today's hardcoded constants (first-boot seed)."""
    by_name: dict[str, PublicCatalogRow] = {}
    order = 0

    def upsert(
        *,
        name: str,
        alias: Optional[str] = None,
        description: Optional[str] = None,
        visible_in_prod: bool,
        visible_in_non_prod: bool,
        kind: PublicCatalogKind = "qdrant",
        applies_eve_filters: bool = False,
    ) -> None:
        nonlocal order
        if not name:
            return
        existing = by_name.get(name)
        if existing is None:
            by_name[name] = PublicCatalogRow(
                name=name,
                alias=alias,
                description=description,
                enabled=True,
                sort_order=order,
                visible_in_prod=visible_in_prod,
                visible_in_non_prod=visible_in_non_prod,
                kind=kind,
                applies_eve_filters=applies_eve_filters,
            )
            order += 1
            return
        if alias and not existing.alias:
            existing.alias = alias
        if description and not existing.description:
            existing.description = description
        existing.visible_in_prod = existing.visible_in_prod or visible_in_prod
        existing.visible_in_non_prod = (
            existing.visible_in_non_prod or visible_in_non_prod
        )
        if applies_eve_filters:
            existing.applies_eve_filters = True
        if kind != "qdrant":
            existing.kind = kind

    for item in WILEY_PUBLIC_COLLECTIONS:
        if not isinstance(item, dict):
            continue
        upsert(
            name=item.get("name") or "",
            alias=item.get("alias"),
            description=item.get("description"),
            visible_in_prod=True,
            visible_in_non_prod=True,
            kind=WILEY_KIND,
        )

    for item in PUBLIC_COLLECTIONS:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or ""
        upsert(
            name=name,
            alias=item.get("alias"),
            description=item.get("description"),
            visible_in_prod=True,
            visible_in_non_prod=True,
            kind=QDRANT_KIND,
            applies_eve_filters=name == EVE_PUBLIC_COLLECTION_NAME_PROD,
        )

    return sorted(by_name.values(), key=lambda row: row.sort_order)


def _row_from_doc(doc: PublicCatalogDoc) -> PublicCatalogRow:
    return PublicCatalogRow(
        name=doc.name,
        alias=doc.alias,
        description=doc.description,
        enabled=doc.enabled,
        sort_order=doc.sort_order,
        visible_in_prod=doc.visible_in_prod,
        visible_in_non_prod=doc.visible_in_non_prod,
        kind=doc.kind,
        applies_eve_filters=doc.applies_eve_filters,
    )


def iter_visible_rows(
    rows: Sequence[PublicCatalogRow], *, is_prod: bool
) -> list[PublicCatalogRow]:
    """Enabled rows visible in this environment, in ``sort_order``."""
    out: list[PublicCatalogRow] = []
    for row in rows:
        if not row.enabled:
            continue
        if is_prod and not row.visible_in_prod:
            continue
        if not is_prod and not row.visible_in_non_prod:
            continue
        out.append(row)
    return out


def catalog_rows_as_dicts(rows: Sequence[PublicCatalogRow]) -> list[dict]:
    return [
        {
            "name": row.name,
            "alias": row.alias,
            "description": row.description,
            "kind": row.kind,
        }
        for row in rows
    ]


def _labels_for_rows(rows: Iterable[PublicCatalogRow]) -> set[str]:
    labels: set[str] = set()
    for row in rows:
        if row.name:
            labels.add(row.name)
        if row.alias:
            labels.add(row.alias)
    return labels


def kind_labels(rows: Sequence[PublicCatalogRow], kind: PublicCatalogKind) -> set[str]:
    return _labels_for_rows(row for row in rows if row.kind == kind)


def eve_filter_labels(rows: Sequence[PublicCatalogRow]) -> set[str]:
    return _labels_for_rows(row for row in rows if row.applies_eve_filters)


def cached_or_seed_rows() -> tuple[PublicCatalogRow, ...]:
    """Last loaded catalog, or the in-memory seed if the cache is cold."""
    if _cache is not None:
        return _cache
    return tuple(default_public_catalog_seed())


def all_catalog_labels() -> set[str]:
    """All names and aliases across environments (for public vs private split)."""
    return _labels_for_rows(cached_or_seed_rows())


def catalog_name_to_alias() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in cached_or_seed_rows():
        if row.name and row.alias:
            mapping[row.name] = row.alias
    return mapping


def wiley_public_collection_names() -> set[str]:
    return kind_labels(cached_or_seed_rows(), WILEY_KIND)


def satcom_public_collection_names() -> set[str]:
    return kind_labels(cached_or_seed_rows(), SATCOM_KIND)


def eve_public_collection_names() -> set[str]:
    return eve_filter_labels(cached_or_seed_rows())


def is_wiley_public_collection(name: str) -> bool:
    return bool(name) and name in wiley_public_collection_names()


def is_satcom_public_collection(name: str) -> bool:
    return bool(name) and name in satcom_public_collection_names()


def is_eve_public_collection(name: str) -> bool:
    return bool(name) and name in eve_public_collection_names()


def invalidate_public_catalog_cache() -> None:
    global _cache, _cache_expires_at
    _cache = None
    _cache_expires_at = 0.0


def clear_public_catalog_cache_for_tests() -> None:
    global _cache_lock
    invalidate_public_catalog_cache()
    _cache_lock = asyncio.Lock()


async def ensure_public_catalog_seeded() -> None:
    """Seed Mongo ``public_catalog`` from constants when empty (first deploy/dev)."""
    count = await PublicCatalogDoc.count_documents({})
    if count > 0:
        return
    seed = default_public_catalog_seed()
    for row in seed:
        doc = PublicCatalogDoc(
            name=row.name,
            alias=row.alias,
            description=row.description,
            enabled=row.enabled,
            sort_order=row.sort_order,
            visible_in_prod=row.visible_in_prod,
            visible_in_non_prod=row.visible_in_non_prod,
            kind=row.kind,
            applies_eve_filters=row.applies_eve_filters,
        )
        try:
            await doc.save()
        except ValueError:
            logger.info(
                "Public catalog %r already seeded by another worker, skipping",
                row.name,
            )
    logger.info("Seeded %d public catalog row(s) into public_catalog", len(seed))


async def _fetch_catalog_from_mongo() -> tuple[PublicCatalogRow, ...]:
    docs = await PublicCatalogDoc.find_all(
        filter_dict={"enabled": True},
        sort=[("sort_order", 1), ("name", 1)],
    )
    if not docs:
        return tuple(default_public_catalog_seed())
    return tuple(_row_from_doc(doc) for doc in docs)


async def load_public_catalog() -> tuple[PublicCatalogRow, ...]:
    """Return enabled catalog rows, cached for ``PUBLIC_CATALOG_CACHE_TTL_SECONDS``."""
    global _cache, _cache_expires_at

    now = time.monotonic()
    if _cache is not None and now < _cache_expires_at:
        return _cache

    async with _cache_lock:
        now = time.monotonic()
        if _cache is not None and now < _cache_expires_at:
            return _cache
        try:
            catalog = await _fetch_catalog_from_mongo()
        except Exception:
            logger.warning(
                "Failed to load public_catalog from Mongo; using seed snapshot",
                exc_info=True,
            )
            catalog = tuple(default_public_catalog_seed())
        _cache = catalog
        _cache_expires_at = now + PUBLIC_CATALOG_CACHE_TTL_SECONDS
        return catalog
