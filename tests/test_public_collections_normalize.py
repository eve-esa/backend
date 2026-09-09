"""Tests for public collection alias/name canonicalization."""

import pytest

from src.utils.helpers import (
    iter_public_catalog,
    normalize_public_collections_selection,
)

pytestmark = pytest.mark.no_db

SHARED_CATALOG = [
    "esa-rag-scraped-qwen3-newpipeline",
    "qwen-512-filtered",
    "wikipedia-512",
]


@pytest.mark.parametrize("is_prod", [True, False])
def test_catalog_is_identical_in_every_env(is_prod):
    names = [item["name"] for item in iter_public_catalog(is_prod=is_prod)]
    assert names == SHARED_CATALOG
    aliases = {item["name"]: item.get("alias") for item in iter_public_catalog(is_prod=is_prod)}
    assert aliases == {
        "esa-rag-scraped-qwen3-newpipeline": "ESA EO Knowledge Base",
        "qwen-512-filtered": "EVE open access",
        "wikipedia-512": "Wikipedia EO",
    }


@pytest.mark.parametrize("is_prod", [True, False])
def test_satcom_dropped_in_every_env(is_prod):
    out = normalize_public_collections_selection(
        [
            "SATCOM Technical Knowledge Base",
            "satcom-chunks-collection",
            "qwen-512-filtered",
        ],
        is_prod=is_prod,
    )
    assert out == ["qwen-512-filtered"]


@pytest.mark.parametrize("is_prod", [True, False])
def test_unknown_label_dropped(is_prod):
    out = normalize_public_collections_selection(
        ["not-a-real-collection", "wikipedia-512"],
        is_prod=is_prod,
    )
    assert out == ["wikipedia-512"]


@pytest.mark.parametrize("is_prod", [True, False])
def test_display_aliases_canonicalize_to_qdrant_names(is_prod):
    out = normalize_public_collections_selection(
        [
            "EVE open access",
            "Wikipedia EO",
            "ESA EO Knowledge Base",
            "qwen-512-filtered",
        ],
        is_prod=is_prod,
    )
    assert out == [
        "qwen-512-filtered",
        "wikipedia-512",
        "esa-rag-scraped-qwen3-newpipeline",
    ]


@pytest.mark.parametrize("is_prod", [True, False])
def test_unknown_hyphenated_eve_label_is_dropped(is_prod):
    out = normalize_public_collections_selection(
        ["EVE open-access", "qwen-512-filtered"],
        is_prod=is_prod,
    )
    assert out == ["qwen-512-filtered"]


@pytest.mark.parametrize("is_prod", [True, False])
def test_shared_public_names_allowed(is_prod):
    out = normalize_public_collections_selection(SHARED_CATALOG, is_prod=is_prod)
    assert out == SHARED_CATALOG
