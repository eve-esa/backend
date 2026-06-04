"""Tests for public collection alias/name canonicalization."""

from src.utils.helpers import normalize_public_collections_selection


def test_staging_satcom_alias_and_name_deduped():
    out = normalize_public_collections_selection(
        [
            "SATCOM Technical Knowledge Base",
            "satcom-chunks-collection",
        ],
        is_prod=False,
    )
    assert out == ["satcom-chunks-collection"]


def test_unknown_label_dropped_staging():
    out = normalize_public_collections_selection(
        ["not-a-real-collection", "wikipedia-512"],
        is_prod=False,
    )
    assert out == ["wikipedia-512"]


def test_prod_does_not_allow_staging_only_alias():
    out = normalize_public_collections_selection(
        ["EVE open-access", "qwen-512-filtered"],
        is_prod=True,
    )
    assert "EVE open-access" not in out
    assert out == ["qwen-512-filtered"]
