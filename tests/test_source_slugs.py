"""Provider slugs on Source descriptors (decision 0004).

Every Source carries ``slug``: the lowercase text the UI shows for its
provider. Public and licensed collections take it from the collection
catalog in ``src/constants.py``; a STAC item from its server; private files
have none (the UI shows "Your files"). ``provider`` and ``provider_label``
stay as they were.
"""

from __future__ import annotations

import pytest

from src.services.sources.citations import (
    CitationRewriter,
    SourceRegistry,
    prior_sources_from_rows,
)
from src.services.sources.descriptors import (
    minimal_source,
    public_source,
    source_from_document,
    source_from_stac_item,
)
from src.utils.helpers import extract_document_data
from tests.test_inline_citations import _private_doc, _public_doc, _stac_response, _wiley_doc

pytestmark = pytest.mark.no_db


def _doc_in(collection: str, point: int = 1) -> dict:
    doc = _public_doc(point=point)
    doc["collection_name"] = collection
    return doc


def _source(doc: dict) -> dict:
    source = source_from_document(extract_document_data(doc), rank=0)
    assert source is not None
    return source


class TestSlugPerProvider:
    def test_wiley_is_wiley(self):
        source = _source(_wiley_doc())
        assert source["slug"] == "wiley"
        assert source["provider"] == "wiley"
        assert source["provider_label"] == "Wiley AI Gateway"

    @pytest.mark.parametrize(
        "collection, slug",
        [
            ("wikipedia-512", "wikipedia"),
            ("Wikipedia EO", "wikipedia"),
            ("qwen-512-filtered", "oa"),
            ("EVE open access", "oa"),
            ("esa-rag-scraped-qwen3-newpipeline", "esa"),
            ("ESA EO Knowledge Base", "esa"),
        ],
    )
    def test_public_collection_by_name_or_alias(self, collection, slug):
        assert _source(_doc_in(collection))["slug"] == slug

    def test_provider_and_label_unchanged(self):
        source = _source(_doc_in("EVE open access"))
        assert source["provider"] == "qwen-512-filtered"
        assert source["provider_label"] == "EVE open access"

    def test_any_esa_scraped_collection_is_esa(self):
        assert _source(_doc_in("esa-rag-scraped-v4-pipeline"))["slug"] == "esa"

    def test_cdse_stac_item_is_cdse(self):
        item = _stac_response()["items"][0]
        source = source_from_stac_item(item, provider="cdse", rank=0)
        assert source["slug"] == "cdse"
        assert source["provider"] == "cdse"

    @pytest.mark.parametrize("conversation", [False, True])
    def test_private_file_has_no_slug(self, conversation):
        source = _source(_private_doc(conversation=conversation))
        assert "slug" in source
        assert source["slug"] is None

    def test_unknown_public_collection_derives_a_lowercase_slug(self):
        source = _source(_doc_in("Sentinel Handbook (2024)"))
        assert source["slug"] == "sentinel-handbook-2024"
        assert source["provider"] == "Sentinel Handbook (2024)"


class TestPersisted:
    def test_citation_and_sources_retrieved_carry_the_slug(self):
        registry = SourceRegistry()
        cited = _source(_doc_in("wikipedia-512", point=1))
        uncited = _source(_wiley_doc())
        h1 = registry.register(cited)
        registry.register(uncited)
        rewriter = CitationRewriter(registry)
        rewriter.feed(f"Radar emits pulses [{h1}].")
        rewriter.flush()
        (citation,) = rewriter.citations()
        assert citation["slug"] == "wikipedia"
        (retrieved,) = rewriter.sources_retrieved()
        assert retrieved["slug"] == "wiley"

    def test_public_and_minimal_views_keep_the_slug(self):
        source = _source(_doc_in("qwen-512-filtered"))
        assert public_source(source)["slug"] == "oa"
        assert minimal_source(source)["slug"] == "oa"
        private = _source(_private_doc())
        assert "slug" in minimal_source(private) and minimal_source(private)["slug"] is None


class TestLegacyRows:
    """Messages stored before this change have no ``slug`` on their citations."""

    def _legacy_rows(self) -> list:
        return [
            {
                "metadata": {
                    "citations": [
                        {
                            "id": "src_00000000000000aa",
                            "handle": "S1",
                            "kind": "document",
                            "provider": "qwen-512-filtered",
                            "provider_label": "EVE open access",
                            "visibility": "public",
                            "title": "Old paper",
                            "document_key": "doi:10.1000/old",
                            "locator": {},
                            "open": {"type": "none"},
                            "n": 1,
                            "at": [10],
                            "origin": "model",
                        }
                    ],
                    "sources_retrieved": [
                        {
                            "id": "src_00000000000000bb",
                            "handle": "S2",
                            "kind": "user_file",
                            "provider": "my-collections",
                            "provider_label": "My collections",
                            "visibility": "private",
                            "title": "notes.pdf",
                            "document_key": "doc:abc",
                            "locator": {},
                            "open": {"type": "viewer", "document_id": "abc"},
                        }
                    ],
                }
            }
        ]

    def test_legacy_rows_load_and_resolve(self):
        prior = prior_sources_from_rows(self._legacy_rows())
        assert {p["handle"] for p in prior} == {"S1", "S2"}
        registry = SourceRegistry()
        registry.seed(prior)
        assert registry.get("S1")["title"] == "Old paper"
        new = _source(_doc_in("wikipedia-512"))
        assert registry.register(new) == "S3"

    def test_legacy_source_cited_again_gets_its_slug(self):
        registry = SourceRegistry()
        registry.seed(prior_sources_from_rows(self._legacy_rows()))
        rewriter = CitationRewriter(registry)
        text = rewriter.feed("Old claim [S1] and a file [S2].") + rewriter.flush()
        assert "[1]" in text and "[2]" in text
        by_handle = {c["handle"]: c for c in rewriter.citations()}
        assert by_handle["S1"]["slug"] == "oa"
        assert by_handle["S2"]["slug"] is None
