"""Tests for the curated image catalog service."""

import pytest

from src import config
from src.services import image_catalog
from src.services.image_catalog import (
    append_image_context,
    build_image_context,
    _load_catalog,
)


CATALOG_YAML = """\
images:
  - title: "Sentinel-2 satellite"
    url: "http://localhost:9000/eve-x-demo-images/sentinel2-satellite.jpg"
    description: "Photo of the Sentinel-2 satellite."
    tags: [sentinel-2, satellite, copernicus]
  - title: "Amazon rainforest mosaic"
    url: "http://localhost:9000/eve-x-demo-images/amazon.jpg"
    description: "True-color mosaic of the Amazon basin."
    tags: [amazon, rainforest, forest]
"""


@pytest.fixture(autouse=True)
def _clear_catalog_cache():
    """The catalog is cached per path; keep tests isolated."""
    image_catalog._catalog_cache.clear()
    yield
    image_catalog._catalog_cache.clear()


@pytest.fixture
def catalog_file(tmp_path):
    path = tmp_path / "image_catalog.yaml"
    path.write_text(CATALOG_YAML, encoding="utf-8")
    return str(path)


def _enable(monkeypatch, path):
    monkeypatch.setattr(config, "FEATURE_IMAGE_CATALOG", True)
    monkeypatch.setattr(config, "IMAGE_CATALOG_PATH", path)


def test_load_catalog_happy_path(catalog_file):
    entries = _load_catalog(catalog_file)

    assert len(entries) == 2
    first = entries[0]
    assert first["title"] == "Sentinel-2 satellite"
    assert first["url"].endswith("sentinel2-satellite.jpg")
    assert first["description"] == "Photo of the Sentinel-2 satellite."
    assert "sentinel-2" in first["tags"]


def test_load_catalog_missing_file_returns_empty(tmp_path):
    missing = str(tmp_path / "does_not_exist.yaml")

    # Must degrade gracefully, never raise.
    assert _load_catalog(missing) == []


def test_load_catalog_malformed_file_returns_empty(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("images: [unclosed", encoding="utf-8")

    assert _load_catalog(str(path)) == []


def test_load_catalog_without_images_key_returns_empty(tmp_path):
    path = tmp_path / "no_images.yaml"
    path.write_text("something_else: 1\n", encoding="utf-8")

    assert _load_catalog(str(path)) == []


def test_build_image_context_disabled_returns_empty(monkeypatch, catalog_file):
    monkeypatch.setattr(config, "FEATURE_IMAGE_CATALOG", False)
    monkeypatch.setattr(config, "IMAGE_CATALOG_PATH", catalog_file)

    assert build_image_context("show me sentinel-2 images") == ""


def test_build_image_context_enabled_lists_images(monkeypatch, catalog_file):
    _enable(monkeypatch, catalog_file)

    block = build_image_context("what is sentinel-2? provide some images")

    assert "AVAILABLE IMAGES (local curated catalog):" in block
    assert (
        "http://localhost:9000/eve-x-demo-images/sentinel2-satellite.jpg" in block
    )
    assert "![<title>](<url>)" in block
    # The instructions must forbid bare URLs / "copy the link" phrasing and
    # must not let the model disclaim image display when catalog images exist.
    assert "bare URL" in block
    assert "Never say you cannot display" in block


def test_keyword_filter_narrows_to_relevant(monkeypatch, catalog_file):
    _enable(monkeypatch, catalog_file)

    block = build_image_context("tell me about the amazon rainforest")

    assert "Amazon rainforest mosaic" in block
    assert "Sentinel-2 satellite" not in block


def test_keyword_filter_falls_back_to_full_catalog(monkeypatch, catalog_file):
    _enable(monkeypatch, catalog_file)

    # No overlap with any tag/title: recall-biased fallback returns everything.
    block = build_image_context("quantum chromodynamics")

    assert "Sentinel-2 satellite" in block
    assert "Amazon rainforest mosaic" in block


def test_append_image_context_enabled_contains_block(monkeypatch, catalog_file):
    _enable(monkeypatch, catalog_file)
    base = "You are a helpful assistant.\n\nAnswer:"

    composed = append_image_context(base, "sentinel-2 images")

    assert composed.startswith(base)
    assert "AVAILABLE IMAGES (local curated catalog):" in composed


def test_append_image_context_disabled_is_byte_identical(monkeypatch, catalog_file):
    monkeypatch.setattr(config, "FEATURE_IMAGE_CATALOG", False)
    monkeypatch.setattr(config, "IMAGE_CATALOG_PATH", catalog_file)
    base = "You are a helpful assistant.\n\nAnswer:"

    # With the catalog off, the composed prompt must be unchanged.
    assert append_image_context(base, "sentinel-2 images") == base


class TestRewriteCatalogImageUrls:
    """The model reliably paraphrases catalog URLs into invented ones."""

    def _catalog(self, tmp_path, monkeypatch):
        catalog = tmp_path / "catalog.yaml"
        catalog.write_text(
            "images:\n"
            '  - title: "Sentinel-2 satellite"\n'
            '    url: "/demo-images/sentinel2-satellite.jpg"\n'
            '    tags: [sentinel-2]\n'
        )
        monkeypatch.setattr(config, "FEATURE_IMAGE_CATALOG", True)
        monkeypatch.setattr(config, "IMAGE_CATALOG_PATH", str(catalog))
        image_catalog._catalog_cache.clear()

    def test_rewrites_invented_url_by_title(self, tmp_path, monkeypatch):
        self._catalog(tmp_path, monkeypatch)
        text = "![Sentinel-2 satellite](https://storage.googleapis.com/fake/sat.jpg)"
        assert (
            image_catalog.rewrite_catalog_image_urls(text)
            == "![Sentinel-2 satellite](/demo-images/sentinel2-satellite.jpg)"
        )

    def test_rewrites_mutated_filename(self, tmp_path, monkeypatch):
        self._catalog(tmp_path, monkeypatch)
        text = "![some pic](https://fake.example/img/sentinel-2-satellite.jpg)"
        assert (
            image_catalog.rewrite_catalog_image_urls(text)
            == "![some pic](/demo-images/sentinel2-satellite.jpg)"
        )

    def test_leaves_correct_and_unrelated_urls_alone(self, tmp_path, monkeypatch):
        self._catalog(tmp_path, monkeypatch)
        ok = "![Sentinel-2 satellite](/demo-images/sentinel2-satellite.jpg)"
        other = "![chart](/artifacts/abc123) and ![ext](https://example.com/x.png)"
        assert image_catalog.rewrite_catalog_image_urls(ok) == ok
        assert image_catalog.rewrite_catalog_image_urls(other) == other

    def test_noop_when_disabled(self, tmp_path, monkeypatch):
        self._catalog(tmp_path, monkeypatch)
        monkeypatch.setattr(config, "FEATURE_IMAGE_CATALOG", False)
        text = "![Sentinel-2 satellite](https://fake.example/sat.jpg)"
        assert image_catalog.rewrite_catalog_image_urls(text) == text
