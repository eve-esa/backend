"""Wiley is searched only when it can apply every requested filter."""

import pytest

from src.services.generate_answer import wiley_supports_all_filters

pytestmark = pytest.mark.no_db


def test_no_filter_is_supported():
    assert wiley_supports_all_filters(None) is True
    assert wiley_supports_all_filters({}) is True


def test_year_filter_is_supported():
    filters = {
        "must": [{"key": "year", "range": {"gte": 2015, "lte": 2024}}],
    }
    assert wiley_supports_all_filters(filters) is True


def test_unsupported_filter_skips_wiley():
    filters = {
        "should": None,
        "must": [
            {"key": "year", "range": {"gte": 2015, "lte": 2024}},
            {"key": "n_citations", "range": {"gte": 10}},
        ],
        "must_not": None,
    }
    assert wiley_supports_all_filters(filters) is False


def test_non_must_clause_skips_wiley():
    filters = {"must_not": [{"key": "year", "match": {"value": 1990}}]}
    assert wiley_supports_all_filters(filters) is False
