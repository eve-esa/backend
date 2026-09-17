"""Tests for selecting unique retrieval results after reranking."""

import pytest

from src.services.generate_answer import _select_top_k_unique_results

pytestmark = pytest.mark.no_db


def test_duplicates_are_removed_before_top_k_is_applied():
    formatted_results = [
        {"id": "duplicate-alias-1", "text": "same Wiley chunk"},
        {"id": "duplicate-alias-2", "text": "same Wiley chunk"},
        {"id": "unique-1", "text": "first unique chunk"},
        {"id": "unique-2", "text": "second unique chunk"},
    ]
    reranked = [
        {"index": 0, "reranking_score": 0.99},
        {"index": 1, "reranking_score": 0.98},
        {"index": 2, "reranking_score": 0.97},
        {"index": 3, "reranking_score": 0.96},
    ]

    results = _select_top_k_unique_results(
        formatted_results, reranked, top_k=3
    )

    assert [result["id"] for result in results] == [
        "duplicate-alias-1",
        "unique-1",
        "unique-2",
    ]


def test_selection_returns_fewer_results_when_unique_candidates_are_exhausted():
    formatted_results = [
        {"id": "duplicate-alias-1", "text": "same Wiley chunk"},
        {"id": "duplicate-alias-2", "text": "same Wiley chunk"},
    ]
    reranked = [
        {"index": 0, "reranking_score": 0.99},
        {"index": 1, "reranking_score": 0.98},
    ]

    results = _select_top_k_unique_results(
        formatted_results, reranked, top_k=10
    )

    assert [result["id"] for result in results] == ["duplicate-alias-1"]
