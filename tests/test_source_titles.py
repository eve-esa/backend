"""Placeholder titles from ingestion must reach clients as a missing title.

Open access chunks were ingested with pandas, so a missing title is stored in
Qdrant as the string "nan" (and sometimes "None"). Clients show "Title not
available" only for a null title, so the backend hands them null instead.
"""

from types import SimpleNamespace

import pytest

from src.utils.helpers import (
    extract_document_data,
    extract_documents_from_retrieval_payload,
)

pytestmark = pytest.mark.no_db


@pytest.mark.parametrize("marker", ["nan", "NaN", "None", "null", "", "  "])
def test_placeholder_payload_title_becomes_none(marker):
    doc = extract_document_data({"id": 1, "payload": {"title": marker, "url": "u"}})
    assert doc["payload"] == {"title": None, "url": "u"}


def test_real_title_is_kept():
    doc = extract_document_data({"id": 1, "payload": {"title": "Nan Shan glaciers"}})
    assert doc["payload"]["title"] == "Nan Shan glaciers"


def test_metadata_titles_are_normalised():
    doc = extract_document_data(
        {
            "id": 1,
            "payload": {"title": "Real"},
            "metadata": {"title": "nan", "additionalMetadata": {"title": "None"}},
        }
    )
    assert doc["metadata"]["title"] is None
    assert doc["metadata"]["additionalMetadata"]["title"] is None
    assert doc["payload"]["title"] == "Real"


def test_input_payload_is_not_mutated():
    payload = {"title": "nan"}
    extract_document_data(SimpleNamespace(id=1, payload=payload, metadata={}))
    assert payload == {"title": "nan"}


def test_agentic_retrieval_payload_is_normalised():
    """The agentic path reads documents back from the eve_retrieval tool output."""
    raw = {"retrieved_docs": [{"id": "a", "payload": {"title": "nan"}}]}
    docs = extract_documents_from_retrieval_payload(raw)
    assert docs[0]["payload"]["title"] is None
