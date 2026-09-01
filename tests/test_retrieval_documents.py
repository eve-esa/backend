"""Retrieval tool payloads must become Document-shaped dicts.

The agentic pipeline gets its documents out of a ToolMessage, whose content is
whatever the MCP retrieval tool serialised: a JSON string, an already-parsed
dict, or a list of MCP content blocks. Everything the UI renders (id, score,
collection, payload, text) has to survive that trip.
"""

import json

import pytest

from src.utils.helpers import (
    extract_documents_from_retrieval_payload,
    is_retrieval_error_payload,
)

pytestmark = pytest.mark.no_db


def _doc(doc_id: str = "doc-1") -> dict:
    return {
        "id": doc_id,
        "version": 3,
        "score": 0.87,
        "reranking_score": 0.91,
        "collection_name": "ESA Earth Observation",
        "payload": {"title": "Sentinel-2"},
        "text": "Sentinel-2 carries a multispectral instrument.",
        "metadata": {"page": 4},
    }


def _retrieval_response(*docs: dict) -> dict:
    return {
        "retrieved_docs": list(docs),
        "latencies": {"retrieval_latency": 0.4},
        "original_query": "sentinel",
        "requery": None,
    }


class TestExtractDocumentsFromRetrievalPayload:
    def test_json_string_with_retrieved_docs(self):
        payload = json.dumps(_retrieval_response(_doc(), _doc("doc-2")))

        documents = extract_documents_from_retrieval_payload(payload)

        assert [d["id"] for d in documents] == ["doc-1", "doc-2"]
        assert documents[0]["collection_name"] == "ESA Earth Observation"
        assert documents[0]["payload"] == {"title": "Sentinel-2"}
        assert documents[0]["text"].startswith("Sentinel-2 carries")
        assert documents[0]["score"] == 0.87

    def test_already_parsed_dict(self):
        documents = extract_documents_from_retrieval_payload(
            _retrieval_response(_doc())
        )

        assert len(documents) == 1
        assert documents[0]["id"] == "doc-1"

    def test_list_of_mcp_content_blocks(self):
        blocks = [
            {"type": "text", "text": json.dumps(_retrieval_response(_doc()))},
            {"type": "text", "text": json.dumps(_retrieval_response(_doc("doc-2")))},
        ]

        documents = extract_documents_from_retrieval_payload(blocks)

        assert [d["id"] for d in documents] == ["doc-1", "doc-2"]

    def test_error_payload_yields_no_documents(self):
        payload = json.dumps({"error": "retrieval failed", "detail": "boom"})

        assert extract_documents_from_retrieval_payload(payload) == []

    def test_garbage_string_yields_no_documents(self):
        assert extract_documents_from_retrieval_payload("not json at all") == []

    def test_empty_payloads_yield_no_documents(self):
        assert extract_documents_from_retrieval_payload(None) == []
        assert extract_documents_from_retrieval_payload("") == []
        assert extract_documents_from_retrieval_payload([]) == []

    def test_bare_list_of_documents(self):
        documents = extract_documents_from_retrieval_payload([_doc(), _doc("doc-2")])

        assert [d["id"] for d in documents] == ["doc-1", "doc-2"]

    def test_non_dict_items_are_skipped(self):
        payload = {"retrieved_docs": [_doc(), "just a string", None, 42]}

        documents = extract_documents_from_retrieval_payload(payload)

        assert [d["id"] for d in documents] == ["doc-1"]

    def test_alternative_wrapper_keys(self):
        for key in ("results", "documents", "items", "data"):
            documents = extract_documents_from_retrieval_payload({key: [_doc()]})
            assert [d["id"] for d in documents] == ["doc-1"], key

    def test_single_document_object(self):
        documents = extract_documents_from_retrieval_payload(_doc())

        assert [d["id"] for d in documents] == ["doc-1"]

    def test_empty_retrieved_docs_is_not_an_error(self):
        payload = json.dumps(_retrieval_response())

        assert extract_documents_from_retrieval_payload(payload) == []
        assert is_retrieval_error_payload(payload) is False

    def test_output_is_idempotent(self):
        once = extract_documents_from_retrieval_payload(_retrieval_response(_doc()))
        twice = extract_documents_from_retrieval_payload({"retrieved_docs": once})

        assert once == twice

    def test_normalize_false_keeps_raw_items(self):
        raw = {"results": [{"title": "untouched"}]}

        documents = extract_documents_from_retrieval_payload(raw, normalize=False)

        assert documents == [{"title": "untouched"}]

    def test_keep_text_fallback_wraps_undecodable_text(self):
        documents = extract_documents_from_retrieval_payload(
            [{"type": "text", "text": "plain prose"}],
            normalize=False,
            keep_text_fallback=True,
        )

        assert documents == [{"text": "plain prose"}]


class TestIsRetrievalErrorPayload:
    def test_true_for_error_object(self):
        assert is_retrieval_error_payload(json.dumps({"error": "nope"})) is True

    def test_true_for_error_inside_content_block(self):
        blocks = [{"type": "text", "text": json.dumps({"error": "nope"})}]

        assert is_retrieval_error_payload(blocks) is True

    def test_false_for_documents(self):
        assert is_retrieval_error_payload(_retrieval_response(_doc())) is False

    def test_false_for_garbage(self):
        assert is_retrieval_error_payload("not json at all") is False


def test_python_repr_of_content_blocks_is_parsed():
    """The agent graph persists ``str(result)``: a repr of the content blocks."""
    blocks = [
        {
            "type": "text",
            "text": json.dumps(
                {
                    "retrieved_docs": [
                        {
                            "id": "1",
                            "collection_name": "wikipedia-512",
                            "payload": {"title": "Sentinel-2", "content": "x"},
                        }
                    ],
                    "latencies": {},
                }
            ),
        }
    ]
    docs = extract_documents_from_retrieval_payload(str(blocks))
    assert len(docs) == 1
    assert docs[0]["collection_name"] == "wikipedia-512"
    assert docs[0]["text"] == "x"
    assert not is_retrieval_error_payload(str(blocks))


def test_python_repr_of_error_block_is_an_error():
    blocks = [{"type": "text", "text": json.dumps({"error": "EVE /retrieve returned 401"})}]
    assert extract_documents_from_retrieval_payload(str(blocks)) == []
    assert is_retrieval_error_payload(str(blocks))
