"""Messages to and from document-parser (FEATURE_BACKGROUND_INGEST), in the alias-ocr task's contract."""

import json

from src.services import document_parser_queue as q

KEY = "users/u1/conversations/c1/uploads/d1/original.pdf"


def test_job_carries_the_contract_fields():
    body = json.loads(q.job_body(document_id="d1", key=KEY, filename="paper.pdf", user_id="u1", collection_id="col"))
    assert body["document_id"] == "d1" and body["user_id"] == "u1"
    assert body["conversation_id"] == "c1"
    assert body["s3_key"] == KEY
    assert body["content_type"] == "application/pdf"
    assert body["filename"] == "paper.pdf"


def test_job_without_a_conversation_in_the_key():
    body = json.loads(q.job_body(document_id="d1", key="other/d1/original.md", filename="n.md", user_id="u", collection_id="c"))
    assert body["conversation_id"] is None and body["content_type"] == "text/markdown"


def test_ready_and_failed_results_are_read():
    assert q.parse_result(json.dumps({"document_id": "d1", "status": "ready"}))["status"] == "ready"
    assert q.parse_result(json.dumps({"document_id": "d1", "status": "failed", "error_code": "ocr_rate_limited"}))
    assert q.parse_result(json.dumps({"document_id": "d1", "status": "processing"}))


def test_done_from_an_older_parser_counts_as_ready():
    assert q.parse_result(json.dumps({"document_id": "d1", "status": "done"}))["status"] == "ready"


def test_unknown_results_are_ignored():
    assert q.parse_result("not json") is None
    assert q.parse_result(json.dumps({"document_id": "d1", "status": "maybe"})) is None
    assert q.parse_result(json.dumps({"document_id": "d1", "status": "ready", "version": 2})) is None


def test_ocr_error_codes_are_mapped():
    from src.services import document_ingest as di

    assert di._PARSER_ERRORS["ocr_rate_limited"] == di.ERROR_PARSER_UNAVAILABLE
    assert di._PARSER_ERRORS["ocr_unavailable"] == di.ERROR_PARSER_UNAVAILABLE
    for code in ("ocr_unreadable", "encrypted_file", "file_too_large"):
        assert di._PARSER_ERRORS[code] == di.ERROR_UNREADABLE
