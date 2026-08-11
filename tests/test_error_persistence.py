"""Failure metadata and retry pipeline detection.

A generation failure must leave a machine-readable marker on the message
(metadata.error with a code the frontend keys its copy on), and a retry of a
failed agentic turn must stay on the agentic pipeline even though no trace was
ever persisted.
"""

import pytest

from src.schemas.generation_request import GenerationRequest
from src.services.agentic_utils import is_agentic_generation_request
from src.services.generate_answer import build_error_payload


class _NodeTimeoutError(Exception):
    pass


_NodeTimeoutError.__name__ = "NodeTimeoutError"


def test_timeout_errors_map_to_the_timeout_code():
    assert build_error_payload(TimeoutError("first token"))["code"] == "timeout"
    assert build_error_payload(_NodeTimeoutError("node"))["code"] == "timeout"


def test_other_errors_map_to_upstream_error():
    payload = build_error_payload(ValueError("No generations found in stream."))
    assert payload["code"] == "upstream_error"
    assert payload["type"] == "ValueError"
    assert "No generations" in payload["message"]


def test_error_message_is_truncated():
    payload = build_error_payload(RuntimeError("x" * 2000))
    assert len(payload["message"]) == 500


class _FakeMessage:
    def __init__(self, metadata=None, trace=None):
        self.metadata = metadata
        self.trace = trace


@pytest.mark.no_db
def test_pipeline_marker_keeps_failed_agentic_retries_agentic():
    request = GenerationRequest(query="q")
    failed_agentic = _FakeMessage(metadata={"pipeline": "agentic"}, trace=None)
    assert is_agentic_generation_request(request, failed_agentic) is True


@pytest.mark.no_db
def test_plain_classic_message_stays_classic():
    request = GenerationRequest(query="q")
    classic = _FakeMessage(metadata={}, trace=None)
    assert is_agentic_generation_request(request, classic) is False


@pytest.mark.no_db
def test_trace_still_marks_agentic_for_legacy_messages():
    request = GenerationRequest(query="q")
    legacy = _FakeMessage(metadata={}, trace=[{"node": "agent"}])
    assert is_agentic_generation_request(request, legacy) is True
