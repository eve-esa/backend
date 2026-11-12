"""The hallucination judge sees the clock and the turn's tool results.

An agentic answer that lists Sentinel-2 scenes returned by a STAC search was
judged hallucinated because the judge had no current date (2026 looked like the
future) and never saw the tool output. The judge now gets the same clock as the
agents and the tool results persisted in ``Message.trace``, trimmed and
redacted. A message without a trace keeps the classic prompt.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from src.database.models.conversation import Conversation
from src.database.models.message import Message
from src.schemas.generation_request import GenerationRequest
from src.services import hallucination_detector as hd
from src.services.hallucination_detector import (
    TOOL_EVIDENCE_MAX_CHARS_PER_TOOL,
    TOOL_EVIDENCE_MAX_CHARS_TOTAL,
    HallucinationDetector,
    HallucinationResult,
    LLMResult,
    build_tool_evidence,
    hallucination_answer_prompt,
)
from tests.utils.cleaner import cleanup_models
from tests.utils.utils import create_test_user_and_token

NOW = datetime(2026, 11, 12, 9, 30, 0, tzinfo=timezone.utc)

STAC_OUTPUT = (
    '{"type": "FeatureCollection", "features": [{"id": '
    '"S2B_MSIL2A_20261031T101029_N0511_R022_T32TQM", "properties": '
    '{"datetime": "2026-10-31T10:10:29Z", "eo:cloud_cover": 3.13}}]}'
)

AGENTIC_TRACE = [
    {"node": "agent", "role": "user", "content": "private user turn text"},
    {
        "node": "agent",
        "role": "assistant",
        "content": "",
        "tool_calls": [{"name": "search_stac", "args": {"collections": ["sentinel-2-l2a"]}, "id": "c1"}],
    },
    {"node": "tools", "role": "tool", "name": "search_stac", "content": STAC_OUTPUT, "tool_call_id": "c1"},
    {
        "node": "tools",
        "role": "tool",
        "name": "eve_retrieval_retrieve",
        "content": "RETRIEVAL_PAYLOAD_ALREADY_IN_DOCUMENTS",
    },
    {"node": "agent", "role": "assistant", "content": "Scene of 2026-10-31, 3.13 % cloud."},
]


class _CapturingLLM:
    def __init__(self, result):
        self.result = result
        self.prompts = []

    def with_structured_output(self, _schema):
        return self

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        return self.result


@pytest.fixture
def fixed_clock(monkeypatch):
    monkeypatch.setattr(hd, "_utc_now", lambda: NOW)


def _detector(result):
    detector = HallucinationDetector()
    llm = _CapturingLLM(result)
    detector.llm_manager.get_client_for_model = lambda *_a, **_k: llm
    return detector, llm


@pytest.mark.no_db
async def test_tool_grounded_answer_is_judged_with_the_clock_and_the_tool_evidence(fixed_clock):
    detector, llm = _detector(HallucinationResult(label=0, reason="grounded"))

    await detector.detect(
        query="Find a recent Sentinel-2 scene over Rome",
        model_response="Scene of 2026-10-31, 3.13 % cloud.",
        docs="",
        tool_evidence=build_tool_evidence(AGENTIC_TRACE),
    )

    prompt = llm.prompts[0]
    assert "Current UTC date: 2026-11-12" in prompt
    assert "Current UTC year: 2026" in prompt
    assert "Current UTC timestamp: 2026-11-12T09:30:00+00:00" in prompt
    assert "<<<TOOL RESULTS>>>" in prompt
    assert "[tool: search_stac]" in prompt
    assert "2026-10-31T10:10:29Z" in prompt
    assert '"eo:cloud_cover": 3.13' in prompt
    # JSON braces reach the model as they are, not doubled.
    assert '{"type": "FeatureCollection"' in prompt
    assert "{{" not in prompt
    # The retrieval output is already in the documents; the user turn is not tool evidence.
    assert "RETRIEVAL_PAYLOAD_ALREADY_IN_DOCUMENTS" not in prompt
    assert "private user turn text" not in prompt
    # The clock comes before the material being judged.
    assert prompt.index("Current UTC date") < prompt.index("Question:")


@pytest.mark.no_db
async def test_without_a_trace_the_prompt_is_the_classic_one_plus_the_clock(fixed_clock):
    assert build_tool_evidence(None) == ""
    assert build_tool_evidence([]) == ""
    detector, llm = _detector(HallucinationResult(label=0, reason="grounded"))

    await detector.detect(query="What is NDVI?", model_response="an index", docs="doc text")

    prompt = llm.prompts[0]
    assert "Current UTC date: 2026-11-12" in prompt
    assert "TOOL RESULTS" not in prompt
    assert "Tool results" not in prompt
    assert prompt.rstrip().endswith("Question: What is NDVI?\nAnswer: an index\nsupporting documents: doc text")


@pytest.mark.no_db
def test_tool_evidence_respects_the_per_tool_and_total_caps():
    big = "x" * (TOOL_EVIDENCE_MAX_CHARS_PER_TOOL * 3)
    trace = [{"role": "tool", "name": f"tool_{i}", "content": big} for i in range(20)]

    evidence = build_tool_evidence(trace)

    assert len(evidence) <= TOOL_EVIDENCE_MAX_CHARS_TOTAL
    assert "[tool: tool_0]" in evidence
    assert "tool_19" not in evidence
    first_body = evidence.split("[tool: tool_0]\n", 1)[1].split("\n\n[tool:", 1)[0]
    assert len(first_body) <= TOOL_EVIDENCE_MAX_CHARS_PER_TOOL


@pytest.mark.no_db
def test_tool_evidence_redacts_secrets_and_tolerates_odd_entries():
    trace = [
        "not a dict",
        {"role": "tool", "name": "geocode", "content": "Authorization: Bearer abcdef0123456789 contact ann@example.org"},
        {"role": "tool", "name": "code_sandbox", "content": {"stdout": "42"}, "status": "error"},
        {"role": "tool", "content": "   "},
        {"role": "assistant", "content": "assistant text"},
    ]

    evidence = build_tool_evidence(trace)

    assert "abcdef0123456789" not in evidence
    assert "ann@example.org" not in evidence
    assert "[tool: geocode]" in evidence
    assert "[tool: code_sandbox] (error)" in evidence
    assert '"stdout": "42"' in evidence
    assert "assistant text" not in evidence


@pytest.mark.no_db
async def test_alternative_answer_prompt_carries_the_clock(fixed_clock):
    prompt = hallucination_answer_prompt("Which scenes {exist}?")
    assert "Current UTC date: 2026-11-12" in prompt
    assert "Which scenes {exist}?" in prompt

    detector, llm = _detector(LLMResult(response="answer"))
    await detector.llm_response("Which scenes {exist}?")
    assert "Current UTC date: 2026-11-12" in llm.prompts[0]
    assert "Which scenes {exist}?" in llm.prompts[0]


# ─── routes: the persisted trace reaches the judge ───────────────────────────


class _RecordingDetector:
    calls: list = []

    def __init__(self):
        self.llm_manager = MagicMock()
        self.llm_manager.get_client_for_model.return_value = GenericFakeChatModel(
            messages=iter([AIMessage(content="answer")])
        )

    async def detect(self, **kwargs):
        type(self).calls.append(kwargs)
        return 0, "grounded"

    async def run(self, **kwargs):
        type(self).calls.append(kwargs)
        return 0, "grounded", kwargs["query"], None, None, {"detect": 0.1}


@pytest.mark.parametrize("trace, expect_tool", [(AGENTIC_TRACE, True), (None, False)])
@pytest.mark.parametrize("route", ["hallucination", "stream-hallucination"])
async def test_routes_pass_the_trace_tool_results_to_the_judge(
    async_client, monkeypatch, route, trace, expect_tool
):
    detector = type("_Detector", (_RecordingDetector,), {"calls": []})
    monkeypatch.setattr("src.routers.message.HallucinationDetector", detector)
    monkeypatch.setattr(
        "src.routers.message.setup_rag_and_context",
        AsyncMock(return_value=("", [], {}, [])),
    )
    user, token = await create_test_user_and_token()
    conversation = message = None
    try:
        conversation = Conversation(user_id=user.id, name="hallucination-tool-evidence")
        await conversation.save()
        message = await Message.create(
            conversation_id=conversation.id,
            input="Find a recent Sentinel-2 scene over Rome",
            output="Scene of 2026-10-31, 3.13 % cloud.",
            documents=[],
            use_rag=False,
            metadata={},
            trace=trace,
            request_input=GenerationRequest(query="Find a recent Sentinel-2 scene", llm_type="main"),
        )
        response = await async_client.post(
            f"/conversations/{conversation.id}/messages/{message.id}/{route}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200, response.text
        assert len(detector.calls) == 1
        evidence = detector.calls[0].get("tool_evidence", "")
        if expect_tool:
            assert "[tool: search_stac]" in evidence
            assert "2026-10-31T10:10:29Z" in evidence
        else:
            assert evidence == ""
    finally:
        await cleanup_models([doc for doc in (user, conversation, message) if doc is not None])
