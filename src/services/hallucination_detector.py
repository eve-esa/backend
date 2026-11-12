import json
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional, Tuple
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from src.core.llm_manager import LLMManager
from src.utils.redaction import redact_secrets
from src.utils.template_loader import get_template

import logging
import time

logger = logging.getLogger(__name__)


def _format_prompt(template: str, **values: str) -> str:
    """Interpolate a prompt without treating user text as format fields.

    ``str.format`` never parses the substituted values, so braces in user text
    or in JSON tool output reach the model unchanged.
    """
    return template.format(**{key: str(value or "") for key, value in values.items()})


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def clock_context(now: Optional[datetime] = None) -> str:
    """The ``## Clock`` section, worded as ``agents.graphs.utils.clock_context``.

    The agents build the same block on every model call; the judge needs it
    too, or a date in the current year reads as a future one. Kept local
    because the agents package is not a dependency of every code path here.
    """
    now = now or _utc_now()
    return (
        "## Clock\n"
        f"Current UTC timestamp: {now.isoformat(timespec='seconds')}\n"
        f"Current UTC date: {now.date().isoformat()}\n"
        f"Current UTC year: {now.year}\n"
        "Use this clock for the current date/year and relative dates, "
        "not the knowledge cutoff or dates in conversation history. "
        "For local requests, interpret this instant in the requested "
        "location's timezone.\n"
    )


# Tool results of the judged turn, as evidence next to the documents. One STAC
# page or sandbox run fits in the per-tool budget; the total keeps the judge
# prompt well inside every model window.
TOOL_EVIDENCE_MAX_CHARS_PER_TOOL = 4000
TOOL_EVIDENCE_MAX_CHARS_TOTAL = 12000
_TRUNCATED = " [truncated]"


def _is_retrieval_tool(name: str) -> bool:
    """The retrieval tool's documents already reach the judge as ``docs``."""
    lowered = name.lower()
    return lowered == "retrieve" or lowered.endswith("_retrieve")


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit <= len(_TRUNCATED):
        return text[:limit]
    return text[: limit - len(_TRUNCATED)] + _TRUNCATED


def build_tool_evidence(trace: Optional[Iterable[Any]]) -> str:
    """Tool outputs from a persisted ``Message.trace``, labelled and trimmed.

    Only ``role == "tool"`` entries are read, so user and assistant text in the
    trace never reaches the judge from here. An offloaded result stays as its
    stored stub with the preview: nothing is fetched. Secrets and e-mail
    addresses are redacted. Empty when there is no tool output (classic turns).
    """
    if not trace:
        return ""
    parts = []
    used = 0
    for entry in trace:
        if not isinstance(entry, dict) or entry.get("role") != "tool":
            continue
        name = str(entry.get("name") or "tool")
        if _is_retrieval_tool(name):
            continue
        content = entry.get("content")
        if content is None:
            continue
        if not isinstance(content, str):
            try:
                content = json.dumps(content, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                content = str(content)
        content = content.strip()
        if not content:
            continue
        status = " (error)" if entry.get("status") == "error" else ""
        label = f"[tool: {name}]{status}\n"
        separator = "\n\n" if parts else ""
        room = TOOL_EVIDENCE_MAX_CHARS_TOTAL - used - len(separator) - len(label)
        if room <= len(_TRUNCATED):
            break
        body = _clip(
            redact_secrets(content), min(TOOL_EVIDENCE_MAX_CHARS_PER_TOOL, room)
        )
        piece = f"{separator}{label}{body}"
        parts.append(piece)
        used += len(piece)
    return "".join(parts)


def _tool_evidence_block(tool_evidence: str) -> str:
    """Wrap tool results in markers, or return empty when the turn had none."""
    text = (tool_evidence or "").strip()
    if not text:
        return ""
    return (
        "Tool results of this turn are wrapped in <<<TOOL RESULTS>>> and "
        "<<<END TOOL RESULTS>>>: outputs of the tools the assistant called while "
        "answering (for example catalogue searches or code runs), each labelled "
        "with the tool name. Treat them as evidence, like the supporting "
        "documents: a fact taken from them is grounded, not invented.\n"
        f"<<<TOOL RESULTS>>>\n{text}\n<<<END TOOL RESULTS>>>\n"
    )


def hallucination_answer_prompt(query: str) -> str:
    """The alternative-answer prompt, with the clock."""
    template = get_template(
        "llm_answer_template", filename="hallucination_detector.yaml"
    )
    return _format_prompt(template, query=query, clock=clock_context().rstrip())


def _conversation_block(conversation: str) -> str:
    """Wrap prior dialogue in markers, or return empty when there is no history.

    The markers are the only boundary between the instructions and the summary
    plus recent turns. A first turn stays unmarked.
    """
    text = (conversation or "").strip()
    if not text:
        return ""
    return f"<<<CONVERSATION>>>\n{text}\n<<<END CONVERSATION>>>\n"


async def load_new_turn_history_prefix(
    conversation_id: str,
    *,
    request: Optional[object] = None,
    before_message: Optional[object] = None,
) -> str:
    """Return the history prefix a new agentic turn prepends to its instruction.

    A new turn builds that string with ``_get_conversation_history_from_db``
    and ``AgentGraph.format_history`` (the lead-in inside ``instruction_text``,
    before the YAML system prompt). ``before_message`` is the turn under
    review: the window is the prior messages only, the same ones a new turn
    would have seen when that message was the latest. This is not the
    checkpointed thread and not a retrieval requery.
    """
    try:
        from src.services.agents.core.registry import get_agent_graph
        from src.services.agents.core.runner import _resolve_agent_graph_type
        from src.services.generate_answer import _get_conversation_history_from_db

        history, summary = await _get_conversation_history_from_db(
            conversation_id, before_message=before_message
        )
        # None request resolves to AGENT_GRAPH_TYPE, the same default a new turn uses.
        prefix = get_agent_graph(_resolve_agent_graph_type(request)).format_history(
            history or [], summary
        )
        return prefix or ""
    except Exception:
        logger.warning(
            "Failed to load conversation history prefix for hallucination detection",
            exc_info=True,
        )
        return ""


class LLMResult(BaseModel):
    """Schema for llm  output"""

    response: str = Field(description="LLM response")


class RewriteResult(BaseModel):
    """Schema for rewritten query output"""

    question: str = Field(description="Original user question.")
    rewritten_question: str = Field(
        description="Rewritten question for factual accuracy."
    )


class HallucinationResult(BaseModel):
    """Schema for hallucination detection output"""

    label: int = Field(description="Binary label: 0 for factual, 1 for hallucination")
    reason: str = Field(description="Explanation for the classification")
    # confidence: float = Field(description="Confidence score")


class HallucinationDetector:
    def __init__(self):
        self.llm_manager = LLMManager()
        self.parser_hallucination = PydanticOutputParser(
            pydantic_object=HallucinationResult
        )
        self.parser_response = PydanticOutputParser(pydantic_object=LLMResult)

    async def rewrite_query(
        self,
        query: str,
        answer: str,
        reason: str,
        llm_type: Optional[str] = None,
        conversation: str = "",
    ):
        rewriting_template = get_template(
            "rewriting_template", filename="hallucination_detector.yaml"
        )
        prompt = _format_prompt(
            rewriting_template,
            question=query,
            answer=answer,
            reason=reason,
            conversation=_conversation_block(conversation),
        )
        try:
            llm = self.llm_manager.get_client_for_model(llm_type)
            structured_model = llm.with_structured_output(RewriteResult)
            response = await structured_model.ainvoke(prompt)
        except Exception:
            llm = self.llm_manager.get_fallback_llm()
            structured_model = llm.with_structured_output(RewriteResult)
            response = await structured_model.ainvoke(prompt)
        return response.question, response.rewritten_question

    async def detect(
        self,
        query: str,
        model_response: str,
        docs: str = "",
        llm_type: Optional[str] = None,
        conversation: str = "",
        tool_evidence: str = "",
    ) -> Tuple[int, str]:
        hallucination_binary_with_retrieval = get_template(
            "hallucination_binary_with_retrieval",
            filename="hallucination_detector.yaml",
        )

        prompt = _format_prompt(
            hallucination_binary_with_retrieval,
            question=query,
            answer=model_response,
            docs=docs,
            conversation=_conversation_block(conversation),
            clock=clock_context(),
            tool_evidence=_tool_evidence_block(tool_evidence),
        )
        try:
            llm = self.llm_manager.get_client_for_model(llm_type)
            structured_model = llm.with_structured_output(HallucinationResult)
            response = await structured_model.ainvoke(prompt)
        except Exception:
            llm = self.llm_manager.get_fallback_llm()
            structured_model = llm.with_structured_output(HallucinationResult)
            response = await structured_model.ainvoke(prompt)
        return response.label, response.reason

    async def llm_response(self, query: str, llm_type: Optional[str] = None) -> str:
        prompt = hallucination_answer_prompt(query)
        try:
            llm = self.llm_manager.get_client_for_model(llm_type)
            structured_model = llm.with_structured_output(LLMResult)
            response = await structured_model.ainvoke(prompt)
        except Exception:
            llm = self.llm_manager.get_fallback_llm()
            structured_model = llm.with_structured_output(LLMResult)
            response = await structured_model.ainvoke(prompt)
        return response.response

    async def run(
        self,
        query: str,
        model_response: str,
        docs: str = "",
        llm_type: Optional[str] = None,
        conversation: str = "",
        tool_evidence: str = "",
    ) -> Tuple[int, str, str, Optional[str], Optional[str], Dict[str, Optional[float]]]:
        t0 = time.perf_counter()
        label, reason = await self.detect(
            query=query,
            model_response=model_response,
            docs=docs,
            llm_type=llm_type,
            conversation=conversation,
            tool_evidence=tool_evidence,
        )
        detect_latency = time.perf_counter() - t0

        original_question = query
        if label != 1:
            latencies: Dict[str, Optional[float]] = {
                "detect": detect_latency,
                "rewrite": None,
                "final_answer": None,
            }
            return label, reason, original_question, None, None, latencies

        t1 = time.perf_counter()
        _, rewritten_question = await self.rewrite_query(
            query=query,
            answer=model_response,
            reason=reason,
            llm_type=llm_type,
            conversation=conversation,
        )
        rewrite_latency = time.perf_counter() - t1

        t2 = time.perf_counter()
        final_answer = await self.llm_response(rewritten_question, llm_type=llm_type)
        final_latency = time.perf_counter() - t2

        latencies = {
            "detect": detect_latency,
            "rewrite": rewrite_latency,
            "final_answer": final_latency,
        }
        return (
            label,
            reason,
            original_question,
            rewritten_question,
            final_answer,
            latencies,
        )
