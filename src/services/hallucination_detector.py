from typing import List, Dict, Optional, Tuple
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from src.constants import DEFAULT_EMBEDDING_MODEL
from src.core.llm_manager import LLMManager
from src.core.vector_store_manager import VectorStoreManager
from src.utils.template_loader import get_template

import logging
import time

logger = logging.getLogger(__name__)


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
        self.vector_store_manager = VectorStoreManager()
        self.llm_manager = LLMManager()
        self.parser_hallucination = PydanticOutputParser(
            pydantic_object=HallucinationResult
        )
        self.parser_response = PydanticOutputParser(pydantic_object=LLMResult)

    async def retrieve_context_from_qdrant(
        self,
        collection_names: List[str],
        query: str,
        k: int = 10,
        score_threshold: float = 0.1,
        query_filter: Optional[Dict] = None,
    ) -> tuple[List, Dict[str, Optional[float]]]:
        results, vs_latencies = (
            await self.vector_store_manager.retrieve_documents_with_latencies(
                collection_names=collection_names,
                query=query,
                k=k * 2,
                score_threshold=score_threshold,
                embeddings_model=self.embeddings_model,
                filters=query_filter,
                private_collections_map=None,
            )
        )

        if not results:
            logger.warning(f"No documents found for query: {query}")

        return results, vs_latencies

    async def rewrite_query(
        self, query: str, answer: str, reason: str, llm_type: Optional[str] = None
    ):
        rewriting_template = get_template(
            "rewriting_template", filename="hallucination_detector.yaml"
        )
        prompt = rewriting_template.format(question=query, answer=answer, reason=reason)
        try:
            llm = self.llm_manager.get_client_for_model(llm_type)
            structured_model = llm.with_structured_output(RewriteResult)  # type: ignore[attr-defined]
            response = await structured_model.ainvoke(prompt)
        except Exception:
            llm = self.llm_manager.get_mistral_model()
            structured_model = llm.with_structured_output(RewriteResult)  # type: ignore[attr-defined]
            response = await structured_model.ainvoke(prompt)
        return response.question, response.rewritten_question

    async def detect(
        self,
        query: str,
        model_response: str,
        collection_names: List[str],
        k: int = 10,
        score_threshold: float = 0.1,
        llm_type: Optional[str] = None,
    ) -> Tuple[int, str]:
        results, _latencies = await self.retrieve_context_from_qdrant(
            collection_names=collection_names,
            query=query,
            k=k,
            score_threshold=score_threshold,
        )
        hallucination_binary_with_retrieval = get_template(
            "hallucination_binary_with_retrieval",
            filename="hallucination_detector.yaml",
        )
        # Build a conservative docs string from payloads
        docs_texts: List[str] = []
        try:
            for doc in results:
                payload = getattr(doc, "payload", {}) or {}
                text_val = (
                    payload.get("content")
                    or payload.get("text")
                    or payload.get("page_content")
                    or ""
                )
                if text_val:
                    docs_texts.append(str(text_val))
        except Exception:
            pass

        prompt = hallucination_binary_with_retrieval.format(
            question=query,
            answer=model_response,
            docs="\n".join(docs_texts),
        )
        try:
            llm = self.llm_manager.get_client_for_model(llm_type)
            structured_model = llm.with_structured_output(HallucinationResult)  # type: ignore[attr-defined]
            response = await structured_model.ainvoke(prompt)
        except Exception:
            llm = self.llm_manager.get_mistral_model()
            structured_model = llm.with_structured_output(HallucinationResult)  # type: ignore[attr-defined]
            response = await structured_model.ainvoke(prompt)
        return response.label, response.reason

    async def llm_response(self, query: str, llm_type: Optional[str] = None) -> str:
        prompt = get_template(
            "llm_answer_template", filename="hallucination_detector.yaml"
        )
        try:
            llm = self.llm_manager.get_client_for_model(llm_type)
            structured_model = llm.with_structured_output(LLMResult)  # type: ignore[attr-defined]
            response = await structured_model.ainvoke(prompt.format(query=query))
        except Exception:
            llm = self.llm_manager.get_mistral_model()
            structured_model = llm.with_structured_output(LLMResult)  # type: ignore[attr-defined]
            response = await structured_model.ainvoke(prompt.format(query=query))
        return response.response

    async def run(
        self,
        query: str,
        model_response: str,
        collection_names: List[str],
        k: int = 10,
        score_threshold: float = 0.1,
        llm_type: Optional[str] = None,
    ) -> Tuple[int, str, str, Optional[str], Optional[str], Dict[str, Optional[float]]]:
        t0 = time.perf_counter()
        label, reason = await self.detect(
            query=query,
            model_response=model_response,
            collection_names=collection_names,
            k=k,
            score_threshold=score_threshold,
            llm_type=llm_type,
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
            query=query, answer=model_response, reason=reason, llm_type=llm_type
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
