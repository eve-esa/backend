from typing import List
import time
import logging
from .hallucination import detect_hallucination
from .rewritting import rewrite_query
from .reflection import regenerate_answer
from .ranking import rank_output
from .utils import filter_docs, vector_db_retrieve_context
from .schemas import generation_schema


async def run_hallucination_loop(
    model,
    docs: str,
    generation_response: generation_schema,
    collection_ids: List[str],
    per_span_reprompting: bool = True,
    prob_threshold: float = 0.5,
):
    overall_start = time.perf_counter()

    # Detection timing
    detection_start = time.perf_counter()
    hallucination_response = await detect_hallucination(
        model, generation_response, docs
    )
    detection_latency = time.perf_counter() - detection_start
    has_spans = any(
        span.get("prob", 0.0) > prob_threshold
        for span in hallucination_response.soft_labels or []
    )
    if not has_spans:
        return (
            {
                "final_answer": generation_response.answer,
                "generation_response": generation_response,
                "hallucination_response": hallucination_response,
                "rewrite_response": None,
                "reflected_response": None,
                "ranked_output": None,
                "docs": docs,
            },
            {
                "detection_latency": detection_latency,
                "span_reprompting_latency": None,
                "query_rewriting_latency": None,
                "regeneration_latency": None,
                "overall_latency": time.perf_counter() - overall_start,
            },
        )

    rewritten_response = None
    span_reprompting_latency = None
    query_rewriting_latency = None
    if per_span_reprompting:
        span_loop_start = time.perf_counter()
        for soft in hallucination_response.soft_labels:
            if soft.get("prob", 0.0) < prob_threshold:
                continue
            response_copy = hallucination_response.model_copy()
            response_copy.soft_labels = [soft]
            rewritten_response = await rewrite_query(model, response_copy)
            new_docs = await vector_db_retrieve_context(
                rewritten_response.rewritten_question, collection_ids
            )
            docs = filter_docs(docs, new_docs)
        span_reprompting_latency = time.perf_counter() - span_loop_start
    else:
        rewrite_start = time.perf_counter()
        rewritten_response = await rewrite_query(model, hallucination_response)
        new_docs = await vector_db_retrieve_context(
            rewritten_response.rewritten_question, collection_ids
        )
        docs = filter_docs(docs, new_docs)
        query_rewriting_latency = time.perf_counter() - rewrite_start

    # Regeneration timing
    regeneration_start = time.perf_counter()
    reflected_response = await regenerate_answer(model, hallucination_response, docs)
    regeneration_latency = time.perf_counter() - regeneration_start
    ranked_response = await rank_output(
        model,
        generation_response.question,
        generation_response.answer,
        reflected_response.answer,
        docs,
    )
    final_answer = (
        generation_response.answer
        if ranked_response.answer_a_score > ranked_response.answer_b_score
        else reflected_response.answer
    )

    return (
        {
            "final_answer": final_answer,
            "generation_response": generation_response,
            "hallucination_response": hallucination_response,
            "rewrite_response": rewritten_response,
            "reflected_response": reflected_response,
            "ranked_output": ranked_response,
            "docs": docs,
        },
        {
            "detection_latency": detection_latency,
            "span_reprompting_latency": span_reprompting_latency,
            "query_rewriting_latency": query_rewriting_latency,
            "regeneration_latency": regeneration_latency,
            "overall_latency": time.perf_counter() - overall_start,
        },
    )
