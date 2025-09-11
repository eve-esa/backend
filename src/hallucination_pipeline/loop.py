from typing import List
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
    hallucination_response = detect_hallucination(model, generation_response, docs)
    has_spans = any(
        span.get("prob", 0.0) > prob_threshold
        for span in hallucination_response.soft_labels or []
    )
    if not has_spans:
        return {
            "final_answer": generation_response.answer,
            "generation_response": generation_response,
            "hallucination_response": hallucination_response,
            "rewrite_response": None,
            "reflected_response": None,
            "ranked_output": None,
            "docs": docs,
        }

    rewritten_response = None
    if per_span_reprompting:
        for soft in hallucination_response.soft_labels:
            if soft.get("prob", 0.0) < prob_threshold:
                continue
            response_copy = hallucination_response.model_copy()
            response_copy.soft_labels = [soft]
            rewritten_response = rewrite_query(model, response_copy)
            new_docs = await vector_db_retrieve_context(
                rewritten_response.rewritten_question, collection_ids
            )
            docs = filter_docs(docs, new_docs)
    else:
        rewritten_response = rewrite_query(model, hallucination_response)
        new_docs = await vector_db_retrieve_context(
            rewritten_response.rewritten_question, collection_ids
        )
        docs = filter_docs(docs, new_docs)

    reflected_response = regenerate_answer(model, hallucination_response, docs)
    ranked_response = rank_output(
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

    return {
        "final_answer": final_answer,
        "generation_response": generation_response,
        "hallucination_response": hallucination_response,
        "rewrite_response": rewritten_response,
        "reflected_response": reflected_response,
        "ranked_output": ranked_response,
        "docs": docs,
    }
