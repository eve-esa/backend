from langchain_core.output_parsers import PydanticOutputParser
from .config import PROMPTS
from .schemas import rewrite_schema
from .utils import *


async def rewrite_query(model, hallucination_output) -> rewrite_schema:
    """
    Rewrite a user query to improve factuality or context alignment using retrieved documents.
    Retries up to 5 times if output parsing fails.

    Args:
        model (Pipeline): Hugging Face text-generation pipeline (reprompting module).
        hallucination_output: Output from hallucination detection containing question, answer, and soft_labels.

    Returns:
        rewrite_schema: Parsed rewritten query.
    """
    parser = PydanticOutputParser(pydantic_object=rewrite_schema)
    format_instructions = parser.get_format_instructions()

    question = hallucination_output.question
    answer = hallucination_output.answer
    soft_labels = hallucination_output.soft_labels

    hallucinated_spans = [
        {"span": question[entry["start"] : entry["end"]], "reason": entry["reason"]}
        for entry in soft_labels
        if entry["prob"] > 0.5
    ]

    for attempt in range(5):
        prompt = safe_prompt_format(
            PROMPTS["rewriting_template"],
            {
                "question": question,
                "answer": answer,
                "hallucinated_spans": hallucinated_spans,
                "format_instructions": format_instructions,
            },
        )

        response = await call_model(model, prompt)

        try:
            parsed_output = parser.parse(response)
            return parsed_output
        except Exception as e:
            print(f"[Rewriting Attempt {attempt+1}] Parsing failed: {e}\n")

    raise ValueError("Failed to produce valid rewritten query output after 5 attempts.")
