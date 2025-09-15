from langchain_core.output_parsers import PydanticOutputParser
from .config import PROMPTS
from .schemas import self_reflect_schema
from .utils import *


async def regenerate_answer(
    model, hallucination_output: dict, docs: str
) -> self_reflect_schema:
    """
    Generate a grounded answer to a user query using a LLM pipeline and provided context.
    Retries up to 5 times if the output cannot be parsed into the expected schema.

    Args:
        model: LLM model
        hallucination_output (dict): Includes 'question', 'answer', and 'soft_labels'.
        docs (str): Supporting documents or context.

    Returns:
        self_reflect_schema: Parsed self-reflection output.
    """
    parser = PydanticOutputParser(pydantic_object=self_reflect_schema)
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
            PROMPTS["self_reflect_template"],
            {
                "question": question,
                "answer": answer,
                "hallucinated_spans": hallucinated_spans,
                "docs": docs,
                "format_instructions": format_instructions,
            },
        )

        response = await call_model(model, prompt)

        try:
            parsed_output = parser.parse(response)
            return parsed_output
        except Exception as e:
            print(f"[Regenerating Answer Attempt {attempt+1}] Parsing failed: {e}\n")

    raise ValueError("Failed to produce valid self-reflection output after 5 attempts.")
