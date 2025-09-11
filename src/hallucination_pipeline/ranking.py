from langchain_core.output_parsers import PydanticOutputParser
from .schemas import ranking_schema
from .config import PROMPTS
from .utils import safe_prompt_format, call_model


def rank_output(model, question, answerA, answerB, docs) -> ranking_schema:
    """
    Score two generated answers to the same EO-related question.

    Args:
        model: HuggingFace or OpenAI-compatible LLM generation pipeline.
        qa_pair: Object with `question`, `answer_a`, and `answer_b` attributes.

    Returns:
        ranking_schema: Parsed structured scores and justifications.
    """
    parser = PydanticOutputParser(pydantic_object=ranking_schema)
    format_instructions = parser.get_format_instructions()

    question = question
    answer_a = answerA
    answer_b = answerB
    docs = docs

    for attempt in range(5):
        prompt = safe_prompt_format(
            PROMPTS["ranking_template"],
            {
                "question": question,
                "answer_a": answer_a,
                "answer_b": answer_b,
                "format_instructions": format_instructions,
            },
        )

        response = call_model(model, prompt)

        try:
            parsed_output = parser.parse(response)
            return parsed_output
        except Exception as e:
            print(f"[Ranking Attempt {attempt+1}] Parsing failed: {e}\n")

    raise ValueError("Failed to produce valid scoring output after 5 attempts.")
