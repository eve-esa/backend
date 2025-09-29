from langchain_core.output_parsers import PydanticOutputParser
from .config import PROMPTS
from .schemas import generation_schema
from .utils import *


def generate_answer(model, question: str, docs: str) -> generation_schema:
    """
    Generate a grounded answer to a user query using a LLM pipeline and provided context.
    Retries up to 5 times if the output format is invalid.

    Args:
        model (Pipeline): Hugging Face text-generation pipeline instance.
        question (str): The user query.
        docs (str): Supporting documents or context.

    Returns:
        generation_schema: Parsed generation output.
    """
    parser = PydanticOutputParser(pydantic_object=generation_schema)
    format_instructions = parser.get_format_instructions()

    for attempt in range(5):
        prompt = safe_prompt_format(
            PROMPTS["generation_template"],
            {
                "question": question,
                "docs": docs,
                "format_instructions": format_instructions,
            },
        )

        response = call_model(model, prompt)
        # print(f"[Attempt {attempt+1}] Raw response:\n{response}\n")

        try:
            parsed_output = parser.parse(response)
            return parsed_output
        except Exception as e:
            print(f"[Attempt {attempt + 1}] Parsing failed: {e}\n")

    raise ValueError("Failed to produce valid generation output after 5 attempts.")
