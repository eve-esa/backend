from langchain_core.output_parsers import PydanticOutputParser
from .config import PROMPTS
from .schemas import hallucination_schema
from .utils import *


async def detect_hallucination(
    model, generation_output, docs: str
) -> hallucination_schema:
    """
    Detects hallucinations in a generated output using a Pydantic schema.
    Retries up to 5 times if the output format is invalid.

    Args:
        model (Pipeline): Hugging Face text-generation pipeline (hallucination detector).
        generation_output (dict): Output from the generation module with 'question' and 'answer'.
        docs (str): The original supporting documents.

    Returns:
        hallucination_schema: Parsed hallucination detection result.
    """
    parser = PydanticOutputParser(pydantic_object=hallucination_schema)
    format_instructions = parser.get_format_instructions()

    for attempt in range(5):
        prompt = safe_prompt_format(
            PROMPTS["hallucination_template"],
            {
                "question": generation_output.question,
                "answer": generation_output.answer,
                "docs": docs,
                "format_instructions": format_instructions,
            },
        )

        response = await call_model(model, prompt)

        try:
            parsed_output = parser.parse(response)
            return parsed_output

        except Exception as e:
            print(
                f"[Detecting Hallucination Attempt {attempt + 1}] Parsing failed: {e}\n"
            )

    raise ValueError(
        " Failed to produce valid hallucination detection output after attempts."
    )
