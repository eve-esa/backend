"""Platform-provided LLM catalog for agentic EVE."""

from typing import List

from src.schemas.custom_model import PlatformModel

PLATFORM_MODELS: List[PlatformModel] = [
    PlatformModel(
        id="eve-instruct",
        llm_type="main",
        display_name="EVE-Instruct",
        description="Default EVE instruction-tuned model hosted by the platform.",
    ),
]


def list_platform_models() -> List[PlatformModel]:
    return list(PLATFORM_MODELS)
