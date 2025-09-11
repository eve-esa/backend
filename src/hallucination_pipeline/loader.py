from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from .config import CONFIG
import replicate
import os
from dotenv import load_dotenv
from .utils import remove_think_content

load_dotenv()


def load_model(module_name: str):
    """
    Load a Hugging Face language model pipeline with parameters defined in the CONFIG dictionary.

    Args:
        module_name (str): The name of the module (e.g., "generation", "hallucination") whose model
                           parameters are defined in `src.config.CONFIG`.

    Returns:
        transformers.pipeline: A Hugging Face text-generation pipeline initialized with the specified model.
    """
    if module_name not in CONFIG:
        raise ValueError(f"Module '{module_name}' not found in CONFIG")

    params = CONFIG[module_name]

    required_keys = ["model_id", "max_new_tokens", "temperature", "top_p"]
    for key in required_keys:
        if key not in params:
            raise KeyError(
                f"Missing required config key '{key}' in CONFIG['{module_name}']"
            )

    model_id = params["model_id"]
    load_in_4bit = params.get("load_in_4bit", True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype="auto", load_in_4bit=load_in_4bit
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=params["max_new_tokens"],
        # temperature=params["temperature"],
        # top_p=params["top_p"],
        # do_sample=True
        pad_token_id=tokenizer.pad_token_id,
    )

    return pipe


class deepseek_replicate:
    def __init__(self, model_name: str = None, api_token: str = None):
        self.api_token = api_token or os.getenv("REPLICATE_API_TOKEN")
        self.model_name = model_name or os.getenv("model_name")

        if not self.api_token:
            raise ValueError("REPLICATE_API_TOKEN not provided.")
        if not self.model_name:
            raise ValueError("model_name not provided.")

        self.client = replicate.Client(api_token=self.api_token)

    def forward(self, prompt: str) -> str:
        output = self.client.run(self.model_name, input={"prompt": prompt})
        filtered_output = remove_think_content("".join(output))
        return filtered_output

    def __call__(self, prompt: str) -> str:
        return self.forward(prompt)
