# src/config.py
import yaml
import os
from dotenv import load_dotenv
import runpod
import logging


load_dotenv(override=True)

# ENV VARIABLES
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")


runpod.api_key = RUNPOD_API_KEY


# CONFIG YAML
import yaml


class Config:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

    def get(self, *keys, default=None):
        """Generalized method to get a value from a nested dictionary."""
        value = self.config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def get_completion_llm_id(self):
        return self.get("runpod", "completion_llm", "id")

    def get_completion_llm_timeout(self):
        return self.get("runpod", "completion_llm", "timeout")

    def get_instruct_llm_id(self):
        return self.get("runpod", "instruct_llm", "id")

    def get_instruct_llm_timeout(self):
        return self.get("runpod", "instruct_llm", "timeout")
