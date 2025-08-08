# src/config.py
import yaml
import os
from dotenv import load_dotenv
import runpod
import logging
import sys

load_dotenv(override=True)

# ENV VARIABLES
QDRANT_URL = os.getenv("QDRANT_URL").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY").strip()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY").strip()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN").strip()
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY").strip()


runpod.api_key = RUNPOD_API_KEY


def configure_logging(level=logging.INFO):
    """Configure logging for the entire application."""
    # Check if already configured to avoid duplicate handlers
    if not logging.getLogger().hasHandlers():
        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        root_logger.addHandler(console_handler)


class Config:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as file:
            raw_config = yaml.safe_load(file)

        # Expand environment variables in the loaded YAML
        self.config = self._expand_env(raw_config)

    def get(self, *keys, default=None):
        """Generalized method to get a value from a nested dictionary."""
        value = self.config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def _expand_env(self, value):
        """Recursively expand ${VAR} environment placeholders in config values."""
        if isinstance(value, dict):
            return {k: self._expand_env(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._expand_env(v) for v in value]
        if isinstance(value, str):
            try:
                return os.path.expandvars(value)
            except Exception:
                return value
        return value

    # MCP
    def get_mcp_server_url(self):
        return self.get("mcp", "server_url")

    def get_mcp_headers(self):
        return self.get("mcp", "headers", default={}) or {}

    def get_instruct_llm_id(self):
        return self.get("runpod", "instruct_llm", "id")

    def get_instruct_llm_timeout(self):
        return self.get("runpod", "instruct_llm", "timeout")

    def get_indus_embedder_id(self):
        return self.get("runpod", "indus_embedder", "id")

    def get_indus_embedder_timeout(self):
        return self.get("runpod", "indus_embedder", "timeout")

    def get_completion_llm_id(self):
        return self.get("runpod", "instruct_llm", "id")

    def get_completion_llm_timeout(self):
        return self.get("runpod", "instruct_llm", "llm")
