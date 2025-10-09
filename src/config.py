# src/config.py
import yaml
import os
from dotenv import load_dotenv
import runpod
import logging
import sys
from typing import Dict, Any

load_dotenv(override=True)

# ENV VARIABLES
QDRANT_URL = os.getenv("QDRANT_URL").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY").strip()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY").strip()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN").strip()
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY").strip()
DEEPINFRA_API_TOKEN = os.getenv("DEEPINFRA_API_TOKEN", "").strip()
INFERENCE_API_KEY = os.getenv("INFERENCE_API_KEY", "").strip()
SILICONFLOW_API_TOKEN = os.getenv("SILICONFLOW_API_TOKEN", "").strip()

MONGO_HOST = os.getenv("MONGO_HOST", "localhost").strip()
MONGO_PORT = int(os.getenv("MONGO_PORT", 27017))
MONGO_USERNAME = os.getenv("MONGO_USERNAME", "").strip()
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD", "").strip()
MONGO_DATABASE = os.getenv("MONGO_DATABASE", "").strip()
MONGO_PARAMS = os.getenv("MONGO_PARAMS", "?authSource=admin").strip()

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY").strip()
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256").strip()
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", 15))
JWT_REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", 7))
JWT_AUDIENCE_ACCESS = os.getenv("JWT_AUDIENCE_ACCESS", "access").strip()
JWT_AUDIENCE_REFRESH = os.getenv("JWT_AUDIENCE_REFRESH", "refresh").strip()

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "").strip()
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "").strip()
EMAIL_FROM_ADDRESS = os.getenv("EMAIL_FROM_ADDRESS", "noreply@eve-ai.com").strip()
EMAIL_FROM_NAME = os.getenv("EMAIL_FROM_NAME", "EVE AI").strip()

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173").strip()

CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:5173").split(",")
]

FORGOT_PASSWORD_CODE_EXPIRE_MINUTES = int(
    os.getenv("FORGOT_PASSWORD_CODE_EXPIRE_MINUTES", 10)
)

WILEY_AUTH_TOKEN = os.getenv("WILEY_AUTH_TOKEN", "").strip()
IS_PROD = os.getenv("IS_PROD", "").strip().lower() == "true"

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

    def get_mistral_model(self):
        return self.get("mistral", "model")

    def get_mistral_timeout(self):
        return self.get("mistral", "timeout")

    # MCP
    def get_mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """Get dictionary of configured MCP servers for MultiServerMCPClient."""
        servers = self.get("mcp", "servers", default={})
        if not servers:
            # Fallback to legacy single server configuration
            legacy_url = self.get("mcp", "server_url")
            legacy_headers = self.get("mcp", "headers", default={})
            if legacy_url:
                return {
                    "legacy-server": {
                        "url": legacy_url,
                        "transport": "streamable_http",
                        "headers": legacy_headers,
                        "enabled": True,
                    }
                }
        return servers

    def get_mcp_server_url(self):
        """Legacy method for backward compatibility."""
        servers = self.get_mcp_servers()
        if servers:
            # Get the first server's URL
            first_server = next(iter(servers.values()))
            return first_server.get("url")
        return None

    def get_mcp_headers(self):
        """Legacy method for backward compatibility."""
        servers = self.get_mcp_servers()
        if servers:
            # Get the first server's headers
            first_server = next(iter(servers.values()))
            return first_server.get("headers", {})
        return {}

    def get_reranker_id(self):
        return self.get("runpod", "reranker", "id")

    def get_reranker_timeout(self):
        return self.get("runpod", "reranker", "timeout")


# Expose a module-level config instance for convenient imports
# Allow overriding the config file location via EVE_CONFIG_PATH
CONFIG_PATH = os.getenv("EVE_CONFIG_PATH", "config.yaml")
config = Config(CONFIG_PATH)
