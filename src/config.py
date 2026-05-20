# src/config.py
import json
import logging
import os
import sys
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

load_dotenv(override=True)

# ENV VARIABLES
QDRANT_URL = os.getenv("QDRANT_URL").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY").strip()
SATCOM_QDRANT_URL = os.getenv("SATCOM_QDRANT_URL").strip()
SATCOM_QDRANT_API_KEY = os.getenv("SATCOM_QDRANT_API_KEY").strip()
DEEPINFRA_API_TOKEN = os.getenv("DEEPINFRA_API_TOKEN", "").strip()
SILICONFLOW_API_TOKEN = os.getenv("SILICONFLOW_API_TOKEN", "").strip()
SATCOM_RUNPOD_API_KEY = os.getenv("SATCOM_RUNPOD_API_KEY", "").strip()

# Main and Fallback LLM URLs (OpenAI-compatible format)
MAIN_MODEL_URL = os.getenv("MAIN_MODEL_URL", "").strip()
FALLBACK_MODEL_URL = os.getenv("FALLBACK_MODEL_URL", "").strip()

MAIN_MODEL_API_KEY = os.getenv("MAIN_MODEL_API_KEY", "").strip()
FALLBACK_MODEL_API_KEY = os.getenv("FALLBACK_MODEL_API_KEY", "").strip()

MAIN_MODEL_NAME = os.getenv("MAIN_MODEL_NAME", "eve-esa/eve_v0.1").strip()
FALLBACK_MODEL_NAME = os.getenv("FALLBACK_MODEL_NAME", "mistral-medium-latest").strip()

MODEL_TIMEOUT = int(os.getenv("MODEL_TIMEOUT", 13))

EMBEDDING_URL = os.getenv(
    "EMBEDDING_URL", "https://api.deepinfra.com/v1/openai"
).strip()
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "").strip()

EMBEDDING_FALLBACK_URL = os.getenv(
    "EMBEDDING_FALLBACK_URL", "https://api.inference.net/v1"
).strip()
EMBEDDING_FALLBACK_API_KEY = os.getenv("EMBEDDING_FALLBACK_API_KEY", "").strip()

SATCOM_SMALL_MODEL_NAME = os.getenv(
    "SATCOM_SMALL_MODEL_NAME", "esa-sceva/satcom-chat-8b"
).strip()
SATCOM_LARGE_MODEL_NAME = os.getenv(
    "SATCOM_LARGE_MODEL_NAME", "esa-sceva/satcom-chat-70b"
).strip()
SATCOM_SMALL_BASE_URL = os.getenv("SATCOM_SMALL_BASE_URL", "").strip()
SATCOM_LARGE_BASE_URL = os.getenv("SATCOM_LARGE_BASE_URL", "").strip()

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

SCRAPING_DOG_API_KEY = os.getenv("SCRAPING_DOG_API_KEY", "").strip()

# Optional Redis URL for cross-process cancel/pubsub
REDIS_URL = os.getenv("REDIS_URL", "").strip()

# Langfuse observability
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "http://localhost:3000").strip()

# ─── Agentic pipeline configuration ───────────────────────────────────────────
# MODEL_TIMEOUT (defined above) is the per-step answer generation timeout used
# by the streaming agentic graph. The vars below are specific to the agentic
# pipeline and its MCP tool integrations.

# Override the LLM type for the agentic graph.  Set to "fallback" to force
# Mistral (reliable tool use) regardless of the request's llm_type.
AGENTIC_LLM_TYPE = os.getenv("AGENTIC_LLM_TYPE", "").strip() or None

# AWS Cognito credentials for AgentCore MCP server authentication.
AGENTCORE_TOKEN_URL = os.getenv("AGENTCORE_TOKEN_URL", "").strip()
AGENTCORE_CLIENT_ID = os.getenv("AGENTCORE_CLIENT_ID", "").strip()
AGENTCORE_CLIENT_SECRET = os.getenv("AGENTCORE_CLIENT_SECRET", "").strip()
MCP_PROXY_BASE_URL = os.getenv("MCP_PROXY_BASE_URL", "").strip()
MCP_PROXY_INTERNAL_BASE_URL = os.getenv("MCP_PROXY_INTERNAL_BASE_URL", "").strip()
# OpenAI-compatible proxy — upstream base URL (e.g. "https://example.com") and optional API key
OPENAI_PROXY_UPSTREAM_URL = os.getenv("OPENAI_PROXY_UPSTREAM_URL", "").strip()
OPENAI_PROXY_API_KEY = os.getenv("OPENAI_PROXY_API_KEY", "").strip()
# ──────────────────────────────────────────────────────────────────────────────


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
            loaded = yaml.safe_load(file) or {}
            self.config = loaded if isinstance(loaded, dict) else {}
        self._apply_token_rate_limit_env_overrides()

    @staticmethod
    def _parse_bool_env(env_name: str) -> Optional[bool]:
        raw = os.getenv(env_name)
        if raw is None:
            return None
        value = raw.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
        logging.warning("Ignoring invalid boolean value for %s.", env_name)
        return None

    @staticmethod
    def _parse_int_env(env_name: str) -> Optional[int]:
        raw = os.getenv(env_name)
        if raw is None or not raw.strip():
            return None
        try:
            return int(raw.strip())
        except ValueError:
            logging.warning("Ignoring invalid integer value for %s.", env_name)
            return None

    @staticmethod
    def _parse_json_object_env(env_name: str) -> Optional[Dict[str, Any]]:
        raw = os.getenv(env_name)
        if raw is None or not raw.strip():
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logging.warning("Ignoring invalid JSON value for %s.", env_name)
            return None
        if not isinstance(parsed, dict):
            logging.warning("Ignoring non-object JSON value for %s.", env_name)
            return None
        return parsed

    @staticmethod
    def _merge_flat_group_env(
        groups: Dict[str, Any], env_prefix: str, group_key: str, default_label: str
    ) -> None:
        max_tokens = Config._parse_int_env(f"{env_prefix}_TOKENS")
        period_months = Config._parse_int_env(f"{env_prefix}_PERIOD_MONTHS")
        if max_tokens is None and period_months is None:
            return

        existing = groups.get(group_key)
        group = dict(existing) if isinstance(existing, dict) else {}
        group.setdefault("label", default_label)
        if max_tokens is not None:
            group["max_tokens"] = max_tokens
        if period_months is not None:
            group["period_months"] = period_months
        groups[group_key] = group

    def _apply_token_rate_limit_env_overrides(self) -> None:
        token_cfg = self.config.get("token_rate_limit")
        token_cfg = dict(token_cfg) if isinstance(token_cfg, dict) else {}

        enabled = self._parse_bool_env("TOKEN_RATE_LIMIT_ENABLED")
        if enabled is not None:
            token_cfg["enabled"] = enabled

        default_group = os.getenv("TOKEN_RATE_LIMIT_DEFAULT_GROUP")
        if default_group is not None and default_group.strip():
            token_cfg["default_group"] = default_group.strip()

        aliases = token_cfg.get("aliases")
        aliases = dict(aliases) if isinstance(aliases, dict) else {}
        aliases_override = self._parse_json_object_env("TOKEN_RATE_LIMIT_ALIASES")
        if aliases_override is not None:
            aliases = {str(key): str(value) for key, value in aliases_override.items()}
        if aliases:
            token_cfg["aliases"] = aliases

        groups = token_cfg.get("groups")
        groups = dict(groups) if isinstance(groups, dict) else {}
        groups_override = self._parse_json_object_env("TOKEN_RATE_LIMIT_GROUPS")
        if groups_override is not None:
            groups = groups_override

        self._merge_flat_group_env(groups, "FREE", "eve_free", "Free")
        self._merge_flat_group_env(groups, "PRO", "eve_standard", "Pro")
        self._merge_flat_group_env(groups, "PRO_PLUS", "eve_advanced", "Pro+")
        self._merge_flat_group_env(groups, "ULTRA", "eve_enterprise", "Ultra")

        if groups:
            token_cfg["groups"] = groups
        else:
            token_cfg.pop("groups", None)

        if token_cfg:
            self.config["token_rate_limit"] = token_cfg

    def get(self, *keys, default=None):
        """Generalized method to get a value from a nested dictionary."""
        value = self.config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

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


# Expose a module-level config instance for convenient imports
# Allow overriding the config file location via EVE_CONFIG_PATH
CONFIG_PATH = os.getenv("EVE_CONFIG_PATH", "config.yaml")
config = Config(CONFIG_PATH)
