# src/config.py
import json
import logging
import os
import sys
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

load_dotenv(override=True)


def getenv_or(name: str, default: str = "") -> str:
    """Read an env var, treating a blank value as absent.

    The infrastructure seeds intentionally-unused Secrets Manager secrets with a
    single space (infra/docs/RUNBOOK.md), because ECS cannot resolve a secret
    that has no version. ``os.getenv(name, default)`` returns that space, so the
    default never applies and the caller ends up with "" after stripping.
    """
    return (os.getenv(name) or "").strip() or default


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
FALLBACK_MODEL_NAME = os.getenv("FALLBACK_MODEL_NAME", "mistral-small-latest").strip()

MODEL_TIMEOUT = int(os.getenv("MODEL_TIMEOUT", 13))
AGENTIC_TIMEOUT = int(os.getenv("AGENTIC_TIMEOUT", 120))

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

EVE_JSC_BASE_URL = os.getenv("EVE_JSC_BASE_URL", "").strip()
EVE_JSC_MODEL_NAME = os.getenv("EVE_JSC_MODEL_NAME", "alias-eve").strip()
# No fallback to MAIN_MODEL_API_KEY: that is a RunPod key, and JSC is a different
# provider (Jülich), so inheriting it only turns a missing-config error into a
# confusing 401 from the upstream. Blank means unusable, and says so.
EVE_JSC_API_KEY = getenv_or("EVE_JSC_API_KEY")

# Ordered EVE endpoint chain, comma-separated llm_type names. Unconfigured
# entries are dropped at resolution time and "fallback" is always appended
# last, so a chain can never end without a last resort.
EVE_ENDPOINT_ORDER = os.getenv("EVE_ENDPOINT_ORDER", "eve_jsc,main,fallback").strip()
EVE_ENDPOINT_COOLDOWN_S = float(os.getenv("EVE_ENDPOINT_COOLDOWN_S", "120"))
# Per-endpoint first-token budgets. RunPod is serverless with 1-3 min cold
# starts; JSC is a warm vLLM. One MODEL_TIMEOUT cannot serve both.
EVE_JSC_TIMEOUT = int(os.getenv("EVE_JSC_TIMEOUT", MODEL_TIMEOUT))
MAIN_MODEL_TIMEOUT = int(os.getenv("MAIN_MODEL_TIMEOUT", "120"))

# Feature flags. A flag names the feature, never the environment it runs in.
# Self-registration is no longer one of them: whether strangers may create an
# account is now a realm / app-client setting at the identity provider, which is
# the only place that can actually open or close registration.
#
# On: GET /models lists the JSC-hosted EVE model first, and the frontend takes the
# first entry as its default. A blank EVE_JSC_BASE_URL wins over the flag: an
# endpoint that is not configured must not be offered, whatever the flag says.
FEATURE_JSC_MODEL = getenv_or("FEATURE_JSC_MODEL").lower() == "true"

# Off: POST /mcp-servers and PATCH /mcp-servers/{id} answer 404, as if the routes
# did not exist. Reading and deleting stay open, so a user can still see and drop
# what is already registered. Default on, because local compose and dev register
# servers; the environments that only ever serve the managed EVE retrieve server
# set it to "false". Treat only "false"/"0" (case-insensitively) as off.
FEATURE_MCP_SERVER_REGISTRATION = getenv_or(
    "FEATURE_MCP_SERVER_REGISTRATION", "true"
).lower() not in (
    "false",
    "0",
)

MONGO_HOST = os.getenv("MONGO_HOST", "localhost").strip()
MONGO_PORT = int(os.getenv("MONGO_PORT", 27017))
MONGO_USERNAME = os.getenv("MONGO_USERNAME", "").strip()
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD", "").strip()
MONGO_DATABASE = os.getenv("MONGO_DATABASE", "").strip()
MONGO_PARAMS = os.getenv("MONGO_PARAMS", "?authSource=admin").strip()

# ─── Identity provider (OIDC) ─────────────────────────────────────────────────
# The application knows an issuer, an audience and a JWT. Which product answers
# at that issuer (Keycloak locally, Cognito in the cloud) is not its business.
#
#   AUTH_ISSUER   exact value of the ``iss`` claim, and the value the discovery
#                 document must declare as its own ``issuer`` before its
#                 ``jwks_uri`` is trusted.
#   AUTH_CLIENT_ID  the public client the browser signs in with. Used as the
#                 expected audience: present in ``aud`` (Keycloak, via the realm
#                 audience mapper) or equal to the ``client_id`` claim (Cognito
#                 access tokens carry no ``aud`` at all).
#
# Both are required, the way JWT_SECRET_KEY used to be: an application that
# cannot name its issuer and its audience cannot verify a token, and failing at
# import is louder than accepting everything at runtime.
AUTH_ISSUER = os.getenv("AUTH_ISSUER").strip()
AUTH_CLIENT_ID = os.getenv("AUTH_CLIENT_ID").strip()
# Override only when the API audience differs from the browser client id.
AUTH_AUDIENCE = getenv_or("AUTH_AUDIENCE") or AUTH_CLIENT_ID
# Set when the container must reach the provider on an address the issuer does
# not use: local compose fetches from http://keycloak:8080 while the issuer
# stays http://localhost:8080 so browser and backend agree on one ``iss``.
AUTH_DISCOVERY_URL = getenv_or("AUTH_DISCOVERY_URL")
AUTH_JWKS_CACHE_TTL_SECONDS = int(getenv_or("AUTH_JWKS_CACHE_TTL_SECONDS", "3600"))
# Local compose only. Every deployed environment talks https to its provider.
AUTH_ALLOW_INSECURE_HTTP = getenv_or("AUTH_ALLOW_INSECURE_HTTP").lower() == "true"
# Attach a first OIDC identity to the pre-existing EVE account with the same
# verified email. Off means every first sign-in provisions a new account.
AUTH_LINK_BY_VERIFIED_EMAIL = getenv_or(
    "AUTH_LINK_BY_VERIFIED_EMAIL", "true"
).lower() == "true"
# TEMPORARY, deleted with src/routers/migration.py once the production migration
# window closes. Unset means the endpoint answers 503 rather than existing
# quietly: it reads legacy password hashes, so absent configuration is closed.
MIGRATION_SHARED_SECRET = getenv_or("MIGRATION_SHARED_SECRET")
# ──────────────────────────────────────────────────────────────────────────────

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

WILEY_AUTH_TOKEN = os.getenv("WILEY_AUTH_TOKEN", "").strip()

# Which build is running. The two halves are set at different moments and neither is ever
# required: unset means "unknown", never a crash.
#   APP_GIT_SHA is baked into the image at build time (Dockerfile ARG GIT_SHA), because the
#     image is built once on push to main and promoted by digest without a rebuild.
#   APP_VERSION is injected on the container at deploy time (deploy-ecs.yml), because no
#     version tag exists yet when the image is built.
APP_VERSION = os.getenv("APP_VERSION", "").strip() or "unknown"
APP_GIT_SHA = os.getenv("APP_GIT_SHA", "").strip() or "unknown"

# Which environment this is. Terraform has always known the answer (var.environment, one of
# dev/staging/prod, with no default), but it used to reach the container collapsed into a
# boolean IS_PROD, so dev and staging were indistinguishable from inside and /health could
# only ever answer "non-prod". The tri-state now travels intact and IS_PROD is derived from
# it, which is the direction that loses no information.
#
# Unknown values read as non-production on purpose: a typo must not silently grant a
# container production behaviour. Terraform validates the value at plan time as well.
_KNOWN_ENVIRONMENTS = ("dev", "staging", "prod")
APP_ENVIRONMENT = getenv_or("APP_ENVIRONMENT").lower()
if APP_ENVIRONMENT and APP_ENVIRONMENT not in _KNOWN_ENVIRONMENTS:
    logging.getLogger(__name__).warning(
        "APP_ENVIRONMENT=%r is not one of %s; treating this as non-production",
        APP_ENVIRONMENT,
        ", ".join(_KNOWN_ENVIRONMENTS),
    )
if not APP_ENVIRONMENT:
    # Transitional: infra still sends IS_PROD until the tfvars change is applied, and the two
    # deploy independently. Delete this branch, and IS_PROD from backend_environment_managed,
    # one release after APP_ENVIRONMENT is live in all three environments.
    _legacy_is_prod = getenv_or("IS_PROD").lower() == "true"
    APP_ENVIRONMENT = "prod" if _legacy_is_prod else "non-prod"

IS_PROD = APP_ENVIRONMENT == "prod"

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


# Which agent graph to use.  Short names (e.g. "react") resolve to built-in
# graphs in src/services/agents/graphs/.  Dotted paths (e.g.
# "my_package.MyAgent") are imported dynamically for external graphs.
def _normalize_agent_graph_type(raw: str) -> str:
    s = raw.strip().lstrip("\ufeff").strip()
    if (s.startswith('"') and s.endswith('"')) or (
        s.startswith("'") and s.endswith("'")
    ):
        s = s[1:-1].strip()
    return s


AGENT_GRAPH_TYPE = _normalize_agent_graph_type(
    os.getenv("AGENT_GRAPH_TYPE", "react") or "react"
)

# AWS Cognito credentials for AgentCore MCP server authentication.
AGENTCORE_TOKEN_URL = os.getenv("AGENTCORE_TOKEN_URL", "").strip()
AGENTCORE_CLIENT_ID = os.getenv("AGENTCORE_CLIENT_ID", "").strip()
AGENTCORE_CLIENT_SECRET = os.getenv("AGENTCORE_CLIENT_SECRET", "").strip()
MCP_PROXY_BASE_URL = os.getenv("MCP_PROXY_BASE_URL", "").strip()
MCP_PROXY_INTERNAL_BASE_URL = os.getenv("MCP_PROXY_INTERNAL_BASE_URL", "").strip()
MCP_TOOLS_CACHE_TTL = float(os.getenv("MCP_TOOLS_CACHE_TTL", "300"))
# OpenAI-compatible proxy — upstream base URL (e.g. "https://example.com") and optional API key
OPENAI_PROXY_UPSTREAM_URL = os.getenv("OPENAI_PROXY_UPSTREAM_URL", "").strip()
OPENAI_PROXY_API_KEY = os.getenv("OPENAI_PROXY_API_KEY", "").strip()

# User custom model API keys (BYOM). Stored as an AES-256-GCM envelope-encrypted
# blob on the Mongo row (src/services/custom_model_cipher.py); AWS_REGION and
# AWS_ENDPOINT_URL are also used for the KMS client. CUSTOM_MODEL_SECRET_PREFIX
# is legacy: it only matters for rows that still carry a Secrets Manager
# secret_arn and haven't been swept by migrate_custom_model_secrets yet.
AWS_REGION = os.getenv("AWS_REGION", "eu-central-1").strip()
AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL", "").strip() or None
CUSTOM_MODEL_SECRET_PREFIX = os.getenv("CUSTOM_MODEL_SECRET_PREFIX", "eve/dev").strip()
CUSTOM_MODEL_MAX_PER_USER = int(os.getenv("CUSTOM_MODEL_MAX_PER_USER", "10"))
# DEK-wrapping backend for the envelope cipher. KMS wins when both are set.
#   CUSTOM_MODEL_KMS_KEY_ID: a KMS CMK id/ARN/alias (cloud) -- DEKs are wrapped
#     with kms:Encrypt/Decrypt under this key.
#   BYOK_LOCAL_KEK: a static 256-bit key, base64 or hex (local compose/CI, no
#     AWS needed) -- DEKs are wrapped with AES-256-GCM under this key.
CUSTOM_MODEL_KMS_KEY_ID = os.getenv("CUSTOM_MODEL_KMS_KEY_ID", "").strip() or None
BYOK_LOCAL_KEK = os.getenv("BYOK_LOCAL_KEK", "").strip() or None
# ──────────────────────────────────────────────────────────────────────────────

def redis_client_kwargs() -> Dict[str, Any]:
    """
    Connection kwargs for Redis clients that use blocking pub/sub reads.

    socket_timeout must be None so listen/get_message can wait for the next
    chunk during long RAG/setup phases (default timeouts break SSE streaming).

    health_check_interval is the counterpart of that None: both pub/sub consumers
    (cancel_manager, stream_bus) block on get_message(timeout=None), so a peer
    that dies without sending a FIN would hang the read until TCP keepalive
    noticed. A ping every 30 s of idle surfaces the dead connection and lets
    redis-py reconnect instead of leaving the SSE stream and Stop wedged.
    """
    return {
        "socket_timeout": None,
        "socket_connect_timeout": float(os.getenv("REDIS_CONNECT_TIMEOUT", "10")),
        "health_check_interval": 30,
    }


# S3 / Artifact storage
# S3_ENDPOINT_URL empty -> real AWS S3; set to http://minio:9000 for local MinIO.
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "eve-x-artifacts-local").strip()
S3_REGION = os.getenv("S3_REGION", "eu-central-1").strip()
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "").strip()
# Dedicated credentials (do not collide with the AWS_* vars used by the CLI).
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID", "").strip()
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY", "").strip()
S3_PRESIGN_TTL_SECONDS = int(os.getenv("S3_PRESIGN_TTL_SECONDS", 300))

# Legacy user-upload path knobs (POST /artifacts). Superseded by the
# generalized ARTIFACT_UPLOAD_* knobs below; kept here only because their env
# values are read as a fallback so already-deployed hosts keep working
# unchanged after the image-only upload gate was generalized to any artifact.
IMAGE_MAX_BYTES = int(os.getenv("IMAGE_MAX_BYTES", 10 * 1024 * 1024))
IMAGE_ALLOWED_TYPES = [
    t.strip().lower()
    for t in os.getenv("IMAGE_ALLOWED_TYPES", "png,jpeg,webp,gif").split(",")
    if t.strip()
]
IMAGE_UPLOADS_PER_DAY = int(os.getenv("IMAGE_UPLOADS_PER_DAY", 100))

# Generic artifact knobs, consumed by MCP tool-output ingestion.
ARTIFACT_MAX_BYTES = int(os.getenv("ARTIFACT_MAX_BYTES", 25 * 1024 * 1024))
ARTIFACT_MAX_PER_TOOL_CALL = int(os.getenv("ARTIFACT_MAX_PER_TOOL_CALL", 10))
ARTIFACT_RESOURCE_READ_TIMEOUT_S = int(
    os.getenv("ARTIFACT_RESOURCE_READ_TIMEOUT_S", 30)
)

# Generalized user-upload path knobs (POST /artifacts with any allowed file
# type: images, pdf, csv, txt, json, geojson). Each falls back to the legacy
# IMAGE_* env value when its ARTIFACT_* counterpart isn't set, so hosts that
# never migrate their environment keep the old image-only behaviour.
# NOTE: the byte cap is named ARTIFACT_UPLOAD_MAX_BYTES rather than
# ARTIFACT_MAX_BYTES because that name is already taken above by the
# unrelated MCP tool-output ingestion cap (different default, different
# purpose); reusing it here would silently couple two independent limits.
ARTIFACT_UPLOAD_ALLOWED_TYPES = [
    t.strip().lower()
    for t in os.getenv(
        "ARTIFACT_UPLOAD_ALLOWED_TYPES",
        os.getenv("IMAGE_ALLOWED_TYPES", "png,jpeg,webp,gif,pdf,csv,txt,json,geojson"),
    ).split(",")
    if t.strip()
]
ARTIFACT_UPLOAD_MAX_BYTES = int(
    os.getenv(
        "ARTIFACT_UPLOAD_MAX_BYTES", os.getenv("IMAGE_MAX_BYTES", 10 * 1024 * 1024)
    )
)
ARTIFACT_UPLOADS_PER_DAY = int(
    os.getenv("ARTIFACT_UPLOADS_PER_DAY", os.getenv("IMAGE_UPLOADS_PER_DAY", 100))
)

# Curated image catalog (demo). When enabled, a small static catalog of images
# is offered to the answer-generation prompt so the model can embed them as
# markdown. Disabled -> the composed generation prompts are byte-identical to
# the catalog-free behaviour. Treat only "false"/"0" (case-insensitively) as off.
FEATURE_IMAGE_CATALOG = getenv_or(
    "FEATURE_IMAGE_CATALOG", getenv_or("IMAGE_CATALOG_ENABLED", "true")
).lower() not in (
    "false",
    "0",
)
_DEFAULT_IMAGE_CATALOG_PATH = os.path.join(
    os.path.dirname(__file__), "templates", "image_catalog.yaml"
)
IMAGE_CATALOG_PATH = (
    os.getenv("IMAGE_CATALOG_PATH", "").strip() or _DEFAULT_IMAGE_CATALOG_PATH
)

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

        # Transitional: the old name is still what the deployed tfvars send, and infra and
        # code deploy independently. Drop the fallback one release after the rename lands.
        enabled = self._parse_bool_env("FEATURE_TOKEN_RATE_LIMIT")
        if enabled is None:
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
