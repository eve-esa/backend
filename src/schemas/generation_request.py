from typing import Any, Dict, List, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, PrivateAttr

from src.constants import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_K,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_QUERY,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_TEMPERATURE,
)


class GenerationRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    query: str = DEFAULT_QUERY
    year: Optional[List[int]] = None
    filters: Optional[Dict[str, Any]] = None
    llm_type: Optional[str] = Field(
        default=None,
        description=(
            "LLM type to use. Options: 'main', 'fallback', 'satcom_small', 'satcom_large', 'eve_jsc'. "
            "Legacy options 'runpod' and 'mistral' are also supported. "
            "Defaults to None, which means environment-based behavior."
        ),
    )
    embeddings_model: str = DEFAULT_EMBEDDING_MODEL
    k: int = Field(DEFAULT_K, ge=0, le=10)
    temperature: float = Field(DEFAULT_TEMPERATURE, ge=0.0, le=1.0)
    score_threshold: float = Field(DEFAULT_SCORE_THRESHOLD, ge=0.0, le=1.0)
    max_new_tokens: int = Field(DEFAULT_MAX_NEW_TOKENS, ge=100, le=100_000)
    public_collections: List[str] = Field(
        default_factory=list,
        description="List of public collection names to include in the search",
    )
    private_collections: List[str] = Field(
        default_factory=list,
        description="List of private collection IDs to include in the search",
    )
    public_mcp_servers: List[str] = Field(
        default_factory=list,
        description="List of MCP server names to attach as tools for the agentic pipeline",
    )
    agent: Optional[str] = Field(
        default=None,
        description=(
            "Optional agent graph selector for the agentic pipeline "
            "(e.g. 'react', 'simple', or dotted module path). "
            "When omitted, backend uses AGENT_GRAPH_TYPE from environment."
        ),
    )
    artifact_ids: Optional[List[str]] = Field(
        default=None,
        max_length=20,
        validation_alias=AliasChoices("artifact_ids", "image_ids"),
        description=(
            "IDs of previously uploaded artifacts to attach to the message. "
            "The legacy field name 'image_ids' is still accepted."
        ),
    )
    custom_model_id: Optional[str] = Field(
        default=None,
        description=(
            "User-owned custom model ID. When set, overrides llm_type for agentic generation."
        ),
    )

    _collection_ids: List[str] = PrivateAttr(default_factory=list)
    _private_collections_map: Dict[str, str] = PrivateAttr(default_factory=dict)
    _mcp_server_configs: List[Any] = PrivateAttr(default_factory=list)
    _mcp_proxy_bearer_token: Optional[str] = PrivateAttr(default=None)
    _resolved_custom_model: Any = PrivateAttr(default=None)
    _mcp_user_id: Optional[str] = PrivateAttr(default=None)

    @property
    def collection_ids(self) -> List[str]:
        return self._collection_ids

    @collection_ids.setter
    def collection_ids(self, value: List[str]) -> None:
        self._collection_ids = list(value) if value else []

    @property
    def private_collections_map(self) -> Dict[str, str]:
        return self._private_collections_map

    @private_collections_map.setter
    def private_collections_map(self, value: Dict[str, str]) -> None:
        self._private_collections_map = value

    @property
    def mcp_server_configs(self) -> List[Any]:
        return self._mcp_server_configs

    @mcp_server_configs.setter
    def mcp_server_configs(self, value: List[Any]) -> None:
        self._mcp_server_configs = list(value) if value else []

    @property
    def mcp_proxy_bearer_token(self) -> Optional[str]:
        """Inbound access JWT for ``/mcp/{name}`` when using the MCP proxy from the backend."""
        return self._mcp_proxy_bearer_token

    @mcp_proxy_bearer_token.setter
    def mcp_proxy_bearer_token(self, value: Optional[str]) -> None:
        self._mcp_proxy_bearer_token = value

    @property
    def resolved_custom_model(self) -> Any:
        """User-owned custom model loaded during agentic request preparation."""
        return self._resolved_custom_model

    @resolved_custom_model.setter
    def resolved_custom_model(self, value: Any) -> None:
        self._resolved_custom_model = value

    @property
    def mcp_user_id(self) -> Optional[str]:
        """Authenticated user id for MCP tool discovery cache keys."""
        return self._mcp_user_id

    @mcp_user_id.setter
    def mcp_user_id(self, value: Optional[str]) -> None:
        self._mcp_user_id = value
