from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, PrivateAttr

from src.constants import (
    DEFAULT_QUERY,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_K,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
)


class GenerationRequest(BaseModel):
    query: str = DEFAULT_QUERY
    year: Optional[List[int]] = None
    filters: Optional[Dict[str, Any]] = None
    llm_type: Optional[str] = Field(
        default=None,
        description=(
            "LLM type to use. Options: 'main', 'fallback', 'satcom_small', 'satcom_large', 'ship', 'eve_v05'. "
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
    public_mcp_servers: List[str] = Field(
        default_factory=list,
        description="List of MCP server names to attach as tools for the agentic pipeline",
    )

    _collection_ids: List[str] = PrivateAttr(default_factory=list)
    _private_collections_map: Dict[str, str] = PrivateAttr(default_factory=dict)
    _mcp_server_configs: List[Any] = PrivateAttr(default_factory=list)
    _mcp_proxy_bearer_token: Optional[str] = PrivateAttr(default=None)

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
