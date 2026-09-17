"""Model for storing error logs in MongoDB."""

from typing import Any, ClassVar, Dict, Optional

from pydantic import Field, model_validator

from src.database.mongo_model import MongoModel


def _args_are_redundant(
    args: Any, *, message: Optional[str], description: str
) -> bool:
    if not args:
        return True
    if not isinstance(args, list) or len(args) != 1:
        return False
    only = args[0]
    only_str = only if isinstance(only, str) else str(only)
    return only_str == message or (bool(description) and only_str == description)


class ErrorLog(MongoModel):
    """Model for storing error logs from the application."""

    user_id: Optional[str] = Field(default=None, description="User ID associated with the error")
    conversation_id: Optional[str] = Field(
        default=None, description="Conversation ID associated with the error"
    )
    message_id: Optional[str] = Field(
        default=None, description="Message ID associated with the error"
    )
    logger_name: Optional[str] = Field(
        default=None,
        description="Python module for RAG logs. Omitted when it equals source.",
    )
    component: Optional[str] = Field(
        default=None,
        description="Legacy RAG label (LLM, RETRIEVAL, …). Unused on agentic rows.",
    )
    graph: Optional[str] = Field(
        default=None, description="Agent graph name (e.g. react, simple)"
    )
    node: Optional[str] = Field(
        default=None, description="LangGraph node id when the error is in-graph"
    )
    kind: Optional[str] = Field(
        default=None,
        description=(
            "Always set on new writes: timeout, retry, tool_error, rag, frontend, …"
        ),
    )
    policy: Optional[str] = Field(
        default=None,
        description="Deprecated; old rows used this beside kind=policy. Flattened into kind.",
    )
    source: Optional[str] = Field(
        default=None,
        description="Origin when there is no node (runner, mcp_load, router, frontend)",
    )
    error: Optional[Dict[str, Any]] = Field(
        default=None, description="Exception payload; omits fields mirrored elsewhere"
    )
    error_type: Optional[str] = Field(
        default=None,
        description="Deprecated copy of error['type']; omitted on new writes.",
    )
    pipeline_stage: Optional[str] = Field(
        default=None,
        description="Legacy RAG stage. Unused on agentic rows.",
    )
    description: str = Field(..., description="Human-readable description of the error")

    collection_name: ClassVar[str] = "error_logs"

    @model_validator(mode="after")
    def collapse_mirrored_fields(self) -> "ErrorLog":
        """Drop mirrored payload copies; flatten legacy kind=policy + policy into kind."""
        if self.policy and (not self.kind or self.kind == "policy"):
            self.kind = self.policy
        self.policy = None
        if not self.kind:
            if self.source == "frontend":
                self.kind = "frontend"
            elif self.component or self.pipeline_stage:
                self.kind = "rag"
        if self.source and self.logger_name == self.source:
            self.logger_name = None

        payload = dict(self.error) if self.error else {}
        if payload:
            message = payload.get("message")
            message_str = None if message is None else str(message)
            if _args_are_redundant(
                payload.get("args"),
                message=message_str,
                description=self.description,
            ):
                payload.pop("args", None)
            if message_str is not None and message_str == self.description:
                payload.pop("message", None)

            attrs = payload.get("attributes")
            if isinstance(attrs, dict):
                mirrored = {
                    "node": self.node,
                    "graph": self.graph,
                    "source": self.source,
                }
                attrs = {
                    key: value
                    for key, value in attrs.items()
                    if mirrored.get(key) != value
                }
                if attrs:
                    payload["attributes"] = attrs
                else:
                    payload.pop("attributes", None)

            if payload.get("type") == "PolicyEvent":
                payload.pop("type", None)

        self.error = payload or None
        return self

    def to_dict(self) -> Dict[str, Any]:
        doc = super().to_dict()
        return {key: value for key, value in doc.items() if value is not None}
