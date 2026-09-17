"""
Error logging utility for storing errors in MongoDB.

Batches inserts in the background. Agentic call sites pass ``kind`` / ``node`` /
``policy``; legacy RAG still uses :class:`Component` and :class:`PipelineStage`.
"""

import logging
import asyncio
import re
from enum import Enum
from typing import Optional, List, Dict, Any, Union

from contextvars import ContextVar

from src.database.models.error_log import ErrorLog
from src.utils.langfuse_helper import record_error_kind

# Graph failures also go to Langfuse when keys are set (no-op otherwise).
_LANGFUSE_KINDS = frozenset(
    {
        "timeout",
        "run_timeout",
        "retry",
        "error_handler",
        "fallback",
        "tool_error",
    }
)

logger = logging.getLogger(__name__)

conversation_id_context: ContextVar[Optional[str]] = ContextVar(
    "conversation_id", default=None
)
message_id_context: ContextVar[Optional[str]] = ContextVar("message_id", default=None)
user_id_context: ContextVar[Optional[str]] = ContextVar("user_id", default=None)


class Component(str, Enum):
    """Legacy RAG component labels. Agentic logs use ``node`` / ``source`` instead."""

    LLM = "LLM"
    LLM_FALLBACK = "LLM_FALLBACK"
    RETRIEVAL = "RETRIEVAL"
    RETRIEVAL_FALLBACK = "RETRIEVAL_FALLBACK"
    RE_RANKER = "RE-RANKER"
    RE_RANKER_FALLBACK = "RE-RANKER_FALLBACK"
    ROUTER = "ROUTER"
    MCP_TOOL = "MCP_TOOL"


class PipelineStage(str, Enum):
    """Legacy RAG pipeline stages. Agentic logs use ``kind`` / ``policy`` instead."""

    RE_QUERYING = "re-querying"
    GENERATION = "generation"
    RETRIEVAL = "retrieval"
    HALLUCINATION = "hallucination"
    ROUTER = "router"
    TOOL_EXECUTION = "tool_execution"


class PolicyEvent(Exception):
    """Stand-in exception when logging a policy that is not a Python failure."""


_SECRET_KEY_RE = re.compile(
    r"(?i)^(authorization|password|secret|token|access_token|api_key|jwt|credential)$"
)
_SECRET_VALUE_RE = re.compile(
    r"(?i)(bearer\s+[a-z0-9._\-+=/]+|"
    r"eyJ[a-zA-Z0-9_\-]+=*\.[a-zA-Z0-9_\-]+=*\.[a-zA-Z0-9_\-+=/.]*|"
    r"(?:api[_-]?key|access[_-]?token|secret|password)\s*[:=]\s*\S+)"
)
_REDACTED = "[REDACTED]"
_MAX_ERROR_STRING = 500


def redact_secrets(value: str) -> str:
    """Strip bearer tokens / JWTs / api-key assignments from a log string."""
    if not value:
        return value
    return _SECRET_VALUE_RE.sub(_REDACTED, value)


def _label(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


class ErrorLogger:
    """Utility class for logging errors to MongoDB with bulk insert and background tasks."""

    BATCH_SIZE = 100
    FLUSH_INTERVAL = 5.0

    def __init__(self):
        self._buffer: List[ErrorLog] = []
        self._buffer_lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None

    def _serialize_error_value(self, value: Any, *, key: Optional[str] = None) -> Any:
        if key and _SECRET_KEY_RE.match(str(key)):
            return _REDACTED
        if isinstance(value, Exception):
            return {
                "type": type(value).__name__,
                "message": redact_secrets(str(value))[:_MAX_ERROR_STRING],
                "args": [
                    self._serialize_error_value(arg) for arg in value.args
                ]
                if hasattr(value, "args")
                else [],
            }
        elif isinstance(value, (list, tuple)):
            return [self._serialize_error_value(item) for item in value]
        elif isinstance(value, dict):
            return {
                k: self._serialize_error_value(v, key=str(k)) for k, v in value.items()
            }
        elif isinstance(value, str):
            return redact_secrets(value)[:_MAX_ERROR_STRING]
        elif isinstance(value, (int, float, bool, type(None))):
            return value
        else:
            return redact_secrets(str(value))[:_MAX_ERROR_STRING]

    def _create_error_document(
        self,
        error: Exception,
        description: str,
        *,
        component: Optional[Union[Component, str]] = None,
        pipeline_stage: Optional[Union[PipelineStage, str]] = None,
        error_type: Optional[str] = None,
        logger_name: str = "src.services.generate_answer",
        graph: Optional[str] = None,
        node: Optional[str] = None,
        kind: Optional[str] = None,
        policy: Optional[str] = None,
        source: Optional[str] = None,
        error_extra: Optional[Dict[str, Any]] = None,
    ) -> ErrorLog:
        user_id = user_id_context.get()
        conversation_id = conversation_id_context.get()
        message_id = message_id_context.get()

        serialized_args = []
        if hasattr(error, "args"):
            serialized_args = [self._serialize_error_value(arg) for arg in error.args]

        error_repr: Dict[str, Any] = {
            "type": error_type or type(error).__name__,
            "message": redact_secrets(str(error))[:_MAX_ERROR_STRING],
            "args": serialized_args,
        }

        if hasattr(error, "__dict__"):
            error_repr["attributes"] = {
                k: self._serialize_error_value(v, key=k)
                for k, v in error.__dict__.items()
                if not k.startswith("_")
            }

        if error_extra:
            error_repr.update(self._serialize_error_value(error_extra))

        if policy and (not kind or kind == "policy"):
            kind = policy
        return ErrorLog(
            user_id=user_id,
            conversation_id=conversation_id,
            message_id=message_id,
            logger_name=logger_name,
            component=_label(component),
            graph=graph,
            node=node,
            kind=kind,
            policy=None,
            source=source,
            error=error_repr or None,
            error_type=None,
            pipeline_stage=_label(pipeline_stage),
            description=redact_secrets(description)[:_MAX_ERROR_STRING],
        )

    async def log_error(
        self,
        error: Exception,
        component: Optional[Union[Component, str]] = None,
        pipeline_stage: Optional[Union[PipelineStage, str]] = None,
        description: str = "",
        error_type: Optional[str] = None,
        logger_name: str = "src.services.generate_answer",
        graph: Optional[str] = None,
        node: Optional[str] = None,
        kind: Optional[str] = None,
        policy: Optional[str] = None,
        source: Optional[str] = None,
        error_extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an error to the buffer. Errors are batched and inserted in bulk.

        ``component`` and ``pipeline_stage`` remain for RAG and frontend logs.
        Agentic call sites omit them and pass ``node`` / ``kind``.
        ``kind`` is the only discriminator (``timeout``, ``retry``, ``tool_error``,
        ``rag``, ``frontend``, …). A ``policy=`` argument is accepted as an
        alias and stored as ``kind``.

        Agentic kinds are always inserted into Mongo ``error_logs``. When
        Langfuse is enabled they are also emitted as child events on the
        generation trace (``record_error_kind`` is a no-op without keys).
        """
        try:
            error_doc = self._create_error_document(
                error=error,
                description=description,
                component=component,
                pipeline_stage=pipeline_stage,
                error_type=error_type,
                logger_name=logger_name,
                graph=graph,
                node=node,
                kind=kind,
                policy=policy,
                source=source,
                error_extra=error_extra,
            )
            if error_doc.kind in _LANGFUSE_KINDS:
                record_error_kind(
                    error_doc.kind,
                    node=error_doc.node,
                    graph=error_doc.graph,
                    source=error_doc.source,
                    description=error_doc.description,
                    extra=error_extra,
                )

            async with self._buffer_lock:
                self._buffer.append(error_doc)

                if len(self._buffer) >= self.BATCH_SIZE:
                    await self._flush_buffer()

                if self._flush_task is None or self._flush_task.done():
                    self._flush_task = asyncio.create_task(self._periodic_flush())

        except Exception as e:
            logger.error(f"Failed to log error to buffer: {e}", exc_info=True)

    async def _flush_buffer(self) -> None:
        if not self._buffer:
            return

        async with self._buffer_lock:
            if not self._buffer:
                return

            errors_to_insert = self._buffer.copy()
            self._buffer.clear()

        try:
            if errors_to_insert:
                await ErrorLog.bulk_create(errors_to_insert)
                logger.info(
                    f"Bulk inserted {len(errors_to_insert)} error logs to MongoDB"
                )
        except Exception as e:
            logger.error(f"Failed to bulk insert error logs: {e}", exc_info=True)

    async def _periodic_flush(self) -> None:
        try:
            await asyncio.sleep(self.FLUSH_INTERVAL)
            await self._flush_buffer()
        except asyncio.CancelledError:
            await self._flush_buffer()
            raise
        except Exception as e:
            logger.error(f"Error in periodic flush: {e}", exc_info=True)

    async def flush(self) -> None:
        await self._flush_buffer()

    async def log_error_sync(
        self,
        error: Exception,
        component: Optional[Union[Component, str]] = None,
        pipeline_stage: Optional[Union[PipelineStage, str]] = None,
        description: str = "",
        error_type: Optional[str] = None,
        logger_name: str = "src.services.generate_answer",
        graph: Optional[str] = None,
        node: Optional[str] = None,
        kind: Optional[str] = None,
        policy: Optional[str] = None,
        source: Optional[str] = None,
        error_extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        kwargs = dict(
            error=error,
            description=description,
            component=component,
            pipeline_stage=pipeline_stage,
            error_type=error_type,
            logger_name=logger_name,
            graph=graph,
            node=node,
            kind=kind,
            policy=policy,
            source=source,
            error_extra=error_extra,
        )
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.log_error(**kwargs))
            else:
                await self.log_error(**kwargs)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                await self.log_error(**kwargs)
                await self.flush()
            finally:
                loop.close()


_error_logger_instance: Optional[ErrorLogger] = None


def get_error_logger() -> ErrorLogger:
    """Get the global ErrorLogger singleton instance."""
    global _error_logger_instance
    if _error_logger_instance is None:
        _error_logger_instance = ErrorLogger()
    return _error_logger_instance


def set_conversation_context(conversation_id: Optional[str]) -> None:
    conversation_id_context.set(conversation_id)


def set_message_context(message_id: Optional[str]) -> None:
    message_id_context.set(message_id)


def set_user_context(user_id: Optional[str]) -> None:
    user_id_context.set(user_id)


def get_conversation_context() -> Optional[str]:
    return conversation_id_context.get()


def get_message_context() -> Optional[str]:
    return message_id_context.get()


def get_user_context() -> Optional[str]:
    return user_id_context.get()


async def persist_policy_event(
    *,
    policy: str,
    description: str,
    node: Optional[str] = None,
    graph: Optional[str] = None,
    source: Optional[str] = None,
    error: Optional[BaseException] = None,
    attempt: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
    logger_name: str = "src.services.agents.core.runner",
) -> None:
    """Write a row whose ``kind`` is the policy name (timeout, retry, …)."""
    if isinstance(error, Exception):
        exc: Exception = error
    else:
        exc = PolicyEvent(description)
    error_extra: Dict[str, Any] = dict(extra or {})
    if attempt is not None:
        error_extra["attempt"] = attempt
    await get_error_logger().log_error(
        error=exc,
        description=description,
        kind=policy,
        node=node,
        graph=graph,
        source=source,
        logger_name=logger_name,
        error_extra=error_extra or None,
    )
