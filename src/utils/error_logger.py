"""
Error logging utility for storing errors in MongoDB.

This module provides functionality to log errors from ErrorException
to MongoDB using bulk insert and background tasks.
"""

import logging
import asyncio
from typing import Optional, List, Dict, Any
from contextvars import ContextVar
from enum import Enum

from src.database.models.error_log import ErrorLog

logger = logging.getLogger(__name__)

# Context variables for conversation_id and message_id
conversation_id_context: ContextVar[Optional[str]] = ContextVar(
    "conversation_id", default=None
)
message_id_context: ContextVar[Optional[str]] = ContextVar("message_id", default=None)
user_id_context: ContextVar[Optional[str]] = ContextVar("user_id", default=None)


class Component(str, Enum):
    """Component types where errors can occur."""

    LLM = "LLM"
    LLM_FALLBACK = "LLM_FALLBACK"
    RETRIEVAL = "RETRIEVAL"
    RETRIEVAL_FALLBACK = "RETRIEVAL_FALLBACK"
    RE_RANKER = "RE-RANKER"
    RE_RANKER_FALLBACK = "RE-RANKER_FALLBACK"
    ROUTER = "ROUTER"


class PipelineStage(str, Enum):
    """Pipeline stages where errors can occur."""

    RE_QUERYING = "re-querying"
    GENERATION = "generation"
    RETRIEVAL = "retrieval"
    HALLUCINATION = "hallucination"
    ROUTER = "router"


class ErrorException(Exception):
    """Custom exception class for errors that should be logged."""

    def __init__(
        self,
        error: Exception,
        component: Component,
        pipeline_stage: PipelineStage,
        description: str,
        error_type: Optional[str] = None,
        logger_name: str = "src.services.generate_answer",
    ):
        """
        Initialize ErrorException.

        Args:
            error: The original exception/error
            component: Component where error occurred
            pipeline_stage: Pipeline stage where error occurred
            description: Human-readable description of the error
            error_type: Type of error (e.g., 'TimeoutError', 'ValueError')
            logger_name: Name of the logger/module where error occurred
        """
        self.error = error
        self.component = component
        self.pipeline_stage = pipeline_stage
        self.description = description
        self.error_type = error_type or type(error).__name__
        self.logger_name = logger_name
        super().__init__(str(error))


class ErrorLogger:
    """Utility class for logging errors to MongoDB with bulk insert and background tasks."""

    BATCH_SIZE = 100  # Number of errors to batch before bulk insert
    FLUSH_INTERVAL = 5.0  # Seconds to wait before flushing pending errors

    def __init__(self):
        """Initialize ErrorLogger with an internal buffer for batching."""
        self._buffer: List[ErrorLog] = []
        self._buffer_lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None

    def _serialize_error_value(self, value: Any) -> Any:
        """
        Recursively serialize error values to make them BSON-compatible.
        
        Converts exception objects to dictionaries, and ensures all nested
        structures are serializable.
        """
        if isinstance(value, Exception):
            return {
                "type": type(value).__name__,
                "message": str(value),
                "args": [self._serialize_error_value(arg) for arg in value.args] if hasattr(value, "args") else [],
            }
        elif isinstance(value, (list, tuple)):
            return [self._serialize_error_value(item) for item in value]
        elif isinstance(value, dict):
            return {k: self._serialize_error_value(v) for k, v in value.items()}
        elif isinstance(value, (str, int, float, bool, type(None))):
            return value
        else:
            return str(value)

    def _create_error_document(
        self,
        error: Exception,
        component: Component,
        pipeline_stage: PipelineStage,
        description: str,
        error_type: Optional[str] = None,
        logger_name: str = "src.services.generate_answer",
    ) -> ErrorLog:
        """
        Create an error log document with the specified structure.

        Args:
            error: The original exception/error
            component: Component where error occurred
            pipeline_stage: Pipeline stage where error occurred
            description: Human-readable description of the error
            error_type: Type of error (e.g., 'TimeoutError', 'ValueError')
            logger_name: Name of the logger/module where error occurred

        Returns:
            ErrorLog instance representing the error log document
        """
        user_id = user_id_context.get()
        conversation_id = conversation_id_context.get()
        message_id = message_id_context.get()

        # Serialize error args recursively to handle nested exceptions
        serialized_args = []
        if hasattr(error, "args"):
            serialized_args = [self._serialize_error_value(arg) for arg in error.args]

        error_repr = {
            "type": type(error).__name__,
            "message": str(error),
            "args": serialized_args,
        }

        if hasattr(error, "__dict__"):
            error_repr["attributes"] = {
                k: self._serialize_error_value(v) for k, v in error.__dict__.items() if not k.startswith("_")
            }

        return ErrorLog(
            user_id=user_id,
            conversation_id=conversation_id,
            message_id=message_id,
            logger_name=logger_name,
            component=component.value,
            error=error_repr,
            error_type=error_type or type(error).__name__,
            pipeline_stage=pipeline_stage.value,
            description=description,
        )

    async def log_error(
        self,
        error: Exception,
        component: Component,
        pipeline_stage: PipelineStage,
        description: str,
        error_type: Optional[str] = None,
        logger_name: str = "src.services.generate_answer",
    ) -> None:
        """
        Log an error to the buffer. Errors are batched and inserted in bulk.

        Args:
            error: The original exception/error
            component: Component where error occurred
            pipeline_stage: Pipeline stage where error occurred
            description: Human-readable description of the error
            error_type: Type of error (e.g., 'TimeoutError', 'ValueError')
            logger_name: Name of the logger/module where error occurred
        """
        try:
            error_doc = self._create_error_document(
                error=error,
                component=component,
                pipeline_stage=pipeline_stage,
                description=description,
                error_type=error_type,
                logger_name=logger_name,
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
        """Flush buffered errors to MongoDB using bulk insert."""
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
        """Periodically flush the buffer to MongoDB."""
        try:
            await asyncio.sleep(self.FLUSH_INTERVAL)
            await self._flush_buffer()
        except asyncio.CancelledError:
            await self._flush_buffer()
            raise
        except Exception as e:
            logger.error(f"Error in periodic flush: {e}", exc_info=True)

    async def flush(self) -> None:
        """Manually flush all pending errors to MongoDB."""
        await self._flush_buffer()

    async def log_error_sync(
        self,
        error: Exception,
        component: Component,
        pipeline_stage: PipelineStage,
        description: str,
        error_type: Optional[str] = None,
        logger_name: str = "src.services.generate_answer",
    ) -> None:
        """
        Synchronously log an error (for use in background tasks).

        This method creates a new event loop if needed and logs the error.

        Args:
            error: The original exception/error
            component: Component where error occurred
            pipeline_stage: Pipeline stage where error occurred
            description: Human-readable description of the error
            error_type: Type of error (e.g., 'TimeoutError', 'ValueError')
            logger_name: Name of the logger/module where error occurred
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(
                    self.log_error(
                        error=error,
                        component=component,
                        pipeline_stage=pipeline_stage,
                        description=description,
                        error_type=error_type,
                        logger_name=logger_name,
                    )
                )
            else:
                await self.log_error(
                    error=error,
                    component=component,
                    pipeline_stage=pipeline_stage,
                    description=description,
                    error_type=error_type,
                    logger_name=logger_name,
                )
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                await self.log_error(
                    error=error,
                    component=component,
                    pipeline_stage=pipeline_stage,
                    description=description,
                    error_type=error_type,
                    logger_name=logger_name,
                )
                await self.flush()
            finally:
                loop.close()


# Global singleton instance
_error_logger_instance: Optional[ErrorLogger] = None


def get_error_logger() -> ErrorLogger:
    """Get the global ErrorLogger singleton instance."""
    global _error_logger_instance
    if _error_logger_instance is None:
        _error_logger_instance = ErrorLogger()
    return _error_logger_instance


def set_conversation_context(conversation_id: Optional[str]) -> None:
    """Set conversation_id in context."""
    conversation_id_context.set(conversation_id)


def set_message_context(message_id: Optional[str]) -> None:
    """Set message_id in context."""
    message_id_context.set(message_id)


def set_user_context(user_id: Optional[str]) -> None:
    """Set user_id in context."""
    user_id_context.set(user_id)


def get_conversation_context() -> Optional[str]:
    """Get conversation_id from context."""
    return conversation_id_context.get()


def get_message_context() -> Optional[str]:
    """Get message_id from context."""
    return message_id_context.get()


def get_user_context() -> Optional[str]:
    """Get user_id from context."""
    return user_id_context.get()

