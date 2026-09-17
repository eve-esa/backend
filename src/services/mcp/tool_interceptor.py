"""Deprecated shim — implementation lives in agents.core.interceptors."""

from src.services.agents.core.interceptors import (  # noqa: F401
    ErrorLoggingInterceptor,
    ObservabilityInterceptor,
)

__all__ = ["ErrorLoggingInterceptor", "ObservabilityInterceptor"]
