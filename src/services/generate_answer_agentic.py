"""Agentic answer generation — thin facade.

Delegates to ``src.services.agents.core.runner`` which implements all backend
integration logic (streaming, persistence, MCP tool loading, etc.) while
using pluggable graph definitions from ``src.services.agents.graphs``.

Public API is unchanged — all existing imports continue to work.
"""

from src.services.agents.core.runner import (  # noqa: F401
    generate_answer_agentic,
    generate_answer_agentic_json_stream,
    generate_answer_agentic_stream,
    generate_answer_agentic_stream_helper,
    run_agentic_generation_to_bus,
)

__all__ = [
    "generate_answer_agentic",
    "generate_answer_agentic_stream",
    "generate_answer_agentic_stream_helper",
    "generate_answer_agentic_json_stream",
    "run_agentic_generation_to_bus",
]
