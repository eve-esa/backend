"""Pluggable agent graphs package.

Folder structure:

- :mod:`src.services.agents.graphs` — standalone code (no backend imports),
  contains :class:`AgentGraph`, :class:`LatencyInterceptor`, shared utilities,
  and graph implementations.
- :mod:`src.services.agents.core` — backend-bound integration (runner,
  registry, error-logging interceptor).

Top-level re-exports keep the public API stable.
"""

from src.services.agents.core.registry import get_agent_graph
from src.services.agents.graphs.base import AgentGraph, LatencyInterceptor

__all__ = ["AgentGraph", "LatencyInterceptor", "get_agent_graph"]
