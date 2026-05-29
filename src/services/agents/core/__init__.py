"""Backend-bound integration layer for agent graphs.

Contains the runner (streaming, persistence, SSE, bus), the registry,
and backend-specific MCP interceptors.  These modules import from the
rest of the backend (database, FastAPI, etc.) and are NOT portable.

For the standalone parts (AgentGraph base class, utilities, graph
implementations), see :mod:`src.services.agents.graphs`.
"""
