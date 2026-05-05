"""Agent graph registry — auto-discovery + dotted-path import.

Resolution rules for ``AGENT_GRAPH_TYPE``:
  - Short name (e.g. "react"): scans ``src/services/agents/graphs/`` for a
    class whose ``name`` attribute matches.  Graphs can be single files
    (``graphs/my_agent.py``) **or** directories with an ``__init__.py``
    (``graphs/react/``).
  - Dotted path (e.g. "my_package.agents.PlannerAgent"): uses importlib to
    load the class directly.
"""

import importlib
import inspect
import logging
import pkgutil
from functools import lru_cache
from typing import Dict, Optional, Type

from src.services.agents.graphs.base import AgentGraph

logger = logging.getLogger(__name__)


def _discover_local_graphs() -> Dict[str, Type[AgentGraph]]:
    """Scan the ``graphs`` sub-package and collect all AgentGraph subclasses.

    Discovers both single-file modules (``graphs/simple.py``) and directory
    packages (``graphs/react/``).  For directory packages the AgentGraph
    subclass must be importable from the package's ``__init__.py``.
    """
    import src.services.agents.graphs as graphs_pkg

    registry: Dict[str, Type[AgentGraph]] = {}
    package_path = graphs_pkg.__path__
    package_prefix = graphs_pkg.__name__ + "."

    for _importer, module_name, _is_pkg in pkgutil.iter_modules(
        package_path, prefix=package_prefix
    ):
        if module_name.endswith("__"):
            continue
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            logger.warning("Failed to import graph module %s: %s", module_name, exc)
            continue

        for _attr_name, obj in inspect.getmembers(module, inspect.isclass):
            if obj is AgentGraph:
                continue
            if issubclass(obj, AgentGraph) and hasattr(obj, "name"):
                graph_name = getattr(obj, "name", None)
                if graph_name and graph_name != "base":
                    registry[graph_name] = obj

    return registry


def _import_dotted_path(dotted: str) -> Type[AgentGraph]:
    """Import a class from a dotted module path like 'pkg.module.ClassName'."""
    parts = dotted.rsplit(".", 1)
    if len(parts) != 2:
        raise ImportError(
            f"AGENT_GRAPH_TYPE={dotted!r} must be 'module.ClassName' format"
        )
    module_path, class_name = parts
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ImportError(f"Class {class_name!r} not found in module {module_path!r}")
    return cls


@lru_cache(maxsize=1)
def _get_local_registry() -> Dict[str, Type[AgentGraph]]:
    """Cached local graph discovery (runs once at startup)."""
    registry = _discover_local_graphs()
    logger.info(
        "Discovered %d local agent graph(s): %s",
        len(registry),
        list(registry.keys()),
    )
    return registry


def get_agent_graph(graph_type: Optional[str] = None) -> AgentGraph:
    """Resolve and instantiate an agent graph by name or dotted path.

    Falls back to "react" if graph_type is None or empty.
    """
    graph_type = (graph_type or "react").strip()

    if "." in graph_type and not graph_type.startswith("src.services.agents.graphs"):
        cls = _import_dotted_path(graph_type)
        logger.info("Loaded external agent graph: %s", graph_type)
        return cls()

    local_registry = _get_local_registry()
    cls = local_registry.get(graph_type)
    if cls is None:
        available = list(local_registry.keys())
        raise ValueError(
            f"Agent graph {graph_type!r} not found. Available: {available}. "
            f"Or use a dotted path like 'my_package.MyAgent'."
        )

    return cls()
