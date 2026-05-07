"""Agent graph registry — auto-discovery + dotted-path import.

Resolution rules for ``AGENT_GRAPH_TYPE``:

- **``react``** (case-insensitive): prefer ``ReactAgent`` from the active graphs
  tree selected by :mod:`graphs_bundle`; if unavailable, fall back to vendored.
- **Other short names**: merge discoverable graphs from both sources:
  ``agents.graphs`` and ``src.services.agents.graphs``.
- **Dotted path** (``module.Class``): ``importlib`` (e.g. external packages).

This avoids cases where an older installed pip package shadows newer local
graphs (e.g. ``simple``).
"""

import importlib
import inspect
import logging
import pkgutil
from typing import Dict, Iterable, Optional, Type

from src.services.agents.graphs_bundle import (
    clear_graphs_bundle_cache,
    graphs_base_module,
    graphs_prefix,
)

AgentGraph = graphs_base_module().AgentGraph

logger = logging.getLogger(__name__)

_local_registry: Optional[Dict[str, Type[AgentGraph]]] = None
_DISCOVERY_PREFIXES = ("agents.graphs", "src.services.agents.graphs")


def _react_graph_module_path() -> str:
    return f"{graphs_prefix()}.react.graph"


def _iter_discovery_prefixes() -> Iterable[str]:
    """Yield active prefix first, then remaining known fallbacks."""
    active = graphs_prefix()
    yielded = set()
    if active in _DISCOVERY_PREFIXES:
        yielded.add(active)
        yield active
    for prefix in _DISCOVERY_PREFIXES:
        if prefix in yielded:
            continue
        yield prefix


def _react_agent_type() -> Type[AgentGraph]:
    """Return ReactAgent, trying active source first then fallback source."""
    tried = []
    for prefix in _iter_discovery_prefixes():
        module_path = f"{prefix}.react.graph"
        tried.append(module_path)
        try:
            mod = importlib.import_module(module_path)
            return mod.ReactAgent
        except Exception:
            continue
    raise ImportError(f"Could not import ReactAgent from any source: {tried}")


def _register_builtin_react(registry: Dict[str, Type[AgentGraph]]) -> None:
    """Register the shipped ReAct graph under its ``name`` key."""
    mod_path = _react_graph_module_path()
    try:
        cls = _react_agent_type()
        key = getattr(cls, "name", None) or "react"
        registry[key] = cls
    except Exception as exc:
        logger.error("Could not import built-in ReactAgent from %s: %s", mod_path, exc)


def _is_agent_graph_like(obj: object) -> bool:
    """Return True for subclasses of an AgentGraph class from either source."""
    if not inspect.isclass(obj):
        return False
    if getattr(obj, "__name__", "") == "AgentGraph":
        return False
    for base in inspect.getmro(obj)[1:]:
        if getattr(base, "__name__", "") == "AgentGraph":
            return True
    return False


def _discover_graphs_from_prefix(
    prefix: str, registry: Dict[str, Type[AgentGraph]]
) -> None:
    """Discover graph classes under a specific package prefix."""
    try:
        graphs_pkg = importlib.import_module(prefix)
    except Exception as exc:
        logger.debug("Skipping graph prefix %s (import failed: %s)", prefix, exc)
        return

    package_path = getattr(graphs_pkg, "__path__", None)
    if not package_path:
        return
    package_prefix = prefix + "."

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
            if not _is_agent_graph_like(obj):
                continue
            graph_name = getattr(obj, "name", None)
            if graph_name and graph_name != "base":
                registry.setdefault(graph_name, obj)


def _discover_local_graphs() -> Dict[str, Type[AgentGraph]]:
    """Collect AgentGraph subclasses from active + fallback prefixes."""
    registry: Dict[str, Type[AgentGraph]] = {}

    _register_builtin_react(registry)
    for prefix in _iter_discovery_prefixes():
        _discover_graphs_from_prefix(prefix, registry)

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


def _get_local_registry() -> Dict[str, Type[AgentGraph]]:
    """Lazy singleton registry."""
    global _local_registry
    if _local_registry is None:
        _local_registry = _discover_local_graphs()
        if not _local_registry:
            logger.error(
                "Graph registry is empty after discovery — check that %s is importable.",
                _react_graph_module_path(),
            )
        logger.info(
            "Discovered %d local agent graph(s): %s",
            len(_local_registry),
            list(_local_registry.keys()),
        )
    return _local_registry


def clear_graph_registry_cache() -> None:
    """Clear the in-process graph registry (e.g. for tests)."""
    global _local_registry
    _local_registry = None
    clear_graphs_bundle_cache()


def _normalize_graph_type(graph_type: Optional[str]) -> str:
    """Strip whitespace, UTF-8 BOM, and surrounding quotes from env values."""
    s = (graph_type or "react").strip()
    s = s.lstrip("\ufeff").strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    return s


def get_agent_graph(graph_type: Optional[str] = None) -> AgentGraph:
    """Resolve and instantiate an agent graph by name or dotted path.

    Falls back to ``react`` if *graph_type* is None or empty after normalization.
    """
    graph_type = _normalize_graph_type(graph_type)

    # Built-in react: same tree as graphs_bundle (pip agents or monorepo).
    if graph_type.casefold() == "react":
        return _react_agent_type()()

    if "." in graph_type:
        cls = _import_dotted_path(graph_type)
        logger.info("Loaded agent graph from dotted path: %s", graph_type)
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
