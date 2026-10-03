"""Resolve agent graphs from the ``eve-esa-agents`` package (import name ``agents``).

A local checkout under ``repos/eve-esa-agents`` is put on ``sys.path`` first when
present, so a dev container can run unreleased graphs without a pip install.
"""

from __future__ import annotations

import importlib
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import List, Tuple

_CANDIDATES: Tuple[Tuple[str, str], ...] = (
    (
        "agents.graphs",
        "agents.graphs.base",
    ),
)


def _maybe_add_local_agents_checkout_to_syspath() -> None:
    """Best-effort local checkout fallback for dev/docker setups.

    If ``repos/eve-esa-agents/agents`` exists in this backend workspace, add
    ``repos/eve-esa-agents`` to ``sys.path`` so ``import agents.graphs`` works
    without requiring a pip install inside the running container.
    """
    here = Path(__file__).resolve()
    backend_root = here.parents[3]  # .../backend
    checkout_root = backend_root / "repos" / "eve-esa-agents"
    package_dir = checkout_root / "agents"
    if package_dir.is_dir():
        path_str = str(checkout_root)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


@lru_cache(maxsize=1)
def graphs_prefix() -> str:
    """Dotted package prefix for the active graphs tree (no trailing dot)."""
    _maybe_add_local_agents_checkout_to_syspath()
    errors: List[str] = []
    for prefix, base_mod in _CANDIDATES:
        try:
            importlib.import_module(prefix)
            importlib.import_module(base_mod)
            return prefix
        except ImportError as exc:
            errors.append(f"{prefix}: {exc}")
            continue
    raise ImportError(
        "No agent graphs package found. Install `eve-esa-agents` "
        "(pip install git+https://github.com/eve-esa/agents.git). "
        f"Import errors: {errors}"
    )


@lru_cache(maxsize=1)
def graphs_base_module() -> ModuleType:
    return importlib.import_module(f"{graphs_prefix()}.base")


@lru_cache(maxsize=1)
def graphs_utils_module() -> ModuleType:
    return importlib.import_module(f"{graphs_prefix()}.utils")
