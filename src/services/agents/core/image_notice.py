"""The text a model without vision reads in place of an image file.

deepagents' ``read_file`` returns an image as an image block. Its
``UnsupportedContentMiddleware`` swaps that block for a text notice when the
model's profile takes no images (``platform_model_profile``): "was not
attached because this model does not support image content". A model that
reads the chart it just saved concludes from that text that it cannot show
the chart, and says so next to the rendered image (dev, 2026-11-19).

The notice built here says what is true and what to do: the model cannot see
the image, must not describe it, and can still show it with markdown the
answer link rewrite turns into a link the chat loads (a workspace file
relative to the workspace, an artifact by its ``/artifacts/{id}/{name}``
path). Only image blocks of a ``read_file`` result change; every other block
keeps the deepagents placeholder. A model whose profile takes images never
reaches this code. The thread keeps the image block, so the notice is built
again on every model call and nothing is stored.

The eve-esa/agents deep graph builds its own middleware list and takes none
from the backend, so the notice is set on the deepagents class, once per
process, by :func:`install_image_read_notice`.
"""

from __future__ import annotations

import logging
import posixpath
import urllib.parse
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# The deepagents method replaced by the install, kept to delegate to.
_DEEPAGENTS_REPLACE: Optional[Callable[..., Any]] = None

_WORKSPACE = "/workspace/"
_ARTIFACTS = "/artifacts/"


def _size_text(size: int) -> str:
    if size < 1024:
        return f"{size} bytes"
    if size < 1024 * 1024:
        return f"{round(size / 1024)} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def markdown_target(path: str) -> Optional[str]:
    """The markdown target the answer link rewrite turns into a chat link.

    ``/workspace/x`` is ``x`` (``session_path_for_markdown`` reads a relative
    path as a workspace file). ``/artifacts/{id}/{name}`` stays as it is, for
    the artifact rewrite. Any other tree has no published link: ``None``.
    """
    raw = str(path or "").strip()
    if not raw or ".." in raw.split("/"):
        return None
    if raw.startswith("workspace/"):
        raw = "/" + raw
    if raw.startswith(_WORKSPACE):
        relative = raw[len(_WORKSPACE) :]
        return urllib.parse.quote(relative, safe="/") if relative else None
    if raw.startswith(_ARTIFACTS) and raw.count("/") >= 3:
        return urllib.parse.quote(raw, safe="/")
    return None


def image_read_notice(path: str, size: Optional[int]) -> str:
    """What a model without vision reads after ``read_file`` on an image."""
    name = posixpath.basename(str(path or "").rstrip("/")) or "image"
    about = f"{name}, {_size_text(size)}" if size is not None else name
    parts = [
        f"[read_file: {path}] This is an image file ({about}).",
        "This model cannot see image content, so do not describe it or guess "
        "what it shows.",
        "If the user asks what it shows, tell them you cannot see images with "
        "the current model.",
    ]
    target = markdown_target(path)
    if target is not None:
        label = name.translate(str.maketrans("", "", "[]()"))
        parts.append(
            f"You can still show it to the user: write ![{label}]({target}) in "
            "your answer."
        )
    return " ".join(parts)


def _base64_size(data: Any) -> Optional[int]:
    """Decoded size of a base64 string, without decoding it."""
    if not isinstance(data, str):
        return None
    text = data.strip()
    padding = len(text) - len(text.rstrip("="))
    return max(len(text) * 3 // 4 - padding, 0)


def _replace(self: Any, block: Any, message: Any) -> Any:
    """``UnsupportedContentMiddleware._replace`` with the image read notice."""
    kwargs = getattr(message, "additional_kwargs", None) or {}
    path = kwargs.get("read_file_path")
    if (
        isinstance(block, dict)
        and block.get("type") == "image"
        and getattr(message, "type", None) == "tool"
        and isinstance(path, str)
        and path
    ):
        return {
            "type": "text",
            "text": image_read_notice(path, _base64_size(block.get("base64"))),
        }
    return _DEEPAGENTS_REPLACE(self, block, message)


def install_image_read_notice() -> bool:
    """Put the image read notice in deepagents' middleware. Idempotent.

    Returns False when the installed deepagents has no ``_replace`` to wrap:
    the model then reads the deepagents placeholder, as before.
    """
    global _DEEPAGENTS_REPLACE
    try:
        from deepagents.middleware.unsupported_content import (
            UnsupportedContentMiddleware,
        )
    except ImportError:
        return False
    current = getattr(UnsupportedContentMiddleware, "_replace", None)
    if current is None:
        logger.warning("image_notice.not_installed reason=no_replace_method")
        return False
    if current is _replace:
        return True
    _DEEPAGENTS_REPLACE = current
    UnsupportedContentMiddleware._replace = _replace
    return True
