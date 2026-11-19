"""The text a model without vision reads in place of an image file.

deepagents' ``read_file`` returns an image as an image block. Its
``UnsupportedContentMiddleware`` swaps that block for a text notice when the
model's profile takes no images (``platform_model_profile``): "was not
attached because this model does not support image content". A model that
reads the chart it just saved concludes from that text that it cannot show
the chart, and says so next to the rendered image (dev, 2026-11-19).

The notice built here says what is true and what to do: the model cannot see
the image and must not describe it. For an image the agent wrote in its
workspace it adds how to show it, with markdown the answer link rewrite turns
into a link the chat loads (relative to the workspace). An upload or an
artifact (anything under an ``artifacts`` directory) is never offered back:
the user already has it, tool images are appended by the answer net, and a
model told it may show an upload answered with the upload alone (U8, dev,
2026-11-19). Only image blocks of a ``read_file`` result change; every other block
keeps the deepagents placeholder. A model whose profile takes images never
reaches this code. The thread keeps the image block, so the notice is built
again on every model call and nothing is stored.

The eve-esa/agents deep graph builds its own middleware list and takes none
from the backend, so the notice is set on the deepagents class, once per
process, by :func:`install_image_read_notice`.
"""

from __future__ import annotations

import inspect
import logging
import posixpath
import re
import urllib.parse
from typing import Any, Callable, Optional

from src.services.agents.core.sandbox import known_session_roots, place_in_workspace
from src.services.agents.core.workspace import _workspace_file, session_path_for_markdown

logger = logging.getLogger(__name__)

# The deepagents method replaced by the install, kept to delegate to.
_DEEPAGENTS_REPLACE: Optional[Callable[..., Any]] = None
# The parameters of the deepagents method the install knows how to wrap.
_EXPECTED_PARAMETERS = ("self", "block", "message")
# One warning per process when the install skips.
_SKIP_LOGGED = False

# An ``artifacts`` directory anywhere in the path: the uploads and artifacts
# tree beside the workspace (``/artifacts``, ``<root>/artifacts``), the same
# tree as the model may name it (``artifacts/...``, ``workspace/artifacts/...``).
_ARTIFACTS_DIR = re.compile(r"(?:^|/)artifacts(?:/|$)")
# A file name is user or model text inside an instruction: no line breaks,
# no markdown or tag syntax, bounded length.
_LABEL_DROP = re.compile(r"[`\[\]()<>]")
_LABEL_SPACE = re.compile(r"[\x00-\x1f\x7f]")
_LABEL_MAX = 80
_PATH_MAX = 200


def _size_text(size: int) -> str:
    if size < 1024:
        return f"{size} bytes"
    if size < 1024 * 1024:
        return f"{round(size / 1024)} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def _plain(text: str, limit: int) -> str:
    """``text`` without control characters or markdown syntax, cut at ``limit``."""
    cleaned = " ".join(_LABEL_DROP.sub("", _LABEL_SPACE.sub(" ", text)).split())
    return cleaned[:limit].rstrip()


def _session_root(raw: str) -> str:
    for root in known_session_roots():
        root = root.rstrip("/")
        if root and (raw == root or raw.startswith(root + "/")):
            return root
    return ""


def _published(target: str) -> Optional[str]:
    """The workspace file the answer publishing links for ``target``."""
    return _workspace_file(session_path_for_markdown(target))


def markdown_target(path: str) -> Optional[str]:
    """The markdown target the answer link rewrite turns into a chat link.

    A workspace file is linked relative to the workspace, through the parser
    the answer publishing uses (``session_path_for_markdown``,
    ``_workspace_file``). deepagents passes a bare ``ndvi.png`` as
    ``/ndvi.png`` and the sandbox reads it from the workspace
    (``place_in_workspace``), so that path is read the same way. A target is
    kept only when publishing it links the same file. A path under an
    ``artifacts`` directory (an upload or an artifact) is never a target.
    Anything else: ``None``.
    """
    raw = str(path or "").strip()
    if not raw or ".." in raw.split("/") or _ARTIFACTS_DIR.search(raw):
        return None
    # What the sandbox read first, then the path as written (a real session
    # path whose root this process has not seen).
    for candidate in (place_in_workspace(raw, _session_root(raw)), raw):
        relative = _published(candidate)
        if relative is None:
            continue
        target = urllib.parse.quote(relative, safe="/")
        if _published(target) == relative:
            return target
    return None


def image_read_notice(path: str, size: Optional[int]) -> str:
    """What a model without vision reads after ``read_file`` on an image."""
    raw = str(path or "")
    name = _plain(posixpath.basename(raw.rstrip("/")), _LABEL_MAX) or "image"
    about = f"{name}, {_size_text(size)}" if size is not None else name
    parts = [
        f"[read_file: {_plain(raw, _PATH_MAX)}] This is an image file ({about}).",
        "This model cannot see image content, so do not describe it or guess "
        "what it shows.",
        "If the user asks what it shows, tell them you cannot see images with "
        "the current model.",
    ]
    target = markdown_target(raw)
    if target is not None:
        parts.append(
            f"You can still show it to the user: write ![{name}]({target}) in "
            "your answer."
        )
    return " ".join(parts)


def _base64_size(data: Any) -> Optional[int]:
    """Decoded size of a base64 string, without decoding or copying it."""
    if not isinstance(data, str):
        return None
    length = len(data) - sum(data.count(c) for c in " \t\r\n")
    tail = data[-4:].rstrip()
    padding = min(len(tail) - len(tail.rstrip("=")), 2)
    return max(length * 3 // 4 - padding, 0)


def _notice_block(block: Any, message: Any) -> Optional[dict]:
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
    return None


def _replace(self: Any, block: Any, message: Any) -> Any:
    """``UnsupportedContentMiddleware._replace`` with the image read notice.

    Any error while building the notice falls back to the deepagents
    placeholder: a turn never fails because of the notice.
    """
    try:
        notice = _notice_block(block, message)
    except Exception as exc:  # noqa: BLE001
        logger.warning("image_notice.fallback error=%s", type(exc).__name__)
        notice = None
    if notice is not None:
        return notice
    return _DEEPAGENTS_REPLACE(self, block, message)


def _skip(reason: str) -> bool:
    global _SKIP_LOGGED
    if not _SKIP_LOGGED:
        _SKIP_LOGGED = True
        logger.warning("image_notice.not_installed reason=%s", reason)
    return False


def install_image_read_notice() -> bool:
    """Put the image read notice in deepagents' middleware. Idempotent.

    Returns False, and the model reads the deepagents placeholder as before,
    when the installed deepagents has no ``_replace`` or its parameters are
    not ``(self, block, message)``.
    """
    global _DEEPAGENTS_REPLACE
    try:
        from deepagents.middleware.unsupported_content import (
            UnsupportedContentMiddleware,
        )
    except ImportError:
        return False
    current = getattr(UnsupportedContentMiddleware, "_replace", None)
    if current is _replace:
        return True
    if current is None:
        return _skip("no_replace_method")
    try:
        parameters = tuple(inspect.signature(current).parameters)
    except (TypeError, ValueError):
        return _skip("signature")
    if parameters != _EXPECTED_PARAMETERS:
        return _skip("signature")
    _DEEPAGENTS_REPLACE = current
    UnsupportedContentMiddleware._replace = _replace
    return True
