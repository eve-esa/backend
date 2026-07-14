"""
Curated image catalog for the demo.

A deliberately minimal, fully static "image source": a small YAML catalog of
publicly reachable images served from the local MinIO bucket. It lets the
answer-generation prompt offer relevant images to embed as markdown, without
any real retrieval. This is intended to be easy to evolve later into proper
retrieval; do not treat it as one today.

Fail-safe by design: any problem reading or parsing the catalog degrades to an
empty catalog and an empty prompt block. Nothing in this module raises.
"""

import logging
import re
from typing import Any, Dict, List

import yaml

from src import config

logger = logging.getLogger(__name__)

# Parsed catalog cached per file path so the YAML is read from disk only once.
_catalog_cache: Dict[str, List[Dict[str, Any]]] = {}

# Minimum query token length considered for keyword matching (drops noise like
# "a", "s", "of").
_MIN_TOKEN_LEN = 2


def _tokenize(text: str) -> List[str]:
    """Split text into lowercase alphanumeric tokens."""
    return [tok for tok in re.split(r"[^a-z0-9]+", text.lower()) if tok]


def _load_catalog(path: str) -> List[Dict[str, Any]]:
    """Load and cache the catalog entries from a YAML file.

    A missing or malformed file is logged as a warning and treated as an empty
    catalog. This function never raises.

    Args:
        path: Filesystem path to the catalog YAML file.

    Returns:
        A list of normalized entries, each with ``title``, ``url``,
        ``description`` and a list of lowercase ``tags``.
    """
    if path in _catalog_cache:
        return _catalog_cache[path]

    entries: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        raw_images = data.get("images") if isinstance(data, dict) else None
        if isinstance(raw_images, list):
            for item in raw_images:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title", "")).strip()
                url = str(item.get("url", "")).strip()
                if not title or not url:
                    continue
                entries.append(
                    {
                        "title": title,
                        "url": url,
                        "description": str(item.get("description", "")).strip(),
                        "tags": [
                            str(tag).strip().lower()
                            for tag in (item.get("tags") or [])
                            if str(tag).strip()
                        ],
                    }
                )
        else:
            logger.warning(
                "Image catalog at %s has no 'images' list; using empty catalog.",
                path,
            )
    except FileNotFoundError:
        logger.warning(
            "Image catalog file not found at %s; using empty catalog.", path
        )
    except Exception:
        logger.warning(
            "Failed to load image catalog from %s; using empty catalog.",
            path,
            exc_info=True,
        )

    _catalog_cache[path] = entries
    return entries


def _select_entries(
    query: str, entries: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Return catalog entries whose title/tags overlap the query keywords.

    The catalog is tiny, so recall beats precision: if the keyword filter would
    return nothing, the full catalog is returned instead.
    """
    query_tokens = {tok for tok in _tokenize(query) if len(tok) >= _MIN_TOKEN_LEN}
    if not query_tokens:
        return entries

    matched: List[Dict[str, Any]] = []
    for entry in entries:
        keywords = set(_tokenize(entry["title"]))
        for tag in entry["tags"]:
            keywords.update(_tokenize(tag))
        if query_tokens & keywords:
            matched.append(entry)
    return matched or entries


def build_image_context(query: str) -> str:
    """Build a compact prompt block advertising relevant curated images.

    Args:
        query: The user's question, used for best-effort keyword filtering.

    Returns:
        An empty string when the catalog is disabled or empty, otherwise a
        block listing candidate images and instructing the model on how to
        embed them as markdown.
    """
    if not config.IMAGE_CATALOG_ENABLED:
        return ""

    entries = _load_catalog(config.IMAGE_CATALOG_PATH)
    if not entries:
        return ""

    selected = _select_entries(query or "", entries)

    lines = ["AVAILABLE IMAGES (local curated catalog):"]
    for entry in selected:
        lines.append(f"- {entry['title']} | {entry['description']} | {entry['url']}")
    lines.append(
        "If the user asks for images/pictures/visuals, or an image would "
        "materially help the answer, you MUST embed 1-3 of the MOST relevant "
        "images above using exact markdown: ![<title>](<url>). Use ONLY urls "
        "from this list, never invent or modify image URLs. If none are "
        "relevant, embed none and do not mention this catalog.\n"
        "Embedding an image this way IS how you display it: the chat renders "
        "the markdown as an inline image, so this is not affected by any "
        "general statement elsewhere about not being able to view or generate "
        "images - that limitation is about interpreting images the user "
        "uploads or creating new pixels, not about presenting these catalog "
        "images. Never say you cannot display, show, view or open images when "
        "catalog images are available. Never print a bare URL, a 'copy this "
        "link' instruction, or ask the user to open the URL themselves - the "
        "markdown embed above is the only acceptable way to present these "
        "images."
    )
    return "\n".join(lines)


def append_image_context(text: str, query: str) -> str:
    """Append the image prompt block to ``text`` when the catalog is active.

    Returns ``text`` unchanged when the block is empty (catalog disabled or
    empty), guaranteeing byte-identical prompts in that case.
    """
    block = build_image_context(query)
    if not block:
        return text
    return f"{text}\n\n{block}"
