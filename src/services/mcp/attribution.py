"""Turn a stored CITATION.cff into the public ``attribution`` payload.

The registry (eve-esa/mcp-tool-registry) requires every server to ship a
CITATION.cff in Citation File Format 1.2.0 and renders it into the server
README with ``scripts/citation.py``. The name and GitHub-handle rules below
follow that script so the app and the README credit authors the same way.

Contact details (``email``, ``tel``, addresses) are never exposed.

References:
    https://github.com/citation-file-format/citation-file-format/blob/1.2.0/schema-guide.md
    https://github.com/eve-esa/mcp-tool-registry/blob/main/scripts/citation.py
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import ValidationError

from src.schemas.mcp_server import (
    MCPServerAttribution,
    MCPServerAuthor,
    MCPServerPreferredCitation,
)

logger = logging.getLogger(__name__)

_GITHUB_HANDLE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")


def _text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _http_url(value: Any) -> Optional[str]:
    """A URL safe to hand to a browser as a link, or None.

    CITATION.cff is only checked against the CFF schema, and that check does
    not enforce the ``uri`` format, so a ``website`` can be ``javascript:...``.
    """
    text = _text(value)
    if not text:
        return None
    parsed = urlparse(text)
    if parsed.scheme in ("http", "https") and parsed.netloc:
        return text
    return None


def _doi_url(doi: str) -> Optional[str]:
    """Link for a DOI: use it unchanged when it is already an http(s) URL."""
    if doi.lower().startswith(("http://", "https://")):
        return _http_url(doi)
    if ":" in doi:
        return None
    return f"https://doi.org/{doi}"


def _author_name(author: Dict[str, Any]) -> Optional[str]:
    if author.get("name"):
        return _text(author["name"])
    parts = [
        author.get("given-names"),
        author.get("name-particle"),
        author.get("family-names"),
        author.get("name-suffix"),
    ]
    return _text(" ".join(str(p) for p in parts if p))


def _authors(raw: Any) -> List[MCPServerAuthor]:
    if not isinstance(raw, list):
        return []
    authors: List[MCPServerAuthor] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = _author_name(entry)
        if not name:
            continue
        alias = _text(entry.get("alias"))
        authors.append(
            MCPServerAuthor(
                name=name,
                given_names=_text(entry.get("given-names")),
                family_names=_text(entry.get("family-names")),
                alias=alias,
                github_url=(
                    f"https://github.com/{alias}"
                    if alias and _GITHUB_HANDLE.match(alias)
                    else None
                ),
                affiliation=_text(entry.get("affiliation")),
                orcid=_http_url(entry.get("orcid")),
                website=_http_url(entry.get("website")),
            )
        )
    return authors


def _license(raw: Any) -> Optional[str]:
    if isinstance(raw, list):
        return _text(" OR ".join(str(item) for item in raw if item))
    return _text(raw)


def _preferred_citation(raw: Any) -> Optional[MCPServerPreferredCitation]:
    if not isinstance(raw, dict) or not _text(raw.get("title")):
        return None
    doi = _text(raw.get("doi"))
    year = _text(raw.get("year")) or (_text(raw.get("date-published")) or "")[:4] or None
    link = _doi_url(doi) if doi else None
    return MCPServerPreferredCitation(
        title=_text(raw["title"]),
        authors=[author.name for author in _authors(raw.get("authors"))],
        year=year,
        doi=doi,
        url=link or _http_url(raw.get("url")),
    )


def attribution_from_citation(
    citation: Any, *, server_name: str = ""
) -> Optional[MCPServerAttribution]:
    """Map a CITATION.cff dict to ``MCPServerAttribution``, or None if absent or unusable.

    A malformed citation is logged and dropped rather than raised: it comes from
    a row another system writes, and one bad row must not fail a list of servers.
    """
    if citation is None:
        return None
    if not isinstance(citation, dict):
        logger.warning("Ignoring non-mapping citation on MCP server '%s'", server_name)
        return None
    try:
        attribution = MCPServerAttribution(
            title=_text(citation.get("title")),
            abstract=_text(citation.get("abstract")),
            authors=_authors(citation.get("authors")),
            license=_license(citation.get("license")),
            repository_url=_http_url(citation.get("repository-code"))
            or _http_url(citation.get("repository"))
            or _http_url(citation.get("url")),
            preferred_citation=_preferred_citation(citation.get("preferred-citation")),
        )
    except (TypeError, ValueError, ValidationError) as exc:
        logger.warning("Ignoring malformed citation on MCP server '%s': %s", server_name, exc)
        return None
    if not attribution.authors and not attribution.title:
        return None
    return attribution
