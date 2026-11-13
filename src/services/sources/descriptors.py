"""Source descriptors: one JSON object per citable unit (sources standard, section 1).

A unit is a retrieved passage or a STAC item. Two passages of one document are
two Sources that share ``document_key``. The ``retrieval`` block (rank, scores,
point id) stays on the server: ``public_source`` drops it before a Source
leaves the process, and nothing here ever reaches the model.

Pure: no I/O, no config, no logging of payload text.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional

from src.constants import (
    PUBLIC_COLLECTION_SLUG_PREFIXES,
    PUBLIC_COLLECTIONS,
    WILEY_PUBLIC_COLLECTIONS,
)

PASSAGE_MAX_CHARS = 600
# Licensed content (Wiley) until the licence answer: half the general cap.
LICENSED_PASSAGE_MAX_CHARS = 300

# Same value as src/services/conversation_documents.py (not imported: that
# module pulls the database models into this pure one).
CONVERSATION_COLLECTION_NAME = "Attached documents"

_SERVER_ONLY_KEYS = ("retrieval",)
_MINIMAL_KEYS = (
    "id",
    "handle",
    "kind",
    "provider",
    "provider_label",
    "slug",
    "visibility",
    "title",
    "document_key",
    "locator",
    "open",
    "tool_call_id",
)
# Locator keys that tell two passages of one document apart.
_POSITIONAL_LOCATOR_KEYS = frozenset(
    {"page", "page_end", "section", "table", "figure", "line_start", "line_end", "stac_item_id", "chunk"}
)
# The chunk position ingestion appends to a document id (Wiley:
# ``10.1155/2022/7936392_0000``): not part of the DOI or the document.
_CHUNK_SUFFIX = re.compile(r"_(\d{4})$")
# Markdown the passage drops (sources standard: ``passage`` is plain text).
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$")
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_REF_LINK = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")
_MD_BOLD = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1")
_MD_ITALIC_STAR = re.compile(r"(?<![\w*])\*(?=\S)([^*]+?)(?<=\S)\*(?![\w*])")
_MD_ITALIC_UNDERSCORE = re.compile(r"(?<![\w_])_(?=\S)([^_]+?)(?<=\S)_(?![\w_])")
_MD_STRIKE = re.compile(r"~~(?=\S)(.+?)(?<=\S)~~")
_MD_CODE = re.compile(r"`+([^`]*)`+")
_MD_LINE_MARK = re.compile(r"^\s*(?:>\s?)+|^\s*(?:[-*+]|\d{1,3}[.)])\s+")
_HTML_TAG = re.compile(r"</?[A-Za-z][^<>]*>")
_WEB_PAGE_PROVIDERS = frozenset({"esa-rag-scraped-qwen3-newpipeline", "wikipedia-512"})
_SENTENCE_END = re.compile(r"[.!?](?=\s)|\n")
_TILE = re.compile(r"_T(\d{2}[A-Z]{3})_")
_LEVEL = re.compile(r"-(l[0-9][a-z]?)(?:-|$)", re.IGNORECASE)
_SLUG_SEPARATORS = re.compile(r"[^a-z0-9]+")
# Providers of the user's own files: no slug, the UI names them "Your files".
_PRIVATE_PROVIDERS = frozenset({"my-collections", "conversation"})


def _wiley_labels() -> Dict[str, str]:
    return {c["name"]: c["name"] for c in WILEY_PUBLIC_COLLECTIONS}


def _public_by_label() -> Dict[str, Dict[str, str]]:
    """Public collection by its alias or its name: retrieval reports the alias."""
    out: Dict[str, Dict[str, str]] = {}
    for c in PUBLIC_COLLECTIONS:
        entry = {"key": c["name"], "label": c.get("alias") or c["name"], "slug": _catalog_slug(c)}
        out[c["name"]] = entry
        if c.get("alias"):
            out[c["alias"]] = entry
    return out


def derive_slug(value: Any) -> Optional[str]:
    """Lowercase letters and digits, every other run of characters one ``-``; None when empty."""
    slug = _SLUG_SEPARATORS.sub("-", str(value or "").lower()).strip("-")
    return slug or None


def _catalog_slug(entry: Mapping[str, Any]) -> Optional[str]:
    return entry.get("slug") or derive_slug(entry.get("alias") or entry.get("name"))


def collection_slug(name: Any) -> Optional[str]:
    """The provider slug of a public or licensed collection, by name or alias.

    Catalog entries (``src/constants.py``) give theirs; a name outside the
    catalog takes the slug of a known prefix, else one derived from the name.
    """
    name = str(name or "")
    for entry in list(WILEY_PUBLIC_COLLECTIONS) + list(PUBLIC_COLLECTIONS):
        if name in (entry.get("name"), entry.get("alias")):
            return _catalog_slug(entry)
    for prefix, slug in PUBLIC_COLLECTION_SLUG_PREFIXES.items():
        if name.startswith(prefix):
            return slug
    return derive_slug(name)


def slug_for_provider(provider: Any, kind: Any = None) -> Optional[str]:
    """The slug of a stored Source's ``provider`` key: None for the user's own files.

    For Sources stored before ``slug`` existed.
    """
    if kind == "user_file" or provider in _PRIVATE_PROVIDERS:
        return None
    return collection_slug(provider)


def source_id(provider: str, document_key: str, locator: Mapping[str, Any], passage: str = "") -> str:
    """``src_`` plus 16 hex of SHA-256 over provider, document key and locator.

    When the locator holds nothing positional (no page, section, line or STAC
    id), the passage joins the digest, so two passages of one document still
    get two ids.
    """
    parts: List[Any] = [provider, document_key, dict(locator)]
    if not _POSITIONAL_LOCATOR_KEYS.intersection(locator):
        parts.append(passage)
    digest = hashlib.sha256(
        json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()
    return f"src_{digest[:16]}"


def split_chunk_suffix(value: str) -> tuple:
    """``(value without the chunk suffix, chunk index or None)``."""
    match = _CHUNK_SUFFIX.search(value)
    if not match:
        return value, None
    return value[: match.start()], int(match.group(1))


def markdown_to_text(text: str) -> tuple:
    """``(plain text, last heading or None)`` for a markdown passage.

    Headings leave the text (the last one is returned for the locator),
    links and images keep their label, emphasis, code marks, quote and list
    marks and HTML tags go. Lines are joined with one space. Snake case and
    arithmetic (``band_ratio``, ``2*3``) are left alone.
    """
    heading: Optional[str] = None
    lines: List[str] = []
    for line in str(text or "").splitlines():
        match = _MD_HEADING.match(line)
        if match:
            heading = match.group(1).strip() or heading
            continue
        line = _MD_LINE_MARK.sub("", line)
        if line.strip() in ("", "---", "***", "___"):
            continue
        lines.append(line.strip())
    plain = " ".join(lines)
    plain = _MD_IMAGE.sub(r"\1", plain)
    plain = _MD_LINK.sub(r"\1", plain)
    plain = _MD_REF_LINK.sub(r"\1", plain)
    plain = _MD_CODE.sub(r"\1", plain)
    plain = _HTML_TAG.sub("", plain)
    plain = _MD_BOLD.sub(r"\2", plain)
    plain = _MD_STRIKE.sub(r"\1", plain)
    plain = _MD_ITALIC_STAR.sub(r"\1", plain)
    plain = _MD_ITALIC_UNDERSCORE.sub(r"\1", plain)
    plain = re.sub(r"\s+", " ", plain).strip()
    if not plain and heading:
        plain = heading
    return plain, heading


def cut_passage(text: str, limit: int) -> str:
    """At most ``limit`` characters, cut after the last sentence end that fits.

    No sentence end in the window: cut at the last space and mark the cut.
    """
    text = re.sub(r"[ \t]+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    window = text[:limit]
    ends = [m.end() for m in _SENTENCE_END.finditer(window)]
    if ends and ends[-1] >= limit // 3:
        return window[: ends[-1]].strip()
    space = window.rfind(" ")
    head = window[: space if space > 0 else limit - 3].rstrip()
    return f"{head}..."


def _first(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def _http(url: Any) -> Optional[str]:
    if isinstance(url, str) and url.lower().startswith(("http://", "https://")):
        return url
    return None


def _doi_from(*values: Any) -> Optional[str]:
    for value in values:
        if not isinstance(value, str) or not value:
            continue
        value = value.strip()
        lowered = value.lower()
        for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "doi:"):
            if lowered.startswith(prefix):
                value = value[len(prefix):]
                break
        value = split_chunk_suffix(value.split("#", 1)[0])[0]
        if value.startswith("10.") and "/" in value:
            return value
    return None


def _int_or_none(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _merged_metadata(doc: Mapping[str, Any]) -> Dict[str, Any]:
    """Payload, metadata and Wiley ``additionalMetadata`` read as one dict (first wins)."""
    payload = doc.get("payload") if isinstance(doc.get("payload"), Mapping) else {}
    metadata = doc.get("metadata") if isinstance(doc.get("metadata"), Mapping) else {}
    nested = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
    merged: Dict[str, Any] = {}
    for layer in (
        payload,
        metadata,
        nested,
        metadata.get("additionalMetadata") if isinstance(metadata.get("additionalMetadata"), Mapping) else {},
        nested.get("additionalMetadata") if isinstance(nested.get("additionalMetadata"), Mapping) else {},
    ):
        for key, value in layer.items():
            if value not in (None, "", [], {}) and key not in merged:
                merged[key] = value
    return merged


def _is_private(doc: Mapping[str, Any], merged: Mapping[str, Any]) -> bool:
    payload = doc.get("payload") if isinstance(doc.get("payload"), Mapping) else {}
    return bool(payload.get("user_id") or payload.get("collection_id") or (
        merged.get("document_id") and (merged.get("source_name") or merged.get("filename"))
    ))


def _provider_for(doc: Mapping[str, Any], merged: Mapping[str, Any]) -> Dict[str, Any]:
    """``{key, label, slug, visibility, kind}`` for one retrieval document."""
    name = str(doc.get("collection_name") or "")
    if _is_private(doc, merged):
        if name == CONVERSATION_COLLECTION_NAME:
            return {
                "key": "conversation",
                "label": "This conversation",
                "slug": None,
                "visibility": "private",
                "kind": "user_file",
            }
        return {
            "key": "my-collections",
            "label": "My collections",
            "slug": None,
            "visibility": "private",
            "kind": "user_file",
        }
    if name in _wiley_labels():
        return {
            "key": "wiley",
            "label": name,
            "slug": collection_slug(name),
            "visibility": "licensed",
            "kind": "document",
        }
    public = _public_by_label().get(name)
    if public is not None:
        kind = "web_page" if public["key"] in _WEB_PAGE_PROVIDERS else "document"
        return {
            "key": public["key"],
            "label": public["label"],
            "slug": public["slug"],
            "visibility": "public",
            "kind": kind,
        }
    key = name or "unknown"
    return {
        "key": key,
        "label": name or "Unknown source",
        "slug": collection_slug(key),
        "visibility": "public",
        "kind": "document",
    }


def _passage_text(doc: Mapping[str, Any]) -> str:
    payload = doc.get("payload") if isinstance(doc.get("payload"), Mapping) else {}
    for candidate in (doc.get("text"), payload.get("text"), payload.get("content")):
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return ""


def source_from_document(
    doc: Mapping[str, Any],
    *,
    rank: int,
    tool_call_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """A Source for one Document-shaped retrieval record (``extract_document_data``).

    None for a record with no text: there is no passage to cite.
    """
    if not isinstance(doc, Mapping):
        return None
    text = _passage_text(doc)
    if not text:
        return None
    merged = _merged_metadata(doc)
    provider = _provider_for(doc, merged)
    kind = provider["kind"]

    file_name = _first(merged.get("source_name"), merged.get("filename"), merged.get("source"))
    url = _http(_first(merged.get("url"), merged.get("link")))
    if url and "doi.org/" in url.lower():
        # A DOI link carries the same chunk suffix as the DOI.
        url = split_chunk_suffix(url)[0]
    doi = _doi_from(merged.get("doi"), merged.get("link"), merged.get("url"), doc.get("id"))
    document_id = merged.get("document_id") if isinstance(merged.get("document_id"), str) else None
    if kind == "user_file":
        title = str(file_name or merged.get("title") or "Untitled file")
    else:
        title = str(_first(merged.get("title"), url, doi and f"doi:{doi}", "Untitled"))

    if kind == "user_file" and document_id:
        document_key = f"doc:{document_id}"
    elif doi:
        document_key = f"doi:{doi}"
    elif url:
        document_key = f"url:{url}"
    else:
        document_key = f"title:{title}"

    plain, heading = markdown_to_text(text)
    if not plain:
        return None

    locator: Dict[str, Any] = {}
    chunk = next(
        (
            index
            for index in (
                split_chunk_suffix(v)[1] for v in (doc.get("id"), merged.get("doi")) if isinstance(v, str)
            )
            if index is not None
        ),
        None,
    )
    page = _int_or_none(_first(merged.get("page"), merged.get("page_number")))
    if page is not None:
        # As stored by ingestion: whether chunk payloads keep a PDF page, and
        # from which base, is an open point of the standard (section 5).
        locator["page"] = page
    headers = merged.get("headers")
    if isinstance(headers, list) and headers:
        locator["section"] = str(headers[-1])
    elif isinstance(merged.get("section"), str):
        locator["section"] = merged["section"]
    elif heading:
        locator["section"] = heading
    if chunk is not None:
        locator["chunk"] = chunk
    if url:
        locator["url"] = url
    if doi:
        locator["doi"] = doi

    limit = LICENSED_PASSAGE_MAX_CHARS if provider["visibility"] == "licensed" else PASSAGE_MAX_CHARS
    passage = cut_passage(plain, limit)

    if kind == "user_file":
        open_target: Dict[str, Any] = {"type": "viewer", "document_id": document_id}
        if "page" in locator:
            open_target["page"] = locator["page"]
    elif url:
        open_target = {"type": "url", "href": url}
    elif doi:
        open_target = {"type": "url", "href": f"https://doi.org/{doi}"}
    else:
        open_target = {"type": "none"}

    source: Dict[str, Any] = {
        "id": source_id(provider["key"], document_key, locator, passage),
        "kind": kind,
        "provider": provider["key"],
        "provider_label": provider["label"],
        "slug": provider["slug"],
        "visibility": provider["visibility"],
        "title": title,
        "document_key": document_key,
        "passage": passage,
        "locator": locator,
        "open": open_target,
    }
    meta: Dict[str, Any] = {}
    authors = merged.get("authors")
    if isinstance(authors, list) and authors:
        meta["authors"] = [str(a) for a in authors][:10]
    year = _int_or_none(merged.get("year"))
    if year:
        meta["year"] = year
    published = merged.get("publicationDate") or merged.get("published")
    if isinstance(published, str):
        meta["published"] = published
        if "year" not in meta and published[:4].isdigit():
            meta["year"] = int(published[:4])
    venue = _first(merged.get("journal"), merged.get("journalTitle"), merged.get("venue"))
    if isinstance(venue, str):
        meta["venue"] = venue
    if meta:
        source["meta"] = meta
    if kind == "user_file":
        file_block: Dict[str, Any] = {"name": title}
        mime = merged.get("file_type") or merged.get("mime")
        if isinstance(mime, str):
            file_block["mime"] = mime
        if document_id:
            file_block["document_id"] = document_id
        payload = doc.get("payload") if isinstance(doc.get("payload"), Mapping) else {}
        collection_id = payload.get("collection_id") or merged.get("collection_id")
        if provider["key"] == "conversation" and isinstance(collection_id, str):
            # The conversation collection's id is the conversation id.
            file_block["conversation_id"] = collection_id
        elif isinstance(collection_id, str):
            file_block["collection_id"] = collection_id
        source["file"] = file_block
    if tool_call_id:
        source["tool_call_id"] = tool_call_id
    source["retrieval"] = {
        "rank": rank,
        "score": doc.get("score"),
        "reranking_score": doc.get("reranking_score"),
        "collection": doc.get("collection_name"),
        "point_id": doc.get("id"),
    }
    return source


def _stac_title(item: Mapping[str, Any]) -> str:
    item_id = str(item.get("id") or "")
    platform = item.get("platform")
    collection = str(item.get("collection") or "")
    parts: List[str] = []
    if isinstance(platform, str) and platform:
        name = platform.replace("sentinel-", "Sentinel-").replace("landsat-", "Landsat-")
        if name[-1:].isalpha() and name[-2:-1].isdigit():
            name = name[:-1] + name[-1].upper()
        level = _LEVEL.search(collection)
        parts.append(f"{name} {level.group(1).upper()}" if level else name)
    elif collection:
        parts.append(collection)
    tile = _TILE.search(item_id)
    if tile:
        parts.append(f"tile {tile.group(1)}")
    when = item.get("datetime")
    if isinstance(when, str):
        try:
            day = datetime.fromisoformat(when.replace("Z", "+00:00"))
            parts.append(f"{day.day} {day.strftime('%B %Y')}")
        except ValueError:
            pass
    return ", ".join(parts) if parts else (item_id or "STAC item")


def source_from_stac_item(
    item: Mapping[str, Any],
    *,
    provider: str,
    rank: int,
    tool_call_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """A Source for one item of a ``search_stac`` result (``summarise_item`` shape)."""
    if not isinstance(item, Mapping) or not isinstance(item.get("id"), str) or not item["id"]:
        return None
    item_id = item["id"]
    collection = item.get("collection") if isinstance(item.get("collection"), str) else ""
    locator: Dict[str, Any] = {"stac_item_id": item_id}
    if collection:
        locator["stac_collection"] = collection
    if isinstance(item.get("datetime"), str):
        locator["datetime"] = item["datetime"]
    if isinstance(item.get("bbox"), list):
        locator["bbox"] = item["bbox"]
    document_key = f"stac:{collection}/{item_id}" if collection else f"stac:{item_id}"
    href = _http(item.get("self"))
    source: Dict[str, Any] = {
        "id": source_id(provider, document_key, locator),
        "kind": "stac_item",
        "provider": provider,
        "provider_label": provider,
        "slug": derive_slug(provider),
        "visibility": "public",
        "title": _stac_title(item),
        "document_key": document_key,
        "locator": locator,
        "open": {"type": "stac", "href": href} if href else {"type": "none"},
    }
    meta: Dict[str, Any] = {}
    cloud = item.get("cloud_cover")
    if isinstance(cloud, (int, float)) and not isinstance(cloud, bool):
        meta["cloud_cover"] = cloud
    if isinstance(item.get("platform"), str):
        meta["platform"] = item["platform"]
    if meta:
        source["meta"] = meta
    if tool_call_id:
        source["tool_call_id"] = tool_call_id
    source["retrieval"] = {"rank": rank}
    return source


# Keys a Scholar Gateway result may hold its passage under (same list as the
# frontend's ``messageDocuments.ts``).
_CHUNK_TEXT_KEYS = ("text", "content", "document", "chunk", "passage", "snippet", "page_content")


def wiley_envelope(value: Any) -> Optional[Dict[str, Any]]:
    """The Scholar Gateway envelope ``{success, query, count, results}`` in ``value``, or None.

    ``value`` is a document's text: the envelope object, or its JSON string.
    Same test as ``isWileyEnvelope`` in the frontend.
    """
    if isinstance(value, str):
        if not value.lstrip().startswith("{"):
            return None
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(value, dict):
        return None
    results = value.get("results")
    if not isinstance(results, list) or not results:
        return None
    if value.get("success") is True or isinstance(value.get("query"), str) or (
        isinstance(value.get("count"), (int, float)) and not isinstance(value.get("count"), bool)
    ):
        return value
    for item in results:
        if isinstance(item, dict) and (
            isinstance(item.get("chunk_index"), int)
            or (isinstance(item.get("metadata"), dict) and isinstance(item["metadata"].get("additionalMetadata"), dict))
        ):
            return value
    return None


def envelope_result_text(result: Any) -> str:
    """The passage of one envelope result, "" when it has none."""
    if not isinstance(result, Mapping):
        return ""
    for key in _CHUNK_TEXT_KEYS:
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def envelope_result_document(result: Any, collection_name: Any) -> Optional[Dict[str, Any]]:
    """One envelope result as a Document-shaped record ``source_from_document`` reads."""
    text = envelope_result_text(result)
    if not text:
        return None
    metadata = result.get("metadata") if isinstance(result.get("metadata"), Mapping) else {}
    return {
        "id": result.get("id"),
        "collection_name": collection_name,
        "text": text,
        "payload": {},
        "metadata": dict(metadata),
    }


def is_licensed_collection(name: Any) -> bool:
    return isinstance(name, str) and name in _wiley_labels()


def cap_licensed_document(doc: Any) -> Any:
    """A licensed (Wiley) document with every passage cut to the licence cap.

    For what the user sees: ``Message.documents`` and the Sources panel. The
    model context is not cut. Any other document is returned unchanged.
    """
    if not isinstance(doc, dict) or not is_licensed_collection(doc.get("collection_name")):
        return doc

    def cut(text: str) -> str:
        return text if len(text) <= LICENSED_PASSAGE_MAX_CHARS else cut_passage(text, LICENSED_PASSAGE_MAX_CHARS)

    def cap(value: Any) -> Any:
        if isinstance(value, str):
            envelope = wiley_envelope(value)
            if envelope is not None:
                return json.dumps(cap(envelope), ensure_ascii=False)
            return cut(value)
        envelope = wiley_envelope(value)
        if envelope is not None:
            results = []
            for item in envelope["results"]:
                if isinstance(item, dict):
                    item = {
                        k: (cut(v) if k in _CHUNK_TEXT_KEYS and isinstance(v, str) else v)
                        for k, v in item.items()
                    }
                results.append(item)
            return {**envelope, "results": results}
        return value

    capped = dict(doc)
    if "text" in capped:
        capped["text"] = cap(capped["text"])
    payload = capped.get("payload")
    if isinstance(payload, dict):
        capped["payload"] = {k: (cap(v) if k in ("text", "content") else v) for k, v in payload.items()}
    return capped


def public_source(source: Mapping[str, Any]) -> Dict[str, Any]:
    """The Source as the browser and the stored message see it: no ``retrieval``."""
    return {k: v for k, v in source.items() if k not in _SERVER_ONLY_KEYS}


def minimal_source(source: Mapping[str, Any]) -> Dict[str, Any]:
    """The few fields an uncited Source keeps: no passage, no meta, no retrieval."""
    return {k: source[k] for k in _MINIMAL_KEYS if k in source}


def sources_by_handle(sources: Iterable[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    return {s["handle"]: s for s in sources if isinstance(s.get("handle"), str)}
