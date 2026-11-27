"""Chunks from a DoclingDocument (document.json written by eve-esa/document-parser).

Reads the JSON directly, so the backend needs no Docling package. The parts used are stable
across DoclingDocument 1.x: ``body`` and ``groups`` hold the reading order as ``$ref``s to
``texts``, ``tables`` and ``pictures``; every item carries ``prov`` with its page number.

Chunks follow the document: text is grouped under its heading up to ``max_chars`` and never
crosses a heading; a table is one chunk of its own, as Markdown with its caption, so its labels
stay next to its numbers; a figure contributes its caption. Page headers and footers
(``content_layer: furniture``) are left out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Optional

HEADING_LABELS = {"title", "section_header"}
# ATTACHED-DOCS-RETRIEVAL (2026-11-27): a text chunk with fewer words than this joins the
# chunk after it (see _merge_tiny_chunks). Revert: delete this, the function and its call.
MIN_CHUNK_WORDS = 12
SKIPPED_LABELS = {"page_header", "page_footer"}


@dataclass
class Chunk:
    text: str
    section: str
    pages: List[int] = field(default_factory=list)
    element_type: str = "text"

    @property
    def page(self) -> Optional[int]:
        return self.pages[0] if self.pages else None


def _resolve(doc: dict, ref: str) -> Optional[dict]:
    # "#/texts/3" -> doc["texts"][3]; "#/body" -> doc["body"]
    parts = ref.lstrip("#/").split("/")
    node = doc
    try:
        for part in parts:
            node = node[int(part)] if isinstance(node, list) else node[part]
    except (KeyError, IndexError, ValueError, TypeError):
        return None
    return node if isinstance(node, dict) else None


def _pages(item: dict) -> List[int]:
    return sorted({int(p["page_no"]) for p in item.get("prov") or [] if p.get("page_no")})


def _walk(doc: dict, node: dict) -> Iterator[dict]:
    """Items in reading order. Tables and pictures are yielded whole (their captions with them)."""
    for child in node.get("children") or []:
        item = _resolve(doc, child.get("$ref", ""))
        if item is None:
            continue
        ref = item.get("self_ref", "")
        if ref.startswith(("#/tables/", "#/pictures/")):
            yield item
        elif ref.startswith("#/groups/"):
            yield from _walk(doc, item)
        else:
            yield item
            yield from _walk(doc, item)


def _caption(doc: dict, item: dict) -> str:
    parts = []
    for ref in item.get("captions") or []:
        cap = _resolve(doc, ref.get("$ref", ""))
        if cap and cap.get("text"):
            parts.append(cap["text"].strip())
    return " ".join(parts)


def _cell(text: str) -> str:
    return " ".join((text or "").split()).replace("|", "\\|")


def table_markdown(item: dict) -> str:
    data = item.get("data") or {}
    rows, cols = int(data.get("num_rows") or 0), int(data.get("num_cols") or 0)
    if not rows or not cols:
        return ""
    grid = [["" for _ in range(cols)] for _ in range(rows)]
    header_rows = set()
    for cell in data.get("table_cells") or []:
        for r in range(cell.get("start_row_offset_idx", 0), min(cell.get("end_row_offset_idx", 0), rows)):
            for c in range(cell.get("start_col_offset_idx", 0), min(cell.get("end_col_offset_idx", 0), cols)):
                grid[r][c] = _cell(cell.get("text", ""))
            if cell.get("column_header"):
                header_rows.add(r)
    head = 1 if 0 in header_rows or rows == 1 else 0
    lines = []
    if head:
        lines.append("| " + " | ".join(grid[0]) + " |")
    else:
        lines.append("| " + " | ".join(f"Column {i + 1}" for i in range(cols)) + " |")
    lines.append("|" + "---|" * cols)
    for row in grid[head:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def chunks_from_docling(doc: dict, max_chars: int = 1500) -> List[Chunk]:
    max_chars = max(200, int(max_chars))
    chunks: List[Chunk] = []
    section = ""
    buf: List[str] = []
    buf_pages: set = set()

    def flush():
        nonlocal buf, buf_pages
        text = "\n".join(buf).strip()
        if text:
            chunks.append(Chunk(text=text, section=section, pages=sorted(buf_pages)))
        buf, buf_pages = [], set()

    for item in _walk(doc, doc.get("body") or {}):
        if item.get("content_layer", "body") != "body":
            continue
        ref = item.get("self_ref", "")
        label = item.get("label", "")
        if ref.startswith("#/tables/"):
            flush()
            md = table_markdown(item)
            caption = _caption(doc, item)
            text = "\n".join(part for part in (caption, md) if part)
            if text:
                chunks.append(Chunk(text=text, section=section, pages=_pages(item), element_type="table"))
            continue
        if ref.startswith("#/pictures/"):
            caption = _caption(doc, item)
            if caption:
                buf.append(caption)
                buf_pages.update(_pages(item))
            continue
        if label in SKIPPED_LABELS or label == "caption" and item.get("parent", {}).get("$ref", "").startswith(
            ("#/tables/", "#/pictures/")
        ):
            continue
        text = (item.get("text") or "").strip()
        if not text:
            continue
        if label in HEADING_LABELS:
            flush()
            section = text
            continue
        if label == "formula":
            text = f"$${text}$$"
        elif label == "list_item":
            text = f"- {text}"
        if buf and sum(len(b) + 1 for b in buf) + len(text) > max_chars:
            flush()
        # A single paragraph longer than a chunk is split at sentence-ish boundaries.
        while len(text) > max_chars:
            cut = text.rfind(". ", 0, max_chars)
            cut = cut + 1 if cut > max_chars // 2 else max_chars
            buf.append(text[:cut].strip())
            buf_pages.update(_pages(item))
            flush()
            text = text[cut:].strip()
        if text:
            buf.append(text)
            buf_pages.update(_pages(item))
    flush()
    return _merge_tiny_chunks(chunks)  # ATTACHED-DOCS-RETRIEVAL: was ``return chunks``


# ATTACHED-DOCS-RETRIEVAL (2026-11-27). Every chunk is embedded with the file name and section
# in front, so a chunk with almost no text of its own (a page number, a paper id such as
# "Paper L 4 / E73-10202" left alone under a heading) matched any question naming the file
# better than the chunks holding the answer, and was all the model got
# (conversation 6b090b07c755a9689cd90846).
def _merge_tiny_chunks(chunks: List[Chunk]) -> List[Chunk]:
    """Join each text chunk under MIN_CHUNK_WORDS to the next text chunk, else the previous one.

    Tables keep their own chunk. The joined chunk keeps the section of the chunk it joined and
    the pages of both.
    """
    def words(chunk: Chunk) -> int:
        return len(chunk.text.split())

    out: List[Chunk] = []
    pending: Optional[Chunk] = None
    for chunk in chunks:
        if chunk.element_type != "text":
            out.append(chunk)
            continue
        if pending is not None:
            chunk = Chunk(
                text=f"{pending.text}\n{chunk.text}",
                section=chunk.section,
                pages=sorted(set(pending.pages) | set(chunk.pages)),
            )
            pending = None
        if words(chunk) < MIN_CHUNK_WORDS:
            pending = chunk
            continue
        out.append(chunk)
    if pending is not None:
        last = next((i for i in range(len(out) - 1, -1, -1) if out[i].element_type == "text"), None)
        if last is None:
            out.append(pending)
        else:
            prev = out[last]
            out[last] = Chunk(
                text=f"{prev.text}\n{pending.text}",
                section=prev.section,
                pages=sorted(set(prev.pages) | set(pending.pages)),
            )
    return out


def page_count(doc: dict) -> Optional[int]:
    pages = doc.get("pages") or {}
    return len(pages) or None
