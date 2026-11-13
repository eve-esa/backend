"""Conversation handles and the inline citation rewrite (sources standard, section 4).

``SourceRegistry`` gives every Source of one run a handle, ``S1``, ``S2``, ...,
in arrival order; the same Source seen twice keeps its first handle. Handles
are conversation scoped: the tool results that carry them stay in the
checkpoint thread the model reads on the next turn, so a run is seeded with
the Sources earlier turns stored (``seed``), continues after the highest
handle, and resolves an earlier handle to its stored Source.
``CitationRewriter`` sits on the answer token stream: it holds back a possible
``[S..]`` marker until it is complete, turns each known handle into ``[n]``,
records where it put the marker, and drops an unknown handle. Every recorded
marker maps to a Source of this run by construction.

Numbers go to cited documents (``document_key``) in order of first citation:
two passages of one document share ``n``. A marker that repeats the number
written right before it (``[S1][S2]`` on two passages of one document) is not
written twice; both passages list that one offset.

Answer text format (what the frontend parses): ``[n]`` at a code point offset
listed in a citation's ``at``. Only those offsets are markers; a literal
``[1]`` the model wrote elsewhere is plain text.

Pure: no I/O, no config. Logging is the caller's, with counts only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

from src.services.sources.descriptors import minimal_source, public_source, slug_for_provider

# The key the handle rides under in the model view, written after every other
# field so a source field of the same name cannot replace it.
HANDLE_KEY = "eve_ref"

CITATION_INSTRUCTION = (
    "Each passage carries an eve_ref such as S1. After a sentence that uses a passage, "
    "write its eve_ref in square brackets, for example [S1] or [S1][S3]. "
    "Only use eve_ref values that appear in tool results."
)

# A complete marker: one or more handles, separated by commas or semicolons.
_MARKER = re.compile(r"\[(S\d{1,4}(?:\s*[,;]\s*S\d{1,4})*)\]")
# Any prefix of a marker that could still complete.
_MARKER_PREFIX = re.compile(r"\[(?:S\d{0,4}(?:\s*[,;]\s*(?:S\d{0,4})?)*)?")
_HANDLE = re.compile(r"S\d{1,4}")
_HANDLE_FULL = re.compile(r"S(\d{1,4})")
# A handle written in source text, in a form the rewriter would read.
_LITERAL_MARKER = re.compile(r"\[\s*(S\d{1,4}(?:\s*[,;]\s*S\d{1,4})*)\s*\]")
_LITERAL_OPEN = re.compile(r"\[(?=\s*S\d)")
# Fields a stored citation carries that are not part of the Source.
_MARKER_FIELDS = ("n", "at", "origin", "alias_handles")
# The longest prefix held back before it is let through as text.
MAX_HOLD_CHARS = 40


@dataclass
class SourceRegistry:
    """The Sources of one run, by handle and by id, over the conversation's earlier ones."""

    sources: List[Dict[str, Any]] = field(default_factory=list)
    _by_id: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _by_handle: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Sources earlier turns stored, by id and by handle: resolvable, not listed.
    _prior_by_id: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _prior_by_handle: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _next: int = 1

    def seed(self, prior: Iterable[Mapping[str, Any]]) -> None:
        """Earlier turns' Sources (``prior_sources_from_rows``): new handles start after theirs."""
        for source in prior:
            handle, sid = source.get("handle"), source.get("id")
            number = _handle_number(handle)
            if number is None or not isinstance(sid, str) or not sid:
                continue
            entry = dict(source)
            self._prior_by_id.setdefault(sid, entry)
            self._prior_by_handle.setdefault(handle, entry)
            self._next = max(self._next, number + 1)

    def register(self, source: Dict[str, Any]) -> str:
        """Add ``source`` and return its handle; a known id keeps its handle."""
        known = self._by_id.get(source["id"])
        if known is not None:
            return known["handle"]
        earlier = self._prior_by_id.get(source["id"])
        if earlier is not None and earlier["handle"] not in self._by_handle:
            # Retrieved again: the thread already names it by this handle.
            handle = earlier["handle"]
        else:
            handle = f"S{self._next}"
            self._next += 1
        source["handle"] = handle
        self.sources.append(source)
        self._by_id[source["id"]] = source
        self._by_handle[handle] = source
        return handle

    def get(self, handle: str) -> Optional[Dict[str, Any]]:
        return self._by_handle.get(handle) or self._prior_by_handle.get(handle)

    def by_id(self, source_id: str) -> Optional[Dict[str, Any]]:
        return self._by_id.get(source_id) or self._prior_by_id.get(source_id)


def _handle_number(handle: Any) -> Optional[int]:
    match = _HANDLE_FULL.fullmatch(handle) if isinstance(handle, str) else None
    return int(match.group(1)) if match else None


def prior_sources_from_rows(rows: Iterable[Any]) -> List[Dict[str, Any]]:
    """The Sources earlier turns stored (``metadata.citations`` and ``sources_retrieved``).

    One entry per id and handle, first seen wins, without the marker fields;
    each ``alias_handles`` value gives one more entry for the same Source under
    that handle. Entries with no id or no ``S<k>`` handle are skipped. An
    entry stored before ``slug`` existed gets it from its ``provider``.
    """
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for row in rows or []:
        metadata = row.get("metadata") if isinstance(row, Mapping) else None
        if not isinstance(metadata, Mapping):
            continue
        for key in ("citations", "sources_retrieved"):
            entries = metadata.get(key)
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                sid = entry.get("id")
                if not isinstance(sid, str) or not sid:
                    continue
                base = {k: v for k, v in entry.items() if k not in _MARKER_FIELDS}
                if "slug" not in base:
                    base["slug"] = slug_for_provider(base.get("provider"), base.get("kind"))
                aliases = entry.get("alias_handles")
                for handle in [entry.get("handle"), *(aliases if isinstance(aliases, list) else [])]:
                    if _handle_number(handle) is None or (sid, handle) in seen:
                        continue
                    seen.add((sid, handle))
                    out.append({**base, "handle": handle})
    return out


def neutralise_handles(value: Any) -> Any:
    """``value`` with every handle written in source text taken out of marker form.

    ``[S3]`` becomes ``(S3)`` and a lone ``[S`` becomes ``(S``, in every string
    of a nested structure, so a passage that quotes a handle cannot be copied
    into the answer and rewritten to a real Source.
    """
    if isinstance(value, str):
        if "[" not in value:
            return value
        return _LITERAL_OPEN.sub("(", _LITERAL_MARKER.sub(r"(\1)", value))
    if isinstance(value, dict):
        return {k: neutralise_handles(v) for k, v in value.items()}
    if isinstance(value, list):
        return [neutralise_handles(v) for v in value]
    return value


class CitationRewriter:
    """Rewrites ``[S<k>]`` handles in streamed answer text to ``[n]`` markers."""

    def __init__(self, registry: SourceRegistry) -> None:
        self._registry = registry
        self._held = ""
        self._emitted = 0  # code points written so far
        self.numbers: Dict[str, int] = {}  # source id -> display number
        self._doc_numbers: Dict[str, int] = {}  # document key -> display number
        self._last: Optional[tuple] = None  # (n, offset, end) of the last marker written
        self.offsets: Dict[str, List[int]] = {}  # source id -> marker offsets
        self.invalid = 0
        self.markers_seen = 0
        # Bumped whenever ``citations()`` would change: a new number or offset.
        self.version = 0

    def feed(self, text: str) -> str:
        """Rewritten text that is safe to send now; a possible marker is held back."""
        if not text:
            return ""
        buffer = self._held + text
        self._held = ""
        out: List[str] = []
        i = 0
        while i < len(buffer):
            start = buffer.find("[", i)
            if start < 0:
                out.append(self._take(buffer[i:]))
                break
            out.append(self._take(buffer[i:start]))
            match = _MARKER.match(buffer, start)
            if match:
                out.append(self._markers(match.group(1)))
                i = match.end()
                continue
            rest = buffer[start:]
            prefix = _MARKER_PREFIX.match(rest)
            if prefix and prefix.end() == len(rest) and len(rest) <= MAX_HOLD_CHARS:
                self._held = rest
                break
            out.append(self._take("["))
            i = start + 1
        return "".join(out)

    @property
    def held(self) -> str:
        """Text held back as a possible marker, not yet sent."""
        return self._held

    def flush(self) -> str:
        """The held text, as plain text: the stream ended inside a possible marker."""
        held, self._held = self._held, ""
        return self._take(held)

    def _take(self, text: str) -> str:
        self._emitted += len(text)
        return text

    def _markers(self, body: str) -> str:
        parts: List[str] = []
        for handle in _HANDLE.findall(body):
            self.markers_seen += 1
            source = self._registry.get(handle)
            if source is None:
                self.invalid += 1
                continue
            sid = source["id"]
            doc_key = source.get("document_key") or sid
            n = self._doc_numbers.setdefault(doc_key, len(self._doc_numbers) + 1)
            if sid not in self.numbers:
                self.numbers[sid] = n
                self.version += 1
            if self._last is not None and self._last[0] == n and self._last[2] == self._emitted:
                at = self._last[1]
            else:
                at = self._emitted
                parts.append(self._take(f"[{n}]"))
                self._last = (n, at, self._emitted)
            offsets = self.offsets.setdefault(sid, [])
            if at not in offsets:
                offsets.append(at)
                self.version += 1
        return "".join(parts)

    # ── what the client and the stored message read ──────────────────────

    def citations(self) -> List[Dict[str, Any]]:
        """Cited Sources in display order, each with ``n``, ``at`` and ``origin``.

        The first entry of a document also lists, under ``alias_handles``, the
        handles of its passages nobody cited (see ``_listed``).
        """
        entries, aliases = self._listed()
        out: List[Dict[str, Any]] = []
        for source in entries:
            entry = public_source(source)
            sid = source["id"]
            entry.update({"n": self.numbers[sid], "at": list(self.offsets.get(sid, [])), "origin": "model"})
            if aliases.get(sid):
                entry["alias_handles"] = aliases[sid]
            out.append(entry)
        return out

    def sources_retrieved(self) -> List[Dict[str, Any]]:
        """One entry per document of the run nobody cited, minimal fields, in arrival order."""
        _, aliases = self._listed()
        cited_keys = self._cited_keys()
        out: List[Dict[str, Any]] = []
        seen: set = set()
        for source in self._registry.sources:
            key = _doc_key(source)
            if source["id"] in self.numbers or key in cited_keys or key in seen:
                continue
            seen.add(key)
            entry = minimal_source(source)
            if aliases.get(source["id"]):
                entry["alias_handles"] = aliases[source["id"]]
            out.append(entry)
        return out

    def _cited_keys(self) -> set:
        keys = set()
        for sid in self.numbers:
            source = self._registry.by_id(sid)
            if source is not None:
                keys.add(_doc_key(source))
        return keys

    def _listed(self) -> tuple:
        """``(cited Sources in display order, {listed id: alias handles})``.

        A document is listed once: by its cited passages, or by its first
        passage in ``sources_retrieved``. The handles of its other passages
        ride the first listed entry as ``alias_handles``, so the next turn can
        still resolve them and never hands their numbers out again.
        """
        order = sorted(
            self.numbers.items(),
            key=lambda kv: (kv[1], (self.offsets.get(kv[0]) or [0])[0]),
        )
        cited: List[Dict[str, Any]] = []
        representative: Dict[str, str] = {}  # document key -> listed id
        for sid, _n in order:
            source = self._registry.by_id(sid)
            if source is None:
                continue
            cited.append(source)
            representative.setdefault(_doc_key(source), sid)
        aliases: Dict[str, List[str]] = {}
        for source in self._registry.sources:
            key = _doc_key(source)
            if source["id"] in self.numbers:
                continue
            listed = representative.setdefault(key, source["id"])
            if listed != source["id"]:
                aliases.setdefault(listed, []).append(source["handle"])
        return cited, aliases


def _doc_key(source: Mapping[str, Any]) -> str:
    return source.get("document_key") or source["id"]


def strip_handles(text: str) -> str:
    """``text`` without any ``[S<k>]`` handle: the rewrite with no known Source.

    For the paths that hand out no handles (the sync agentic route, the flag
    off) but whose model may still copy one from the checkpoint thread.
    """
    rewriter = CitationRewriter(SourceRegistry())
    return rewriter.feed(text or "") + rewriter.flush()


def citation_payload(rewriter: Optional[CitationRewriter]) -> Dict[str, Any]:
    """``{"citations": [...], "sources_retrieved": [...]}``; empty dict when off."""
    if rewriter is None:
        return {}
    return {"citations": rewriter.citations(), "sources_retrieved": rewriter.sources_retrieved()}


def marker_spans(answer: str, citations: List[Mapping[str, Any]]) -> List[str]:
    """The text at every recorded offset (tests and the frontend contract check)."""
    spans: List[str] = []
    for entry in citations:
        marker = f"[{entry['n']}]"
        for at in entry.get("at") or []:
            spans.append(answer[at : at + len(marker)])
    return spans
