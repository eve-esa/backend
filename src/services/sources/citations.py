"""Run-local handles and the inline citation rewrite (sources standard, section 4).

``SourceRegistry`` gives every Source of one run a handle, ``S1``, ``S2``, ...,
in arrival order; the same Source seen twice keeps its first handle.
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
from typing import Any, Dict, List, Mapping, Optional

from src.services.sources.descriptors import minimal_source, public_source

CITATION_INSTRUCTION = (
    "Each passage carries a ref such as S1. After a sentence that uses a passage, "
    "write its ref in square brackets, for example [S1] or [S1][S3]. "
    "Only use refs that appear in tool results."
)

# A complete marker: one or more handles, separated by commas or semicolons.
_MARKER = re.compile(r"\[(S\d{1,4}(?:\s*[,;]\s*S\d{1,4})*)\]")
# Any prefix of a marker that could still complete.
_MARKER_PREFIX = re.compile(r"\[(?:S\d{0,4}(?:\s*[,;]\s*(?:S\d{0,4})?)*)?")
_HANDLE = re.compile(r"S\d{1,4}")
# The longest prefix held back before it is let through as text.
MAX_HOLD_CHARS = 40


@dataclass
class SourceRegistry:
    """The Sources of one run, by handle and by id."""

    sources: List[Dict[str, Any]] = field(default_factory=list)
    _by_id: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _by_handle: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def register(self, source: Dict[str, Any]) -> str:
        """Add ``source`` and return its handle; a known id keeps its handle."""
        known = self._by_id.get(source["id"])
        if known is not None:
            return known["handle"]
        handle = f"S{len(self.sources) + 1}"
        source["handle"] = handle
        self.sources.append(source)
        self._by_id[source["id"]] = source
        self._by_handle[handle] = source
        return handle

    def get(self, handle: str) -> Optional[Dict[str, Any]]:
        return self._by_handle.get(handle)


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
            self.numbers.setdefault(sid, n)
            if self._last is not None and self._last[0] == n and self._last[2] == self._emitted:
                at = self._last[1]
            else:
                at = self._emitted
                parts.append(self._take(f"[{n}]"))
                self._last = (n, at, self._emitted)
            offsets = self.offsets.setdefault(sid, [])
            if at not in offsets:
                offsets.append(at)
        return "".join(parts)

    # ── what the client and the stored message read ──────────────────────

    def citations(self) -> List[Dict[str, Any]]:
        """Cited Sources in display order, each with ``n``, ``at`` and ``origin``."""
        by_id = {s["id"]: s for s in self._registry.sources}
        out: List[Dict[str, Any]] = []
        order = sorted(
            self.numbers.items(),
            key=lambda kv: (kv[1], (self.offsets.get(kv[0]) or [0])[0]),
        )
        for sid, n in order:
            source = by_id.get(sid)
            if source is None:
                continue
            entry = public_source(source)
            entry.update({"n": n, "at": list(self.offsets.get(sid, [])), "origin": "model"})
            out.append(entry)
        return out

    def sources_retrieved(self) -> List[Dict[str, Any]]:
        """Sources of the run nobody cited, minimal fields, in arrival order."""
        return [minimal_source(s) for s in self._registry.sources if s["id"] not in self.numbers]


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
