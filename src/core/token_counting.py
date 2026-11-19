"""Token counts from a model's own tokenizer, cached per message.

Only model types named in ``MODEL_TOKENIZERS`` get one. Everyone else keeps
the graph's character estimate. The tokenizer file is loaded from disk once
per process and is never downloaded here: a missing file logs once and the
counter is ``None``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections import OrderedDict
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Per-message overhead of langchain's count_tokens_approximately.
_PER_MESSAGE = 3
_CACHE_SIZE = 4096

_DEFAULT_TOKENIZERS = {
    "main": "tokenizers/eve-instruct/tokenizer.json",
    "eve_jsc": "tokenizers/eve-instruct/tokenizer.json",
}

_counters: dict[str, Any] = {}
_warned: set[str] = set()


def clear_tokenizer_cache_for_tests() -> None:
    _counters.clear()
    _warned.clear()


def tokenizer_paths() -> dict[str, str]:
    raw = os.getenv("MODEL_TOKENIZERS", "").strip()
    if not raw:
        return dict(_DEFAULT_TOKENIZERS)
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("MODEL_TOKENIZERS must be a JSON object")
    return {str(key): str(value) for key, value in parsed.items()}


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        parts = [content]
    elif isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text") or ""))
            else:
                parts.append(str(block))
    else:
        parts = [str(content or "")]
    for call in getattr(message, "tool_calls", None) or []:
        if isinstance(call, dict):
            parts.append(json.dumps(call.get("args", {}), sort_keys=True, default=str))
    return "\n".join(parts)


class TokenizerCounter:
    """Callable ``(messages, tools=None) -> int`` with a per-text cache."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._cache: OrderedDict[bytes, int] = OrderedDict()
        self._lock = threading.Lock()

    def __call__(self, messages: Any, tools: Any = None) -> int:
        pending: list[tuple[bytes, str]] = []
        known: list[int] = []
        with self._lock:
            for message in messages:
                key = hashlib.sha256(_message_text(message).encode()).digest()
                count = self._cache.get(key)
                if count is None:
                    pending.append((key, _message_text(message)))
                else:
                    self._cache.move_to_end(key)
                    known.append(count)
        if pending:
            encoded = self._tokenizer.encode_batch(
                [text for _, text in pending], add_special_tokens=False
            )
            with self._lock:
                for (key, _), piece in zip(pending, encoded):
                    count = len(piece.ids)
                    self._cache[key] = count
                    self._cache.move_to_end(key)
                    known.append(count)
                while len(self._cache) > _CACHE_SIZE:
                    self._cache.popitem(last=False)
        total = sum(known) + _PER_MESSAGE * len(list(messages))
        if tools:
            dumped = json.dumps(tools, default=str, sort_keys=True)
            total += len(
                self._tokenizer.encode(dumped, add_special_tokens=False).ids
            )
        return total


def counter_for(llm_type: Optional[str]) -> Optional[Callable]:
    """Tokenizer counter for *llm_type*, or ``None`` when it has none."""
    path = tokenizer_paths().get(llm_type or "")
    if not path:
        return None
    if path in _counters:
        return _counters[path]
    if not os.path.isfile(path):
        if path not in _warned:
            logger.warning("tokenizer file missing: %s", path)
            _warned.add(path)
        _counters[path] = None
        return None
    from tokenizers import Tokenizer

    counter = TokenizerCounter(Tokenizer.from_file(path))
    _counters[path] = counter
    return counter
