"""Split one model call's stream into reasoning and answer text.

Three shapes reach this process, and none of them is OpenAI's:

- vLLM and OpenRouter put reasoning on the delta as ``reasoning_content``
  or ``reasoning``. ``ChatOpenAI`` drops both (langchain-openai 1.6.7).
  The client in ``llm_manager`` copies them onto
  ``additional_kwargs["reasoning_content"]`` before this splitter runs.
  https://docs.vllm.ai/en/latest/features/reasoning_outputs.html
- Mistral sends ``thinking`` content parts. The same client lifts those
  parts into ``reasoning_content`` and leaves the answer text.
  https://docs.mistral.ai/capabilities/reasoning/
- A model with no reasoning parser writes ``<think>`` / ``[THINK]`` tags
  in the content. A chat template may also inject the opening tag, so the
  stream begins inside the block and the first tag is the close
  (Qwen3). https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html

The splitter is the tag half. Provider reasoning is passed in already
extracted and is not scanned for tags. A suffix that could still grow
into a tag is held, so a tag split across chunks is not emitted as text.
"""

from __future__ import annotations

from typing import Tuple

# Opening tag, closing tag. Matched exactly: models emit these spellings.
_TAGS: Tuple[Tuple[str, str], ...] = (
    ("<think>", "</think>"),
    ("[THINK]", "[/THINK]"),
)
_ALL_TAGS: Tuple[str, ...] = tuple(tag for pair in _TAGS for tag in pair)


def _strict_prefix_suffix(text: str, tags: Tuple[str, ...] = _ALL_TAGS) -> str:
    """Longest suffix of ``text`` that is a strict prefix of one of ``tags``."""
    if not text:
        return ""
    limit = max(len(tag) for tag in tags) - 1
    best = ""
    for start in range(max(0, len(text) - limit), len(text)):
        suffix = text[start:]
        if any(tag.startswith(suffix) for tag in tags) and len(suffix) > len(best):
            best = suffix
    return best


def _could_be_close(suffix: str) -> bool:
    """True when ``suffix`` can still grow into a closing tag."""
    return any(close.startswith(suffix) for _, close in _TAGS)


def _earliest_open(text: str) -> Tuple[int, str, str]:
    """Index of the first opening tag, that tag, and its close. -1 if none."""
    found = -1
    open_tag = ""
    close_tag = ""
    for open_, close in _TAGS:
        index = text.find(open_)
        if index != -1 and (found == -1 or index < found):
            found = index
            open_tag = open_
            close_tag = close
    return found, open_tag, close_tag


def _earliest_close(text: str) -> Tuple[int, str]:
    """Index of the first closing tag, and that tag. -1 if none."""
    found = -1
    tag = ""
    for _, close in _TAGS:
        index = text.find(close)
        if index != -1 and (found == -1 or index < found):
            found = index
            tag = close
    return found, tag


class ReasoningSplitter:
    """One model call. ``feed`` returns ``(reasoning, answer)`` for new text.

    Answer text is released as soon as it cannot be part of a tag. Reasoning
    inside an opening tag is released the same way. A stream that is already
    inside a think block (the template wrote the opening tag) stays held
    until the closing tag arrives, so the text before that tag is reasoning
    and not the answer. Once any answer text has been released, a later
    closing tag is ordinary text: streamed tokens are not rewritten.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Drop state. The next ``feed`` is a new model call."""
        self._mode = "answer"
        self._close = ""
        self._hold = ""
        self._answer_emitted = False
        self._drop_newlines = False

    def feed(self, text: str, provider_reasoning: str = "") -> Tuple[str, str]:
        """Take one chunk. ``provider_reasoning`` is already reasoning."""
        provider = provider_reasoning if isinstance(provider_reasoning, str) else ""
        buf = self._hold + (text or "")
        self._hold = ""
        if self._drop_newlines:
            stripped = buf.lstrip("\n")
            if stripped:
                self._drop_newlines = False
            buf = stripped
        reasoning: list[str] = []
        answer: list[str] = []
        while buf:
            if self._mode == "reasoning":
                index = buf.find(self._close) if self._close else -1
                if index == -1:
                    keep = (
                        _strict_prefix_suffix(buf, (self._close,)) if self._close else ""
                    )
                    reasoning.append(buf[: len(buf) - len(keep)] if keep else buf)
                    self._hold = keep
                    break
                reasoning.append(buf[:index])
                buf = self._after_close(buf[index + len(self._close) :])
                continue
            open_at, open_tag, close_tag = _earliest_open(buf)
            close_at, found_close = (
                _earliest_close(buf) if not self._answer_emitted else (-1, "")
            )
            if close_at != -1 and (open_at == -1 or close_at < open_at):
                reasoning.append(buf[:close_at])
                buf = self._after_close(buf[close_at + len(found_close) :])
                continue
            if open_at != -1:
                before = buf[:open_at]
                if before:
                    answer.append(before)
                    self._answer_emitted = True
                buf = buf[open_at + len(open_tag) :]
                self._mode = "reasoning"
                self._close = close_tag
                continue
            keep = _strict_prefix_suffix(buf)
            if keep and not self._answer_emitted and _could_be_close(keep):
                # The closing tag may still finish. Hold the lead so a
                # template-opened think block is not streamed as the answer.
                self._hold = buf
                break
            emit = buf[: len(buf) - len(keep)] if keep else buf
            if emit:
                answer.append(emit)
                self._answer_emitted = True
            self._hold = keep
            break
        return provider + "".join(reasoning), "".join(answer)

    def flush(self) -> Tuple[str, str]:
        """Release the held tail. An unfinished tag is answer text."""
        hold = self._hold
        self._hold = ""
        self._drop_newlines = False
        if not hold:
            return "", ""
        if self._mode == "reasoning" and self._close and self._close.startswith(hold):
            return "", ""
        if self._mode == "reasoning":
            return hold, ""
        self._answer_emitted = True
        return "", hold

    def _after_close(self, rest: str) -> str:
        """Back to answer text. Drop the newline the template puts after the tag."""
        self._mode = "answer"
        self._close = ""
        self._drop_newlines = True
        stripped = rest.lstrip("\n")
        if stripped:
            self._drop_newlines = False
        return stripped


def split_complete(text: str, provider_reasoning: str = "") -> Tuple[str, str]:
    """``(reasoning, answer)`` for one finished message, tags included."""
    splitter = ReasoningSplitter()
    reasoning, answer = splitter.feed(text, provider_reasoning)
    more_reasoning, more_answer = splitter.flush()
    return reasoning + more_reasoning, answer + more_answer
