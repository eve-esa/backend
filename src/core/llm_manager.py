"""
LLM Manager module that handles different language model interactions.
"""

import contextlib
import contextvars
import logging
from enum import Enum
from typing import Any, AsyncGenerator, Dict, Iterator, Mapping, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import PrivateAttr

from src.config import (
    EVE_ENDPOINT_COOLDOWN_S,
    EVE_ENDPOINT_ORDER,
    EVE_JSC_TIMEOUT,
    JSC_LARGE_MODEL_NAME,
    JSC_LARGE_TIMEOUT,
    FALLBACK_MODEL_API_KEY,
    FALLBACK_MODEL_NAME,
    FALLBACK_MODEL_URL,
    MAIN_MODEL_API_KEY,
    MAIN_MODEL_NAME,
    MAIN_MODEL_TIMEOUT,
    MAIN_MODEL_URL,
    MODEL_TIMEOUT,
    SATCOM_LARGE_BASE_URL,
    SATCOM_LARGE_MODEL_NAME,
    SATCOM_RUNPOD_API_KEY,
    SATCOM_SMALL_BASE_URL,
    SATCOM_SMALL_MODEL_NAME,
    Config,
    EVE_JSC_BASE_URL,
    EVE_JSC_MODEL_NAME,
    EVE_JSC_API_KEY,
    IMAGE_INPUT_MODEL_NAMES,
)
from src.constants import DEFAULT_MAX_NEW_TOKENS, MODEL_CONTEXT_SIZE
from src import config as _config
from src.core.llm_health import (
    EndpointHealth,
    EndpointRateLimited,
    RateLimitPermit,
    is_endpoint_failure,
    retry_after_header,
)
from src.services.image_catalog import append_image_context
from src.utils.helpers import (
    str_token_counter,
    trim_text_to_token_limit,
)
from src.utils.template_loader import format_template, get_template

logger = logging.getLogger(__name__)


def platform_model_profile(model_name: Optional[str]) -> Dict[str, bool]:
    """The image fields of a platform client's LangChain ``profile``.

    deepagents' ``UnsupportedContentMiddleware`` reads them: when they are
    False, an image that ``read_file`` returns reaches the model as a text
    notice naming the file and its type. Without a profile the image goes to
    the model as an ``image_url`` part, which EVE-Instruct refuses with a 400,
    and the retry after it says only "Unsupported content", which the model
    answered with an invented description. A model listed in
    ``IMAGE_INPUT_MODEL_NAMES`` keeps the image.
    """
    listed = {name.strip() for name in IMAGE_INPUT_MODEL_NAMES.split(",") if name.strip()}
    accepts = bool(model_name) and model_name in listed
    return {"image_inputs": accepts, "image_tool_message": accepts}


class LLMType(Enum):
    """Enum for supported LLM types."""

    Main = "main"
    Fallback = "fallback"
    Satcom_Small = "satcom_small"
    Satcom_Large = "satcom_large"
    Eve_Jsc = "eve_jsc"
    Jsc_Large = "jsc_large"
    # Legacy aliases for backward compatibility
    Runpod = "main"
    Mistral = "fallback"


# Enum aliases collapse onto their canonical member, so LLMType("runpod") raises
# and the legacy names need their own mapping.
_LEGACY_LLM_TYPES = {"runpod": LLMType.Main.value, "mistral": LLMType.Fallback.value}
_KNOWN_LLM_TYPES = {member.value for member in LLMType}

# The Satcom models are optional and unset in every env (D03), and LLMManager is built
# per turn by the hallucination paths: warn about the missing URLs once per process.
_satcom_missing_warned = False


def canonical_llm_type(llm_type: Optional[str]) -> Optional[str]:
    """Map a requested llm_type, legacy spelling included, onto its canonical name."""
    if not llm_type:
        return None
    return _LEGACY_LLM_TYPES.get(llm_type, llm_type)


def endpoint_timeout(llm_type: Optional[str]) -> int:
    """First-token budget for one endpoint.

    RunPod is serverless and cold-starts for minutes; JSC is a warm vLLM. A
    single MODEL_TIMEOUT either kills warm requests or waits out every cold
    start, so the two endpoints carry their own budgets.
    """
    canonical = canonical_llm_type(llm_type)
    if canonical == LLMType.Main.value:
        return MAIN_MODEL_TIMEOUT
    if canonical == LLMType.Eve_Jsc.value:
        return EVE_JSC_TIMEOUT
    if canonical == LLMType.Jsc_Large.value:
        return JSC_LARGE_TIMEOUT
    return MODEL_TIMEOUT


class _GuardedCall:
    """One call of a :class:`BreakerAwareChatOpenAI`: its permit, and whether
    the endpoint has answered it yet."""

    def __init__(self, client: "BreakerAwareChatOpenAI", permit: RateLimitPermit) -> None:
        self.client = client
        self.permit = permit
        self._answered = False

    def answered(self) -> None:
        """The endpoint produced output: a probe closes its rate limit window.

        Call it only with real output (text or a tool call): Blablador's
        throttling answer is a ``200 text/plain`` body that yields no chunk,
        and it must not close the window.
        """
        if self._answered:
            return
        self._answered = True
        if self.permit.probe:
            self.client._eve_health.close_rate_limit_window(
                self.client._eve_llm_type, self.permit.started_at
            )


def _has_output(message: Any) -> bool:
    """True when a message or chunk carries text or a tool call."""
    if message is None:
        return False
    if getattr(message, "content", None):
        return True
    if getattr(message, "tool_calls", None) or getattr(message, "tool_call_chunks", None):
        return True
    extra = getattr(message, "additional_kwargs", None) or {}
    return bool(
        extra.get("tool_calls")
        or extra.get("function_call")
        or extra.get("reasoning_content")
    )


def _result_has_output(result: Any) -> bool:
    """True when a non-streaming result holds text or a tool call."""
    return any(
        _has_output(getattr(generation, "message", None))
        for generation in getattr(result, "generations", None) or []
    )


# The call a guarded client is running in this context. ChatOpenAI._agenerate
# calls _astream when streaming: the inner call joins the outer one instead of
# asking for a second permit (it would find the outer call's probe lease).
_CURRENT_CALL: contextvars.ContextVar[Optional[_GuardedCall]] = contextvars.ContextVar(
    "eve_guarded_call", default=None
)


# Mistral puts reasoning in content parts of these types. The text lives under
# ``thinking`` (Mistral) or ``reasoning``; ``text`` is the odd server.
_THINKING_PART_TYPES = frozenset({"thinking", "reasoning"})


def _reasoning_field(payload: Mapping[str, Any]) -> str:
    """vLLM ``reasoning_content``, else the newer ``reasoning`` field.

    ChatDeepSeek reads the same two keys. An empty or non-string value is
    not reasoning: servers send ``null`` on ordinary chunks.
    https://github.com/langchain-ai/langchain/blob/master/libs/partners/deepseek/langchain_deepseek/chat_models.py
    """
    text = payload.get("reasoning_content")
    if isinstance(text, str) and text:
        return text
    text = payload.get("reasoning")
    return text if isinstance(text, str) and text else ""


def _text_of(value: Any) -> str:
    """A string, or the text fields of a nested chunk list.

    Mistral's ThinkChunk stores the trace as ``thinking: [{"type": "text",
    "text": "..."}]``, not as a string.
    https://docs.mistral.ai/inference/conversations/reasoning/
    """
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if isinstance(item, str):
            parts.append(item)
            continue
        if isinstance(item, dict) and isinstance(item.get("text"), str):
            parts.append(item["text"])
    return "".join(parts)


def _thinking_part_text(part: Any) -> Optional[str]:
    """Text of a thinking part, or None when ``part`` is not one.

    ``""`` means the part is thinking but carries no text, so it still leaves
    the answer content.
    """
    if not isinstance(part, dict) or part.get("type") not in _THINKING_PART_TYPES:
        return None
    for key in ("thinking", "reasoning", "text"):
        if key not in part:
            continue
        text = _text_of(part.get(key))
        if text:
            return text
    return ""


def _without_thinking_parts(content: Any) -> tuple[Any, str]:
    """Drop thinking parts from a content list. Other content stays.

    A list that is only text afterwards becomes one string, which is what
    ``_convert_delta_to_message_chunk`` and the stream expect. A list that
    still holds a non-text part (an image) stays a list.
    """
    if not isinstance(content, list):
        return content, ""
    kept: list[Any] = []
    thinking: list[str] = []
    found = False
    for part in content:
        text = _thinking_part_text(part)
        if text is None:
            kept.append(part)
            continue
        found = True
        if text:
            thinking.append(text)
    if not found:
        return content, ""
    joined = "".join(thinking)
    texts: list[str] = []
    for part in kept:
        if isinstance(part, str):
            texts.append(part)
            continue
        if (
            isinstance(part, dict)
            and part.get("type") == "text"
            and isinstance(part.get("text"), str)
            and set(part) <= {"type", "text"}
        ):
            texts.append(part["text"])
            continue
        return kept, joined
    return "".join(texts), joined


def _delta_of(chunk: Mapping[str, Any]) -> Optional[dict]:
    """The first choice delta, or None when this chunk has none."""
    choices = chunk.get("choices") or (chunk.get("chunk") or {}).get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return None
    delta = choices[0].get("delta")
    return delta if isinstance(delta, dict) else None


def _with_delta_content(chunk: dict, content: Any) -> dict:
    """Copy of ``chunk`` whose first delta content is ``content``."""
    patched = dict(chunk)
    key = "choices" if chunk.get("choices") else "chunk"
    if key == "chunk":
        inner = dict(chunk.get("chunk") or {})
        choices = list(inner.get("choices") or [])
    else:
        inner = None
        choices = list(chunk.get("choices") or [])
    if not choices or not isinstance(choices[0], dict):
        return chunk
    choice = dict(choices[0])
    delta = dict(choice.get("delta") or {})
    delta["content"] = content
    choice["delta"] = delta
    choices[0] = choice
    if inner is None:
        patched["choices"] = choices
    else:
        inner["choices"] = choices
        patched["chunk"] = inner
    return patched


class ReasoningChatOpenAI(ChatOpenAI):
    """ChatOpenAI that keeps the reasoning the stock client drops.

    langchain-openai 1.6.7 extracts only official OpenAI fields. Its module
    docstring says ``reasoning_content`` and ``reasoning`` are not preserved.
    vLLM and OpenRouter send one of those on the delta. Mistral sends
    ``thinking`` content parts. Both land on
    ``additional_kwargs["reasoning_content"]`` and leave the answer content,
    which is what the agent stream reads.

    Field copy follows ChatDeepSeek._convert_chunk_to_generation_chunk:
    https://github.com/langchain-ai/langchain/blob/master/libs/partners/deepseek/langchain_deepseek/chat_models.py
    """

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> Any:
        delta = _delta_of(chunk)
        patched = chunk
        reasoning = ""
        if delta is not None:
            reasoning = _reasoning_field(delta)
            content, from_parts = _without_thinking_parts(delta.get("content"))
            if not reasoning:
                reasoning = from_parts
            if content is not delta.get("content"):
                patched = _with_delta_content(chunk, content)
        generation = super()._convert_chunk_to_generation_chunk(
            patched, default_chunk_class, base_generation_info
        )
        if generation is not None and reasoning:
            generation.message.additional_kwargs["reasoning_content"] = reasoning
        return generation

    def _create_chat_result(
        self,
        response: dict | Any,
        generation_info: dict | None = None,
    ) -> Any:
        result = super()._create_chat_result(response, generation_info)
        if isinstance(response, dict):
            payload = response
        else:
            payload = response.model_dump(warnings=False)
        choices = payload.get("choices") or []
        for generation, choice in zip(result.generations, choices):
            message = choice.get("message") if isinstance(choice, dict) else None
            if not isinstance(message, dict):
                continue
            reasoning = _reasoning_field(message)
            original = message.get("content")
            content, from_parts = _without_thinking_parts(original)
            if not reasoning:
                reasoning = from_parts
            if content is not original:
                generation.message.content = content
            if reasoning:
                generation.message.additional_kwargs["reasoning_content"] = reasoning
        return result


class BreakerAwareChatOpenAI(ReasoningChatOpenAI):
    """ChatOpenAI that keeps its endpoint out of a rate limit window.

    Every call asks the breaker first (:meth:`EndpointHealth.admit`): inside a
    window, or while another request is the probe after it, the call refuses
    with :class:`EndpointRateLimited` before a request is sent. This is what
    keeps a running agentic turn, whose graph holds this client for every
    step, off the endpoint after a 429. A 429 records the window and comes
    back as :class:`EndpointRateLimited`, which no retry policy retries. The
    probe's first output closes the window. Every other error passes through
    untouched. The probe's window closes only on output: the first chunk
    with text or a tool call, or a non-streaming reply that holds one. A
    stream that ends with no such chunk, or an error, closes nothing. Without
    :meth:`guard` it is a :class:`ReasoningChatOpenAI`.
    """

    _eve_health: Any = PrivateAttr(default=None)
    _eve_llm_type: str = PrivateAttr(default="")

    def guard(self, health: EndpointHealth, llm_type: str) -> "BreakerAwareChatOpenAI":
        """Tie this client to *llm_type*'s rate limit window on *health*."""
        self._eve_health = health
        self._eve_llm_type = llm_type
        return self

    @contextlib.contextmanager
    def _call(self, *, outer: bool) -> Iterator[Optional[_GuardedCall]]:
        """Admit, run and account one call; *outer* marks the generate entry
        points, whose streaming delegation joins this call."""
        if self._eve_health is None:
            yield None
            return
        current = _CURRENT_CALL.get()
        if current is not None and current.client is self:
            yield current
            return
        call = _GuardedCall(self, self._eve_health.admit(self._eve_llm_type))
        token = _CURRENT_CALL.set(call) if outer else None
        try:
            yield call
        except Exception as exc:
            if getattr(exc, "status_code", None) != 429 or isinstance(
                exc, EndpointRateLimited
            ):
                raise
            window_s = self._eve_health.record_rate_limit(
                self._eve_llm_type, retry_after_header(exc)
            )
            # from None: the upstream error, whose body can echo the key or
            # the prompt, must not ride along into span events and log lines.
            raise EndpointRateLimited(self._eve_llm_type, window_s) from None
        finally:
            if token is not None:
                _CURRENT_CALL.reset(token)

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        with self._call(outer=True) as call:
            result = super()._generate(*args, **kwargs)
            if call is not None and _result_has_output(result):
                call.answered()
            return result

    async def _agenerate(self, *args: Any, **kwargs: Any) -> Any:
        with self._call(outer=True) as call:
            result = await super()._agenerate(*args, **kwargs)
            if call is not None and _result_has_output(result):
                call.answered()
            return result

    def _stream(self, *args: Any, **kwargs: Any) -> Any:
        with self._call(outer=False) as call:
            for chunk in super()._stream(*args, **kwargs):
                if call is not None and _has_output(getattr(chunk, "message", None)):
                    call.answered()
                yield chunk

    async def _astream(self, *args: Any, **kwargs: Any) -> Any:
        with self._call(outer=False) as call:
            async for chunk in super()._astream(*args, **kwargs):
                if call is not None and _has_output(getattr(chunk, "message", None)):
                    call.answered()
                yield chunk


class LLMManager:
    """Manages interactions with different language models."""

    def __init__(self, health: Optional[EndpointHealth] = None):
        """Initialize the LLM Manager with configuration.

        *health* shares another manager's breaker: a manager built per request
        must see the circuits and rate limit windows of the process.
        """
        self.config = Config()
        self._setup_api_keys()
        self._selected_llm_type = None
        # Read through the module at construction so the suite can keep each
        # manager on its own breaker (tests/conftest.py).
        self._health = health or EndpointHealth(
            EVE_ENDPOINT_COOLDOWN_S,
            redis_url=_config.REDIS_URL if _config.ENDPOINT_BREAKER_SHARED else None,
            # The half-open probe gets the endpoint's own first-token budget.
            probe_s=endpoint_timeout,
        )
        self._init_langchain_clients()
        # Load system prompt once
        try:
            self._system_prompt: Optional[str] = get_template(
                "system_prompt", filename="system.yaml"
            )
        except Exception:
            self._system_prompt = None

    def _get_current_system_prompt(self) -> Optional[str]:
        """Return system prompt based on the currently selected LLM type.

        Uses satcom-specific system prompt when `self._selected_llm_type` is satcom,
        otherwise falls back to the default system prompt loaded at initialization.
        """
        try:
            if self._selected_llm_type in (
                LLMType.Satcom_Small.value,
                LLMType.Satcom_Large.value,
            ):
                return get_template("system_prompt", filename="satcom/system.yaml")
        except Exception:
            # If satcom template missing or fails to load, fall back to default
            pass
        return self._system_prompt

    def _setup_api_keys(self):
        """Set up API keys for different LLM providers."""
        # OpenAI key is read by the OpenAI SDK from env; nothing to set explicitly here
        # Runpod key will be passed directly to LangChain's ChatOpenAI client
        pass

    def _init_langchain_clients(self) -> None:
        """Initialize LangChain ChatOpenAI client configured for Main and Fallback OpenAI-compatible endpoints."""
        # Main model — independent try block so Satcom misconfiguration can't clobber it
        try:
            if not MAIN_MODEL_URL:
                raise ValueError("MAIN_MODEL_URL environment variable not configured")
            self._main_base_url = MAIN_MODEL_URL
            self._main_model_name = MAIN_MODEL_NAME
            self._main_chat_openai: ChatOpenAI | None = None
        except Exception as e:
            logger.error(f"Failed to initialize LangChain Main client: {e}")
            self._main_base_url = None
            self._main_model_name = None
            self._main_chat_openai = None

        # Satcom models — optional; missing URLs are non-fatal for other models
        try:
            if not SATCOM_SMALL_BASE_URL:
                raise ValueError(
                    "SATCOM_SMALL_BASE_URL environment variable not configured"
                )
            if not SATCOM_LARGE_BASE_URL:
                raise ValueError(
                    "SATCOM_LARGE_BASE_URL environment variable not configured"
                )
            self._satcom_small_base_url = SATCOM_SMALL_BASE_URL
            self._satcom_small_model_name = SATCOM_SMALL_MODEL_NAME
            self._satcom_large_base_url = SATCOM_LARGE_BASE_URL
            self._satcom_large_model_name = SATCOM_LARGE_MODEL_NAME
            self._satcom_small_chat_openai: ChatOpenAI | None = None
            self._satcom_large_chat_openai: ChatOpenAI | None = None
        except Exception as e:
            global _satcom_missing_warned
            if not _satcom_missing_warned:
                _satcom_missing_warned = True
                logger.warning(f"Satcom clients not available (optional): {e}")
            self._satcom_small_base_url = None
            self._satcom_large_base_url = None
            self._satcom_small_model_name = None
            self._satcom_large_model_name = None
            self._satcom_small_chat_openai = None
            self._satcom_large_chat_openai = None

        try:
            self._eve_jsc_base_url = EVE_JSC_BASE_URL or None
            self._eve_jsc_model_name = EVE_JSC_MODEL_NAME
            self._eve_jsc_chat_openai: ChatOpenAI | None = None
            self._jsc_large_model_name = JSC_LARGE_MODEL_NAME or None
            self._jsc_large_chat_openai: ChatOpenAI | None = None
        except Exception as e:
            logger.error(f"Failed to initialize EVE-JSC client config: {e}")
            self._eve_jsc_base_url = None
            self._eve_jsc_model_name = None
            self._eve_jsc_chat_openai = None
            self._jsc_large_model_name = None
            self._jsc_large_chat_openai = None

        # Configure Fallback client lazily
        try:
            if not FALLBACK_MODEL_URL:
                raise ValueError(
                    "FALLBACK_MODEL_URL environment variable not configured"
                )

            # Fallback model uses OpenAI-compatible API
            self._fallback_base_url = FALLBACK_MODEL_URL
            # Model name can be provided via env FALLBACK_MODEL_NAME, else use config
            self._fallback_model_name = FALLBACK_MODEL_NAME
            self._fallback_chat: ChatOpenAI | None = None
        except Exception as e:
            logger.error(f"Failed to initialize Fallback client config: {e}")
            self._fallback_base_url = None
            self._fallback_model_name = None
            self._fallback_chat = None

    def set_selected_llm_type(self, llm_type: str) -> None:
        """Set the selected LLM type."""
        self._selected_llm_type = llm_type

    def _get_main_llm(self) -> ChatOpenAI:
        """Return a configured ChatOpenAI client for Main model."""
        if self._main_chat_openai is None:
            if not self._main_base_url:
                raise RuntimeError("Main model base URL is not configured")
            if not MAIN_MODEL_API_KEY:
                raise RuntimeError(
                    "MAIN_MODEL_API_KEY (or RUNPOD_API_KEY) is not set (required for main model)"
                )
            self._main_chat_openai = ReasoningChatOpenAI(
                api_key=MAIN_MODEL_API_KEY,
                base_url=self._main_base_url,
                model=self._main_model_name,
                profile=platform_model_profile(self._main_model_name),
                temperature=0.3,
                timeout=MAIN_MODEL_TIMEOUT,
                # SDK-level retries would burn the caller's first-token budget
                # and hide the 5xx that must open this endpoint's circuit.
                max_retries=0,
            )
        return self._main_chat_openai

    def _get_fallback_llm(self) -> ChatOpenAI:
        """Return a configured ChatOpenAI client for Fallback model (OpenAI-compatible)."""
        if self._fallback_chat is None:
            if not self._fallback_base_url:
                raise RuntimeError("Fallback model base URL is not configured")
            if not FALLBACK_MODEL_API_KEY:
                raise RuntimeError(
                    "FALLBACK_MODEL_API_KEY is not set (required for fallback model)"
                )
            self._fallback_chat = ReasoningChatOpenAI(
                api_key=FALLBACK_MODEL_API_KEY,
                base_url=self._fallback_base_url,
                model=self._fallback_model_name,
                profile=platform_model_profile(self._fallback_model_name),
                temperature=0.3,
                timeout=MODEL_TIMEOUT,
                max_retries=0,
            )
        return self._fallback_chat

    def _get_satcom_small_llm(self) -> ChatOpenAI:
        """Return a configured ChatOpenAI client for Satcom Small."""
        if self._satcom_small_chat_openai is None:
            if not self._satcom_small_base_url:
                raise RuntimeError("Satcom small base URL is not configured")
            if not SATCOM_RUNPOD_API_KEY:
                raise RuntimeError("SATCOM_API_KEY is not set")
            self._satcom_small_chat_openai = ReasoningChatOpenAI(
                api_key=SATCOM_RUNPOD_API_KEY,
                base_url=self._satcom_small_base_url,
                model=self._satcom_small_model_name,
                profile=platform_model_profile(self._satcom_small_model_name),
                temperature=0.3,
                timeout=MODEL_TIMEOUT,
                max_retries=0,
            )
        return self._satcom_small_chat_openai

    def _get_satcom_large_llm(self) -> ChatOpenAI:
        """Return a configured ChatOpenAI client for Satcom Large."""
        if self._satcom_large_chat_openai is None:
            if not self._satcom_large_base_url:
                raise RuntimeError("Satcom large base URL is not configured")
            if not SATCOM_RUNPOD_API_KEY:
                raise RuntimeError("SATCOM_API_KEY is not set")
            self._satcom_large_chat_openai = ReasoningChatOpenAI(
                api_key=SATCOM_RUNPOD_API_KEY,
                base_url=self._satcom_large_base_url,
                model=self._satcom_large_model_name,
                profile=platform_model_profile(self._satcom_large_model_name),
                temperature=0.3,
                timeout=MODEL_TIMEOUT,
                max_retries=0,
            )
        return self._satcom_large_chat_openai

    def _get_eve_jsc_llm(self) -> ChatOpenAI:
        """Return a configured ChatOpenAI client for the EVE-JSC model."""
        if self._eve_jsc_chat_openai is None:
            if not self._eve_jsc_base_url:
                raise RuntimeError("EVE_JSC_BASE_URL is not configured")
            if not EVE_JSC_API_KEY:
                raise RuntimeError("EVE_JSC_API_KEY is not set")
            # Blablador doubles a block for every request sent during it, so
            # the rate limit window must hold mid-turn too.
            self._eve_jsc_chat_openai = BreakerAwareChatOpenAI(
                api_key=EVE_JSC_API_KEY,
                base_url=self._eve_jsc_base_url,
                model=self._eve_jsc_model_name,
                profile=platform_model_profile(self._eve_jsc_model_name),
                temperature=0.3,
                timeout=EVE_JSC_TIMEOUT,
                max_retries=0,
            ).guard(self._health, LLMType.Eve_Jsc.value)
        return self._eve_jsc_chat_openai

    def _get_jsc_large_llm(self) -> ChatOpenAI:
        """Return a configured ChatOpenAI client for the large model on JSC."""
        if self._jsc_large_chat_openai is None:
            if not self._eve_jsc_base_url:
                raise RuntimeError("EVE_JSC_BASE_URL is not configured")
            if not EVE_JSC_API_KEY:
                raise RuntimeError("EVE_JSC_API_KEY is not set")
            if not self._jsc_large_model_name:
                raise RuntimeError("JSC_LARGE_MODEL_NAME is not configured")
            self._jsc_large_chat_openai = BreakerAwareChatOpenAI(
                api_key=EVE_JSC_API_KEY,
                base_url=self._eve_jsc_base_url,
                model=self._jsc_large_model_name,
                profile=platform_model_profile(self._jsc_large_model_name),
                temperature=0.3,
                timeout=JSC_LARGE_TIMEOUT,
                max_retries=0,
            ).guard(self._health, LLMType.Jsc_Large.value)
        return self._jsc_large_chat_openai

    def build_custom_client(
        self, *, base_url: str, model_name: str, api_key: str
    ) -> ChatOpenAI:
        """Build a one-off ChatOpenAI client for a user-owned custom model."""
        return ReasoningChatOpenAI(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            model=model_name,
            temperature=0.3,
            timeout=MODEL_TIMEOUT,
            max_retries=0,
        )

    def _is_configured(self, llm_type: str) -> bool:
        """True when this endpoint has both a base URL and a key.

        Answers without building a client, so chain resolution stays free of
        network configuration side effects.
        """
        if llm_type == LLMType.Main.value:
            return bool(self._main_base_url and MAIN_MODEL_API_KEY)
        if llm_type == LLMType.Fallback.value:
            return bool(self._fallback_base_url and FALLBACK_MODEL_API_KEY)
        if llm_type == LLMType.Eve_Jsc.value:
            return bool(self._eve_jsc_base_url and EVE_JSC_API_KEY)
        if llm_type == LLMType.Jsc_Large.value:
            return bool(
                self._eve_jsc_base_url and EVE_JSC_API_KEY and self._jsc_large_model_name
            )
        if llm_type == LLMType.Satcom_Small.value:
            return bool(self._satcom_small_base_url and SATCOM_RUNPOD_API_KEY)
        if llm_type == LLMType.Satcom_Large.value:
            return bool(self._satcom_large_base_url and SATCOM_RUNPOD_API_KEY)
        return False

    def is_configured(self, llm_type: str) -> bool:
        """True when ``llm_type`` could answer here, for callers outside the chain."""
        return self._is_configured(canonical_llm_type(llm_type) or "")

    def resolve_chain(self, llm_type: Optional[str] = None) -> list[str]:
        """Return the ordered endpoint candidates for a request.

        An explicit request is honoured as asked and never promoted to another
        endpoint; only ``None`` expands to EVE_ENDPOINT_ORDER. Unconfigured
        entries drop out, the fallback model is always the last resort, and
        endpoints whose circuit is open sink to the back rather than out, so a
        chain with nothing but open circuits still answers. An endpoint a 429
        keeps out (``is_rate_limited``) drops out instead, explicit requests
        included, whenever something else can answer: calling it inside its
        window would lengthen the block.
        """
        candidates: list[str] = []
        for name in self._requested_names(llm_type):
            if name in candidates:
                continue
            if name not in _KNOWN_LLM_TYPES:
                logger.warning(
                    "Unknown llm_type %r in the endpoint chain, dropping it", name
                )
                continue
            if not self._is_configured(name):
                logger.info(
                    "Endpoint %s is not configured, dropping it from the chain", name
                )
                continue
            candidates.append(name)

        fallback_configured = self._is_configured(LLMType.Fallback.value)
        chain = [
            name
            for name in candidates
            if name != LLMType.Fallback.value
            and not (fallback_configured and self._health.is_rate_limited(name))
        ]
        # Stable sort: closed circuits keep their configured order and open ones
        # sink to the back. When every candidate is open they all compare equal
        # and come back untouched, so a chain entirely in cooldown still answers.
        chain.sort(key=self._health.is_open)
        if fallback_configured:
            # The last resort is last by definition, whatever its own circuit
            # says: callers treat the tail of the chain as the final attempt.
            chain.append(LLMType.Fallback.value)
        if not chain:
            raise RuntimeError(f"No endpoint is configured for llm_type={llm_type!r}")
        return chain

    def open_circuits(self, llm_type: Optional[str], chain: list[str]) -> list[str]:
        """Endpoints of this request whose circuit is open, for metadata.

        The chain's own, then the requested ones a 429 took out of it.
        """
        names = list(chain)
        for name in self._requested_names(llm_type):
            if name not in names and self._health.is_rate_limited(name):
                names.append(name)
        return [name for name in names if self._health.is_open(name)]

    def _requested_names(self, llm_type: Optional[str]) -> list[str]:
        """The endpoints a request names, canonical, in order: the explicit one,
        or EVE_ENDPOINT_ORDER for a request that names none. Legacy spellings
        are valid in the env var too, not only in stored requests."""
        requested = canonical_llm_type(llm_type)
        raw = EVE_ENDPOINT_ORDER.split(",") if requested is None else [requested]
        return [name for name in (canonical_llm_type(n.strip()) for n in raw) if name]

    def _skips_rate_limited(self, name: str) -> bool:
        """True when *name* must leave the chain: a rate limit window keeps it
        out and the fallback model can answer instead. Calling it inside its
        window would lengthen the block."""
        return (
            name != LLMType.Fallback.value
            and self._health.is_rate_limited(name)
            and self._is_configured(LLMType.Fallback.value)
        )

    def client_for_candidate(self, llm_type: str) -> ChatOpenAI:
        """Return the client for one chain candidate, without substituting.

        ``get_client_for_model`` answers with the fallback client when anything
        goes wrong, which is the right last resort for a single-shot caller but
        would hide a dead endpoint from a caller that has a chain left to walk.
        """
        canonical = canonical_llm_type(llm_type)
        if canonical == LLMType.Fallback.value:
            client = self._get_fallback_llm()
        elif canonical == LLMType.Main.value:
            client = self._get_main_llm()
        elif canonical == LLMType.Eve_Jsc.value:
            client = self._get_eve_jsc_llm()
        elif canonical == LLMType.Jsc_Large.value:
            client = self._get_jsc_large_llm()
        elif canonical == LLMType.Satcom_Small.value:
            client = self._get_satcom_small_llm()
        elif canonical == LLMType.Satcom_Large.value:
            client = self._get_satcom_large_llm()
        else:
            raise RuntimeError(f"Invalid model selection: {llm_type}")
        self._selected_llm_type = canonical
        return client

    def get_client_for_model(self, llm_type: Optional[str] = None):
        """Return an LLM client instance based on the requested model/provider.

        A request without an llm_type takes the head of the configured chain.
        Anything that fails to resolve falls back to the Fallback model, which
        is the last resort for callers that have no chain of their own.
        Supports legacy 'runpod' and 'mistral' values for backward compatibility.
        """
        try:
            if llm_type is None:
                llm_type = self.resolve_chain(None)[0]
            canonical = canonical_llm_type(llm_type)
            if canonical and self._skips_rate_limited(canonical):
                self._selected_llm_type = LLMType.Fallback.value
                return self._get_fallback_llm()
            return self.client_for_candidate(llm_type)
        except Exception as e:
            logger.error(f"Failed to get client for model {llm_type!r}: {e}")
            # The substitution stays silent to the caller, so the breaker is the
            # only place this failure can still be seen; callers that must know
            # they were substituted resolve their own chain instead.
            canonical = canonical_llm_type(llm_type)
            if canonical and is_endpoint_failure(e):
                self._health.record_failure(canonical, e)
            self._selected_llm_type = LLMType.Fallback.value
            return self._get_fallback_llm()

    def get_selected_llm_type(self) -> str:
        """Return the selected LLM type."""
        return self._selected_llm_type

    @property
    def health(self) -> EndpointHealth:
        """Circuit breaker shared by every caller of this manager, and by every
        worker when REDIS_URL is set."""
        return self._health

    def health_snapshot(self) -> dict:
        """Per-endpoint circuit state, for operational readouts."""
        return self._health.snapshot()

    def get_model(self) -> ChatOpenAI:
        """Public accessor for the primary ChatOpenAI client with fallback to Fallback."""
        try:
            return self._get_main_llm()
        except Exception as e:
            logger.warning(
                f"Falling back to Fallback model due to Main model error: {e}"
            )
            return self._get_fallback_llm()

    def get_fallback_llm(self) -> ChatOpenAI:
        """Public accessor for the fallback ChatOpenAI client."""
        return self._get_fallback_llm()

    def __call__(self, *args, **kwargs):
        """Make the class callable, delegating to generate_answer."""
        return self.generate_answer(*args, **kwargs)

    def _generate_prompt(self, query: str, context: str) -> str:
        """
        Generate a prompt for the language model using template.

        Args:
            query: The user's question
            context: Contextual information to assist the model

        Returns:
            A formatted prompt string
        """
        if self._selected_llm_type in (
            LLMType.Satcom_Small.value,
            LLMType.Satcom_Large.value,
        ):
            return format_template(
                "basic_prompt",
                filename="satcom/prompts.yaml",
                query=query,
                context=context,
            )
        else:
            return format_template(
                "basic_prompt",
                filename="prompts.yaml",
                query=query,
                context=context,
            )

    def _generate_prompt_with_history(
        self, query: str, context: str, conversation_context: str
    ) -> str:
        """
        Generate a prompt with conversation history for multi-turn context using template.

        Args:
            query: The user's question
            context: Contextual information to assist the model
            conversation_context: Previous conversation messages

        Returns:
            A formatted prompt string with conversation history
        """
        if context:
            if self._selected_llm_type in (
                LLMType.Satcom_Small.value,
                LLMType.Satcom_Large.value,
            ):
                return format_template(
                    "rag_prompt",
                    filename="satcom/prompts.yaml",
                    query=query,
                    context=context,
                    conversation=conversation_context,
                )
            else:
                return format_template(
                    "rag_prompt",
                    filename="prompts.yaml",
                    query=query,
                    context=context,
                    conversation=conversation_context,
                )
        else:
            if self._selected_llm_type in (
                LLMType.Satcom_Small.value,
                LLMType.Satcom_Large.value,
            ):
                return format_template(
                    "no_rag_prompt",
                    filename="satcom/prompts.yaml",
                    query=query,
                    conversation=conversation_context,
                )
            else:
                return format_template(
                    "no_rag_prompt",
                    filename="prompts.yaml",
                    query=query,
                    conversation=conversation_context,
                )

    def _call_eve_instruct(self, prompt: str, max_new_tokens: int = 150) -> str:
        """
        Call the Eve Instruct model via Main model using LangChain ChatOpenAI (OpenAI-compatible API).

        Args:
            prompt: The formatted prompt
            max_new_tokens: Maximum new tokens for the response

        Returns:
            The generated response

        Raises:
            TimeoutError: If the job times out
            Exception: For other errors
        """
        try:
            base_llm = self._get_main_llm()
            llm = base_llm.bind(max_tokens=max_new_tokens)
            system_and_prompt = prompt
            current_system_prompt = self._get_current_system_prompt()
            if current_system_prompt:
                system_and_prompt = (
                    f"System:\n{current_system_prompt}\n\nUser:\n{prompt}"
                )
            response = llm.invoke(system_and_prompt)
            return getattr(response, "content", str(response))
        except Exception as e:
            logger.error(f"Eve Instruct Main model API call failed: {str(e)}")
            raise

    async def _call_eve_instruct_fallback(
        self, prompt: str, max_new_tokens: int = 150, temperature: float | None = None
    ) -> str:
        """
        Call the Eve Instruct model via Fallback Model using LangChain ChatOpenAI.
        """
        try:
            base_llm = self._get_fallback_llm()
            bind_kwargs = {}
            # bind_kwargs = {"max_tokens": max_new_tokens}
            if temperature is not None:
                bind_kwargs["temperature"] = temperature
                bind_kwargs["timeout"] = 150
            llm = base_llm.bind(**bind_kwargs)
            current_system_prompt = self._get_current_system_prompt()
            if current_system_prompt:
                messages = [
                    SystemMessage(content=current_system_prompt),
                    HumanMessage(content=prompt),
                ]
                response = await llm.ainvoke(messages)
            else:
                response = await llm.ainvoke(prompt)
            content = getattr(response, "content", str(response))
            return content
        except Exception as e:
            logger.error(f"Fallback model call failed: {str(e)}")
            raise

    async def _call_eve_instruct_async(
        self, prompt: str, max_new_tokens: int = 150, temperature: float | None = None
    ) -> str:
        """Async version using LangChain ChatOpenAI."""
        try:
            base_llm = self._get_main_llm()
            bind_kwargs = {}
            # bind_kwargs = {"max_tokens": max_new_tokens}
            if temperature is not None:
                bind_kwargs["temperature"] = temperature
            llm = base_llm.bind(**bind_kwargs)
            system_and_prompt = prompt
            current_system_prompt = self._get_current_system_prompt()
            if current_system_prompt:
                system_and_prompt = (
                    f"System:\n{current_system_prompt}\n\nUser:\n{prompt}"
                )
            response = await llm.ainvoke(system_and_prompt)
            content = getattr(response, "content", str(response))
            return content
        except Exception as e:
            logger.error(
                f"Eve Instruct async API call failed: {str(e)}. Trying Fallback model."
            )
            raise

    def _process_eve_response(self, response) -> str:
        """Kept for backward compatibility; now simply returns content if present."""
        return getattr(response, "content", str(response))

    async def summarize_context_in_all(
        self,
        transcript: str,
        max_tokens: int = 5000,
        is_force: bool = False,
        llm: Optional[Any] = None,
    ) -> str:
        """Summarize entire conversation history.
        """
        if not transcript:
            return ""
        if is_force is False and str_token_counter(transcript) <= max_tokens:
            return transcript

        system = (
            "You are an AI assistant specialized in summarizing chat histories. "
            "Your role is to read a transcript of a conversation and produce a clear, "
            "concise, and neutral summary of the main points."
        )
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=f"Conversation transcript:\n{transcript}"),
        ]
        if llm is not None:
            resp = await llm.ainvoke(messages)
            return getattr(resp, "content", str(resp))

        try:
            llm = self.get_client_for_model(self._selected_llm_type)
            resp = await llm.ainvoke(messages)
            return getattr(resp, "content", str(resp))
        except Exception:
            llm = self._get_fallback_llm()
            resp = await llm.ainvoke(messages)
            return getattr(resp, "content", str(resp))

    # summarize_context_with_map_reduce removed — load_summarize_chain blocks on import

    async def generate_answer(
        self,
        query: str,
        context: str,
        max_new_tokens: int = 150,
        temperature: float | None = None,
    ) -> str:
        """
        Generate an answer using the main model.

        Args:
            query: The user's question
            context: Contextual information to assist the model
            max_new_tokens: Maximum new tokens for the response

        Returns:
            The generated answer

        Raises:
            Exception: For other errors during generation
        """
        try:
            prompt = self._generate_prompt(query=query, context=context)
            max_context_len = MODEL_CONTEXT_SIZE - max_new_tokens
            if len(context) > max_context_len:
                logger.info(
                    f"Truncating context from {len(context)} to {max_context_len} characters"
                )
                context = context[:max_context_len]

            return await self._call_eve_instruct_async(
                prompt, max_new_tokens=max_new_tokens, temperature=temperature
            )

        except Exception as e:
            logger.error(f"Failed to generate answer using main model: {str(e)}")
            raise

    async def generate_answer_fallback(
        self,
        query: str,
        context: str,
        max_new_tokens: int = 150,
        temperature: float | None = None,
        conversation_context: str = "",
    ) -> tuple[str, str]:
        """
        Generate an answer using the fallback model (legacy method name).

        Args:
            query: The user's question
            context: Contextual information to assist the model
            max_new_tokens: Maximum new tokens for the response
            temperature: Temperature for generation
            conversation_context: Previous conversation messages for context

        Returns:
            The generated answer

        Raises:
            Exception: For other errors during generation
        """
        try:
            # If conversation history or summary is provided, use it for multi-turn context
            if conversation_context:
                available_tokens = DEFAULT_MAX_NEW_TOKENS
                if str_token_counter(conversation_context) > available_tokens:
                    conversation_context = trim_text_to_token_limit(
                        text=conversation_context, max_tokens=available_tokens
                    )
                prompt = self._generate_prompt_with_history(
                    query=query,
                    context=context,
                    conversation_context=conversation_context,
                )
            else:
                prompt = self._generate_prompt(query=query, context=context)

            # Offer the curated image catalog to the model (no-op when disabled).
            prompt = append_image_context(prompt, query)

            max_new_tokens = MODEL_CONTEXT_SIZE - str_token_counter(prompt)
            answer = await self._call_eve_instruct_fallback(
                prompt, max_new_tokens=max_new_tokens, temperature=temperature
            )
            return answer, prompt

        except Exception as e:
            logger.error(f"Failed to generate answer using fallback model: {str(e)}")
            raise

    async def generate_answer_fallback_stream(
        self,
        query: str,
        context: str,
        max_new_tokens: int = 150,
        temperature: float | None = None,
        conversation_context: str = "",
    ) -> AsyncGenerator[tuple[str, str], None]:
        """Generate an answer using the mistral model, streaming tokens."""
        try:
            if conversation_context:
                available_tokens = DEFAULT_MAX_NEW_TOKENS
                if str_token_counter(conversation_context) > available_tokens:
                    conversation_context = trim_text_to_token_limit(
                        text=conversation_context, max_tokens=available_tokens
                    )
                prompt = self._generate_prompt_with_history(
                    query=query,
                    context=context,
                    conversation_context=conversation_context,
                )
            else:
                prompt = self._generate_prompt(query=query, context=context)

            # Offer the curated image catalog to the model (no-op when disabled).
            prompt = append_image_context(prompt, query)

            max_new_tokens = MODEL_CONTEXT_SIZE - str_token_counter(prompt)
            async for chunk in self._call_eve_instruct_fallback_stream(
                prompt, max_new_tokens=max_new_tokens, temperature=temperature
            ):
                yield chunk, prompt
        except Exception as e:
            logger.error(f"Failed to generate answer using fallback model: {str(e)}")
            raise

    async def _call_eve_instruct_fallback_stream(
        self, prompt: str, max_new_tokens: int = 150, temperature: float | None = None
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from the Fallback model via LangChain ChatOpenAI."""
        try:
            base_llm = self._get_fallback_llm()
            bind_kwargs = {}
            # bind_kwargs = {"max_tokens": max_new_tokens}
            if temperature is not None:
                bind_kwargs["temperature"] = temperature
            llm = base_llm.bind(**bind_kwargs)

            # Prefer astream which yields message chunks with `.content`
            current_system_prompt = self._get_current_system_prompt()
            if current_system_prompt:
                input_payload = [
                    SystemMessage(content=current_system_prompt),
                    HumanMessage(content=prompt),
                ]
            else:
                input_payload = prompt
            async for event in llm.astream(input_payload):
                try:
                    text = getattr(event, "content", None)
                    if not text and isinstance(event, dict):
                        text = event.get("content")
                    if text:
                        yield str(text)
                except Exception:
                    continue
        except Exception as e:
            logger.error(f"Fallback streaming call failed: {str(e)}")
            raise
