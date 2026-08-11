"""
LLM Manager module that handles different language model interactions.
"""

import logging
from enum import Enum
from typing import Any, AsyncGenerator, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import (
    EVE_ENDPOINT_COOLDOWN_S,
    EVE_ENDPOINT_ORDER,
    EVE_JSC_TIMEOUT,
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
)
from src.constants import DEFAULT_MAX_NEW_TOKENS, MODEL_CONTEXT_SIZE
from src.core.llm_health import EndpointHealth, is_endpoint_failure
from src.services.image_catalog import append_image_context
from src.utils.helpers import (
    str_token_counter,
    trim_text_to_token_limit,
)
from src.utils.template_loader import format_template, get_template

logger = logging.getLogger(__name__)


class LLMType(Enum):
    """Enum for supported LLM types."""

    Main = "main"
    Fallback = "fallback"
    Satcom_Small = "satcom_small"
    Satcom_Large = "satcom_large"
    Eve_Jsc = "eve_jsc"
    # Legacy aliases for backward compatibility
    Runpod = "main"
    Mistral = "fallback"


# Enum aliases collapse onto their canonical member, so LLMType("runpod") raises
# and the legacy names need their own mapping.
_LEGACY_LLM_TYPES = {"runpod": LLMType.Main.value, "mistral": LLMType.Fallback.value}
_KNOWN_LLM_TYPES = {member.value for member in LLMType}


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
    return MODEL_TIMEOUT


class LLMManager:
    """Manages interactions with different language models."""

    def __init__(self):
        """Initialize the LLM Manager with configuration."""
        self.config = Config()
        self._setup_api_keys()
        self._init_langchain_clients()
        self._selected_llm_type = None
        self._health = EndpointHealth(EVE_ENDPOINT_COOLDOWN_S)
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
        except Exception as e:
            logger.error(f"Failed to initialize EVE-JSC client config: {e}")
            self._eve_jsc_base_url = None
            self._eve_jsc_model_name = None
            self._eve_jsc_chat_openai = None

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
            self._main_chat_openai = ChatOpenAI(
                api_key=MAIN_MODEL_API_KEY,
                base_url=self._main_base_url,
                model=self._main_model_name,
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
            self._fallback_chat = ChatOpenAI(
                api_key=FALLBACK_MODEL_API_KEY,
                base_url=self._fallback_base_url,
                model=self._fallback_model_name,
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
            self._satcom_small_chat_openai = ChatOpenAI(
                api_key=SATCOM_RUNPOD_API_KEY,
                base_url=self._satcom_small_base_url,
                model=self._satcom_small_model_name,
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
            self._satcom_large_chat_openai = ChatOpenAI(
                api_key=SATCOM_RUNPOD_API_KEY,
                base_url=self._satcom_large_base_url,
                model=self._satcom_large_model_name,
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
            self._eve_jsc_chat_openai = ChatOpenAI(
                api_key=EVE_JSC_API_KEY,
                base_url=self._eve_jsc_base_url,
                model=self._eve_jsc_model_name,
                temperature=0.3,
                timeout=EVE_JSC_TIMEOUT,
                max_retries=0,
            )
        return self._eve_jsc_chat_openai

    def build_custom_client(
        self, *, base_url: str, model_name: str, api_key: str
    ) -> ChatOpenAI:
        """Build a one-off ChatOpenAI client for a user-owned custom model."""
        return ChatOpenAI(
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
        if llm_type == LLMType.Satcom_Small.value:
            return bool(self._satcom_small_base_url and SATCOM_RUNPOD_API_KEY)
        if llm_type == LLMType.Satcom_Large.value:
            return bool(self._satcom_large_base_url and SATCOM_RUNPOD_API_KEY)
        return False

    def resolve_chain(self, llm_type: Optional[str] = None) -> list[str]:
        """Return the ordered endpoint candidates for a request.

        An explicit request is honoured as asked and never promoted to another
        endpoint; only ``None`` expands to EVE_ENDPOINT_ORDER. Unconfigured
        entries drop out, the fallback model is always the last resort, and
        endpoints whose circuit is open sink to the back rather than out, so a
        chain with nothing but open circuits still answers.
        """
        requested = canonical_llm_type(llm_type)
        names = (
            [name.strip() for name in EVE_ENDPOINT_ORDER.split(",")]
            if requested is None
            else [requested]
        )

        candidates: list[str] = []
        for name in names:
            if not name or name in candidates:
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

        chain = [name for name in candidates if name != LLMType.Fallback.value]
        # Stable sort: closed circuits keep their configured order and open ones
        # sink to the back. When every candidate is open they all compare equal
        # and come back untouched, so a chain entirely in cooldown still answers.
        chain.sort(key=self._health.is_open)
        if self._is_configured(LLMType.Fallback.value):
            # The last resort is last by definition, whatever its own circuit
            # says: callers treat the tail of the chain as the final attempt.
            chain.append(LLMType.Fallback.value)
        if not chain:
            raise RuntimeError(f"No endpoint is configured for llm_type={llm_type!r}")
        return chain

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
        """Circuit breaker shared by every caller of this manager."""
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
