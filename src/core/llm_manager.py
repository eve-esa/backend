"""
LLM Manager module that handles different language model interactions.
"""

import logging
from enum import Enum
from typing import AsyncGenerator, List, Any
import os

from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI

from src.config import Config, RUNPOD_API_KEY, MISTRAL_API_KEY
from typing import Optional
from src.constants import MODEL_CONTEXT_SIZE
from src.utils.template_loader import format_template
from src.utils.helpers import str_token_counter, trim_text_to_token_limit
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.chains.summarize import load_summarize_chain
from langchain_core.messages import SystemMessage, HumanMessage


logger = logging.getLogger(__name__)


class LLMType(Enum):
    """Enum for supported LLM types."""

    OPENAI = "openai"
    EVE_INSTRUCT = "eve-instruct-v0.1"
    LLAMA = "llama-3.1"


class LLMManager:
    """Manages interactions with different language models."""

    def __init__(self):
        """Initialize the LLM Manager with configuration."""
        self.config = Config()
        self._setup_api_keys()
        self._init_langchain_clients()

    def _setup_api_keys(self):
        """Set up API keys for different LLM providers."""
        # OpenAI key is read by the OpenAI SDK from env; nothing to set explicitly here
        # Runpod key will be passed directly to LangChain's ChatOpenAI client
        pass

    def _init_langchain_clients(self) -> None:
        """Initialize LangChain ChatOpenAI client configured for Runpod OpenAI-compatible endpoint."""
        try:
            runpod_endpoint_id = self.config.get_instruct_llm_id()
            if not runpod_endpoint_id:
                raise ValueError("Runpod endpoint id not configured")

            # Build OpenAI-compatible base URL for Runpod vLLM worker
            self._runpod_base_url = (
                f"https://api.runpod.ai/v2/{runpod_endpoint_id}/openai/v1"
            )

            # Model name can be provided via env RUNPOD_MODEL_NAME, else rely on worker override
            self._runpod_model_name = os.getenv(
                "RUNPOD_MODEL_NAME", "eve-esa/mistral_small_v1"
            )

            # Lazily initialized; create on first use to avoid unnecessary startup cost
            self._runpod_chat_openai: ChatOpenAI | None = None
        except Exception as e:
            logger.error(f"Failed to initialize LangChain Runpod client: {e}")
            self._runpod_base_url = None
            self._runpod_model_name = None
            self._runpod_chat_openai = None

        # Configure Mistral native client lazily
        try:
            # Mistral provides an OpenAI-compatible Chat Completions API
            self._mistral_base_url = os.getenv(
                "MISTRAL_BASE_URL", "https://api.mistral.ai/v1"
            )
            self._mistral_model_name = self.config.get_mistral_model()
            self._mistral_chat: ChatMistralAI | None = None
        except Exception as e:
            logger.error(f"Failed to initialize Mistral fallback client config: {e}")
            self._mistral_base_url = None
            self._mistral_model_name = None
            self._mistral_chat = None

    def _get_runpod_llm(self) -> ChatOpenAI:
        """Return a configured ChatOpenAI client for Runpod."""
        if self._runpod_chat_openai is None:
            if not self._runpod_base_url:
                raise RuntimeError("Runpod base URL is not configured")
            if not RUNPOD_API_KEY:
                raise RuntimeError("RUNPOD_API_KEY is not set")
            instruct_llm_timeout = self.config.get_instruct_llm_timeout()
            self._runpod_chat_openai = ChatOpenAI(
                api_key=RUNPOD_API_KEY,
                base_url=self._runpod_base_url,
                model="eve-esa/mistral_small_v1",
                temperature=0.3,
                timeout=instruct_llm_timeout,
                max_retries=0,
            )
        return self._runpod_chat_openai

    def _get_mistral_llm(self) -> ChatMistralAI:
        """Return a configured ChatMistralAI client for Mistral (fallback)."""
        if self._mistral_chat is None:
            if not MISTRAL_API_KEY:
                raise RuntimeError("MISTRAL_API_KEY is not set")
            mistral_timeout = self.config.get_mistral_timeout()
            self._mistral_chat = ChatMistralAI(
                api_key=MISTRAL_API_KEY,
                model=self._mistral_model_name or "mistral-small-latest",
                temperature=0.3,
                timeout=mistral_timeout,
                streaming=True,
            )
        return self._mistral_chat

    def get_model(self) -> ChatOpenAI:
        """Public accessor for the primary ChatOpenAI client with fallback to Mistral."""
        try:
            return self._get_runpod_llm()
        except Exception as e:
            logger.warning(
                f"Falling back to Mistral ChatMistralAI due to Runpod error: {e}"
            )
            return self._get_mistral_llm()

    def get_mistral_model(self) -> ChatMistralAI:
        """Public accessor for the Mistral model."""
        return self._get_mistral_llm()

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
        return format_template("basic_prompt", query=query, context=context)

    def _build_conversation_context(
        self, conversation_history: List[Any], summary: Optional[str] = None
    ) -> str:
        """
        Build conversation context from message history and optional summary.

        Args:
            conversation_history: List of previous messages
            summary: Optional conversation summary

        Returns:
            Formatted conversation context string
        """
        context_parts = []

        # Add summary if available
        if summary and summary.strip():
            context_parts.append(f"Conversation Summary: {summary}")
            context_parts.append("")  # Empty line for separation

        # Add recent message history
        if conversation_history:
            for msg in conversation_history:
                try:
                    # Handle LangChain message objects
                    if hasattr(msg, "content") and hasattr(msg, "__class__"):
                        role = (
                            "User" if "Human" in msg.__class__.__name__ else "Assistant"
                        )
                        content = msg.content
                    # Handle dict-style messages
                    elif isinstance(msg, dict):
                        role = msg.get("role", "User").title()
                        content = msg.get("content", "")
                    else:
                        continue

                    if content.strip():
                        context_parts.append(f"{role}: {content}")
                except Exception:
                    continue

        return "\n".join(context_parts)

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
        return format_template(
            "conversation_prompt",
            query=query,
            context=context,
            conversation_context=conversation_context,
        )

    def _call_eve_instruct(self, prompt: str, max_new_tokens: int = 150) -> str:
        """
        Call the Eve Instruct model via Runpod using LangChain ChatOpenAI (OpenAI-compatible API).

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
            base_llm = self._get_runpod_llm()
            llm = base_llm.bind(max_tokens=max_new_tokens)
            response = llm.invoke(prompt)
            return getattr(response, "content", str(response))
        except Exception as e:
            logger.error(f"Eve Instruct Runpod API call failed: {str(e)}")
            raise

    async def _call_eve_instruct_mistral(
        self, prompt: str, max_new_tokens: int = 150, temperature: float | None = None
    ) -> str:
        """
        Call the Eve Instruct model via Mistral using LangChain ChatMistralAI.
        """
        try:
            base_llm = self._get_mistral_llm()
            bind_kwargs = {"max_tokens": max_new_tokens}
            if temperature is not None:
                bind_kwargs["temperature"] = temperature
            llm = base_llm.bind(**bind_kwargs)
            response = await llm.ainvoke(prompt)
            return getattr(response, "content", str(response))
        except Exception as e:
            logger.error(f"Mistral model call failed: {str(e)}")
            raise

    async def _call_eve_instruct_async(
        self, prompt: str, max_new_tokens: int = 150, temperature: float | None = None
    ) -> str:
        """Async version using LangChain ChatOpenAI."""
        try:
            base_llm = self._get_runpod_llm()
            bind_kwargs = {"max_tokens": max_new_tokens}
            if temperature is not None:
                bind_kwargs["temperature"] = temperature
            llm = base_llm.bind(**bind_kwargs)
            response = await llm.ainvoke(prompt)
            return getattr(response, "content", str(response))
        except Exception as e:
            logger.error(
                f"Eve Instruct async API call failed: {str(e)}. Trying Mistral fallback."
            )
            raise

    def _process_eve_response(self, response) -> str:
        """Kept for backward compatibility; now simply returns content if present."""
        return getattr(response, "content", str(response))

    async def summarize_context_in_all(
        self, transcript: str, max_tokens: int = 50000, is_force: bool = False
    ) -> str:
        """Summarize entire conversation history."""
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
        try:
            llm = self.get_model()
            resp = await llm.bind(max_tokens=max_tokens).ainvoke(messages)
            return getattr(resp, "content", str(resp))
        except Exception:
            llm = self.get_mistral_model()
            resp = await llm.bind(max_tokens=max_tokens).ainvoke(messages)
            return getattr(resp, "content", str(resp))

    async def summarize_context_with_map_reduce(
        self, context: str, max_tokens: int
    ) -> str:
        """Summarize RAG/MCP context using LangChain where possible, respecting max_tokens.

        Uses LangChain's RecursiveCharacterTextSplitter, Document, and load_summarize_chain
        with the Runpod-backed ChatOpenAI from LLMManager. Falls back to a manual
        map-reduce approach if LangChain summarization utilities are unavailable.
        """
        if str_token_counter(context) <= max_tokens:
            return context
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400_000, chunk_overlap=0
        )
        docs = [Document(page_content=t) for t in text_splitter.split_text(context)]

        try:
            llm = self.get_model()
            chain = load_summarize_chain(llm, chain_type="map_reduce")

            summary = chain.invoke(docs)
            trimmed_context = trim_text_to_token_limit(
                summary["output_text"], max_tokens
            )
            return trimmed_context
        except Exception:
            llm = self.get_mistral_model()
            chain = load_summarize_chain(llm, chain_type="map_reduce")
            summary = chain.invoke(docs)
            trimmed_context = trim_text_to_token_limit(
                summary["output_text"], max_tokens
            )
            return trimmed_context

    async def generate_answer(
        self,
        query: str,
        context: str,
        max_new_tokens: int = 150,
        temperature: float | None = None,
    ) -> str:
        """
        Generate an answer using the runpod model.

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
            logger.error(f"Failed to generate answer using runpod model: {str(e)}")
            raise

    async def generate_answer_mistral(
        self,
        query: str,
        context: str,
        max_new_tokens: int = 150,
        temperature: float | None = None,
        conversation_history: Optional[List[Any]] = None,
        conversation_summary: Optional[str] = None,
    ) -> str:
        """
        Generate an answer using the mistral model.

        Args:
            query: The user's question
            context: Contextual information to assist the model
            max_new_tokens: Maximum new tokens for the response
            temperature: Temperature for generation
            conversation_history: Previous conversation messages for context
            conversation_summary: Optional conversation summary for context

        Returns:
            The generated answer

        Raises:
            Exception: For other errors during generation
        """
        try:
            # If conversation history or summary is provided, use it for multi-turn context
            if conversation_history or conversation_summary:
                conversation_context = self._build_conversation_context(
                    conversation_history or [], conversation_summary
                )
                available_tokens = (MODEL_CONTEXT_SIZE - max_new_tokens) // 2
                if str_token_counter(conversation_context) > available_tokens:
                    conversation_context = await self.summarize_context_with_map_reduce(
                        context=conversation_context, max_tokens=available_tokens
                    )
                prompt = self._generate_prompt_with_history(
                    query=query,
                    context=context,
                    conversation_context=conversation_context,
                )
            else:
                prompt = self._generate_prompt(query=query, context=context)

            return await self._call_eve_instruct_mistral(
                prompt, max_new_tokens=max_new_tokens, temperature=temperature
            )

        except Exception as e:
            logger.error(f"Failed to generate answer using mistral model: {str(e)}")
            raise

    async def generate_answer_mistral_stream(
        self,
        query: str,
        context: str,
        max_new_tokens: int = 150,
        temperature: float | None = None,
        conversation_history: Optional[List[Any]] = None,
        conversation_summary: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Generate an answer using the mistral model, streaming tokens."""
        try:
            if conversation_history or conversation_summary:
                conversation_context = self._build_conversation_context(
                    conversation_history or [], conversation_summary
                )
                prompt = self._generate_prompt_with_history(
                    query=query,
                    context=context,
                    conversation_context=conversation_context,
                )
            else:
                prompt = self._generate_prompt(query=query, context=context)

            async for chunk in self._call_eve_instruct_mistral_stream(
                prompt, max_new_tokens=max_new_tokens, temperature=temperature
            ):
                yield chunk
        except Exception as e:
            logger.error(f"Failed to generate answer using mistral model: {str(e)}")
            raise

    async def _call_eve_instruct_mistral_stream(
        self, prompt: str, max_new_tokens: int = 150, temperature: float | None = None
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from the Mistral model via LangChain ChatMistralAI."""
        try:
            base_llm = self._get_mistral_llm()
            bind_kwargs = {"max_tokens": max_new_tokens}
            if temperature is not None:
                bind_kwargs["temperature"] = temperature
            llm = base_llm.bind(**bind_kwargs)

            # Prefer astream which yields message chunks with `.content`
            async for event in llm.astream(prompt):
                try:
                    text = getattr(event, "content", None)
                    if not text and isinstance(event, dict):
                        text = event.get("content")
                    if text:
                        yield str(text)
                except Exception:
                    continue
        except Exception as e:
            logger.error(f"Mistral streaming call failed: {str(e)}")
            raise
