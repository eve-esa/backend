"""
LLM Manager module that handles different language model interactions.
"""

import logging
from enum import Enum
from typing import AsyncGenerator, List, Any
import os

import openai
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI

from src.config import Config, RUNPOD_API_KEY, MISTRAL_API_KEY
from typing import Optional
from src.constants import MODEL_CONTEXT_SIZE
from src.utils.template_loader import format_template


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

    def _call_openai(self, prompt: str, max_tokens: int = 500) -> str:
        """
        Call the OpenAI API with the given prompt.

        Args:
            prompt: The formatted prompt to send
            max_tokens: Maximum tokens for the response

        Returns:
            The generated response

        Raises:
            Exception: If the API call fails
        """
        try:
            messages = [{"role": "user", "content": prompt}]
            response = openai.chat.completions.create(
                messages=messages,
                model="gpt-4",
                temperature=0.3,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise

    async def _stream_openai(
        self, prompt: str, max_tokens: int = 500
    ) -> AsyncGenerator[str, None]:
        """
        Stream response from OpenAI API.

        Args:
            prompt: The formatted prompt to send
            max_tokens: Maximum tokens for the response

        Yields:
            Chunks of the generated response
        """
        try:
            messages = [{"role": "user", "content": prompt}]
            stream = openai.chat.completions.create(
                messages=messages,
                model="gpt-4",
                temperature=0.3,
                max_tokens=max_tokens,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"OpenAI streaming API call failed: {str(e)}")
            raise

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

    def _process_stream_chunk(self, chunk) -> str:
        """
        Process a single stream chunk and extract text content.

        Args:
            chunk: The stream chunk to process

        Returns:
            str: The extracted text content
        """
        if isinstance(chunk, dict):
            if "output" in chunk:
                output = chunk["output"]
                if isinstance(output, str):
                    return output
                elif isinstance(output, dict) and "text" in output:
                    return output["text"]
                elif isinstance(output, list) and len(output) > 0:
                    texts = []
                    for item in output:
                        if isinstance(item, dict):
                            if "text" in item:
                                texts.append(item["text"])
                            elif "tokens" in item:
                                tokens = item["tokens"]
                                if isinstance(tokens, list):
                                    texts.append(
                                        " ".join(str(token) for token in tokens)
                                    )
                                else:
                                    texts.append(str(tokens))
                        else:
                            texts.append(str(item))
                    return " ".join(texts)
            elif "text" in chunk:
                return chunk["text"]
            elif "choices" in chunk and len(chunk["choices"]) > 0:
                choice = chunk["choices"][0]
                if "text" in choice:
                    return choice["text"]
                elif "message" in choice:
                    return choice["message"]["content"]
                elif "tokens" in choice:
                    tokens = choice["tokens"]
                    if isinstance(tokens, list):
                        return " ".join(str(token) for token in tokens)
                    else:
                        return str(tokens)
        elif isinstance(chunk, str):
            return chunk
        else:
            return str(chunk)

    async def _stream_eve_instruct(
        self, prompt: str, max_new_tokens: int = 150
    ) -> AsyncGenerator[str, None]:
        """
        Stream response from Eve Instruct model via Runpod OpenAI-compatible API using LangChain.

        Args:
            prompt: The formatted prompt
            max_new_tokens: Maximum new tokens for the response

        Yields:
            Chunks of the generated response
        """
        try:
            base_llm = self._get_runpod_llm()
            llm = base_llm.bind(max_tokens=max_new_tokens)
            async for chunk in llm.astream(prompt):
                content = getattr(chunk, "content", None)
                if content:
                    yield content
        except Exception as e:
            logger.error(
                f"Eve Instruct streaming API call failed: {str(e)}. Trying Mistral fallback."
            )
            try:
                base_llm = self._get_mistral_llm()
                llm = base_llm.bind(max_tokens=max_new_tokens)
                async for chunk in llm.astream(prompt):
                    content = getattr(chunk, "content", None)
                    if content:
                        yield content
            except Exception as e2:
                logger.error(f"Mistral streaming fallback also failed: {str(e2)}")
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
                prompt = self._generate_prompt_with_history(
                    query=query,
                    context=context,
                    conversation_context=conversation_context,
                )
            else:
                prompt = self._generate_prompt(query=query, context=context)

            max_context_len = MODEL_CONTEXT_SIZE - max_new_tokens
            if len(context) > max_context_len:
                logger.info(
                    f"Truncating context from {len(context)} to {max_context_len} characters"
                )
                context = context[:max_context_len]

            return await self._call_eve_instruct_mistral(
                prompt, max_new_tokens=max_new_tokens, temperature=temperature
            )

        except Exception as e:
            logger.error(f"Failed to generate answer using mistral model: {str(e)}")
            raise

    async def generate_answer_stream(
        self,
        query: str,
        context: str,
        llm: str = "llama-3.1",
        max_new_tokens: int = 150,
    ) -> AsyncGenerator[str, None]:
        """
        Generate an answer stream using the specified language model.

        Args:
            query: The user's question
            context: Contextual information to assist the model
            llm: Which language model to use (default: "llama-3.1")
            max_new_tokens: Maximum new tokens for the response

        Yields:
            Chunks of the generated answer

        Raises:
            ValueError: If an unsupported LLM type is specified
            Exception: For other errors during generation
        """
        try:
            prompt = self._generate_prompt(query=query, context=context)

            if llm == LLMType.OPENAI.value:
                async for chunk in self._stream_openai(
                    prompt, max_tokens=max_new_tokens
                ):
                    yield chunk

            elif llm == LLMType.EVE_INSTRUCT.value:
                # Truncate context if needed
                if context:
                    max_context_len = MODEL_CONTEXT_SIZE - max_new_tokens
                    if len(context) > max_context_len:
                        logger.info(
                            f"Truncating context from {len(context)} to {max_context_len} characters"
                        )
                        context = context[:max_context_len]

                async for chunk in self._stream_eve_instruct(
                    prompt, max_new_tokens=max_new_tokens
                ):
                    yield chunk

            else:
                # Handle other LLM types or raise error
                raise ValueError(f"Unsupported LLM type: {llm}")

        except Exception as e:
            logger.error(f"Failed to generate answer stream: {str(e)}")
            raise
