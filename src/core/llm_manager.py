"""
LLM Manager module that handles different language model interactions.
"""

import logging
from enum import Enum
import asyncio
from typing import AsyncGenerator

import openai
import runpod

from src.config import Config, RUNPOD_API_KEY


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

    def _setup_api_keys(self):
        """Set up API keys for different LLM providers."""
        runpod.api_key = RUNPOD_API_KEY
        # Consider adding OpenAI API key setup here if needed

    def __call__(self, *args, **kwargs):
        """Make the class callable, delegating to generate_answer."""
        return self.generate_answer(*args, **kwargs)

    def _generate_prompt(self, query: str, context: str) -> str:
        """
        Generate a prompt for the language model.

        Args:
            query: The user's question
            context: Contextual information to assist the model

        Returns:
            A formatted prompt string
        """
        return f"""
        You are a helpful assistant named Eve developed by ESA and PiSchool that helps researchers and students understand topics regarding Earth Observation.
        Given the following context: {context}.
        
        Please reply in a precise and accurate manner to this query: {query}
        
        Answer:
        """

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
        Call the Eve Instruct model via RunPod.

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
            endpoint = runpod.Endpoint(self.config.get_instruct_llm_id())
            logger.debug(f"Sending prompt to Eve Instruct: {prompt}")

            response = endpoint.run_sync(
                {
                    "input": {
                        "prompt": prompt,
                        "sampling_params": {"max_tokens": max_new_tokens},
                    }
                },
                timeout=self.config.get_instruct_llm_timeout(),
            )

            # Debug: Log the actual response structure
            logger.debug(f"Eve Instruct response: {response}")

            # Handle different possible response structures
            if isinstance(response, dict):
                # If response is a dictionary, extract the text
                if "output" in response:
                    return response["output"]
                elif "text" in response:
                    return response["text"]
                elif "choices" in response and len(response["choices"]) > 0:
                    choice = response["choices"][0]
                    if "text" in choice:
                        return choice["text"]
                    elif "message" in choice:
                        return choice["message"]["content"]
                    elif "tokens" in choice:
                        # Handle tokens array
                        tokens = choice["tokens"]
                        if isinstance(tokens, list):
                            return " ".join(str(token) for token in tokens)
                        else:
                            return str(tokens)
            elif isinstance(response, list) and len(response) > 0:
                # If response is a list, try to extract from first element
                first_item = response[0]
                if isinstance(first_item, dict):
                    if "choices" in first_item and len(first_item["choices"]) > 0:
                        choice = first_item["choices"][0]
                        if "tokens" in choice:
                            tokens = choice["tokens"]
                            if isinstance(tokens, list):
                                return " ".join(str(token) for token in tokens)
                            else:
                                return str(tokens)
                        elif "text" in choice:
                            return choice["text"]
                        elif "message" in choice:
                            return choice["message"]["content"]
                return str(first_item)
            elif isinstance(response, str):
                # If response is already a string, return it
                return response

            # Fallback: convert to string
            logger.warning(f"Unexpected response format: {type(response)}")
            return str(response)

        except TimeoutError as e:
            logger.error(f"Eve Instruct job timed out: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Eve Instruct API call failed: {str(e)}")
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
        Stream response from Eve Instruct model via RunPod.

        Args:
            prompt: The formatted prompt
            max_new_tokens: Maximum new tokens for the response

        Yields:
            Chunks of the generated response
        """
        try:
            endpoint = runpod.Endpoint(self.config.get_instruct_llm_id())
            logger.debug(f"Streaming prompt to Eve Instruct: {prompt}")

            job = endpoint.run(
                {
                    "input": {
                        "prompt": prompt,
                        "sampling_params": {"max_tokens": max_new_tokens},
                    }
                }
            )

            logger.debug(f"Submitted job with ID: {job.job_id}")

            for chunk in job.stream():
                if chunk:
                    try:
                        text_content = self._process_stream_chunk(chunk)
                        if text_content:
                            yield text_content

                    except Exception as e:
                        logger.error(f"Error processing stream chunk: {e}")
                        continue

        except Exception as e:
            logger.error(f"Eve Instruct streaming API call failed: {str(e)}")
            raise

    async def _call_eve_instruct_async(
        self, prompt: str, max_new_tokens: int = 150
    ) -> str:
        """
        Async version of Eve Instruct call in case run_sync returns a coroutine.

        Args:
            prompt: The formatted prompt
            max_new_tokens: Maximum new tokens for the response

        Returns:
            The generated response
        """
        try:
            endpoint = runpod.Endpoint(self.config.get_instruct_llm_id())
            logger.debug(f"Sending prompt to Eve Instruct (async): {prompt}")

            # If run_sync actually returns a coroutine, await it
            try:
                response = await endpoint.run_sync(
                    {
                        "input": {
                            "prompt": prompt,
                            "sampling_params": {"max_tokens": max_new_tokens},
                        }
                    },
                    timeout=self.config.get_instruct_llm_timeout(),
                )
            except (TypeError, AttributeError) as e:
                if "coroutine" in str(e).lower() or "await" in str(e).lower():
                    # If we get a coroutine error, try the sync version
                    logger.info("Detected coroutine response, trying sync approach")
                    response = endpoint.run_sync(
                        {
                            "input": {
                                "prompt": prompt,
                                "sampling_params": {"max_tokens": max_new_tokens},
                            }
                        },
                        timeout=self.config.get_instruct_llm_timeout(),
                    )
                else:
                    raise

            # Process response similar to sync version
            return self._process_eve_response(response)

        except Exception as e:
            logger.error(f"Eve Instruct async API call failed: {str(e)}")
            raise

    def _process_eve_response(self, response) -> str:
        """
        Process the Eve Instruct response and extract text.

        Args:
            response: The response from Eve Instruct

        Returns:
            Processed text response
        """
        # Debug: Log the actual response structure
        logger.debug(f"Processing Eve response: {response}")

        # Handle different possible response structures
        if isinstance(response, dict):
            if "output" in response:
                return response["output"]
            elif "text" in response:
                return response["text"]
            elif "choices" in response and len(response["choices"]) > 0:
                choice = response["choices"][0]
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
        elif isinstance(response, list) and len(response) > 0:
            first_item = response[0]
            if isinstance(first_item, dict):
                if "choices" in first_item and len(first_item["choices"]) > 0:
                    choice = first_item["choices"][0]
                    if "tokens" in choice:
                        tokens = choice["tokens"]
                        if isinstance(tokens, list):
                            return " ".join(str(token) for token in tokens)
                        else:
                            return str(tokens)
                    elif "text" in choice:
                        return choice["text"]
                    elif "message" in choice:
                        return choice["message"]["content"]
            return str(first_item)
        elif isinstance(response, str):
            return response

        # Fallback
        logger.warning(f"Unexpected response format: {type(response)}")
        return str(response)

    def generate_answer(
        self,
        query: str,
        context: str,
        llm: str = "llama-3.1",
        max_new_tokens: int = 150,
    ) -> str:
        """
        Generate an answer using the specified language model.

        Args:
            query: The user's question
            context: Contextual information to assist the model
            llm: Which language model to use (default: "llama-3.1")
            max_new_tokens: Maximum new tokens for the response

        Returns:
            The generated answer

        Raises:
            ValueError: If an unsupported LLM type is specified
            Exception: For other errors during generation
        """
        try:
            prompt = self._generate_prompt(query=query, context=context)

            if llm == LLMType.OPENAI.value:
                return self._call_openai(prompt, max_tokens=max_new_tokens)

            elif llm == LLMType.EVE_INSTRUCT.value:
                # Truncate context if needed
                if context:
                    max_context_len = (1024 - max_new_tokens) * 4
                    if len(context) > max_context_len:
                        logger.info(
                            f"Truncating context from {len(context)} to {max_context_len} characters"
                        )
                        context = context[:max_context_len]

                try:
                    return self._call_eve_instruct(
                        prompt, max_new_tokens=max_new_tokens
                    )
                except TypeError as e:
                    if "coroutine" in str(e).lower():
                        # If we get a coroutine error, try the async version
                        logger.info(
                            "Detected coroutine response, trying async approach"
                        )
                        return asyncio.run(
                            self._call_eve_instruct_async(prompt, max_new_tokens)
                        )
                    else:
                        raise

            else:
                # Handle other LLM types or raise error
                raise ValueError(f"Unsupported LLM type: {llm}")

        except Exception as e:
            logger.error(f"Failed to generate answer: {str(e)}")
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
                    max_context_len = (1024 - max_new_tokens) * 4
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
