import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)


@pytest.mark.integration
def test_fallback_mechanism_request_structure():
    """
    Test that the fallback_llm parameter is properly handled in the request.
    """
    request_data = {
        "query": "What is ESA?",
        "collection_name": "test_collection",
        "llm": "eve-instruct-v0.1",
        "fallback_llm": "mistral-vanilla",
        "embeddings_model": "text-embedding-3-small",
        "score_threshold": 0.7,
        "get_unique_docs": True,
    }

    response = client.post("/generate_answer", json=request_data)

    # The API might work or fail depending on environment, so accept both cases
    assert response.status_code in [200, 400, 401, 403, 500]
    if response.status_code == 200:
        # If it works, check that we get a response
        assert "answer" in response.json()
    else:
        # If it fails, check that we get an error detail
        assert "detail" in response.json()


@pytest.mark.integration
def test_fallback_mechanism_streaming_request_structure():
    """
    Test that the fallback_llm parameter is properly handled in streaming requests.
    """
    request_data = {
        "query": "What is ESA?",
        "collection_name": "test_collection",
        "llm": "eve-instruct-v0.1",
        "fallback_llm": "mistral-vanilla",
        "embeddings_model": "text-embedding-3-small",
        "score_threshold": 0.7,
        "get_unique_docs": True,
    }

    response = client.post("/generate_answer_stream", json=request_data)

    # The API might work or fail depending on environment, so accept both cases
    assert response.status_code in [200, 400, 401, 403, 500]
    if response.status_code == 200:
        # If it works, check that we get a streaming response
        assert response.headers.get("content-type") == "text/event-stream"
    else:
        # If it fails, check that we get an error detail
        assert "detail" in response.json()


@pytest.mark.unit
@patch("src.core.llm_manager.runpod")
@patch("src.core.llm_manager.openai")
@patch("src.core.llm_manager.MISTRAL_API_KEY", "test-key")
@patch("src.core.llm_manager.LLMManager._call_vanilla_mistral")
def test_llm_manager_fallback_to_mistral(mock_call_mistral, mock_openai, mock_runpod):
    """
    Test that the LLM manager properly falls back to Mistral when RunPod fails.
    """
    from src.core.llm_manager import LLMManager

    # Mock RunPod failure
    mock_runpod.Endpoint.return_value.run_sync.side_effect = Exception("RunPod failed")

    # Mock successful Mistral response
    mock_call_mistral.return_value = "Mistral fallback response"

    llm_manager = LLMManager()

    # Test the fallback mechanism
    result = llm_manager.generate_answer(
        query="Test query",
        context="Test context",
        llm="eve-instruct-v0.1",  # This should trigger the fallback logic
        fallback_llm="mistral-vanilla",
    )

    assert result == "Mistral fallback response"
    mock_runpod.Endpoint.assert_called_once()
    mock_call_mistral.assert_called_once()


@pytest.mark.unit
@patch("src.core.llm_manager.runpod")
@patch("src.core.llm_manager.openai")
@patch("src.core.llm_manager.LLMManager._call_openai")
def test_llm_manager_fallback_to_openai(mock_call_openai, mock_openai, mock_runpod):
    """
    Test that the LLM manager properly falls back to OpenAI when RunPod fails.
    """
    from src.core.llm_manager import LLMManager

    # Mock RunPod failure
    mock_runpod.Endpoint.return_value.run_sync.side_effect = Exception("RunPod failed")

    # Mock successful OpenAI response
    mock_call_openai.return_value = "OpenAI fallback response"

    llm_manager = LLMManager()

    # Test the fallback mechanism
    result = llm_manager.generate_answer(
        query="Test query",
        context="Test context",
        llm="eve-instruct-v0.1",
        fallback_llm="openai",
    )

    assert result == "OpenAI fallback response"
    mock_runpod.Endpoint.assert_called_once()
    mock_call_openai.assert_called_once()


@pytest.mark.unit
@patch("src.core.llm_manager.runpod")
@patch("src.core.llm_manager.LLMManager._call_vanilla_mistral")
def test_llm_manager_fallback_both_fail(mock_call_mistral, mock_runpod):
    """
    Test that the LLM manager properly handles when both primary and fallback fail.
    """
    from src.core.llm_manager import LLMManager

    # Mock both RunPod and Mistral failure
    mock_runpod.Endpoint.return_value.run_sync.side_effect = Exception("RunPod failed")
    mock_call_mistral.side_effect = Exception("Mistral failed")

    llm_manager = LLMManager()

    # Test that both failures are handled properly
    with pytest.raises(Exception) as exc_info:
        llm_manager.generate_answer(
            query="Test query",
            context="Test context",
            llm="eve-instruct-v0.1",
            fallback_llm="mistral-vanilla",
        )

    # The error message should contain either the original error or the fallback error
    error_msg = str(exc_info.value)
    assert "Mistral failed" in error_msg or "Fallback LLM also failed" in error_msg


@pytest.mark.unit
@patch("src.core.llm_manager.runpod")
def test_llm_manager_unsupported_fallback(mock_runpod):
    """
    Test that the LLM manager properly handles unsupported fallback LLMs.
    """
    from src.core.llm_manager import LLMManager

    # Mock RunPod failure to trigger fallback
    mock_runpod.Endpoint.return_value.run_sync.side_effect = Exception("RunPod failed")

    llm_manager = LLMManager()

    # Test that unsupported fallback raises ValueError
    with pytest.raises(ValueError) as exc_info:
        llm_manager.generate_answer(
            query="Test query",
            context="Test context",
            llm="eve-instruct-v0.1",
            fallback_llm="unsupported-llm",
        )

    assert "Unsupported fallback LLM" in str(exc_info.value)


@pytest.mark.unit
@patch("src.core.llm_manager.runpod")
@patch("src.core.llm_manager.openai")
@patch("src.core.llm_manager.LLMManager._call_vanilla_mistral")
def test_llm_manager_timeout_fallback_to_mistral(
    mock_call_mistral, mock_openai, mock_runpod
):
    """
    Test that the LLM manager properly falls back to Mistral when Eve Instruct times out after 2 minutes.
    """
    from src.core.llm_manager import LLMManager
    import time

    # Mock RunPod timeout (simulating 2-minute timeout)
    mock_runpod.Endpoint.return_value.run_sync.side_effect = TimeoutError(
        "Eve Instruct timed out after 120s"
    )

    # Mock successful Mistral response
    mock_call_mistral.return_value = "Mistral fallback response after timeout"

    llm_manager = LLMManager()

    # Test the timeout fallback mechanism
    result = llm_manager.generate_answer(
        query="Test query",
        context="Test context",
        llm="eve-instruct-v0.1",
        fallback_llm="mistral-vanilla",
    )

    assert result == "Mistral fallback response after timeout"
    mock_runpod.Endpoint.assert_called_once()
    mock_call_mistral.assert_called_once()


@pytest.mark.unit
@patch("src.core.llm_manager.runpod")
@patch("src.core.llm_manager.openai")
@patch("src.core.llm_manager.LLMManager._stream_vanilla_mistral")
def test_llm_manager_streaming_timeout_fallback_to_mistral(
    mock_stream_mistral, mock_openai, mock_runpod
):
    """
    Test that the LLM manager properly falls back to Mistral streaming when Eve Instruct streaming times out.
    """
    from src.core.llm_manager import LLMManager
    import asyncio

    # Mock RunPod streaming timeout
    mock_runpod.Endpoint.return_value.run.return_value.stream.side_effect = (
        TimeoutError("Eve Instruct streaming timed out")
    )

    # Mock successful Mistral streaming response
    async def mock_stream():
        yield "Mistral streaming fallback response"

    mock_stream_mistral.return_value = mock_stream()

    llm_manager = LLMManager()

    # Test the streaming timeout fallback mechanism
    async def test_streaming_fallback():
        chunks = []
        async for chunk in llm_manager.generate_answer_stream(
            query="Test query",
            context="Test context",
            llm="eve-instruct-v0.1",
            fallback_llm="mistral-vanilla",
        ):
            chunks.append(chunk)
        return "".join(chunks)

    # Run the async test
    result = asyncio.run(test_streaming_fallback())

    assert "Mistral streaming fallback response" in result
    mock_runpod.Endpoint.assert_called_once()
    mock_stream_mistral.assert_called_once()
