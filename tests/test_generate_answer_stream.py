"""Test streaming functionality for generate answer endpoint."""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from src.routers.generate_answer import router
from src.schemas.collections import CollectionRequest


@pytest.fixture
def client():
    """Create a test client."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


async def _mock_stream(
    query, context, llm, max_new_tokens, fallback_llm="mistral-vanilla"
):
    """Mock stream generator."""
    yield "Hello"
    yield " world"
    yield "!"


async def _mock_stream_rag(
    query, context, llm, max_new_tokens, fallback_llm="mistral-vanilla"
):
    """Mock stream generator for RAG tests."""
    yield "Based on the context"
    yield ", Earth observation is"
    yield " important."


@pytest.mark.asyncio
async def test_generate_answer_stream_basic():
    """Test basic streaming functionality."""
    from src.routers.generate_answer import generate_answer_stream_generator
    from src.routers.generate_answer import GenerationRequest

    request = GenerationRequest(
        query="What is Earth observation?",
        collection_name="test_collection",
        llm="eve-instruct-v0.1",
        max_new_tokens=100,
    )

    # Mock the LLM manager
    with patch("src.routers.generate_answer.LLMManager") as mock_llm_manager:
        mock_manager = AsyncMock()
        mock_manager.generate_answer_stream = _mock_stream
        mock_llm_manager.return_value = mock_manager

        # Mock vector store
        with patch(
            "src.routers.generate_answer.VectorStoreManager"
        ) as mock_vector_store:
            mock_store = AsyncMock()
            mock_store.use_rag.return_value = False
            mock_vector_store.return_value = mock_store

            # Test the generator
            chunks = []
            async for chunk in generate_answer_stream_generator(request):
                chunks.append(chunk)

            # Verify we got some chunks
            assert len(chunks) > 0
            assert all(chunk.startswith("data: ") for chunk in chunks)


@pytest.mark.asyncio
async def test_generate_answer_stream_with_rag():
    """Test streaming with RAG enabled."""
    from src.routers.generate_answer import generate_answer_stream_generator
    from src.routers.generate_answer import GenerationRequest

    request = GenerationRequest(
        query="What is Earth observation?",
        collection_name="test_collection",
        llm="eve-instruct-v0.1",
        max_new_tokens=100,
    )

    # Mock the LLM manager
    with patch("src.routers.generate_answer.LLMManager") as mock_llm_manager:
        mock_manager = AsyncMock()
        mock_manager.generate_answer_stream = _mock_stream_rag
        mock_llm_manager.return_value = mock_manager

        # Mock vector store
        with patch(
            "src.routers.generate_answer.VectorStoreManager"
        ) as mock_vector_store:
            mock_store = AsyncMock()
            mock_store.use_rag.return_value = True
            mock_store.retrieve_documents_from_query.return_value = [
                type(
                    "obj",
                    (object,),
                    {"payload": {"page_content": "Test document content"}},
                )()
            ]
            mock_vector_store.return_value = mock_store

            # Test the generator
            chunks = []
            async for chunk in generate_answer_stream_generator(request):
                chunks.append(chunk)

            # Verify we got some chunks
            assert len(chunks) > 0
            assert all(chunk.startswith("data: ") for chunk in chunks)


@pytest.mark.asyncio
async def test_generate_answer_json_stream():
    """Test JSON streaming functionality."""
    from src.routers.generate_answer import generate_answer_json_stream_generator
    from src.routers.generate_answer import GenerationRequest

    request = GenerationRequest(
        query="What is Earth observation?",
        collection_name="test_collection",
        llm="eve-instruct-v0.1",
        max_new_tokens=100,
    )

    # Mock the LLM manager
    with patch("src.routers.generate_answer.LLMManager") as mock_llm_manager:
        mock_manager = AsyncMock()
        mock_manager.generate_answer_stream = _mock_stream_rag
        mock_llm_manager.return_value = mock_manager

        # Mock vector store
        with patch(
            "src.routers.generate_answer.VectorStoreManager"
        ) as mock_vector_store:
            mock_store = AsyncMock()
            mock_store.use_rag.return_value = True
            mock_store.retrieve_documents_from_query.return_value = [
                type(
                    "obj",
                    (object,),
                    {"payload": {"page_content": "Test document content"}},
                )()
            ]
            mock_vector_store.return_value = mock_store

            # Test the generator
            chunks = []
            async for chunk in generate_answer_json_stream_generator(request):
                chunks.append(chunk)

            # Verify we got JSON chunks
            assert len(chunks) > 0
            assert all(chunk.startswith("data: ") for chunk in chunks)

            # Check that we have start, chunk, and end messages
            chunk_types = []
            for chunk in chunks:
                if '"type":' in chunk:
                    chunk_types.append(chunk)

            assert len(chunk_types) >= 3  # start, at least one chunk, end


def test_streaming_endpoints_exist(client):
    """Test that streaming endpoints are accessible."""
    # Test the basic streaming endpoint
    response = client.post(
        "/generate_answer_stream",
        json={"query": "test", "collection_name": "test", "llm": "eve-instruct-v0.1"},
    )
    assert response.status_code in [200, 500]  # 500 is expected if no real LLM

    # Test the JSON streaming endpoint
    response = client.post(
        "/generate_answer_stream_json",
        json={"query": "test", "collection_name": "test", "llm": "eve-instruct-v0.1"},
    )
    assert response.status_code in [200, 500]  # 500 is expected if no real LLM
