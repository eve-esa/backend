"""
Test cases for document update functionality.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi import HTTPException

from src.services.document import DocumentService
from src.schemas.documents import UpdateDocumentRequest


class TestUpdateDocuments:
    """Test cases for document update functionality."""

    @pytest.fixture
    def document_service(self):
        """Create a DocumentService instance for testing."""
        return DocumentService()

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock VectorStoreManager."""
        mock_store = MagicMock()

        # Mock the update result
        class MockUpdateResult:
            def __init__(self, updated_count):
                self.updated = updated_count

        mock_store.update_documents_by_metadata_filter.return_value = MockUpdateResult(
            3
        )
        return mock_store

    @pytest.fixture
    def update_request(self):
        """Create a sample update request."""
        return UpdateDocumentRequest(
            embeddings_model="nasa-ai/nasa-ai-embedding-v1",
            source_name="test_document.pdf",
            new_metadata={"category": "technical", "priority": "high"},
        )

    async def test_update_documents_success(
        self, document_service, mock_vector_store, update_request
    ):
        """Test successful document update."""
        # Mock the _get_vector_store_manager method
        document_service._get_vector_store_manager = MagicMock(
            return_value=mock_vector_store
        )

        result = await document_service.update_documents(
            collection_name="test_collection", request=update_request
        )

        # Verify the result
        assert result.success is True
        assert result.message == "Documents updated successfully"
        assert result.data["collection"] == "test_collection"
        assert result.data["source_name"] == "test_document.pdf"
        assert result.data["updated_count"] == 3
        assert result.data["new_metadata"] == {
            "category": "technical",
            "priority": "high",
        }

        # Verify the vector store method was called correctly
        mock_vector_store.update_documents_by_metadata_filter.assert_called_once_with(
            collection_name="test_collection",
            metadata_filter={"source_name": "test_document.pdf"},
            new_metadata={"category": "technical", "priority": "high"},
        )

    async def test_update_documents_no_documents_found(
        self, document_service, mock_vector_store, update_request
    ):
        """Test update when no documents match the filter."""

        # Mock the update result to return 0 updated documents
        class MockUpdateResult:
            def __init__(self, updated_count):
                self.updated = updated_count

        mock_vector_store.update_documents_by_metadata_filter.return_value = (
            MockUpdateResult(0)
        )
        document_service._get_vector_store_manager = MagicMock(
            return_value=mock_vector_store
        )

        result = await document_service.update_documents(
            collection_name="test_collection", request=update_request
        )

        # Verify the result
        assert result.success is False
        assert result.message == "No documents found to update"
        assert "No documents found with source_name 'test_document.pdf'" in result.error
        assert result.data["updated_count"] == 0

    async def test_update_documents_exception(self, document_service, update_request):
        """Test update when an exception occurs."""
        # Mock the _get_vector_store_manager to raise an exception
        document_service._get_vector_store_manager = MagicMock(
            side_effect=Exception("Test error")
        )

        result = await document_service.update_documents(
            collection_name="test_collection", request=update_request
        )

        # Verify the result
        assert result.success is False
        assert result.message == "Error updating documents"
        assert "Test error" in result.error
        assert result.data["collection"] == "test_collection"

    async def test_update_documents_empty_metadata(
        self, document_service, mock_vector_store
    ):
        """Test update with empty metadata."""
        update_request = UpdateDocumentRequest(
            embeddings_model="nasa-ai/nasa-ai-embedding-v1",
            source_name="test_document.pdf",
            new_metadata={},
        )

        document_service._get_vector_store_manager = MagicMock(
            return_value=mock_vector_store
        )

        result = await document_service.update_documents(
            collection_name="test_collection", request=update_request
        )

        # Verify the result - should still work with empty metadata
        assert result.success is True
        assert result.message == "Documents updated successfully"
        assert result.data["new_metadata"] == {}

    async def test_update_documents_none_metadata(
        self, document_service, mock_vector_store
    ):
        """Test update with None metadata."""
        update_request = UpdateDocumentRequest(
            embeddings_model="nasa-ai/nasa-ai-embedding-v1",
            source_name="test_document.pdf",
            new_metadata=None,
        )

        document_service._get_vector_store_manager = MagicMock(
            return_value=mock_vector_store
        )

        result = await document_service.update_documents(
            collection_name="test_collection", request=update_request
        )

        # Verify the result - should still work with None metadata
        assert result.success is True
        assert result.message == "Documents updated successfully"
        assert result.data["new_metadata"] is None
