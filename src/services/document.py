"""
Document Service for managing documents in Qdrant collections.

This module provides a service layer for adding, retrieving, updating, and deleting
documents in the Qdrant vector store.
"""

import os
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass

from fastapi import UploadFile
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.core.vector_store_manager import VectorStoreManager
from src.utils.helpers import save_upload_file_to_temp
from src.utils.file_parser import FileParser
from src.schemas.documents import (
    RetrieveRequest,
    DeleteRequest,
    AddDocumentRequest,
    UpdateDocumentRequest,
)

logger = logging.getLogger(__name__)


@dataclass
class DocumentResult:
    """Represents the result of a document operation."""

    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class DocumentService:
    """
    Service for managing documents in the vector store.

    This class provides methods to add, retrieve, update, and delete documents
    using the VectorStoreManager.
    """

    def __init__(self):
        """Initialize the document service."""
        self.vector_store_manager = None
        self.file_parser = FileParser()

    def _get_vector_store_manager(self, embeddings_model: str) -> VectorStoreManager:
        """
        Get or create a VectorStoreManager instance.

        Args:
            embeddings_model: The embeddings model to use

        Returns:
            VectorStoreManager: The vector store manager instance
        """
        if (
            self.vector_store_manager is None
            or self.vector_store_manager.embeddings_model != embeddings_model
        ):
            self.vector_store_manager = VectorStoreManager(
                embeddings_model=embeddings_model
            )
        return self.vector_store_manager

    def _process_metadata(
        self, metadata_input: Optional[List[str] | str], file_count: int
    ) -> List[str]:
        """Process metadata (URLs or names) into a list matching file count."""
        if not metadata_input:
            return [""] * file_count

        if isinstance(metadata_input, str):
            parts = [part.strip() for part in metadata_input.split(",")]
            return (parts + [""] * (file_count - len(parts)))[:file_count]

        if isinstance(metadata_input, list):
            return [
                (
                    item.split(",")[0].strip()
                    if isinstance(item, str) and "," in item
                    else (item.strip() if item else "")
                )
                for item in metadata_input[:file_count]
            ] + [""] * (file_count - len(metadata_input))

        return [""] * file_count

    async def add_documents(
        self,
        collection_name: str,
        files: List[UploadFile],
        request: AddDocumentRequest,
        metadata_urls: Optional[List[str] | str] = None,
        metadata_names: Optional[List[str] | str] = None,
    ) -> DocumentResult:
        """
        Add documents to a collection.

        Args:
            collection_name: Name of the collection
            files: List of uploaded files
            request: Add document request with configuration
            metadata_urls: Optional metadata URLs
            metadata_names: Optional metadata names

        Returns:
            DocumentResult: Result of the operation
        """
        # Initialize temp_files at the beginning to ensure it's always defined
        temp_files = []

        try:
            logger.info(
                f"Processing {len(files)} files for collection '{collection_name}'"
            )

            # Process metadata
            processed_urls = self._process_metadata(metadata_urls, len(files))
            processed_names = self._process_metadata(metadata_names, len(files))

            # Initialize components
            vector_store = self._get_vector_store_manager(request.embeddings_model)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=request.chunk_size, chunk_overlap=request.chunk_overlap
            )

            all_documents = []

            for file, url, name in zip(files, processed_urls, processed_names):
                if not name.strip():
                    logger.warning(f"Skipping {file.filename} - no valid source_name")
                    continue

                # Save and parse file
                temp_path = await save_upload_file_to_temp(file)
                temp_files.append(temp_path)
                extension = os.path.splitext(file.filename)[1].lower()
                documents = await self.file_parser(temp_path, extension)

                if not documents:
                    logger.warning(f"No documents parsed from {file.filename}")
                    continue

                # Add metadata and split
                for doc in documents:
                    doc.metadata = {
                        "source": url,
                        "source_name": name,
                        "filename": file.filename,
                        "file_type": extension.lstrip("."),
                        "upload_time": datetime.now().isoformat(),
                    }
                split_docs = text_splitter.split_documents(documents)
                all_documents.extend(split_docs)
                logger.info(f"Processed {len(split_docs)} chunks from {file.filename}")

            # Handle results
            if not all_documents:
                return DocumentResult(
                    success=False,
                    message="No documents processed",
                    data={"collection": collection_name},
                )

            valid_documents = [
                doc
                for doc in all_documents
                if doc.metadata.get("source_name", "").strip()
            ]
            if len(valid_documents) != len(all_documents):
                logger.warning(
                    f"Filtered out {len(all_documents) - len(valid_documents)} documents with invalid source_name"
                )

            if valid_documents:
                vector_store.add_document_list(
                    collection_name=collection_name, document_list=valid_documents
                )
                return DocumentResult(
                    success=True,
                    message=f"Successfully processed {len(valid_documents)} chunks from {len(files)} files",
                    data={
                        "collection": collection_name,
                        "chunk_count": len(valid_documents),
                        "file_count": len(files),
                    },
                )

            return DocumentResult(
                success=False,
                message="No valid documents with source_name",
                data={"collection": collection_name},
            )

        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}", exc_info=True)
            return DocumentResult(
                success=False,
                message="Error processing documents",
                error=str(e),
                data={"collection": collection_name},
            )
        finally:
            # Clean up temp files
            for temp_path in temp_files:
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except Exception as e:
                    logger.error(f"Failed to remove temp file {temp_path}: {str(e)}")

    async def retrieve_documents(
        self, collection_name: str, request: RetrieveRequest
    ) -> DocumentResult:
        """
        Retrieve documents from a collection based on a query.

        Args:
            collection_name: Name of the collection
            request: Retrieve request with query parameters

        Returns:
            DocumentResult: Result with retrieved documents
        """
        try:
            vector_store = self._get_vector_store_manager(request.embeddings_model)
            results = await vector_store.retrieve_documents_from_query(
                query=request.query,
                embeddings_model=request.embeddings_model,
                collection_name=collection_name,
                k=request.k,
                score_threshold=request.score_threshold,
                get_unique_docs=request.get_unique_docs,
            )

            if not results:
                return DocumentResult(
                    success=False,
                    message="No documents found",
                    data={"collection": collection_name, "results": []},
                )

            return DocumentResult(
                success=True,
                message="Documents retrieved successfully",
                data={
                    "collection": collection_name,
                    "results": results,
                    "count": len(results),
                },
            )

        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}", exc_info=True)
            return DocumentResult(
                success=False,
                message="Error retrieving documents",
                error=str(e),
                data={"collection": collection_name},
            )

    async def delete_documents(
        self, collection_name: str, request: DeleteRequest
    ) -> DocumentResult:
        """
        Delete documents from a collection.

        Args:
            collection_name: Name of the collection
            request: Delete request with document list

        Returns:
            DocumentResult: Result of the deletion operation
        """
        try:
            vector_store = self._get_vector_store_manager(request.embeddings_model)
            errors = []
            total_deleted_count = 0

            for source in request.document_list:
                try:
                    result = vector_store.delete_docs_by_metadata_filter(
                        collection_name=collection_name,
                        metadata={"source_name": source},
                    )
                    # Get the actual number of documents deleted
                    deleted_count = getattr(result, "deleted", 0)
                    total_deleted_count += deleted_count
                    logger.info(
                        f"Deleted {deleted_count} documents for source '{source}'"
                    )
                except Exception as e:
                    errors.append(f"Failed to delete document {source}: {str(e)}")

            if errors:
                return DocumentResult(
                    success=False,
                    message="Some documents could not be deleted",
                    error="; ".join(errors),
                    data={
                        "collection": collection_name,
                        "deleted_count": total_deleted_count,
                        "total_requested": len(request.document_list),
                        "errors": errors,
                    },
                )

            return DocumentResult(
                success=True,
                message="Documents deleted successfully",
                data={
                    "collection": collection_name,
                    "deleted_documents": request.document_list,
                    "deleted_count": total_deleted_count,
                },
            )

        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}", exc_info=True)
            return DocumentResult(
                success=False,
                message="Error deleting documents",
                error=str(e),
                data={"collection": collection_name},
            )

    async def update_documents(
        self, collection_name: str, request: UpdateDocumentRequest
    ) -> DocumentResult:
        """
        Update document metadata in a collection.

        Args:
            collection_name: Name of the collection
            request: Update request with new metadata

        Returns:
            DocumentResult: Result of the update operation
        """
        try:
            vector_store = self._get_vector_store_manager(request.embeddings_model)

            metadata_filter = {"source_name": request.source_name}

            result = vector_store.update_documents_by_metadata_filter(
                collection_name=collection_name,
                metadata_filter=metadata_filter,
                new_metadata=request.new_metadata,
            )

            updated_count = getattr(result, "updated", 0)

            if updated_count == 0:
                return DocumentResult(
                    success=False,
                    message="No documents found to update",
                    error=f"No documents found with source_name '{request.source_name}'",
                    data={
                        "collection": collection_name,
                        "source_name": request.source_name,
                        "updated_count": updated_count,
                    },
                )

            return DocumentResult(
                success=True,
                message="Documents updated successfully",
                data={
                    "collection": collection_name,
                    "source_name": request.source_name,
                    "updated_count": updated_count,
                    "new_metadata": request.new_metadata,
                },
            )

        except Exception as e:
            logger.error(f"Error updating documents: {str(e)}", exc_info=True)
            return DocumentResult(
                success=False,
                message="Error updating documents",
                error=str(e),
                data={"collection": collection_name},
            )
