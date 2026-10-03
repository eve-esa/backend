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
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.vector_store_manager import VectorStoreManager
from src.utils.helpers import save_upload_file_to_temp
from src.utils.file_parser import FileParser
from src.schemas.documents import (
    AddDocumentRequest,
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
        collection_id: str,
        user_id: str,
        files: List[UploadFile],
        request: AddDocumentRequest,
        metadata_urls: Optional[List[str] | str] = None,
        metadata_names: Optional[List[str] | str] = None,
        document_ids: Optional[List[str]] = None,
    ) -> DocumentResult:
        """
        Add documents to a collection.

        Args:
            collection_id: Logical Mongo collection ID
            user_id: Owner user ID
            files: List of uploaded files
            request: Add document request with configuration
            metadata_urls: Optional metadata URLs
            metadata_names: Optional metadata names
            document_ids: Optional Mongo document IDs aligned with ``files``

        Returns:
            DocumentResult: Result of the operation
        """
        # Initialize temp_files at the beginning to ensure it's always defined
        temp_files = []

        try:
            logger.info(
                f"Processing {len(files)} files for collection '{collection_id}'"
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
            ingested_document_ids: List[str] = []
            ingested_file_count = 0
            aligned_ids = list(document_ids or [])
            if len(aligned_ids) < len(files):
                aligned_ids.extend([None] * (len(files) - len(aligned_ids)))

            for file, url, name, document_id in zip(
                files, processed_urls, processed_names, aligned_ids
            ):
                if not name.strip():
                    logger.warning(f"Skipping {file.filename} - no valid source_name")
                    continue

                # Save and parse file
                temp_path = await save_upload_file_to_temp(file)
                temp_files.append(temp_path)
                filename = file.filename or ""
                extension = os.path.splitext(filename)[1].lower()
                documents = await self.file_parser.parse_file(temp_path, extension)

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
                    if document_id:
                        doc.metadata["document_id"] = document_id
                split_docs = text_splitter.split_documents(documents)
                if not split_docs:
                    logger.warning(f"No chunks produced from {file.filename}")
                    continue
                all_documents.extend(split_docs)
                ingested_file_count += 1
                if document_id:
                    ingested_document_ids.append(document_id)
                logger.info(f"Processed {len(split_docs)} chunks from {file.filename}")

            # Handle results
            if not all_documents:
                return DocumentResult(
                    success=False,
                    message="No documents processed",
                    data={"collection": collection_id},
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
                await vector_store.add_document_list(
                    valid_documents,
                    user_id=user_id,
                    collection_id=collection_id,
                )

                return DocumentResult(
                    success=True,
                    message=(
                        f"Successfully processed {len(valid_documents)} chunks "
                        f"from {ingested_file_count} files"
                    ),
                    data={
                        "collection": collection_id,
                        "chunk_count": len(valid_documents),
                        "file_count": ingested_file_count,
                        "ingested_document_ids": ingested_document_ids,
                    },
                )

            return DocumentResult(
                success=False,
                message="No valid documents with source_name",
                data={"collection": collection_id},
            )

        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}", exc_info=True)
            return DocumentResult(
                success=False,
                message="Error processing documents",
                error=str(e),
                data={"collection": collection_id},
            )
        finally:
            # Clean up temp files
            for temp_path in temp_files:
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except Exception as e:
                    logger.error(f"Failed to remove temp file {temp_path}: {str(e)}")
