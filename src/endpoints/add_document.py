"""FastAPI router for uploading and processing documents into a Qdrant vector store.

Supports PDF/TXT/MD file parsing, chunking, and metadata via /add_document_list.
"""

import os
import logging
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Form
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.config import QDRANT_API_KEY, QDRANT_URL
from src.services.vector_store_manager import VectorStoreManager
from src.services.utils import save_upload_file_to_temp
from src.services.file_parser import FileParser

# Constants
DEFAULT_COLLECTION = "esa-nasa-workshop"
# TODO: this was changed verify if it is correct
DEFAULT_EMBEDDINGS = "nasa-impact/nasa-smd-ibm-st-v2"
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 0

# Setup
router = APIRouter()
logger = logging.getLogger(__name__)


def process_metadata(metadata_input: Optional[List[str] | str], file_count: int) -> List[str]:
    """Process metadata (URLs or names) into a list matching file count."""
    if not metadata_input:
        return [""] * file_count

    if isinstance(metadata_input, str):
        parts = [part.strip() for part in metadata_input.split(",")]
        return (parts + [""] * (file_count - len(parts)))[:file_count]

    if isinstance(metadata_input, list):
        return [
            item.split(",")[0].strip() if isinstance(item, str) and "," in item else (item.strip() if item else "")
            for item in metadata_input[:file_count]
        ] + [""] * (file_count - len(metadata_input))

    return [""] * file_count

@router.post("/add_document_list")
async def add_document_list(
    collection_name: str = Form(default=DEFAULT_COLLECTION),
    embeddings_model: str = Form(default=DEFAULT_EMBEDDINGS),
    files: List[UploadFile] = File(...),
    chunk_size: int = Form(default=DEFAULT_CHUNK_SIZE),
    chunk_overlap: int = Form(default=DEFAULT_CHUNK_OVERLAP),
    metadata_urls: Optional[List[str] | str] = Form(default=None),
    metadata_names: Optional[List[str] | str] = Form(default=None),
) -> dict:
    """Add a list of documents with metadata to a vector store."""
    logger.info(f"Received {len(files)} files for processing")

    # Process metadata
    processed_urls = process_metadata(metadata_urls, len(files))
    processed_names = process_metadata(metadata_names, len(files))
    logger.debug(f"Processed metadata - URLs: {processed_urls}, Names: {processed_names}")

    # Initialize vector store and text splitter
    vector_store = VectorStoreManager(embeddings_model=embeddings_model,)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    file_parser = FileParser()

    all_documents, temp_files = [], []

    try:
        for file, url, name in zip(files, processed_urls, processed_names):
            if not name.strip():
                logger.warning(f"Skipping {file.filename} - no valid source_name")
                continue

            # Save and parse file
            temp_path = await save_upload_file_to_temp(file)
            temp_files.append(temp_path)
            extension = os.path.splitext(file.filename)[1].lower()
            documents = await file_parser(temp_path, extension)  # Fixed: Added extension

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
            return {"message": "No documents processed", "collection": collection_name}

        valid_documents = [doc for doc in all_documents if doc.metadata.get("source_name", "").strip()]
        if len(valid_documents) != len(all_documents):
            logger.warning(f"Filtered out {len(all_documents) - len(valid_documents)} documents with invalid source_name")

        if valid_documents:
            vector_store.add_document_list(collection_name=collection_name, document_list=valid_documents)
            return {
                "message": f"Successfully processed {len(valid_documents)} chunks from {len(files)} files",
                "collection": collection_name,
                "chunk_count": len(valid_documents),
            }
        return {"message": "No valid documents with source_name", "collection": collection_name}

    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}", exc_info=True)
        return {"message": "Error processing documents", "error": str(e), "collection": collection_name}

    finally:
        # Clean up temp files
        for temp_path in temp_files:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception as e:
                logger.error(f"Failed to remove temp file {temp_path}: {str(e)}")
