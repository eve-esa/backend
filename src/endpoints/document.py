"""RESTful document endpoints for collections."""

import os
import logging
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Path
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.database.models.collection import Collection
from src.database.models.user import User
from src.database.models.document import Document as DocumentModel
from src.middlewares.auth import get_current_user
from src.services.vector_store_manager import VectorStoreManager
from src.services.utils import save_upload_file_to_temp
from src.services.file_parser import FileParser
from src.database.mongo_model import PaginatedResponse
from pydantic import BaseModel

# Constants
DEFAULT_EMBEDDINGS = "nasa-impact/nasa-smd-ibm-v0.1"
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 0

# Setup
router = APIRouter()
logger = logging.getLogger(__name__)


class Pagination(BaseModel):
    page: int = 1
    limit: int = 10


class DocumentUploadRequest(BaseModel):
    embeddings_model: str = DEFAULT_EMBEDDINGS
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP


def process_metadata(
    metadata_input: Optional[List[str] | str], file_count: int
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


async def get_collection_and_validate_ownership(
    collection_id: str, requesting_user: User
) -> Collection:
    """Get collection and validate user ownership."""
    collection = await Collection.find_by_id(collection_id)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    if collection.user_id != requesting_user.id:
        raise HTTPException(
            status_code=403, detail="You are not allowed to access this collection"
        )

    return collection


@router.get(
    "/collections/{collection_id}/documents",
    response_model=PaginatedResponse[DocumentModel],
)
async def list_documents(
    collection_id: str = Path(..., description="Collection ID"),
    pagination: Pagination = Depends(),
    requesting_user: User = Depends(get_current_user),
):
    """List documents in a collection."""
    await get_collection_and_validate_ownership(collection_id, requesting_user)

    return await DocumentModel.find_all_with_pagination(
        filter_dict={"collection_id": collection_id},
        limit=pagination.limit,
        page=pagination.page,
        sort=[("timestamp", -1)],
    )


@router.get(
    "/collections/{collection_id}/documents/{document_id}", response_model=DocumentModel
)
async def get_document(
    collection_id: str = Path(..., description="Collection ID"),
    document_id: str = Path(..., description="Document ID"),
    requesting_user: User = Depends(get_current_user),
):
    """Get a specific document from a collection."""
    await get_collection_and_validate_ownership(collection_id, requesting_user)

    document = await DocumentModel.find_by_id(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    if document.collection_id != collection_id:
        raise HTTPException(
            status_code=400, detail="Document does not belong to this collection"
        )

    if document.user_id != requesting_user.id:
        raise HTTPException(
            status_code=403, detail="You are not allowed to access this document"
        )

    return document


@router.post("/collections/{collection_id}/documents")
async def upload_documents(
    collection_id: str = Path(..., description="Collection ID"),
    files: List[UploadFile] = File(...),
    metadata_urls: Optional[List[str] | str] = Form(default=None),
    metadata_names: Optional[List[str] | str] = Form(default=None),
    embeddings_model: str = Form(default=DEFAULT_EMBEDDINGS),
    chunk_size: int = Form(default=DEFAULT_CHUNK_SIZE),
    chunk_overlap: int = Form(default=DEFAULT_CHUNK_OVERLAP),
    requesting_user: User = Depends(get_current_user),
):
    """Upload documents to a collection."""
    collection = await get_collection_and_validate_ownership(
        collection_id, requesting_user
    )

    logger.info(
        f"Received {len(files)} files for processing in collection {collection_id}"
    )

    # Process metadata
    processed_urls = process_metadata(metadata_urls, len(files))
    processed_names = process_metadata(metadata_names, len(files))
    logger.debug(
        f"Processed metadata - URLs: {processed_urls}, Names: {processed_names}"
    )

    # Initialize vector store and text splitter
    vector_store = VectorStoreManager(embeddings_model=embeddings_model)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    file_parser = FileParser()

    all_documents, temp_files = [], []
    uploaded_documents = []

    try:
        for file, url, name in zip(files, processed_urls, processed_names):
            if not name.strip():
                logger.warning(f"Skipping {file.filename} - no valid source_name")
                continue

            preliminary_doc = await DocumentModel.create(
                user_id=requesting_user.id,
                collection_id=collection_id,
                name=name or file.filename,
                filename=file.filename,
                file_type=os.path.splitext(file.filename)[1].lstrip("."),
                source_url=url,
            )

            # Save and parse file
            temp_path = await save_upload_file_to_temp(file)
            temp_files.append(temp_path)
            extension = os.path.splitext(file.filename)[1].lower()
            documents = await file_parser(temp_path, extension)

            if not documents:
                logger.warning(f"No documents parsed from {file.filename}")
                # cleanup: delete empty preliminary doc
                await preliminary_doc.delete()
                continue

            # Add metadata and split
            for doc in documents:
                doc.metadata = {
                    "document_id": preliminary_doc.id,
                    "source": url,
                    "source_name": name,
                    "filename": file.filename,
                    "file_type": extension.lstrip("."),
                    "upload_time": datetime.now().isoformat(),
                }
            split_docs = text_splitter.split_documents(documents)
            all_documents.extend(split_docs)

            vector_ids = vector_store.add_document_list(
                collection_name=collection.id,
                document_list=split_docs,
            )

            # Update document record with stats
            preliminary_doc.chunk_count = len(split_docs)
            preliminary_doc.file_size = (
                len(file.file.read()) if hasattr(file.file, "read") else 0
            )
            preliminary_doc.vector_ids = vector_ids
            await preliminary_doc.save()
            uploaded_documents.append(preliminary_doc)

            logger.info(f"Processed {len(split_docs)} chunks from {file.filename}")

        if not all_documents:
            return {"message": "No documents processed", "collection_id": collection_id}

        valid_documents = [
            doc for doc in all_documents if doc.metadata.get("source_name", "").strip()
        ]
        if len(valid_documents) != len(all_documents):
            logger.warning(
                f"Filtered out {len(all_documents) - len(valid_documents)} documents with invalid source_name"
            )

        if valid_documents:
            return {
                "message": f"Successfully processed {len(valid_documents)} chunks from {len(files)} files",
                "collection_id": collection_id,
                "chunk_count": len(valid_documents),
                "documents": uploaded_documents,
            }
        return {
            "message": "No valid documents with source_name",
            "collection_id": collection_id,
        }

    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error processing documents: {str(e)}"
        )

    finally:
        for temp_path in temp_files:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception as e:
                logger.error(f"Failed to remove temp file {temp_path}: {str(e)}")


@router.delete("/collections/{collection_id}/documents/{document_id}")
async def delete_document(
    collection_id: str = Path(..., description="Collection ID"),
    document_id: str = Path(..., description="Document ID"),
    requesting_user: User = Depends(get_current_user),
):
    """Delete a document from a collection."""
    await get_collection_and_validate_ownership(collection_id, requesting_user)

    document = await DocumentModel.find_by_id(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    if document.collection_id != collection_id:
        raise HTTPException(
            status_code=400, detail="Document does not belong to this collection"
        )

    if document.user_id != requesting_user.id:
        raise HTTPException(
            status_code=403, detail="You are not allowed to delete this document"
        )

    vector_store = VectorStoreManager()
    try:
        vector_store.delete_docs_by_metadata_filter(
            collection_name=collection_id,
            metadata={"document_id": document_id},
        )
    except Exception as e:
        logger.error(f"Failed to delete vectors for document {document_id}: {e}")

    await document.delete()
    return {"message": "Document and embeddings deleted successfully"}
