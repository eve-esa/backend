"""FastAPI router for uploading and processing documents into a Qdrant vector store.

Supports PDF/TXT/MD file parsing, chunking, and metadata via /add_document_list.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status

from src.schemas import (
    CollectionRequest, 
    RetrieveRequest, 
    DeleteRequest, 
    AddDocumentRequest, 
    UpdateDocumentRequest
)
from src.services.document import DocumentService

# Constants
DEFAULT_COLLECTION = "esa-nasa-workshop"

# Setup
router = APIRouter()
logger = logging.getLogger(__name__)
document_service = DocumentService()


@router.put("/collections/{collection_name}/documents")
async def add_document_to_collection(
    collection_name: str,
    request: CollectionRequest,
    files: List[UploadFile] = File(...),
    chunk_size: int = Form(default=1024),
    chunk_overlap: int = Form(default=0),
    metadata_urls: Optional[List[str] | str] = Form(default=None),
    metadata_names: Optional[List[str] | str] = Form(default=None),
) -> dict:
    """Add a list of documents with metadata to a vector store."""
    logger.info(f"Received {len(files)} files for processing")

    # Create AddDocumentRequest from the parameters
    add_request = AddDocumentRequest(
        embeddings_model=request.embeddings_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Use the document service
    result = await document_service.add_documents(
        collection_name=collection_name,
        files=files,
        request=add_request,
        metadata_urls=metadata_urls,
        metadata_names=metadata_names
    )

    if not result.success:
        if result.error:
            raise HTTPException(status_code=500, detail=result.error)
        return result.data

    return result.data


@router.post("/collections/{collection_name}/retrieve")
async def retrieve_documents(
    request: RetrieveRequest,
    collection_name: str = DEFAULT_COLLECTION,
):
    """Retrieve documents from a collection based on a query."""
    result = await document_service.retrieve_documents(
        collection_name=collection_name,
        request=request
    )

    if not result.success:
        if "No documents found" in result.message:
            raise HTTPException(status_code=404, detail="No documents found.")
        raise HTTPException(status_code=500, detail=result.error or "An error occurred")

    return result.data["results"]


@router.delete("/collections/{collection_name}/documents", status_code=status.HTTP_200_OK)
async def delete_document_list(
    request: DeleteRequest,
    collection_name: str = DEFAULT_COLLECTION,
):
    """Delete documents from a collection."""
    result = await document_service.delete_documents(
        collection_name=collection_name,
        request=request
    )

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": result.message,
                "errors": result.data.get("errors", []) if result.data else []
            },
        )

    return result.data


@router.patch("/collections/{collection_name}/documents")
async def update_documents(
    request: UpdateDocumentRequest,
    collection_name: str = DEFAULT_COLLECTION,
):
    """Update document metadata in a collection."""
    result = await document_service.update_documents(
        collection_name=collection_name,
        request=request
    )

    if not result.success:
        raise HTTPException(status_code=500, detail=result.error or "An error occurred")

    return result.data