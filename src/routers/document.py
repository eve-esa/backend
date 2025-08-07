"""RESTful document endpoints for collections."""

import os
import logging
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Path

from src.schemas.documents import AddDocumentRequest
from src.services.document import DocumentService
from src.database.models.collection import Collection
from src.database.models.user import User
from src.database.models.document import Document as DocumentModel
from src.middlewares.auth import get_current_user
from src.core.vector_store_manager import VectorStoreManager
from src.database.mongo_model import PaginatedResponse
from src.constants import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)
from src.schemas.common import Pagination

# Setup
router = APIRouter()
logger = logging.getLogger(__name__)
document_service = DocumentService()


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
    embeddings_model: str = Form(default=DEFAULT_EMBEDDING_MODEL),
    chunk_size: int = Form(default=DEFAULT_CHUNK_SIZE),
    chunk_overlap: int = Form(default=DEFAULT_CHUNK_OVERLAP),
    requesting_user: User = Depends(get_current_user),
):
    """Upload documents to a collection."""
    await get_collection_and_validate_ownership(collection_id, requesting_user)

    logger.info(
        f"Received {len(files)} files for processing in collection {collection_id}"
    )

    docs_data = [
        await DocumentModel.create(
            user_id=requesting_user.id,
            collection_id=collection_id,
            name=file.filename,
            filename=file.filename,
            file_type=os.path.splitext(file.filename)[1].lstrip("."),
            source_url=metadata_urls[i] if metadata_urls else None,
            source_name=metadata_names[i] if metadata_names else None,
        )
        for i, file in enumerate(files)
    ]

    try:
        result = await document_service.add_documents(
            collection_name=collection_id,
            files=files,
            request=AddDocumentRequest(
                embeddings_model=embeddings_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                metadata_urls=metadata_urls,
                metadata_names=metadata_names,
            ),
            metadata_urls=metadata_urls,
            metadata_names=metadata_names,
        )

        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)

        await DocumentModel.bulk_create(docs_data)
        return result.data
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error processing documents: {str(e)}"
        )


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
