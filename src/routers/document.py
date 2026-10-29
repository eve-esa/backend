"""RESTful document endpoints for collections."""

import os
import logging
from functools import partial
from typing import List, Optional

import anyio
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Path
from bson import ObjectId

from src import config
from src.schemas.documents import AddDocumentRequest
from src.services.document import DocumentService
from src.database.models.collection import Collection
from src.database.models.conversation import Conversation
from src.database.models.user import User
from src.database.models.document import Document as DocumentModel
from src.middlewares.auth import get_current_user
from src.core.vector_store_manager import VectorStoreManager
from src.database.mongo_model import PaginatedResponse
from src.constants import (
    CONVERSATION_DOCUMENT_DEADLINE_SECONDS,
    CONVERSATION_DOCUMENT_EXTENSIONS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    MAX_CONVERSATION_DOCUMENT_BYTES,
    MAX_CONVERSATION_DOCUMENT_CHUNKS,
    MAX_CONVERSATION_DOCUMENTS,
)
from src.schemas.common import Pagination
from src.services.private_document_limit import (
    reserve_private_document_slots,
    release_private_document_slots,
)
from src.services.request_rate_limiter import enforce_request_rate
from src.services.conversation_documents import (
    get_or_create_conversation_collection,
)

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


async def _rollback_uploaded_docs(
    user_id: str, collection_id: str, created_docs: List[DocumentModel]
) -> None:
    """Remove Mongo rows and any Qdrant points written for a failed upload."""
    if not created_docs:
        return
    vector_store = VectorStoreManager()
    for doc in created_docs:
        try:
            await anyio.to_thread.run_sync(
                partial(
                    vector_store.delete_private_docs,
                    user_id=user_id,
                    collection_id=collection_id,
                    metadata={"metadata.document_id": doc.id},
                )
            )
        except Exception:
            logger.warning(
                "Failed to roll back Qdrant points for document %s",
                doc.id,
                exc_info=True,
            )
    try:
        await DocumentModel.delete_many(
            {"_id": {"$in": [ObjectId(doc.id) for doc in created_docs]}}
        )
    except Exception:
        logger.warning(
            "Failed to roll back Mongo documents after upload error",
            exc_info=True,
        )


@router.get(
    "/collections/{collection_id}/documents",
    response_model=PaginatedResponse[DocumentModel],
)
async def list_documents(
    collection_id: str = Path(..., description="Collection ID"),
    pagination: Pagination = Depends(),
    requesting_user: User = Depends(get_current_user),
) -> PaginatedResponse[DocumentModel]:
    """
    List documents in a collection.

    Args:
        collection_id (str): Collection identifier.
        pagination (Pagination): Pagination parameters.
        requesting_user (User): Authenticated user injected by dependency.

    Returns:
        Paginated documents for the collection.

    Raises:
        HTTPException: 404 if collection is not found; 403 if access is forbidden.
    """
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
) -> DocumentModel:
    """
    Get a specific document from a collection.

    Args:
        collection_id (str): Collection identifier.
        document_id (str): Document identifier.
        requesting_user (User): Authenticated user injected by dependency.

    Returns:
        Document details.

    Raises:
        HTTPException: 404 if not found; 400 if document not in collection; 403 if access is forbidden.
    """
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


@router.post(
    "/collections/{collection_id}/documents",
    dependencies=[Depends(enforce_request_rate("upload"))],
)
async def upload_documents(
    collection_id: str = Path(..., description="Collection ID"),
    files: List[UploadFile] = File(...),
    metadata_urls: Optional[List[str] | str] = Form(default=None),
    metadata_names: Optional[List[str] | str] = Form(default=None),
    embeddings_model: str = Form(default=DEFAULT_EMBEDDING_MODEL),
    chunk_size: int = Form(default=DEFAULT_CHUNK_SIZE),
    chunk_overlap: int = Form(default=DEFAULT_CHUNK_OVERLAP),
    requesting_user: User = Depends(get_current_user),
) -> dict:
    """
    Upload documents to a collection.

    Stores document records and triggers asynchronous parsing, chunking, and vectorization for retrieval.

    Args:
        collection_id (str): Collection identifier.
        files (list[UploadFile]): One or more files to ingest.
        metadata_urls (list[str] | str | None): Optional list or single URL per file.
        metadata_names (list[str] | str | None): Optional list or single display name per file.
        embeddings_model (str): Embeddings model to use for vectorization.
        chunk_size (int): Chunk size for splitting documents.
        chunk_overlap (int): Overlap between chunks.
        requesting_user (User): Authenticated user injected by dependency.

    Returns:
        Service response with ingestion details.

    Raises:
        HTTPException: 404 if collection is not found; 403 if access is forbidden; 400 if the private document limit is exceeded; 500 for processing errors.
    """
    collection = await get_collection_and_validate_ownership(
        collection_id, requesting_user
    )
    # Attached documents go through POST /conversations/{id}/documents, which
    # enforces the per-conversation caps this route does not know about.
    if collection.conversation_id:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "conversation_collection",
                "message": "Attach documents to this conversation instead.",
            },
        )

    file_count = len(files)
    slots_reserved = False
    created_docs: List[DocumentModel] = []
    try:
        await reserve_private_document_slots(requesting_user.id, file_count)
        slots_reserved = True

        logger.info(
            f"Received {file_count} files for processing in collection {collection_id}"
        )

        docs_data = [
            DocumentModel(
                user_id=requesting_user.id,
                collection_id=collection_id,
                name=file.filename,
                filename=file.filename,
                file_type=os.path.splitext(file.filename)[1].lstrip("."),
                source_url=metadata_urls[i] if metadata_urls else None,
            )
            for i, file in enumerate(files)
        ]

        created_docs = await DocumentModel.bulk_create(docs_data)

        effective_model = collection.embeddings_model or embeddings_model
        result = await document_service.add_documents(
            collection_id=collection_id,
            user_id=requesting_user.id,
            files=files,
            request=AddDocumentRequest(
                embeddings_model=effective_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                metadata_urls=metadata_urls,
                metadata_names=metadata_names,
            ),
            metadata_urls=metadata_urls,
            metadata_names=metadata_names,
            document_ids=[doc.id for doc in created_docs],
        )

        if not result.success:
            await _rollback_uploaded_docs(
                requesting_user.id, collection_id, created_docs
            )
            created_docs = []
            raise HTTPException(status_code=500, detail=result.error)

        result_data = dict(result.data or {})
        result_data.pop("chunk_counts", None)
        ingested_ids = result_data.pop("ingested_document_ids", None)
        if ingested_ids is not None:
            ingested_set = set(ingested_ids)
            skipped_docs = [doc for doc in created_docs if doc.id not in ingested_set]
            if skipped_docs:
                await _rollback_uploaded_docs(
                    requesting_user.id, collection_id, skipped_docs
                )
                await release_private_document_slots(
                    requesting_user.id, len(skipped_docs)
                )
                created_docs = [
                    doc for doc in created_docs if doc.id in ingested_set
                ]

        return result_data
    except Exception as e:
        if created_docs:
            await _rollback_uploaded_docs(
                requesting_user.id, collection_id, created_docs
            )
            created_docs = []
        if slots_reserved:
            await release_private_document_slots(requesting_user.id, file_count)
        if isinstance(e, HTTPException):
            raise
        logger.error(f"Error processing documents: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error processing documents: {str(e)}"
        )


def _conversation_document_error(status_code: int, code: str, message: str):
    return HTTPException(
        status_code=status_code, detail={"code": code, "message": message}
    )


async def _validate_conversation_files(files: List[UploadFile]) -> None:
    """Reject a wrong type (415) or an oversize file (413) before any slot is taken."""
    for file in files:
        extension = os.path.splitext(file.filename or "")[1].lower()
        if extension not in CONVERSATION_DOCUMENT_EXTENSIONS:
            raise _conversation_document_error(
                415,
                "conversation_document_type",
                f"{file.filename} is not a PDF, TXT or MD file.",
            )
        size = file.size
        if size is None:
            size = len(await file.read())
            await file.seek(0)
        if size > MAX_CONVERSATION_DOCUMENT_BYTES:
            raise _conversation_document_error(
                413,
                "conversation_document_size",
                f"{file.filename} is larger than "
                f"{MAX_CONVERSATION_DOCUMENT_BYTES // (1024 * 1024)} MB.",
            )


@router.post(
    "/conversations/{conversation_id}/documents",
    dependencies=[Depends(enforce_request_rate("upload"))],
)
async def upload_conversation_documents(
    conversation_id: str = Path(..., description="Conversation ID"),
    files: List[UploadFile] = File(...),
    requesting_user: User = Depends(get_current_user),
) -> dict:
    """
    Attach documents to one conversation.

    The files are parsed, chunked and embedded into a collection that belongs to
    the conversation: chat in it searches them, and GET /collections does not
    list it. Each file also takes one of the user's private document slots.

    Args:
        conversation_id (str): Conversation identifier.
        files (list[UploadFile]): PDF, TXT or MD files, at most 10 MB each.
        requesting_user (User): Authenticated user injected by dependency.

    Returns:
        The collection id and, per document, its id, name and chunk count.

    Raises:
        HTTPException: 404 if the feature is off or the conversation is not
        found or belongs to another user; 415 for a file type other than pdf,
        txt or md; 413 for a file over 10 MB, or one that splits into more than
        600 chunks; 400 ``conversation_document_limit`` past 10 documents in
        the conversation, or the private document limit; 422 if no file yields
        any text; 504 ``conversation_document_timeout`` past the upload
        deadline; 500 ``conversation_document_failed`` for anything else.
    """
    if not config.FEATURE_CONVERSATION_DOCUMENTS:
        raise HTTPException(status_code=404, detail="Not Found")

    # Primary: a client may create the conversation and attach a file right away,
    # before a secondary has the new row (backend #281, #288).
    conversation = await Conversation.find_by_id_on_primary(conversation_id)
    # Another user's conversation answers like a missing one, so the route never
    # confirms that an id exists.
    if not conversation or conversation.user_id != requesting_user.id:
        raise HTTPException(status_code=404, detail="Conversation not found")

    await _validate_conversation_files(files)

    collection = await get_or_create_conversation_collection(
        requesting_user.id, conversation_id
    )
    # Two uploads racing past this count can exceed the cap by the smaller
    # batch; the private document counter below stays exact either way.
    attached = await DocumentModel.get_primary_collection().count_documents(
        {"collection_id": collection.id}
    )
    if attached + len(files) > MAX_CONVERSATION_DOCUMENTS:
        raise _conversation_document_error(
            400,
            "conversation_document_limit",
            f"A conversation holds at most {MAX_CONVERSATION_DOCUMENTS} documents "
            f"({attached} already attached).",
        )

    file_count = len(files)
    slots_held = 0
    created_docs: List[DocumentModel] = []
    try:
        await reserve_private_document_slots(requesting_user.id, file_count)
        slots_held = file_count

        with anyio.fail_after(CONVERSATION_DOCUMENT_DEADLINE_SECONDS):
            created_docs = await DocumentModel.bulk_create(
                [
                    DocumentModel(
                        user_id=requesting_user.id,
                        collection_id=collection.id,
                        name=file.filename,
                        filename=file.filename,
                        file_type=os.path.splitext(file.filename)[1].lstrip("."),
                        file_size=file.size,
                    )
                    for file in files
                ]
            )
            names = [file.filename for file in files]
            result = await document_service.add_documents(
                collection_id=collection.id,
                user_id=requesting_user.id,
                files=files,
                request=AddDocumentRequest(
                    embeddings_model=collection.embeddings_model,
                    chunk_size=DEFAULT_CHUNK_SIZE,
                    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                    metadata_names=names,
                ),
                # add_documents skips a file without a source name.
                metadata_names=names,
                document_ids=[doc.id for doc in created_docs],
                max_chunks_per_file=MAX_CONVERSATION_DOCUMENT_CHUNKS,
            )

        if not result.success:
            if result.message == "too_many_chunks":
                raise _conversation_document_error(
                    413,
                    "conversation_document_chunks",
                    f"{(result.data or {}).get('filename')} is too long: a document "
                    f"holds at most {MAX_CONVERSATION_DOCUMENT_CHUNKS} chunks.",
                )
            if result.error:
                raise RuntimeError(result.error)
            raise _conversation_document_error(
                422,
                "conversation_document_unreadable",
                "No text could be read from the uploaded files.",
            )

        result_data = result.data or {}
        ingested = set(result_data.get("ingested_document_ids") or [])
        chunk_counts = result_data.get("chunk_counts") or {}
        skipped_docs = [doc for doc in created_docs if doc.id not in ingested]
        if skipped_docs:
            await _rollback_uploaded_docs(
                requesting_user.id, collection.id, skipped_docs
            )
            await release_private_document_slots(
                requesting_user.id, len(skipped_docs)
            )
            slots_held -= len(skipped_docs)
            created_docs = [doc for doc in created_docs if doc.id in ingested]

        for doc in created_docs:
            await DocumentModel.get_collection().update_one(
                {"_id": ObjectId(doc.id)},
                {"$set": {"chunk_count": chunk_counts.get(doc.id, 0)}},
            )
        stored = await DocumentModel.get_primary_collection().find(
            {"_id": {"$in": [ObjectId(doc.id) for doc in created_docs]}}
        ).to_list(length=None)

        return {
            "collection_id": collection.id,
            "documents": [
                {
                    "document_id": str(row["_id"]),
                    "name": row.get("name"),
                    "chunk_count": row.get("chunk_count"),
                }
                for row in stored
            ],
            "skipped": [doc.name for doc in skipped_docs],
        }
    except Exception as e:
        if created_docs:
            await _rollback_uploaded_docs(
                requesting_user.id, collection.id, created_docs
            )
        if slots_held:
            await release_private_document_slots(requesting_user.id, slots_held)
        if isinstance(e, TimeoutError):
            raise _conversation_document_error(
                504,
                "conversation_document_timeout",
                "The documents took too long to process. Try fewer or smaller files.",
            )
        if isinstance(e, HTTPException):
            raise
        logger.error(
            "Error attaching documents to conversation %s",
            conversation_id,
            exc_info=True,
        )
        raise _conversation_document_error(
            500,
            "conversation_document_failed",
            "The documents could not be processed. Try again.",
        )


@router.delete("/collections/{collection_id}/documents/{document_id}")
async def delete_document(
    collection_id: str = Path(..., description="Collection ID"),
    document_id: str = Path(..., description="Document ID"),
    requesting_user: User = Depends(get_current_user),
) -> dict:
    """
    Delete a document from a collection.

    Removes the document record and attempts to delete associated vectors.

    Args:
        collection_id (str): Collection identifier.
        document_id (str): Document identifier.
        requesting_user (User): Authenticated user injected by dependency.

    Returns:
        Confirmation message.

    Raises:
        HTTPException: 404 if not found; 400 if document not in collection; 403 if deletion is forbidden.
    """
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
        await anyio.to_thread.run_sync(
            partial(
                vector_store.delete_private_docs,
                user_id=requesting_user.id,
                collection_id=collection_id,
                metadata={"metadata.document_id": document_id},
            )
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to delete vectors for document {document_id}: {e}")

    await document.delete()
    await release_private_document_slots(requesting_user.id, 1)
    return {"message": "Document and embeddings deleted successfully"}
