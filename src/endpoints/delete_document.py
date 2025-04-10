"""Enfdpoint to delete documents from a vector store."""
from fastapi import APIRouter, HTTPException, status
from typing import List
from src.services.vector_store_manager import VectorStoreManager

router = APIRouter()

@router.delete("/delete_document_list", status_code=status.HTTP_200_OK)
async def delete_document_list(
    collection_name: str = "esa-nasa-workshop",
    embeddings_model: str = "nasa-impact/nasa-smd-ibm-v0.1",
    document_list: List[str] = [],  # List of documents source_name, can be passed as comma-separated values, e.g ["indus"]
):
    
    vector_store = VectorStoreManager(embeddings_model=embeddings_model)
    errors = []
    
    for source in document_list:
        try:
            vector_store.delete_docs_by_metadata_filter(
                collection_name=collection_name, metadata={"source_name": source}
            )
        except Exception as e:
            errors.append(f"Failed to delete document {source}: {str(e)}")

    # Handle cases where errors occurred during deletion
    if errors:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "Some documents could not be deleted.",
                "errors": errors,
            },
        )

    return {
        "message": "Documents deleted successfully",
        "deleted_documents": document_list,
    }
