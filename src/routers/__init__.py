# src/endpoints/__init__.py
from .collections import router as collections_router
from .health_check import router as health_check_router
from .add_document import router as add_document_list_router
from .delete_document import router as delete_document_router
from .retrieve_documents import router as retrieve_documents_router
from .generate_answer import router as generate_answer_router
from .completion_llm import router as completion_llm_router

__all__ = [
    "collections_router",
    "health_check_router",
    "add_document_list_router",
    "delete_document_router",
    "retrieve_documents_router",
    "generate_answer_router",
    "completion_llm_router",
]
