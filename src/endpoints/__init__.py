# src/endpoints/__init__.py
from .create_collection import router as create_collection_router
from .delete_collection import router as delete_collection_router
from .health_check import router as health_check_router
from .add_document import router as add_document_list_router
from .delete_document import router as delete_document_router
from .retrieve_documents import router as retrieve_documents_router
from .generate_answer import router as generate_answer_router
from .completion_llm import router as completion_llm_router
from .list_collections import router as list_collections_llm_router
from .auth import router as auth_router
from .conversation import router as conversation_router
from .message import router as message_router

__all__ = [
    "create_collection_router",
    "delete_collection_router",
    "health_check_router",
    "add_document_list_router",
    "delete_document_router",
    "retrieve_documents_router",
    "generate_answer_router",
    "completion_llm_router",
    "list_collections_llm_router",
    "auth_router",
    "conversation_router",
    "message_router",
]
