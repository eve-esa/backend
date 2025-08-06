# src/endpoints/__init__.py
from .health_check import router as health_check_router
from .auth import router as auth_router
from .conversation import router as conversation_router
from .message import router as message_router
from .user import router as user_router
from .forgot_password import router as forgot_password_router
from .collection import router as collection_router
from .document import router as document_router

__all__ = [
    "health_check_router",
    "auth_router",
    "conversation_router",
    "message_router",
    "user_router",
    "forgot_password_router",
    "collection_router",
    "document_router",
]
