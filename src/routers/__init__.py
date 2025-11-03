# src/endpoints/__init__.py
from .collection import router as collection_router
from .health_check import router as health_check_router
from .document import router as document_router
from .message import router as message_router
from .conversation import router as conversation_router
from .user import router as user_router
from .auth import router as auth_router
from .forgot_password import router as forgot_password_router
from .tool import router as tool_router

__all__ = [
    "collection_router",
    "health_check_router",
    "document_router",
    "message_router",
    "conversation_router",
    "user_router",
    "auth_router",
    "forgot_password_router",
    "tool_router",
]
