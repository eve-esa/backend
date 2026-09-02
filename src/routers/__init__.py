# src/endpoints/__init__.py
from .collection import router as collection_router
from .health_check import router as health_check_router
from .document import router as document_router
from .artifact import router as artifact_router
from .message import router as message_router
from .conversation import router as conversation_router
from .user import router as user_router
from .mcp_server import router as mcp_server_router
from .custom_model import router as custom_model_router
from .error_log import router as error_log_router
from .migration import router as migration_router
from .openai_proxy import OpenAIProxyDispatcher

__all__ = [
    "collection_router",
    "health_check_router",
    "document_router",
    "artifact_router",
    "message_router",
    "conversation_router",
    "user_router",
    "mcp_server_router",
    "custom_model_router",
    "error_log_router",
    "migration_router",
    "OpenAIProxyDispatcher",
]
