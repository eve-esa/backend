# src/endpoints/__init__.py
from .collections import router as collections_router
from .health_check import router as health_check_router
from .documents import router as documents_router
from .generate_answer import router as generate_answer_router
from .completion_llm import router as completion_llm_router

__all__ = [
    "collections_router",
    "health_check_router",
    "documents_router",
    "generate_answer_router",
    "completion_llm_router",
]
