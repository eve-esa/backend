from src.database.mongo import async_mongo_manager
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import logging
from src.config import configure_logging

from src.endpoints import (
    create_collection_router,
    delete_collection_router,
    health_check_router,
    add_document_list_router,
    delete_document_router,
    retrieve_documents_router,
    generate_answer_router,
    completion_llm_router,
    list_collections_llm_router,
    auth_router,
    conversation_router,
    message_router,
    user_router,
)

origins = [
    "http://localhost",
    "http://localhost:6333",
    "http://localhost:5173",
]

configure_logging(level=logging.DEBUG)


def register_routers(app: FastAPI):
    # Collections
    app.include_router(create_collection_router, tags=["Collections"])
    app.include_router(delete_collection_router, tags=["Collections"])
    app.include_router(list_collections_llm_router, tags=["Collections"])

    # Documents
    app.include_router(add_document_list_router, tags=["Documents"])
    app.include_router(delete_document_router, tags=["Documents"])
    app.include_router(retrieve_documents_router, tags=["Documents"])

    # LLM
    app.include_router(generate_answer_router, tags=["LLM"])
    app.include_router(completion_llm_router, tags=["LLM"])

    # Health
    app.include_router(health_check_router, tags=["Health"])

    # Auth
    app.include_router(auth_router, tags=["Auth"])

    # User
    app.include_router(user_router, tags=["User"])

    # Conversations
    app.include_router(conversation_router, tags=["Conversations"])
    app.include_router(message_router, tags=["Messages"])


def create_app(debug=False, **kwargs):
    """Create and configure the FastAPI app instance."""

    logging.info("Creating FastAPI app...")
    app = FastAPI(debug=debug, **kwargs)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get(path="/")
    def main_page():
        return "Welcome to Eve"

    register_routers(app)

    return app


app = create_app()


@app.on_event("startup")
async def startup_event():
    """Initialize database connection on startup."""
    await async_mongo_manager.connect()
    logging.info("Database connection established")


@app.on_event("shutdown")
async def shutdown_event():
    """Close database connection on shutdown."""
    await async_mongo_manager.close()
    logging.info("Database connection closed")
