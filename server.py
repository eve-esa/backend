from src.database.mongo import async_mongo_manager
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import logging
from src.config import CORS_ALLOWED_ORIGINS, configure_logging
from contextlib import asynccontextmanager

from src.routers import (
    health_check_router,
    auth_router,
    conversation_router,
    message_router,
    user_router,
    forgot_password_router,
    collection_router,
    document_router,
    tool_router,
    hallucination_router,
)

configure_logging(level=logging.DEBUG)


def register_routers(app: FastAPI):
    # Health
    app.include_router(health_check_router, tags=["Health"])

    # Auth
    app.include_router(auth_router, tags=["Auth"])

    # User
    app.include_router(user_router, tags=["User"])

    # Forgot Password
    app.include_router(forgot_password_router, tags=["Forgot Password"])

    # Conversations
    app.include_router(conversation_router, tags=["Conversations"])
    app.include_router(message_router, tags=["Messages"])

    # Collections
    app.include_router(collection_router, tags=["Collections"])

    # Documents
    app.include_router(document_router, tags=["Documents"])

    # Tools
    app.include_router(tool_router, tags=["Tools"])

    # Hallucination
    app.include_router(hallucination_router, tags=["Hallucination"])


def create_app(debug=False, **kwargs):
    """Create and configure the FastAPI app instance."""

    logging.info("Creating FastAPI app...")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await async_mongo_manager.connect()
        logging.info("Database connection established")
        try:
            yield
        finally:
            await async_mongo_manager.close()
            logging.info("Database connection closed")

    app = FastAPI(debug=debug, lifespan=lifespan, **kwargs)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ALLOWED_ORIGINS,
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
