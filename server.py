from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import logging
from src.config import configure_logging

from src.routers import (
    collections_router,
    health_check_router,
    documents_router,
    generate_answer_router,
    mcp_client_router,
)

origins = [
    "http://localhost",
    "http://localhost:6333",
]

configure_logging(level=logging.DEBUG)


def register_routers(app: FastAPI):
    # Collections
    app.include_router(collections_router, tags=["Collections"])

    # Documents
    app.include_router(documents_router, tags=["Documents"])

    # LLM
    app.include_router(generate_answer_router, tags=["LLM"])

    # Health
    app.include_router(health_check_router, tags=["Health"])

    # MCP client
    app.include_router(mcp_client_router, tags=["MCP"])


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
