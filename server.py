import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.config import APP_VERSION, CORS_ALLOWED_ORIGINS, configure_logging
from src.database.indexes import ensure_indexes
from src.database.mongo import async_mongo_manager
from src.services.provider_catalog import ensure_provider_catalog_seeded
from src.routers import (
    OpenAIProxyDispatcher,
    artifact_router,
    auth_router,
    collection_router,
    conversation_router,
    custom_model_router,
    document_router,
    error_log_router,
    forgot_password_router,
    health_check_router,
    mcp_server_router,
    message_router,
    user_router,
)
from src.routers.mcp_proxy import MCPProxyDispatcher, shutdown_mcp_proxy_lifespans

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

    # Artifacts
    app.include_router(artifact_router, tags=["Artifacts"])

    # MCP Servers
    app.include_router(mcp_server_router, tags=["MCP Servers"])

    # Custom models
    app.include_router(custom_model_router, tags=["Custom Models"])

    # Error Logs
    app.include_router(error_log_router, tags=["Error Logs"])


def create_app(debug=False, **kwargs):
    """Create and configure the FastAPI app instance."""

    logging.info("Creating FastAPI app...")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await async_mongo_manager.connect()
        await ensure_indexes()
        await ensure_provider_catalog_seeded()
        logging.info("Database connection established")
        try:
            yield
        finally:
            try:
                await shutdown_mcp_proxy_lifespans()
            except Exception:
                logging.exception("MCP proxy sub-app shutdown failed")
            await async_mongo_manager.close()
            logging.info("Database connection closed")

    # Surfaces the running version in /docs and in the OpenAPI document. setdefault so an
    # explicit version= from a caller still wins.
    kwargs.setdefault("version", APP_VERSION)

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

    # Public demo images referenced by the curated image catalog
    # (src/templates/image_catalog.yaml). Served by the backend so the
    # catalog can use environment-independent relative URLs: the frontend
    # resolves them against its API base, which is this app on every
    # environment. Licenses in the folder's ATTRIBUTION.md.
    app.mount(
        "/demo-images",
        StaticFiles(
            directory=os.path.join(
                os.path.dirname(__file__), "src", "templates", "demo_images"
            )
        ),
        name="demo-images",
    )

    register_routers(app)
    app = OpenAIProxyDispatcher(app)
    app = MCPProxyDispatcher(app)
    return app


app = create_app()
