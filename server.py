import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

from src.endpoints.create_collection import router as create_collection_router
from src.endpoints.delete_collection import router as delete_collection_router
from src.endpoints.health_check import router as health_check_router
from src.endpoints.add_document import router as add_document_list_router
from src.endpoints.delete_document import router as delete_document_router
from src.endpoints.retrieve_documents import router as retrieve_documents_router
from src.endpoints.generate_answer import router as generate_answer_router
from src.endpoints.completion_llm import router as completion_llm_router

origins = [
    "http://localhost",
    "http://localhost:6333",
]


def create_app(debug=False, **kwargs):
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
        return "Qdrant vector search API"

    app.include_router(create_collection_router)
    app.include_router(health_check_router)
    app.include_router(delete_collection_router)
    app.include_router(add_document_list_router)
    app.include_router(delete_document_router)
    app.include_router(retrieve_documents_router)
    app.include_router(generate_answer_router)
    app.include_router(completion_llm_router)

    return app


app = create_app()
