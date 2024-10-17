import uvicorn

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

from src.endpoints.create_collection import router as create_collection_router
from src.endpoints.delete_collection import router as delete_collection_router
from src.endpoints.health_check import router as health_check_router
from src.endpoints.add_document import router as add_document_list

# from endpoints.add_document import add_doc_to_existing_collection
# from src.endpoints.delete_collection import router as delete_collection_router
# from src.endpoints.delete_doc_from_collection import (
#     router as delete_doc_from_collection_router,
# )
# from src.endpoints.get_collection_info import router as get_collection_info_router
# from src.endpoints.query_collection import router as query_collection_router
# from src.endpoints.test_add_text_to_existing_collection import (
#     router as add_text_to_existing_collection_router,
# )

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

    # app.include_router(add_doc_to_existing_collection.router)
    app.include_router(create_collection_router)
    app.include_router(health_check_router)
    app.include_router(delete_collection_router)
    app.include_router(add_document_list)
    # app.include_router(delete_collection_router)
    # app.include_router(delete_doc_from_collection_router)
    # app.include_router(get_collection_info_router)
    # app.include_router(query_collection_router)
    # app.include_router(add_text_to_existing_collection_router)
    return app


app = create_app()
