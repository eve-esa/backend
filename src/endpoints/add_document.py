import PyPDF2
from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional
from src.config import QDRANT_API_KEY, QDRANT_URL
from src.services.vector_store_manager import VectorStoreManager
from src.services.doc_parser_manager import DocumentParserManager
from datetime import datetime
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from src.services.utils import save_upload_file_to_temp
from langchain.text_splitter import RecursiveCharacterTextSplitter

router = APIRouter()


async def parse_pdf(file_path: str) -> List:
    """Parse a PDF file into Langchain Document pages."""
    loader = PyPDFLoader(file_path)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    return pages


async def process_uploaded_files(
    files: List[UploadFile],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    metadata_urls: Optional[List[str]] = None,
    metadata_names: Optional[List[str]] = None,
) -> List[Document]:
    documents = []

    for i, file in enumerate(files):
        temp_path = await save_upload_file_to_temp(file)
        pages = await parse_pdf(temp_path)

        full_text = "\n".join(page.page_content for page in pages)

        source = (
            metadata_urls[i]
            if metadata_urls and i < len(metadata_urls) and metadata_urls[i]
            else file.filename
        )
        source_name = (
            metadata_names[i]
            if metadata_names and i < len(metadata_names) and metadata_names[i]
            else file.filename
        )

        metadata = {"source": source, "source_name": source_name}

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        split_docs = text_splitter.create_documents([full_text], metadatas=[metadata])

        documents.extend(split_docs)

    return documents


@router.post("/add_document_list")
async def add_document_list(
    collection_name: str = Form("esa-nasa-workshop"),
    embeddings_model: str = Form("text-embedding-3-small"),
    files: List[UploadFile] = File(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    metadata_urls: Optional[List[str]] = Form(None),
    metadata_names: Optional[List[str]] = Form(None),
):
    vector_store = VectorStoreManager(
        qdrant_api_key=QDRANT_API_KEY,
        qdrant_url=QDRANT_URL,
        embeddings_model=embeddings_model,
    )

    document_list = await process_uploaded_files(
        files,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        metadata_urls=metadata_urls,
        metadata_names=metadata_names,
    )

    vector_store.add_document_list(
        collection_name=collection_name, document_list=document_list
    )

    return {
        "message": f"Parsed {len(document_list)} documents",
        "collection": collection_name,
        "documents": document_list,
    }
