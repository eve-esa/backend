import PyPDF2
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import List
from src.config import QDRANT_API_KEY, QDRANT_URL
from src.services.vector_store_manager import VectorStoreManager
from src.services.doc_parser_manager import DocumentParserManager
from datetime import datetime
from langchain_core.documents import Document

router = APIRouter()


@router.post("/add_document_list")
async def add_document_list(
    collection_name: str = Form(...),
    embeddings_model: str = Form(...),
    files: List[UploadFile] = File(...),
):

    # Initialize vector store manager with collection name and embeddings model
    vector_store = VectorStoreManager(
        qdrant_api_key=QDRANT_API_KEY,
        qdrant_url=QDRANT_URL,
        embeddings_model=embeddings_model,
    )
    document_parser = DocumentParserManager()

    document_list, ignored_files, errors = [], [], []

    # process uploaded files
    for file in files:
        try:
            if file.filename.endswith(".pdf"):
                pass
                # pdf_text = await process_pdf(file)
                # document_list.append({"name": file.filename, "description": pdf_text})

            elif file.filename.endswith(".txt"):
                parsed_list = await document_parser.get_document_vectors_from_txt(file)
                document_list.extend(parsed_list)
            else:
                ignored_files.append(file.filename)

        except Exception as e:
            error_message = f"Error processing file {file.filename}: {str(e)}"
            errors.append(error_message)

    # Check if documents were successfully processed
    if not document_list:
        raise HTTPException(
            status_code=400,
            detail="No valid files were processed. Only PDF and TXT files are supported.",
        )

    try:
        # vector_store.add_document(collection_name=collection_name, documents=document_list)
        vector_store.add_document_list(
            collection_name=collection_name, document_list=document_list
        )
        message = f"Documents uploaded successfully to '{collection_name}'."
        if ignored_files:
            message += f" The following files were ignored: {', '.join(ignored_files)}."
        return {"message": message}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload documents to collection '{collection_name}': {str(e)}",
        )


async def process_pdf(file: UploadFile) -> str:
    """Extracts text from a PDF file."""
    try:
        # Read PDF content
        pdf_reader = PyPDF2.PdfReader(await file.read())
        pdf_text = ""

        # Extract text from each page
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            pdf_text += page.extract_text()

        return pdf_text.strip()

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error extracting text from PDF: {str(e)}"
        )
