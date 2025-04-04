import PyPDF2
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import List
from src.config import QDRANT_API_KEY, QDRANT_URL
from src.services.vector_store_manager import VectorStoreManager
from src.services.doc_parser_manager import DocumentParserManager
from datetime import datetime
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from src.services.utils import save_upload_file_to_temp
from typing import Dict, List

router = APIRouter()


async def parse_pdf(file_path: str) -> List:
    """Parse a PDF file into Langchain Document pages."""
    loader = PyPDFLoader(file_path)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    return pages


async def process_uploaded_files(files: List[UploadFile]) -> List[Document]:
    documents = []

    for file in files:
        temp_path = await save_upload_file_to_temp(file)
        pages = await parse_pdf(temp_path)

        # Se vuoi un solo documento per file (anziché per pagina):
        full_text = "\n".join(page.page_content for page in pages)

        doc = Document(page_content=full_text, metadata={"name": file.filename})

        documents.append(doc)

    return documents


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

    document_list = await process_uploaded_files(files)
    vector_store.add_document_list(
        collection_name=collection_name, document_list=document_list
    )

    return {
        "message": f"Parsed {len(document_list)} documents",
        "collection": collection_name,
        "documents": document_list,
    }


#     # process uploaded files
#     for file in files:
#         try:
#             if file.filename.endswith(".pdf"):
#                 print("EIII")
#                 pdf_text = await process_pdf(file)
#                 print("PDF: ", pdf_text)
#                 document_list.append({"name": file.filename, "description": pdf_text})

#             elif file.filename.endswith(".txt"):
#                 parsed_list = await document_parser.get_document_vectors_from_txt(file)
#                 document_list.extend(parsed_list)
#             else:
#                 ignored_files.append(file.filename)

#         except Exception as e:
#             error_message = f"Error processing file {file.filename}: {str(e)}"
#             errors.append(error_message)

#     # Check if documents were successfully processed
#     if not document_list:
#         raise HTTPException(
#             status_code=400,
#             detail="No valid files were processed. Only PDF and TXT files are supported.",
#         )

#     try:
#         # vector_store.add_document(collection_name=collection_name, documents=document_list)
#         vector_store.add_document_list(
#             collection_name=collection_name, document_list=document_list
#         )
#         message = f"Documents uploaded successfully to '{collection_name}'."
#         if ignored_files:
#             message += f" The following files were ignored: {', '.join(ignored_files)}."
#         return {"message": message}

#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to upload documents to collection '{collection_name}': {str(e)}",
#         )


# async def process_pdf(file: UploadFile) -> str:
#     """Extracts text from a PDF file."""
#     try:
#         # Read PDF content
#         pdf_reader = PyPDF2.PdfReader(await file.read())
#         pdf_text = ""

#         # Extract text from each page
#         for page_num in range(len(pdf_reader.pages)):
#             page = pdf_reader.pages[page_num]
#             pdf_text += page.extract_text()

#         return pdf_text.strip()

#     except Exception as e:
#         raise HTTPException(
#             status_code=500, detail=f"Error extracting text from PDF: {str(e)}"
#         )
