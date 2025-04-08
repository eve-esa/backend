from fastapi import APIRouter, UploadFile, File, Form
from typing import List, Optional
import os
from datetime import datetime

from src.config import QDRANT_API_KEY, QDRANT_URL
from src.services.vector_store_manager import VectorStoreManager
from src.services.utils import save_upload_file_to_temp

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

router = APIRouter()

async def parse_pdf(file_path: str) -> List[Document]:
    """Parse a PDF file into Langchain Document pages."""
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        return pages
    except Exception as e:
        print(f"Error parsing PDF: {str(e)}")
        return []

async def parse_text_file(file_path: str) -> List[Document]:
    """Parse a text file into a Langchain Document."""
    try:
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        return documents
    except Exception as e:
        print(f"Error parsing text file: {str(e)}")
        return []

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
    """
    Add documents with their metadata.
    """
    # Log what we received for debugging
    print(f"Received {len(files)} files")
    for i, file in enumerate(files):
        print(f"File {i}: {file.filename}")
    
    # Get file names in order
    file_names = [file.filename for file in files]
    print(f"File names in order: {file_names}")
    
    # Create a mapping from filename to its index
    filename_to_index = {name: i for i, name in enumerate(file_names)}
    
    # Build metadata arrays
    processed_urls = [""] * len(files)
    processed_names = [""] * len(files)
    
    # Process metadata_urls - handle comma-separated values
    if metadata_urls:
        # Check if we received a single concatenated string
        if isinstance(metadata_urls, str) and ',' in metadata_urls:
            # Split and assign to files in order
            url_parts = metadata_urls.split(',')
            for i, url in enumerate(url_parts):
                if i < len(files):
                    processed_urls[i] = url.strip()
        elif isinstance(metadata_urls, str):
            # Single value - assign to first file
            processed_urls[0] = metadata_urls.strip()
        elif isinstance(metadata_urls, list):
            # Process each URL in the list
            for i, url in enumerate(metadata_urls):
                if i < len(files):
                    # Check if this individual URL contains commas
                    if isinstance(url, str) and ',' in url:
                        url_parts = url.split(',')
                        # Take the first part for this file
                        processed_urls[i] = url_parts[0].strip()
                    else:
                        processed_urls[i] = url.strip() if url else ""
    
    # Process metadata_names - handle comma-separated values
    if metadata_names:
        # Check if we received a single concatenated string
        if isinstance(metadata_names, str) and ',' in metadata_names:
            # Split and assign to files in order
            name_parts = metadata_names.split(',')
            for i, name in enumerate(name_parts):
                if i < len(files):
                    processed_names[i] = name.strip()
        elif isinstance(metadata_names, str):
            # Single value - assign to first file
            processed_names[0] = metadata_names.strip()
        elif isinstance(metadata_names, list):
            # Process each name in the list
            for i, name in enumerate(metadata_names):
                if i < len(files):
                    # Check if this individual name contains commas
                    if isinstance(name, str) and ',' in name:
                        name_parts = name.split(',')
                        # Take the first part for this file
                        processed_names[i] = name_parts[0].strip()
                    else:
                        processed_names[i] = name.strip() if name else ""
    
    print(f"Processed URLs: {processed_urls}")
    print(f"Processed Names: {processed_names}")
    
    # Initialize vector store
    vector_store = VectorStoreManager(
        qdrant_api_key=QDRANT_API_KEY,
        qdrant_url=QDRANT_URL,
        embeddings_model=embeddings_model,
    )
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    all_documents = []
    temp_files = []
    
    try:
        # Process each file with its corresponding metadata
        for i, (file, url, name) in enumerate(zip(files, processed_urls, processed_names)):
            print(f"Processing file {i}: {file.filename} with URL: {url}, Name: {name}")
            
            # Skip files with empty source_name
            if not name or not name.strip():
                print(f"Skipping {file.filename} - no valid source_name provided")
                continue
                
            # Save file to temp
            temp_path = await save_upload_file_to_temp(file)
            temp_files.append(temp_path)
            
            # Determine file type and parse
            ext = os.path.splitext(file.filename)[1].lower()
            file_documents = []
            
            if ext == ".pdf":
                file_documents = await parse_pdf(temp_path)
            elif ext in [".txt", ".md"]:
                file_documents = await parse_text_file(temp_path)
            else:
                print(f"Skipping unsupported file type: {ext}")
                continue
            
            # If parsing failed, skip
            if not file_documents:
                print(f"No documents parsed from {file.filename}")
                continue
            
            # Get metadata for this file
            source = processed_urls[i]
            source_name = processed_names[i]
            
            # Add metadata to each document
            for doc in file_documents:
                doc.metadata = {
                    "source": source,
                    "source_name": source_name,
                    "filename": file.filename,
                    "file_type": ext[1:] if ext.startswith('.') else ext,
                    "upload_time": datetime.now().isoformat()
                }
                print(f"Added metadata to document: source={source}, source_name={source_name}")
            
            # Only add documents if they have a valid source_name from the user input
            if source_name and source_name.strip():
                # Split documents
                split_docs = text_splitter.split_documents(file_documents)
                all_documents.extend(split_docs)
                print(f"Added {len(split_docs)} chunks from {file.filename} with source_name='{source_name}'")
            else:
                print(f"Skipping {len(file_documents)} documents from {file.filename} due to missing source_name")
            
            # Split documents
            split_docs = text_splitter.split_documents(file_documents)
            all_documents.extend(split_docs)
            
        # After processing all files
        if all_documents:
            # Final check: ensure all documents have valid source_name
            valid_documents = [doc for doc in all_documents if doc.metadata.get("source_name") and doc.metadata["source_name"].strip()]
            
            if len(valid_documents) != len(all_documents):
                print(f"Filtered out {len(all_documents) - len(valid_documents)} documents with missing source_name")
            
            if valid_documents:
                vector_store.add_document_list(
                    collection_name=collection_name, 
                    document_list=valid_documents
                )
                
                return {
                    "message": f"Successfully processed {len(valid_documents)} chunks from {len(files)} files",
                    "collection": collection_name,
                    "chunk_count": len(valid_documents)
                }
            else:
                return {
                    "message": "No valid documents were processed (all had missing source_names)",
                    "collection": collection_name
                }
        else:
            return {
                "message": "No documents were processed",
                "collection": collection_name
            }
            
    except Exception as e:
        return {
            "error": str(e),
            "message": "Error processing documents",
            "collection": collection_name
        }
    finally:
        # Clean up temp files
        for temp_path in temp_files:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception as e:
                print(f"Error removing temp file {temp_path}: {str(e)}")