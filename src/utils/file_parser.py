import os
import logging
from dataclasses import dataclass
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

logger = logging.getLogger(__name__)

@dataclass
class FileParser:
   supported_extensions = [".pdf", ".txt", ".md"]
   
   file_parser = {
        ".pdf": lambda path: PyPDFLoader(path).load(),
        ".txt": lambda path: TextLoader(path, encoding="utf-8").load(),
        ".md": lambda path: TextLoader(path, encoding="utf-8").load(),
    }
   
   def __call__(self, file_path:str, extension:str, *args, **kwds) -> List[Document]:
        return self.parse_file(file_path, extension, *args, **kwds)
   
   async def parse_file(self, file_path: str, extension: str) -> List[Document]:
        """Parse a file based on its extension into LangChain Documents."""
        if extension not in self.supported_extensions:
            logger.error(f"Unsupported file type: {extension}")
            return ValueError(f"Unsupported file type: {extension}")
        
        parser = self.file_parser.get(extension.lower())
        if not parser:
            logger.warning(f"Unsupported file type: {extension}")
            return []
        try:
            return parser(file_path)
        except Exception as e:
            logger.error(f"Error parsing {extension} file: {str(e)}")
            return []
    

    