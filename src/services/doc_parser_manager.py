from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
from fastapi import File
from datetime import datetime


class DocumentParserManager:
    def __init__(self) -> None:
        pass

    async def extract_content_from_txt(self, file: File) -> str:
        file_content = await file.read()
        text_content = file_content.decode("utf-8")
        return text_content

    def get_chunks(self, text):
        text_splitter = CharacterTextSplitter(
            separator="\n", chunk_size=1_000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks

    async def get_document_vectors_from_txt(self, file: File):
        text_content = await self.extract_content_from_txt(file)
        chunk_list = self.get_chunks(text_content)

        document_list = []
        for chunk in chunk_list:
            document_list.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": file.filename,
                        "date_added": datetime.now(),
                        "chars_len": len(chunk),
                    },
                )
            )
        return document_list
