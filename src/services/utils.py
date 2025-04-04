from langchain_core.embeddings import FakeEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Union, Tuple
from langchain_mistralai import MistralAIEmbeddings
from src.config import MISTRAL_API_KEY, OPENAI_API_KEY
import tempfile
from fastapi import UploadFile


async def save_upload_file_to_temp(upload_file: UploadFile) -> str:
    """Save an uploaded file to a temporary file and return the path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        contents = await upload_file.read()
        temp_file.write(contents)
        temp_file.flush()
        return temp_file.name


def get_embeddings_model(model: str, return_embeddings_size=False) -> Union[
    FakeEmbeddings,
    OpenAIEmbeddings,
    HuggingFaceEmbeddings,
    MistralAIEmbeddings,
    Tuple[
        Union[
            FakeEmbeddings, OpenAIEmbeddings, HuggingFaceEmbeddings, MistralAIEmbeddings
        ],
        int,
    ],
]:
    openai_embedding_model_list = {
        "text-embedding-ada-002": 1536,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }
    if model == "fake":
        embeddings = FakeEmbeddings(size=4096)
        embeddings_size = 4096
    elif model == "mistral-embed":
        embeddings = MistralAIEmbeddings(model=model, api_key=MISTRAL_API_KEY)
        embeddings_size = 1024
    elif model in openai_embedding_model_list.keys():
        embeddings = OpenAIEmbeddings(model=model, api_key=OPENAI_API_KEY)
        embeddings_size = openai_embedding_model_list[model]
    else:
        embeddings = HuggingFaceEmbeddings(model_name=model)
        embeddings_size = embeddings.client[1].word_embedding_dimension
    if not return_embeddings_size:
        return embeddings
    return embeddings, embeddings_size
