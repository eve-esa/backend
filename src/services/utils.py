from langchain_core.embeddings import FakeEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Union, Tuple

import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


def get_embeddings_model(model: str, return_embeddings_size=False) -> Union[
    FakeEmbeddings,
    OpenAIEmbeddings,
    HuggingFaceEmbeddings,
    Tuple[Union[FakeEmbeddings, OpenAIEmbeddings, HuggingFaceEmbeddings], int],
]:
    openai_embedding_model_list = {
        "text-embedding-ada-002": 1536,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }
    if model == "fake":
        embeddings = FakeEmbeddings(size=4096)
        embeddings_size = 4096
    if model in openai_embedding_model_list.keys():
        embeddings = OpenAIEmbeddings(model=model, api_key=openai_api_key)
        embeddings_size = openai_embedding_model_list[model]
    else:
        embeddings = HuggingFaceEmbeddings(model_name=model)
        embeddings_size = embeddings.client[1].word_embedding_dimension
    if not return_embeddings_size:
        return embeddings
    return embeddings, embeddings_size
