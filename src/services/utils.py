import tempfile
import runpod
from fastapi import UploadFile
from typing import Union, Tuple, List

from langchain_core.embeddings import FakeEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import MistralAIEmbeddings

from src.config import MISTRAL_API_KEY, OPENAI_API_KEY, RUNPOD_API_KEY

runpod.api_key = RUNPOD_API_KEY


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
        embeddings_size = embeddings._client[1].word_embedding_dimension
    if not return_embeddings_size:
        return embeddings
    return embeddings, embeddings_size

def runpod_api_request(endpoint_id: str, model: str, user_input: str, timeout: int = 60) -> List[float]:
    """Return only embeddings as a vector using the RunPod library. Prints errors if they occur."""
    # Prepare the input payload matching your original structure
    payload = {
        "input": {
            "model": model,
            "input": user_input
        }
    }
    
    try:
        # Create an endpoint instance
        endpoint = runpod.Endpoint(endpoint_id)
        
        # Submit the job
        run_request = endpoint.run(payload)
        print(f"Job submitted: {run_request.job_id}")
        
        # Poll for the output with a timeout
        result = run_request.output(timeout=timeout)        
        embedding = result["data"][0]["embedding"]
        
        if not isinstance(embedding, list) or not all(isinstance(x, (int, float)) for x in embedding):
            raise ValueError("Embedding is not a valid list of numbers")
        
        print(f"Job completed: {run_request.job_id}")
        return embedding
    
    except Exception as e:
        print(f"Error: {e}")
        return []