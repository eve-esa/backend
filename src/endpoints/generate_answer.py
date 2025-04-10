from fastapi import APIRouter, HTTPException, Request, Response
from typing import Dict, Any
from pydantic import BaseModel, Field
from src.services.vector_store_manager import VectorStoreManager
from src.config import QDRANT_URL, QDRANT_API_KEY, OPENAI_API_KEY
from openai import OpenAI, Client

router = APIRouter()
openai_client = Client(api_key=OPENAI_API_KEY)


class GenerationRequest(BaseModel):
    query: str = "What is ESA?"
    collection_name: str = "esa-nasa-workshop"
    llm: str = "eve-instruct-v0.1"  # or openai
    embeddings_model: str = "text-embedding-3-small"
    k: int = 3
    score_threshold: float = Field(0.7, ge=0.0, le=1.0)
    get_unique_docs: bool = True
    max_new_tokens: int = Field(1500, ge=100, le=8192)


def use_rag(query: str) -> bool:
    prompt = f"""
    Decide whether to use RAG to answer the given query. Follow these rules:
    - Do NOT use RAG for generic, casual, or non-specific queries, such as "hi", "hello", "how are you", "what can you do", or "tell me a joke".
    - USE RAG for queries related to earth science, space science, climate, space agencies, or similar scientific topics.
    - USE RAG for specific technical or scientific questions, even if the topic is unclear (e.g., "What’s the thermal conductivity of basalt?" or "How does orbital decay work?").
    - If unsure whether RAG is needed, default to USING RAG.
    - Respond only with 'yes' or 'no'.

    Query: {query}
    """

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0,
    )

    if response.choices:
        answer = response.choices[0].message.content.strip()
        if answer.lower() == "yes":
            print("I am using rag...")
            return True
        elif answer.lower() == "no":
            print("I am not using rag...")
            return False
        else:
            raise ValueError("Unexpected response from OpenAI API")


def get_rag_context(
    vector_store: VectorStoreManager, request: GenerationRequest
) -> str:
    vector_store = VectorStoreManager(
        QDRANT_URL, QDRANT_API_KEY, embeddings_model=request.embeddings_model
    )
    results = vector_store.retrieve_documents_from_query(
        query=request.query,
        collection_name=request.collection_name,
        embeddings_model=request.embeddings_model,
        score_threshold=request.score_threshold,
        get_unique_docs=request.get_unique_docs,
        k=request.k,
    )
    if not results:
        print(f"No documents found for query : {request.query}")

    retrieved_documents = [result.payload.get("page_content", "") for result in results]
    context = "\n".join(retrieved_documents)
    return context, results


@router.post("/generate_answer", response_model=Dict[str, Any])
def create_collection(request: GenerationRequest) -> Dict[str, Any]:
    try:
        vector_store = VectorStoreManager(
            QDRANT_URL, QDRANT_API_KEY, embeddings_model=request.embeddings_model
        )
        is_rag: bool = use_rag(request.query)
        context, results = (
            get_rag_context(vector_store, request) if is_rag else ("", [])
        )
        answer = vector_store.generate_answer(
            query=request.query,
            context=context,
            llm=request.llm,
            max_new_tokens=request.max_new_tokens,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"answer": answer, "documents": results, "use_rag": is_rag}


