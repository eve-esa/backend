from fastapi import APIRouter, HTTPException, Request, Response
from typing import Dict, Any
from pydantic import BaseModel, Field
from src.services.vector_store_manager import VectorStoreManager
from src.config import QDRANT_URL, QDRANT_API_KEY
from collections import defaultdict
import uuid
import os
import json
from datetime import datetime

router = APIRouter()

user_conversations = defaultdict(list)

# Directory to store user logs
LOG_DIR = "user_logs"
os.makedirs(LOG_DIR, exist_ok=True)


class GenerationRequest(BaseModel):
    query: str = "What is ESA?"
    collection_name: str = "esa-nasa-workshop"
    llm: str = "eve-instruct-v0.1"
    embeddings_model: str = "text-embedding-3-small"
    k: int = 3
    score_threshold: float = Field(0.7, ge=0.0, le=1.0)
    get_unique_docs: bool = True
    max_new_tokens: int = Field(1500, ge=100, le=8192)


@router.post("/generate_answer", response_model=Dict[str, Any])
def create_collection(
    request_data: GenerationRequest, request: Request, response: Response
) -> Dict[str, Any]:
    try:
        # Get or create user ID from cookie
        user_id = request.cookies.get("user_id")
        if not user_id:
            user_id = str(uuid.uuid4())
            response.set_cookie(
                key="user_id",
                value=user_id,
                httponly=True,
                samesite="Lax",
                path="/",
                max_age=60 * 60 * 24 * 30,  # 30 days
            )

        print(f"[SESSION] Using user_id: {user_id}")  # Debug/log

        user_messages = user_conversations[user_id]
        user_messages.append({"role": "user", "content": request_data.query})

        vector_store = VectorStoreManager(
            QDRANT_URL, QDRANT_API_KEY, embeddings_model=request_data.embeddings_model
        )

        should_use_rag = vector_store.should_use_rag(query=request_data.query)
        if not should_use_rag:
            answer = vector_store.generate_answer(
                query=request_data.query,
                context="",
                llm=request_data.llm,
                max_new_tokens=request_data.max_new_tokens,
                history_messages=user_messages,
            )
            user_messages.append({"role": "assistant", "content": answer})
            _log_user_conversation(user_id, user_messages)
            return {"answer": answer, "documents": [], "use_rag": False}

        results = vector_store.retrieve_documents_from_query(
            query=request_data.query,
            collection_name=request_data.collection_name,
            embeddings_model=request_data.embeddings_model,
            score_threshold=request_data.score_threshold,
            get_unique_docs=request_data.get_unique_docs,
            k=request_data.k,
        )

        retrieved_documents = [
            result.payload.get("page_content", "") for result in results
        ]
        context = "\n".join(retrieved_documents)

        answer = vector_store.generate_answer(
            query=request_data.query,
            context=context,
            llm=request_data.llm,
            max_new_tokens=request_data.max_new_tokens,
            history_messages=user_messages,
        )

        user_messages.append({"role": "assistant", "content": answer})
        _log_user_conversation(user_id, user_messages)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"answer": answer, "documents": results, "use_rag": True}


# Save only the last turn in user logs
def _log_user_conversation(user_id: str, messages: list):
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(LOG_DIR, f"{user_id}.json")

    if len(messages) < 2:
        return

    last_turn = messages[-2:]

    log_entry = {"timestamp": timestamp, "turn": last_turn}

    try:
        if os.path.exists(log_path):
            with open(log_path, "r+", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
                data.append(log_entry)
                f.seek(0)
                json.dump(data, f, indent=2)
        else:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump([log_entry], f, indent=2)
    except Exception as log_err:
        print(f"[LOG ERROR] Failed to write log for {user_id}: {log_err}")
