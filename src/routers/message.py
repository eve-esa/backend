from src.core.vector_store_manager import VectorStoreManager
from src.schemas.message import MessageUpdate
from src.services.generate_answer import GenerationRequest, generate_answer
from src.database.models.conversation import Conversation
from src.database.models.message import Message
from src.database.models.collection import Collection as CollectionModel
from fastapi import APIRouter, HTTPException, Depends
from src.database.models.user import User
from src.middlewares.auth import get_current_user
from typing import Dict, Any

router = APIRouter()


@router.post("/conversations/{conversation_id}/messages", response_model=Dict[str, Any])
async def create_message(
    request: GenerationRequest,
    conversation_id: str,
    requesting_user: User = Depends(get_current_user),
):
    try:
        conversation = await Conversation.find_by_id(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        if conversation.user_id != requesting_user.id:
            raise HTTPException(
                status_code=403,
                detail="You are not allowed to add a message to this conversation",
            )

        # All public collections are used by default
        public_collections = VectorStoreManager().list_public_collections()
        if len(public_collections) > 0:
            request.collection_ids = [c["name"] for c in public_collections]

        # All user collections are used by default
        user_collections = await CollectionModel.find_all(
            filter_dict={"user_id": requesting_user.id}
        )

        if len(user_collections) > 0:
            request.collection_ids = request.collection_ids + [
                c.id for c in user_collections
            ]

        answer, results, is_rag = await generate_answer(request)

        documents_data = []
        if results:
            for result in results:
                doc_data = {
                    "id": (str(result.id) if hasattr(result, "id") else None),
                    "version": (
                        int(result.version) if hasattr(result, "version") else None
                    ),
                    "score": float(result.score) if hasattr(result, "score") else None,
                    "payload": result.payload if hasattr(result, "payload") else {},
                }
                documents_data.append(doc_data)

        message = await Message.create(
            conversation_id=conversation_id,
            input=request.query,
            output=answer,
            documents=documents_data,
            use_rag=is_rag,
        )

        return {
            "id": message.id,
            "query": request.query,
            "answer": answer,
            "documents": documents_data,
            "use_rag": is_rag,
            "conversation_id": conversation_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.patch("/conversations/{conversation_id}/messages/{message_id}")
async def update_message(
    conversation_id: str,
    message_id: str,
    request: MessageUpdate,
    requesting_user: User = Depends(get_current_user),
):
    try:
        conversation = await Conversation.find_by_id(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        message = await Message.find_by_id(message_id)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")

        if message.conversation_id != conversation_id:
            raise HTTPException(
                status_code=404, detail="Message not found in this conversation"
            )

        if conversation.user_id != requesting_user.id:
            raise HTTPException(
                status_code=403,
                detail="You are not allowed to update feedback for this message",
            )

        if request.feedback is not None:
            message.feedback = request.feedback.value

        if request.was_copied is not None:
            message.was_copied = request.was_copied

        await message.save()

        return {"message": "Feedback updated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
