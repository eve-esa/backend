from src.database.models.conversation import Conversation
from fastapi import APIRouter, HTTPException, Depends
from src.database.models.user import User
from src.middlewares.auth import get_current_user
from pydantic import BaseModel
from enum import Enum

router = APIRouter()


class Feedback(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NONE = "none"

class UpdateConversationRequest(BaseModel):
    feedback: Feedback


@router.put("/conversations/{conversation_id}", response_model=Conversation)
async def update_conversation(
    conversation_id: str,
    request: UpdateConversationRequest,
    requesting_user: User = Depends(get_current_user),
):
    try:
        conversation = await Conversation.find_by_id(conversation_id)

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        if conversation.user_id != requesting_user.id:
            raise HTTPException(
                status_code=403,
                detail="You are not allowed to update this conversation",
            )

        conversation.feedback = request.feedback.value
        await conversation.save()

        return conversation

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
