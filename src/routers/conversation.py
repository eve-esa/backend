from src.schemas.conversation import (
    ConversationDetail,
    ConversationCreate,
    ConversationNameUpdate,
)
from src.database.models.conversation import Conversation
from src.database.models.message import Message
from fastapi import APIRouter, HTTPException, Depends
from src.database.models.user import User
from src.middlewares.auth import get_current_user
from src.database.mongo_model import PaginatedResponse
from src.schemas.common import Pagination

router = APIRouter()


@router.get("/conversations", response_model=PaginatedResponse[Conversation])
async def list_conversations(
    request_user: User = Depends(get_current_user), pagination: Pagination = Depends()
):
    try:
        result = await Conversation.find_all_with_pagination(
            filter_dict={"user_id": request_user.id},
            page=pagination.page,
            limit=pagination.limit,
            sort=[("timestamp", -1)],
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
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
                detail="You are not allowed to get this conversation",
            )

        # Fetch messages for this conversation
        messages = await Message.find_all(
            filter_dict={"conversation_id": conversation_id}, sort=[("timestamp", 1)]
        )

        return ConversationDetail(
            id=conversation.id,
            user_id=conversation.user_id,
            name=conversation.name,
            timestamp=conversation.timestamp,
            messages=messages,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.post("/conversations", response_model=Conversation)
async def create_conversation(
    request: ConversationCreate,
    requesting_user: User = Depends(get_current_user),
):
    try:
        return await Conversation.create(
            user_id=requesting_user.id,
            name=request.name,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.patch("/conversations/{conversation_id}")
async def update_conversation_name(
    conversation_id: str,
    request: ConversationNameUpdate,
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

        conversation.name = request.name
        await conversation.save()

        return conversation

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
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
                detail="You are not allowed to delete this conversation",
            )

        await Message.delete_many({"conversation_id": conversation_id})
        await conversation.delete()
        return {"message": "Conversation deleted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
