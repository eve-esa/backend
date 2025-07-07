from src.database.models.conversation import Conversation
from fastapi import APIRouter, HTTPException, Depends
from src.database.models.user import User
from src.middlewares.auth import get_current_user
from pydantic import BaseModel
from src.database.mongo_model import PaginatedResponse

router = APIRouter()


class Pagination(BaseModel):
    page: int = 1
    limit: int = 10


@router.get("/conversations", response_model=PaginatedResponse[Conversation])
async def list_conversations(
    request_user: User = Depends(get_current_user), pagination: Pagination = Depends()
):
    try:
        result = await Conversation.find_all_with_pagination(
            filter_dict={"user_id": request_user.id},
            limit=pagination.limit,
            sort=[("timestamp", -1)],
            skip=(pagination.page - 1) * pagination.limit,
        )

        if not result.data:
            raise HTTPException(status_code=404, detail="No conversations found")

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
