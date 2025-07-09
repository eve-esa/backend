import logging
from src.database.models.user import User
from fastapi import APIRouter, Depends
from src.middlewares.auth import get_current_user
from pydantic import BaseModel

router = APIRouter(prefix="/users")
logger = logging.getLogger(__name__)


class UpdateUserRequest(BaseModel):
    first_name: str
    last_name: str


@router.get("/me", response_model=User)
async def me(user: User = Depends(get_current_user)):
    return user


@router.patch("/", response_model=User)
async def update_user(
    request: UpdateUserRequest, user: User = Depends(get_current_user)
):
    user.first_name = request.first_name
    user.last_name = request.last_name
    await user.save()
    return user
