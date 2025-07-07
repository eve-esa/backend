import logging
from src.config import JWT_ALGORITHM, JWT_SECRET_KEY
from src.database.models.user import User
from pydantic import BaseModel, EmailStr
from fastapi import APIRouter, HTTPException
from src.services.auth import verify_user, create_access_token, create_refresh_token
from jose import jwt, JWTError


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str


class RefreshResponse(BaseModel):
    access_token: str


class RefreshRequest(BaseModel):
    refresh_token: str


router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    if not await verify_user(request.email, request.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    user = await User.find_one({"email": request.email})

    return LoginResponse(
        access_token=create_access_token(sub=user.id),
        refresh_token=create_refresh_token(sub=user.id),
    )

@router.post("/refresh", response_model=RefreshResponse)
async def refresh(request: RefreshRequest):
    try:
        payload = jwt.decode(
            request.refresh_token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM]
        )
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid refresh token")

        # Use find_by_id instead of find_one with _id
        user = await User.find_by_id(user_id)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    return RefreshResponse(
        access_token=create_access_token(sub=user.id),
    )
