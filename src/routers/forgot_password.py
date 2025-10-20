from datetime import datetime, timedelta, timezone
from typing import Dict
import logging
from src.schemas.forgot_password import (
    ForgotPasswordRequest,
    ForgotPasswordConfirmation,
)
from src.services.email import email_service
from src.config import FORGOT_PASSWORD_CODE_EXPIRE_MINUTES, FRONTEND_URL
from src.services.utils import hash_password
from src.database.models.forgot_password import ForgotPassword
from src.database.models.user import User
from fastapi import APIRouter, HTTPException
import random
import string

router = APIRouter(prefix="/forgot-password")
logger = logging.getLogger(__name__)


@router.post("/code", response_model=Dict[str, str])
async def send_forgot_password_code(request: ForgotPasswordRequest):
    user = await User.find_one({"email": request.email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    code = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))

    already_exists = await ForgotPassword.find_one({"email": request.email})
    if already_exists:
        already_exists.code = code
        already_exists.expires_at = datetime.now(timezone.utc) + timedelta(
            minutes=FORGOT_PASSWORD_CODE_EXPIRE_MINUTES
        )
        await already_exists.save()

    reset_url = f"{FRONTEND_URL}/reset-password?code={code}"
    email_service.send_email(
        to_email=request.email,
        subject="Forgot Password",
        template_name="forgot_password.html",
        context={
            "reset_url": reset_url,
            "expiry_minutes": FORGOT_PASSWORD_CODE_EXPIRE_MINUTES,
        },
    )

    await ForgotPassword.create(email=request.email, code=code)

    return {"message": "Code sent"}


@router.post("/confirm", response_model=Dict[str, str])
async def confirm_new_password(request: ForgotPasswordConfirmation):
    forgot_password = await ForgotPassword.find_one({"code": request.code})
    if not forgot_password:
        raise HTTPException(status_code=404, detail="Invalid code")

    current_time = datetime.now(timezone.utc)
    expires_at = forgot_password.expires_at

    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    if not forgot_password or expires_at < current_time:
        raise HTTPException(status_code=404, detail="Invalid code")

    if request.new_password != request.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")

    user = await User.find_one({"email": forgot_password.email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.password_hash = hash_password(request.new_password)
    await user.save()
    await forgot_password.delete()
    return {"message": "Password changed"}
