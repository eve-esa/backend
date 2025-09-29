from src.config import FORGOT_PASSWORD_CODE_EXPIRE_MINUTES
from src.database.mongo_model import MongoModel
from datetime import datetime, timedelta, timezone
from typing import ClassVar
from pydantic import EmailStr, Field


class ForgotPassword(MongoModel):
    email: EmailStr
    code: str
    expires_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
        + timedelta(minutes=FORGOT_PASSWORD_CODE_EXPIRE_MINUTES)
    )

    collection_name: ClassVar[str] = "forgot_passwords"
