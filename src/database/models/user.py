from typing import Optional, ClassVar
from pydantic import Field, EmailStr
from src.database.mongo_model import MongoModel


class User(MongoModel):
    """Base user model for the application."""

    email: EmailStr = Field(..., description="User's email address")
    password_hash: str = Field(..., description="Hashed password")
    first_name: Optional[str] = Field(default=None, description="User's first name")
    last_name: Optional[str] = Field(default=None, description="User's last name")

    collection_name: ClassVar[str] = "users"
