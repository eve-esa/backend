from typing import Optional, ClassVar
from pydantic import Field, EmailStr
import hashlib
from src.database.mongo_model import MongoModel

class User(MongoModel):
    """Base user model for the application."""

    email: EmailStr = Field(..., description="User's email address")
    password_hash: str = Field(..., description="Hashed password")

    collection_name: ClassVar[str] = "users"

    # Override save to hash password
    async def save(self):
        if await User.find_one({"email": self.email}):
            raise ValueError(f"Email {self.email} already exists")

        self.password_hash = hashlib.sha256(self.password_hash.encode()).hexdigest()
        return await super().save()
