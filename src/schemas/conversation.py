from datetime import datetime
from typing import List
from pydantic import BaseModel
from src.database.models.message import Message


class ConversationDetail(BaseModel):
    id: str
    user_id: str
    name: str
    timestamp: datetime
    messages: List[Message] = []


class ConversationCreate(BaseModel):
    name: str


class ConversationNameUpdate(BaseModel):
    name: str
