from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel
from src.database.models.message import Message


class ConversationDetail(BaseModel):
    id: str
    user_id: str
    name: str
    timestamp: datetime
    messages: List[Message] = []
    # The assistant message still being generated, null when no turn runs.
    active_message_id: Optional[str] = None


class ConversationCreate(BaseModel):
    name: str


class ConversationNameUpdate(BaseModel):
    name: str
