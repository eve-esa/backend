from enum import Enum
from typing import Optional
from pydantic import BaseModel


class FeedbackEnum(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class MessageUpdate(BaseModel):
    was_copied: Optional[bool] = None
    feedback: Optional[FeedbackEnum] = None
    feedback_reason: Optional[str] = None
