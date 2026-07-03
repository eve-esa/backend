from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class PlatformModel(BaseModel):
    id: str
    llm_type: str
    display_name: str
    description: Optional[str] = None


class CustomModelPublic(BaseModel):
    id: str
    display_name: str
    model_name: str
    base_url: str
    has_api_key: bool = True
    created_at: datetime
    updated_at: datetime


class ModelListResponse(BaseModel):
    platform: List[PlatformModel]
    custom: List[CustomModelPublic]


class CreateCustomModelRequest(BaseModel):
    display_name: str = Field(..., min_length=1, max_length=120)
    model_name: str = Field(..., min_length=1, max_length=200)
    base_url: str = Field(..., min_length=1, max_length=500)
    api_key: str = Field(..., min_length=1, max_length=500)

    @field_validator("display_name", "model_name", "base_url", "api_key")
    @classmethod
    def strip_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Field cannot be empty")
        return stripped


class UpdateCustomModelRequest(BaseModel):
    display_name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    model_name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    base_url: Optional[str] = Field(default=None, min_length=1, max_length=500)
    api_key: Optional[str] = Field(default=None, min_length=1, max_length=500)

    @field_validator("display_name", "model_name", "base_url", "api_key")
    @classmethod
    def strip_optional_non_empty(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("Field cannot be empty")
        return stripped
