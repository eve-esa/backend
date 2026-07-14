from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class PlatformModel(BaseModel):
    id: str
    llm_type: str
    display_name: str
    description: Optional[str] = None


class ProviderCatalogModelPublic(BaseModel):
    id: str
    display_name: str
    model_name: str


class ProviderCatalogPublic(BaseModel):
    id: str
    display_name: str
    models: List[ProviderCatalogModelPublic]


class CustomModelPublic(BaseModel):
    id: str
    display_name: str
    provider_id: str
    catalog_model_id: str
    provider_display_name: str
    model_display_name: str
    model_name: str
    has_api_key: bool = True
    created_at: datetime
    updated_at: datetime


class ModelListResponse(BaseModel):
    platform: List[PlatformModel]
    providers: List[ProviderCatalogPublic]
    custom: List[CustomModelPublic]


class CreateCustomModelRequest(BaseModel):
    display_name: str = Field(..., min_length=1, max_length=120)
    provider_id: str = Field(..., min_length=1, max_length=80)
    catalog_model_id: str = Field(..., min_length=1, max_length=120)
    api_key: str = Field(..., min_length=1, max_length=500)

    @field_validator("display_name", "provider_id", "catalog_model_id", "api_key")
    @classmethod
    def strip_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Field cannot be empty")
        return stripped


class UpdateCustomModelRequest(BaseModel):
    display_name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    catalog_model_id: Optional[str] = Field(default=None, min_length=1, max_length=120)
    api_key: Optional[str] = Field(default=None, min_length=1, max_length=500)

    @field_validator("display_name", "catalog_model_id", "api_key")
    @classmethod
    def strip_optional_non_empty(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("Field cannot be empty")
        return stripped
