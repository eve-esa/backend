"""Business logic for user-owned custom models."""

from __future__ import annotations

import logging
from typing import Optional
from urllib.parse import urlparse

from fastapi import HTTPException

from src.config import CUSTOM_MODEL_MAX_PER_USER, IS_PROD
from src.database.models.user_custom_model import UserCustomModel
from src.schemas.custom_model import CustomModelPublic

logger = logging.getLogger(__name__)


def validate_custom_model_base_url(base_url: str) -> str:
    """Normalize and validate a custom model base URL."""
    normalized = base_url.strip().rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=422, detail="base_url must be a valid HTTP(S) URL")

    if IS_PROD and parsed.scheme != "https":
        raise HTTPException(
            status_code=422,
            detail="base_url must use HTTPS in production",
        )

    if not IS_PROD and parsed.scheme == "http":
        host = (parsed.hostname or "").lower()
        if host not in {"localhost", "127.0.0.1"}:
            raise HTTPException(
                status_code=422,
                detail="HTTP base_url is only allowed for localhost in non-production",
            )

    return normalized


def to_custom_model_public(model: UserCustomModel) -> CustomModelPublic:
    return CustomModelPublic(
        id=model.id,
        display_name=model.display_name,
        model_name=model.model_name,
        base_url=model.base_url,
        has_api_key=bool(model.secret_arn),
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


async def get_owned_custom_model(
    model_id: str,
    user_id: str,
    *,
    action: str = "access",
) -> UserCustomModel:
    model = await UserCustomModel.find_by_id(model_id)
    if not model or model.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Custom model not found")
    if model.user_id != user_id:
        raise HTTPException(
            status_code=403,
            detail=f"Not allowed to {action} this custom model",
        )
    return model


async def count_active_custom_models(user_id: str) -> int:
    return await UserCustomModel.count_documents(
        {"user_id": user_id, "deleted_at": None}
    )


async def ensure_custom_model_quota(user_id: str) -> None:
    count = await count_active_custom_models(user_id)
    if count >= CUSTOM_MODEL_MAX_PER_USER:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum of {CUSTOM_MODEL_MAX_PER_USER} custom models allowed",
        )
