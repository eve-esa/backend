"""Business logic for user-owned custom models."""

from __future__ import annotations

from fastapi import HTTPException

from src.config import CUSTOM_MODEL_MAX_PER_USER
from src.database.models.user_custom_model import UserCustomModel
from src.schemas.custom_model import CustomModelPublic
from src.services.provider_catalog import resolve_catalog_entry


def resolve_custom_model_endpoints(model: UserCustomModel) -> tuple[str, str]:
    """Resolve base URL and model name from the fixed provider catalog."""
    if model.provider_id and model.catalog_model_id:
        entry = resolve_catalog_entry(model.provider_id, model.catalog_model_id)
        return entry.provider.base_url, entry.model.model_name

    if model.base_url and model.model_name:
        return model.base_url.rstrip("/"), model.model_name

    raise HTTPException(
        status_code=422,
        detail="Custom model configuration is invalid or no longer available",
    )


def to_custom_model_public(model: UserCustomModel) -> CustomModelPublic:
    provider_display_name = model.provider_id
    model_display_name = model.catalog_model_id
    model_name = model.model_name

    if model.provider_id and model.catalog_model_id:
        try:
            entry = resolve_catalog_entry(model.provider_id, model.catalog_model_id)
            provider_display_name = entry.provider.display_name
            model_display_name = entry.model.display_name
            model_name = entry.model.model_name
        except HTTPException:
            pass

    return CustomModelPublic(
        id=model.id,
        display_name=model.display_name,
        provider_id=model.provider_id,
        catalog_model_id=model.catalog_model_id,
        provider_display_name=provider_display_name,
        model_display_name=model_display_name,
        model_name=model_name,
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
