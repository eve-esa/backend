"""Business logic for user-owned custom models."""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import HTTPException
from pydantic import ValidationError

from src.config import CUSTOM_MODEL_MAX_PER_USER
from src.database.models.message import Message
from src.database.models.user_custom_model import UserCustomModel
from src.schemas.custom_model import CustomModelPublic
from src.services.custom_model_secrets import get_custom_model_api_key
from src.services.provider_catalog import resolve_catalog_entry

logger = logging.getLogger(__name__)


def catalog_backed_custom_model_filter(user_id: str) -> dict[str, Any]:
    """Mongo filter for catalog-backed custom models (excludes legacy BYOM rows)."""
    return {
        "user_id": user_id,
        "deleted_at": None,
        "provider_id": {"$exists": True, "$nin": [None, ""]},
        "catalog_model_id": {"$exists": True, "$nin": [None, ""]},
    }


async def list_owned_catalog_custom_models(user_id: str) -> list[UserCustomModel]:
    """List active catalog-backed custom models, skipping invalid legacy documents."""
    collection = UserCustomModel.get_collection()
    cursor = collection.find(catalog_backed_custom_model_filter(user_id)).sort(
        "created_at", -1
    )
    models: list[UserCustomModel] = []
    async for doc in cursor:
        try:
            models.append(UserCustomModel.from_dict(doc))
        except ValidationError:
            logger.warning(
                "Skipping invalid custom model document id=%s user_id=%s",
                doc.get("_id"),
                user_id,
            )
    return models


def resolve_custom_model_endpoints(model: UserCustomModel) -> tuple[str, str]:
    """Resolve base URL and model name from the fixed provider catalog."""
    if not model.provider_id or not model.catalog_model_id:
        raise HTTPException(
            status_code=422,
            detail="Custom model configuration is invalid or no longer available",
        )
    entry = resolve_catalog_entry(model.provider_id, model.catalog_model_id)
    return entry.provider.base_url, entry.model.model_name


def ensure_custom_model_has_credentials(model: UserCustomModel) -> None:
    if not model.secret_arn:
        raise HTTPException(
            status_code=422,
            detail="Custom model has no stored credentials",
        )


def custom_model_prompt_metadata(model: UserCustomModel) -> dict[str, str]:
    """Agentic prompt metadata for a resolved custom model."""
    _, model_name = resolve_custom_model_endpoints(model)
    return {
        "custom_model_id": model.id,
        "custom_model_display_name": model.display_name,
        "custom_model_name": model_name,
    }


async def build_custom_model_llm(model: UserCustomModel) -> Any:
    """Build a ChatOpenAI client for a user-owned custom model."""
    ensure_custom_model_has_credentials(model)
    base_url, model_name = resolve_custom_model_endpoints(model)
    api_key = await get_custom_model_api_key(model.secret_arn)
    from src.services.generate_answer import get_shared_llm_manager

    return get_shared_llm_manager().build_custom_client(
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
    )


async def build_custom_model_llm_for_user(model_id: str, user_id: str) -> Any:
    """Load an owned custom model and build its LLM client."""
    model = await get_owned_custom_model(model_id, user_id, action="use")
    return await build_custom_model_llm(model)


def custom_model_id_from_messages(messages: list[Message]) -> Optional[str]:
    """Return the most recent custom model id referenced by conversation messages."""
    for message in reversed(messages):
        request_input = getattr(message, "request_input", None)
        custom_model_id = getattr(request_input, "custom_model_id", None)
        if custom_model_id:
            return custom_model_id
    return None


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
    if action == "use":
        ensure_custom_model_has_credentials(model)
    return model


async def count_active_custom_models(user_id: str) -> int:
    return await UserCustomModel.count_documents(
        catalog_backed_custom_model_filter(user_id)
    )


async def ensure_custom_model_quota(user_id: str) -> None:
    count = await count_active_custom_models(user_id)
    if count >= CUSTOM_MODEL_MAX_PER_USER:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum of {CUSTOM_MODEL_MAX_PER_USER} custom models allowed",
        )
