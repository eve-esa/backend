import asyncio
import logging
from datetime import datetime, timezone

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException

from src.database.models.user import User
from src.database.models.user_custom_model import UserCustomModel
from src.middlewares.auth import get_current_user
from src.schemas.custom_model import (
    CreateCustomModelRequest,
    CustomModelPublic,
    ModelListResponse,
    UpdateCustomModelRequest,
)
from src.services.custom_model_secrets import (
    create_custom_model_secret,
    delete_custom_model_secret,
    update_custom_model_secret,
)
from src.services.custom_model_service import (
    apply_catalog_model_fields,
    catalog_create_fields,
    ensure_custom_model_has_credentials,
    ensure_custom_model_quota,
    get_owned_custom_model,
    list_custom_models_public,
    to_custom_model_public,
)
from src.services.provider_catalog import (
    list_platform_models,
    list_provider_catalog,
    resolve_catalog_entry,
)

router = APIRouter()
logger = logging.getLogger(__name__)

_DUPLICATE_DISPLAY_NAME_DETAIL = "A custom model with this display name already exists"


def _raise_if_duplicate_display_name(exc: ValueError) -> None:
    if "unique field" in str(exc):
        raise HTTPException(status_code=409, detail=_DUPLICATE_DISPLAY_NAME_DETAIL)
    raise exc


@router.get("/models", response_model=ModelListResponse)
async def list_models(
    requesting_user: User = Depends(get_current_user),
) -> ModelListResponse:
    """List platform models, provider catalog, and the user's custom models."""
    platform, providers, custom = await asyncio.gather(
        list_platform_models(),
        list_provider_catalog(),
        list_custom_models_public(requesting_user.id),
    )
    return ModelListResponse(platform=platform, providers=providers, custom=custom)


@router.post("/users/custom-models", response_model=CustomModelPublic, status_code=201)
async def create_custom_model(
    request: CreateCustomModelRequest,
    requesting_user: User = Depends(get_current_user),
) -> CustomModelPublic:
    """Register a user-owned custom model. The API key is envelope-encrypted on the row."""
    await ensure_custom_model_quota(requesting_user.id)
    entry = await resolve_catalog_entry(request.provider_id, request.catalog_model_id)

    try:
        model = await UserCustomModel.create(
            user_id=requesting_user.id,
            display_name=request.display_name,
            **catalog_create_fields(entry),
        )
    except ValueError as exc:
        _raise_if_duplicate_display_name(exc)

    try:
        model.encrypted_key = await create_custom_model_secret(
            user_id=requesting_user.id,
            provider_id=entry.provider.id,
            model_id=model.id,
            api_key=request.api_key,
        )
        model.updated_at = datetime.now(timezone.utc)
        await model.save()
    except Exception:
        logger.exception("Failed to store custom model credentials for model_id=%s", model.id)
        await UserCustomModel.delete_many({"_id": ObjectId(model.id)})
        raise HTTPException(
            status_code=500,
            detail="Failed to store custom model credentials",
        )

    return await to_custom_model_public(model)


@router.patch("/users/custom-models/{model_id}", response_model=CustomModelPublic)
async def update_custom_model(
    model_id: str,
    request: UpdateCustomModelRequest,
    requesting_user: User = Depends(get_current_user),
) -> CustomModelPublic:
    """Update custom model metadata and optionally rotate the API key."""
    model = await get_owned_custom_model(model_id, requesting_user.id, action="update")

    if request.display_name is not None:
        model.display_name = request.display_name
    if request.catalog_model_id is not None:
        entry = await resolve_catalog_entry(model.provider_id, request.catalog_model_id)
        apply_catalog_model_fields(model, entry)
    if request.api_key is not None:
        ensure_custom_model_has_credentials(model)
        try:
            model.encrypted_key = await update_custom_model_secret(
                user_id=requesting_user.id,
                provider_id=model.provider_id,
                model_id=model.id,
                api_key=request.api_key,
            )
        except Exception:
            logger.exception("Failed to rotate custom model secret model_id=%s", model.id)
            raise HTTPException(
                status_code=500,
                detail="Failed to update custom model credentials",
            )

    model.updated_at = datetime.now(timezone.utc)
    try:
        await model.save()
    except ValueError as exc:
        _raise_if_duplicate_display_name(exc)
    return await to_custom_model_public(model)


@router.delete("/users/custom-models/{model_id}", status_code=204)
async def delete_custom_model(
    model_id: str,
    requesting_user: User = Depends(get_current_user),
) -> None:
    """Soft-delete a custom model, best-effort clearing any legacy Secrets Manager entry."""
    model = await get_owned_custom_model(model_id, requesting_user.id, action="delete")

    try:
        await delete_custom_model_secret(model)
    except Exception:
        logger.exception("Failed to delete custom model secret model_id=%s", model.id)
        raise HTTPException(
            status_code=500,
            detail="Failed to delete custom model credentials",
        )

    # Drop the key material from the soft-deleted row: the ciphertext (or a
    # legacy secret pointer) must not outlive the delete. The row is kept only
    # for its id/audit trail, never for its credentials.
    model.encrypted_key = None
    model.secret_arn = None
    model.deleted_at = datetime.now(timezone.utc)
    model.updated_at = datetime.now(timezone.utc)
    await model.save()
