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
    ensure_custom_model_has_credentials,
    ensure_custom_model_quota,
    get_owned_custom_model,
    to_custom_model_public,
)
from src.services.platform_models import list_platform_models
from src.services.provider_catalog import list_provider_catalog, resolve_catalog_entry

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
    custom_models = await UserCustomModel.find_all(
        filter_dict={"user_id": requesting_user.id, "deleted_at": None},
        sort=[("created_at", -1)],
    )
    return ModelListResponse(
        platform=list_platform_models(),
        providers=list_provider_catalog(),
        custom=[to_custom_model_public(model) for model in custom_models],
    )


@router.post("/users/custom-models", response_model=CustomModelPublic, status_code=201)
async def create_custom_model(
    request: CreateCustomModelRequest,
    requesting_user: User = Depends(get_current_user),
) -> CustomModelPublic:
    """Register a user-owned custom model. The API key is stored in AWS Secrets Manager."""
    await ensure_custom_model_quota(requesting_user.id)
    entry = resolve_catalog_entry(request.provider_id, request.catalog_model_id)

    try:
        model = await UserCustomModel.create(
            user_id=requesting_user.id,
            display_name=request.display_name,
            provider_id=entry.provider.id,
            catalog_model_id=entry.model.id,
            model_name=entry.model.model_name,
            secret_arn="",
        )
    except ValueError as exc:
        _raise_if_duplicate_display_name(exc)

    secret_arn: str | None = None
    try:
        secret_arn = await create_custom_model_secret(
            user_id=requesting_user.id,
            model_id=model.id,
            api_key=request.api_key,
        )
        model.secret_arn = secret_arn
        model.updated_at = datetime.now(timezone.utc)
        await model.save()
    except Exception:
        logger.exception("Failed to create custom model secret for model_id=%s", model.id)
        await UserCustomModel.delete_many({"_id": ObjectId(model.id)})
        if secret_arn:
            try:
                await delete_custom_model_secret(secret_arn)
            except Exception:
                logger.exception(
                    "Failed to clean up secret after model create failure model_id=%s",
                    model.id,
                )
        raise HTTPException(
            status_code=500,
            detail="Failed to store custom model credentials",
        )

    return to_custom_model_public(model)


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
        entry = resolve_catalog_entry(model.provider_id, request.catalog_model_id)
        model.catalog_model_id = entry.model.id
        model.model_name = entry.model.model_name
    if request.api_key is not None:
        ensure_custom_model_has_credentials(model)
        try:
            await update_custom_model_secret(
                secret_arn=model.secret_arn,
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
    return to_custom_model_public(model)


@router.delete("/users/custom-models/{model_id}", status_code=204)
async def delete_custom_model(
    model_id: str,
    requesting_user: User = Depends(get_current_user),
) -> None:
    """Soft-delete a custom model and remove its Secrets Manager entry."""
    model = await get_owned_custom_model(model_id, requesting_user.id, action="delete")

    if model.secret_arn:
        try:
            await delete_custom_model_secret(model.secret_arn)
        except Exception:
            logger.exception("Failed to delete custom model secret model_id=%s", model.id)
            raise HTTPException(
                status_code=500,
                detail="Failed to delete custom model credentials",
            )

    model.deleted_at = datetime.now(timezone.utc)
    model.updated_at = datetime.now(timezone.utc)
    await model.save()
