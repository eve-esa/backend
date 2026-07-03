import logging
from datetime import datetime, timezone

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException
from pymongo.errors import DuplicateKeyError

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
    ensure_custom_model_quota,
    get_owned_custom_model,
    to_custom_model_public,
    validate_custom_model_base_url,
)
from src.services.platform_models import list_platform_models

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/models", response_model=ModelListResponse)
async def list_models(
    requesting_user: User = Depends(get_current_user),
) -> ModelListResponse:
    """List platform models and the authenticated user's custom models."""
    custom_models = await UserCustomModel.find_all(
        filter_dict={"user_id": requesting_user.id, "deleted_at": None},
        sort=[("created_at", -1)],
    )
    return ModelListResponse(
        platform=list_platform_models(),
        custom=[to_custom_model_public(model) for model in custom_models],
    )


@router.post("/users/custom-models", response_model=CustomModelPublic, status_code=201)
async def create_custom_model(
    request: CreateCustomModelRequest,
    requesting_user: User = Depends(get_current_user),
) -> CustomModelPublic:
    """Register a user-owned custom model. The API key is stored in AWS Secrets Manager."""
    await ensure_custom_model_quota(requesting_user.id)
    base_url = validate_custom_model_base_url(request.base_url)

    model = await UserCustomModel.create(
        user_id=requesting_user.id,
        display_name=request.display_name,
        model_name=request.model_name,
        base_url=base_url,
        secret_arn="",
    )

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

    try:
        await model.save()
    except DuplicateKeyError:
        await delete_custom_model_secret(secret_arn)
        await UserCustomModel.delete_many({"_id": ObjectId(model.id)})
        raise HTTPException(
            status_code=409,
            detail="A custom model with this display name already exists",
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
    if request.model_name is not None:
        model.model_name = request.model_name
    if request.base_url is not None:
        model.base_url = validate_custom_model_base_url(request.base_url)
    if request.api_key is not None:
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
    except DuplicateKeyError:
        raise HTTPException(
            status_code=409,
            detail="A custom model with this display name already exists",
        )
    return to_custom_model_public(model)


@router.delete("/users/custom-models/{model_id}", status_code=204)
async def delete_custom_model(
    model_id: str,
    requesting_user: User = Depends(get_current_user),
) -> None:
    """Soft-delete a custom model and remove its Secrets Manager entry."""
    model = await get_owned_custom_model(model_id, requesting_user.id, action="delete")
    model.deleted_at = datetime.now(timezone.utc)
    model.updated_at = datetime.now(timezone.utc)
    await model.save()

    try:
        await delete_custom_model_secret(model.secret_arn)
    except Exception:
        logger.exception("Failed to delete custom model secret model_id=%s", model.id)
        raise HTTPException(
            status_code=500,
            detail="Custom model was archived but credential cleanup failed",
        )
