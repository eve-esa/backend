from fastapi import APIRouter, Depends, HTTPException

from src.middlewares.admin import require_admin_api_key
from src.schemas.user import AdminCreateUserRequest, AdminCreateUserResponse
from src.services.auth import create_user_admin

router = APIRouter()


@router.post("/admin/users", response_model=AdminCreateUserResponse, status_code=201)
async def admin_create_user(
    request: AdminCreateUserRequest,
    _: None = Depends(require_admin_api_key),
) -> AdminCreateUserResponse:
    try:
        user, password = await create_user_admin(
            email=request.email,
            password=request.password,
            first_name=request.first_name,
            last_name=request.last_name,
            rate_limit_group=request.rate_limit_group,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return AdminCreateUserResponse(
        id=user.id,
        email=user.email,
        password=password,
        is_active=user.is_active,
        rate_limit_group=user.rate_limit_group.value,
    )
