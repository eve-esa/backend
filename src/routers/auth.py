import logging
from src.schemas.auth import (
    LoginRequest,
    LoginResponse,
    RefreshRequest,
    RefreshResponse,
    ResendActivationRequest,
    SignupRequest,
    SignupResponse,
    VerifyRequest,
)
from src.config import JWT_ALGORITHM, JWT_SECRET_KEY, JWT_AUDIENCE_REFRESH
from src.database.models.user import User
from fastapi import APIRouter, HTTPException
from src.services.auth import (
    verify_user,
    create_access_token,
    create_refresh_token,
    create_user,
)
from src.services.auth import generate_activation_code
from jose import jwt, JWTError
from src.services.email import email_service
from src.config import FRONTEND_URL


router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    if not await verify_user(request.email, request.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    user = await User.find_one({"email": request.email})
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    if not user.is_active:
        raise HTTPException(
            status_code=403, detail="Account not activated. Please check your email."
        )

    return LoginResponse(
        access_token=create_access_token(sub=user.id),
        refresh_token=create_refresh_token(sub=user.id),
    )


@router.post("/refresh", response_model=RefreshResponse)
async def refresh(request: RefreshRequest):
    try:
        payload = jwt.decode(
            request.refresh_token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
            audience=JWT_AUDIENCE_REFRESH,
        )
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid refresh token")

        # Use find_by_id instead of find_one with _id
        user = await User.find_by_id(user_id)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    return RefreshResponse(
        access_token=create_access_token(sub=user.id),
    )


@router.post("/signup", response_model=SignupResponse)
async def signup(request: SignupRequest):
    try:
        user = await create_user(
            email=request.email,
            password=request.password,
            first_name=request.first_name,
            last_name=request.last_name,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    verification_url = (
        f"{FRONTEND_URL}/verify?email={user.email}&code={user.activation_code}"
    )
    email_service.send_email(
        to_email=user.email,
        subject="Activate your EVE account",
        template_name="activation.html",
        context={
            "verification_url": verification_url,
        },
    )
    return SignupResponse(
        id=user.id,
        email=user.email,
        first_name=user.first_name,
        last_name=user.last_name,
    )


@router.post("/resend-activation")
async def resend_activation(request: ResendActivationRequest):
    user = await User.find_one({"email": request.email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.is_active:
        return {"message": "Account already activated."}
    user.activation_code = generate_activation_code()
    await user.save()
    verification_url = (
        f"{FRONTEND_URL}/verify?email={user.email}&code={user.activation_code}"
    )
    email_service.send_email(
        to_email=user.email,
        subject="Activate your EVE account",
        template_name="activation.html",
        context={
            "verification_url": verification_url,
        },
    )
    return {"message": "Activation code resent."}


@router.post("/verify")
async def verify(request: VerifyRequest):
    user = await User.find_one({"email": request.email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.is_active:
        return {"message": "Account already activated."}
    if user.activation_code != request.activation_code:
        raise HTTPException(status_code=400, detail="Invalid activation code")
    user.is_active = True
    user.activation_code = None
    await user.save()
    return {"message": "Account activated successfully."}
