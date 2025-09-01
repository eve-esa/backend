from pydantic import BaseModel, EmailStr


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str


class RefreshResponse(BaseModel):
    access_token: str


class RefreshRequest(BaseModel):
    refresh_token: str


class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    first_name: str | None = None
    last_name: str | None = None


class SignupResponse(BaseModel):
    id: str
    email: EmailStr
    first_name: str | None = None
    last_name: str | None = None


class ResendActivationRequest(BaseModel):
    email: str


class VerifyRequest(BaseModel):
    email: str
    activation_code: str
