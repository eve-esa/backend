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
