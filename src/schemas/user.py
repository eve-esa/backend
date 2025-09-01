from pydantic import BaseModel


class UpdateUserRequest(BaseModel):
    first_name: str
    last_name: str


class UserCreate(BaseModel):
    email: str
    password: str
    first_name: str | None = None
    last_name: str | None = None


class UserResponse(BaseModel):
    id: str
    email: str
    first_name: str | None = None
    last_name: str | None = None
