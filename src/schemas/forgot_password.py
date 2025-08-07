from pydantic import BaseModel, EmailStr


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ForgotPasswordConfirmation(BaseModel):
    new_password: str
    confirm_password: str
    code: str
