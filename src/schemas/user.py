from pydantic import BaseModel


class UpdateUserRequest(BaseModel):
    first_name: str
    last_name: str
