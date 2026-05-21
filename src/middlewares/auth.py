from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from src.config import JWT_SECRET_KEY, JWT_ALGORITHM, JWT_AUDIENCE_ACCESS
from src.database.models.user import User

security = HTTPBearer()


def verify_access_token(token: str) -> dict:
    """Decode and verify a user access JWT, returning its claims.

    Raises ``jose.JWTError`` on signature/audience/format failures.
    """
    return jwt.decode(
        token,
        JWT_SECRET_KEY,
        algorithms=[JWT_ALGORITHM],
        audience=JWT_AUDIENCE_ACCESS,
    )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    try:
        payload = verify_access_token(credentials.credentials)
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token payload")
        user = await User.find_by_id(user_id)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
