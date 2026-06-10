import secrets

from fastapi import Header, HTTPException

from src.config import ADMIN_API_KEY


async def require_admin_api_key(
    x_admin_api_key: str | None = Header(None, alias="X-Admin-Api-Key"),
) -> None:
    if not ADMIN_API_KEY:
        raise HTTPException(status_code=503, detail="Admin API not configured")
    if not x_admin_api_key or not secrets.compare_digest(x_admin_api_key, ADMIN_API_KEY):
        raise HTTPException(status_code=403, detail="Forbidden")
