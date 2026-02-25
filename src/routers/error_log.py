from fastapi import APIRouter, Depends, HTTPException

from src.database.models.error_log import ErrorLog
from src.database.models.user import User
from src.middlewares.auth import get_current_user
from src.schemas.error_log import FrontendErrorLogRequest

router = APIRouter()


@router.post("/log-error")
async def log_error(
    request: FrontendErrorLogRequest,
    requesting_user: User = Depends(get_current_user),
):
    """
    Log a frontend error to the error_log collection.

    Args:
        request (FrontendErrorLogRequest): Error log details from the frontend.
        requesting_user (User): Authenticated user injected by dependency.

    Returns:
        dict: Confirmation message with error log ID.

    Raises:
        HTTPException: 401 if user is not authenticated.
        HTTPException: 500 for server errors.
    """
    try:
        error_dict = {
            "type": request.error_type,
            "message": request.error_message,
        }
        
        if request.error_stack:
            error_dict["stack"] = request.error_stack
        
        if request.metadata:
            error_dict["metadata"] = request.metadata

        error_log = ErrorLog(
            user_id=requesting_user.id,
            conversation_id=None,
            message_id=None,
            logger_name="frontend",
            component=request.component or "FRONTEND",
            error=error_dict,
            error_type=request.error_type,
            pipeline_stage="CLIENT_ERROR",
            description=request.description or request.error_message,
        )

        if request.url:
            error_log.error["url"] = request.url
        if request.user_agent:
            error_log.error["user_agent"] = request.user_agent

        await error_log.save()
        return {"message": "Error logged successfully", "id": error_log.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

