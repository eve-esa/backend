from fastapi import APIRouter

from src.config import APP_ENVIRONMENT, APP_GIT_SHA, APP_VERSION

router = APIRouter()


@router.get("/health")
def health_check():
    """Liveness probe endpoint.

    "status" keeps its exact value and shape: the ALB target group health check and the
    post-deploy verification both read it. The build identity is additive, and every value
    falls back to "unknown" rather than raising when the variable is not set.

    Returns:
        dict: Service health plus the identity of the running build.
    """
    return {
        "status": "healthy",
        "version": APP_VERSION,
        "commit": APP_GIT_SHA,
        "environment": APP_ENVIRONMENT,
    }
