from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health_check():
    """Liveness probe endpoint.

    Returns:
        dict: Static status payload indicating service health.
    """
    return {"status": "healthy"}
