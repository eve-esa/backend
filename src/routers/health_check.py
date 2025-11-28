from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health_check():
    """
    Liveness probe endpoint.

    :return: Static status payload indicating service health.\n
    :rtype: dict\n
    """
    return {"status": "healthy"}
