from typing import List

from src.database.mongo_model import MongoModel


async def cleanup_models(models: List[MongoModel]) -> None:
    """Delete all models provided, meant to be used ONLY in tests."""
    for doc in reversed(models):
        await doc.delete()
