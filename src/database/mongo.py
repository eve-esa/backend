import os
from motor.motor_asyncio import AsyncIOMotorClient
from motor.motor_asyncio import AsyncIOMotorDatabase
from motor.motor_asyncio import AsyncIOMotorCollection
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class AsyncMongoDBManager:
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None

    async def connect(
        self, connection_string: Optional[str] = None
    ) -> AsyncIOMotorDatabase:
        """Connect to MongoDB and return the database instance."""
        if connection_string is None:
            mongo_host = os.getenv("MONGO_HOST", "localhost")
            mongo_port = os.getenv("MONGO_PORT", "27017")
            mongo_username = os.getenv("MONGO_USERNAME", "root")
            mongo_password = os.getenv("MONGO_PASSWORD", "root")
            mongo_database = os.getenv("MONGO_DATABASE", "eve_backend")

            connection_string = f"mongodb://{mongo_username}:{mongo_password}@{mongo_host}:{mongo_port}/{mongo_database}?authSource=admin"

        try:
            self.client = AsyncIOMotorClient(connection_string)
            # Test the connection
            await self.client.admin.command("ping")

            # Get the database
            self.database = self.client.get_database()
            return self.database

        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    def get_collection(self, collection_name: str) -> AsyncIOMotorCollection:
        """Get a collection from the database."""
        if self.database is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self.database[collection_name]

    async def close(self):
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")


async_mongo_manager = AsyncMongoDBManager()


async def get_database() -> AsyncIOMotorDatabase:
    """Get the database instance."""
    if async_mongo_manager.database is None:
        await async_mongo_manager.connect()
    return async_mongo_manager.database


def get_collection(collection_name: str) -> AsyncIOMotorCollection:
    """Get a collection from the database."""
    return async_mongo_manager.get_collection(collection_name)
