import math
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, TypeVar, Generic, Type, ClassVar
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorCollection
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.errors import DuplicateKeyError
import logging
from datetime import datetime, timezone

from src.database.mongo import get_collection, get_database

T = TypeVar("T", bound="MongoModel")


class PaginationMetadata(BaseModel):
    total_count: int
    current_page: int
    total_pages: int
    has_next: bool


class PaginatedResponse(BaseModel, Generic[T]):
    data: List[T]
    meta: PaginationMetadata


def get_pagination_metadata(
    total_count: int, current_page: int, limit: int
) -> PaginationMetadata:
    if limit == 0:
        limit = 10

    return PaginationMetadata(
        total_count=total_count,
        current_page=current_page,
        total_pages=math.ceil(total_count / limit),
        has_next=current_page < math.ceil(total_count / limit),
    )


class MongoModel(BaseModel):
    """Generic base model for MongoDB documents with collection and instance operations."""

    id: str = Field(None, description="MongoDB document ID")

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Time of creation",
    )

    class Config:
        allow_population_by_field_name = True
        json_encoders = {ObjectId: lambda v: str(v)}

    # Collection name - should be overridden by subclasses
    collection_name: ClassVar[str] = "documents"

    logger: ClassVar[logging.Logger] = logging.getLogger(__name__)
    logger = logging.getLogger(__name__)

    @classmethod
    def get_collection(cls) -> AsyncIOMotorCollection:
        """Get the MongoDB collection for this model."""
        return get_collection(cls.collection_name)

    @classmethod
    async def get_database(cls) -> AsyncIOMotorDatabase:
        """Get the MongoDB database instance."""
        return await get_database()

    # Static methods for collection operations
    @classmethod
    async def find_all(
        cls: Type[T],
        limit: Optional[int] = None,
        skip: Optional[int] = None,
        sort: Optional[List[tuple]] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[T]:
        """Find all documents in the collection."""
        collection = cls.get_collection()
        query = filter_dict or {}

        cursor = collection.find(query)

        if sort:
            cursor = cursor.sort(sort)
        if skip:
            cursor = cursor.skip(skip)
        if limit:
            cursor = cursor.limit(limit)

        documents = await cursor.to_list(length=limit if limit is not None else None)
        return [cls.from_dict(doc) for doc in documents]

    @classmethod
    async def find_all_with_pagination(
        cls: Type[T],
        limit: int = 10,
        page: int = 0,
        sort: Optional[List[tuple]] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> PaginatedResponse[T]:
        """Find all documents in the collection with pagination."""
        if limit == 0:
            cls.logger.warning("Limit is 0, setting to 10 to avoid division by zero")
            limit = 10

        skip = (page - 1) * limit
        total_count = await cls.count_documents(filter_dict)
        total_pages = math.ceil(total_count / limit)
        has_next = skip + limit < total_count
        current_page = (skip // limit) + 1
        documents = await cls.find_all(
            limit=limit, skip=skip, sort=sort, filter_dict=filter_dict
        )

        return PaginatedResponse[T](
            data=documents,
            meta=get_pagination_metadata(total_count, current_page, limit),
        )

    @classmethod
    async def create(cls: Type[T], **kwargs) -> T:
        """Create a new document in the collection and return the id"""
        doc = cls(**kwargs)
        await doc.save()
        return doc

    @classmethod
    async def bulk_create(cls: Type[T], documents: List[T]) -> List[T]:
        """Insert multiple documents and return them with assigned IDs."""
        collection = cls.get_collection()

        docs_dicts = [doc.to_dict() for doc in documents]
        result = await collection.insert_many(docs_dicts)
        for doc, inserted_id in zip(documents, result.inserted_ids):
            doc.id = str(inserted_id)

        return documents

    @classmethod
    async def find_one(cls: Type[T], filter_dict: Dict[str, Any]) -> Optional[T]:
        """Find a single document by filter criteria."""
        collection = cls.get_collection()
        doc = await collection.find_one(filter_dict)
        return cls.from_dict(doc) if doc else None

    @classmethod
    async def find_by_id(cls: Type[T], document_id: str) -> Optional[T]:
        """Find a document by its ID."""
        try:
            return await cls.find_one({"_id": ObjectId(document_id)})
        except Exception as e:
            cls.logger.error(f"Error finding document by ID {document_id}: {e}")
            return None

    @classmethod
    async def count_documents(cls, filter_dict: Optional[Dict[str, Any]] = None) -> int:
        """Count documents in the collection."""
        collection = cls.get_collection()
        query = filter_dict or {}
        return await collection.count_documents(query)

    @classmethod
    async def delete_many(cls, filter_dict: Dict[str, Any]) -> int:
        """Delete multiple documents matching the filter."""
        collection = cls.get_collection()
        result = await collection.delete_many(filter_dict)
        return result.deleted_count

    # Instance methods for single record operations
    async def save(self: T) -> T:
        """Save the current instance to the database."""
        collection = self.__class__.get_collection()

        # Convert to dict for MongoDB
        doc_dict = self.to_dict()

        try:
            if self.id:
                # Update existing document
                result = await collection.replace_one(
                    {"_id": ObjectId(self.id)}, doc_dict
                )
                if result.matched_count == 0:
                    raise ValueError(f"Document with ID {self.id} not found")
            else:
                # Insert new document
                result = await collection.insert_one(doc_dict)
                self.id = str(result.inserted_id)

            return self

        except DuplicateKeyError as e:
            self.logger.error(f"Duplicate key error while saving: {e}")
            raise ValueError("Document with this unique field already exists")
        except Exception as e:
            self.logger.error(f"Error saving document: {e}")
            raise

    async def delete(self) -> bool:
        """Delete the current instance from the database."""
        if not self.id:
            raise ValueError("Cannot delete document without ID")

        collection = self.get_collection()
        result = await collection.delete_one({"_id": ObjectId(self.id)})
        return result.deleted_count > 0

    async def refresh(self: T) -> T:
        """Refresh the instance with data from the database."""
        if not self.id:
            raise ValueError("Cannot refresh document without ID")

        fresh_doc = await self.find_by_id(self.id)
        if not fresh_doc:
            raise ValueError(f"Document with ID {self.id} not found")

        # Update current instance with fresh data
        for field, value in fresh_doc.dict().items():
            setattr(self, field, value)

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for MongoDB storage."""
        doc_dict = self.dict(exclude={"id"})

        for key, value in doc_dict.items():
            if isinstance(value, Enum):
                doc_dict[key] = value.value

        return doc_dict

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create model instance from dictionary (e.g., from MongoDB)."""
        if data and "_id" in data:
            data["id"] = str(data["_id"])
            del data["_id"]
        return cls(**data)
