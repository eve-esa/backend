"""
Vector Store Management for Retrieval-Augmented Generation

This module provides a complete interface for creating, managing, and querying
vector collections using Qdrant as the backend. It handles embedding generation,
document storage, and similarity search operations.
"""

import asyncio
import logging
import re
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from langchain_core.documents import Document
from openai import OpenAI
from qdrant_client import QdrantClient, models
from qdrant_client.conversions import common_types as types
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    IsEmptyCondition,
    MatchAny,
    MatchValue,
    PayloadField,
    VectorParams,
)

from src.config import (
    EMBEDDING_API_KEY,
    EMBEDDING_FALLBACK_API_KEY,
    EMBEDDING_FALLBACK_URL,
    EMBEDDING_URL,
    IS_PROD,
    QDRANT_API_KEY,
    QDRANT_URL,
    Config,
)
from src.constants import (
    DEFAULT_EMBEDDING_MODEL,
    EVE_PUBLIC_COLLECTION_NAME_PROD,
    EVE_PUBLIC_COLLECTION_NAME_STAGING,
    PRIVATE_COLLECTION_NAME,
    PUBLIC_ENV_PROD,
    PUBLIC_ENV_STAGING,
    WILEY_PUBLIC_COLLECTIONS,
)
from src.utils.error_logger import Component, PipelineStage, get_error_logger
from src.utils.helpers import all_known_public_collection_labels, iter_public_catalog

# Setup logging
logger = logging.getLogger(__name__)

# Initialize configuration
config = Config()

_OBJECT_ID_RE = re.compile(r"^[a-fA-F0-9]{24}$")


def looks_like_mongo_id(value: str) -> bool:
    """Return True if ``value`` looks like a Mongo ObjectId hex string."""
    return bool(value) and bool(_OBJECT_ID_RE.fullmatch(value))


def eve_public_collection_names() -> set[str]:
    """Catalog names and aliases that receive EVE client ``filters``."""
    return {
        EVE_PUBLIC_COLLECTION_NAME_PROD,
        EVE_PUBLIC_COLLECTION_NAME_STAGING,
        "EVE open-access",
    }


def is_eve_public_collection(name: str) -> bool:
    """True when ``name`` is the EVE open-access collection (any catalog label)."""
    return bool(name) and name in eve_public_collection_names()


def wiley_public_collection_names() -> set[str]:
    """Catalog names and aliases for Wiley public collections."""
    names: set[str] = set()
    for item in WILEY_PUBLIC_COLLECTIONS:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        alias = item.get("alias")
        if name:
            names.add(name)
        if alias:
            names.add(alias)
    return names


def is_wiley_public_collection(name: str) -> bool:
    """True when ``name`` is a Wiley catalog collection (never env-filtered)."""
    return bool(name) and name in wiley_public_collection_names()


def build_public_env_condition(is_prod: bool) -> FieldCondition:
    """Payload condition for public-collection ``env`` filtering."""
    if is_prod:
        return FieldCondition(
            key="env",
            match=MatchValue(value=PUBLIC_ENV_PROD),
        )
    return FieldCondition(
        key="env",
        match=MatchAny(any=[PUBLIC_ENV_PROD, PUBLIC_ENV_STAGING]),
    )


def build_public_env_filter(is_prod: bool) -> Filter:
    """Require an allowed ``env`` value, or an untagged point (missing ``env``).

    Untagged points stay searchable until backfill. ``env=staging`` points do
    not match prod. Nested as ``must`` so client ``should`` clauses stay intact.

    See https://qdrant.tech/documentation/concepts/filtering/#is-empty
    """
    return Filter(
        should=[
            build_public_env_condition(is_prod),
            IsEmptyCondition(is_empty=PayloadField(key="env")),
        ]
    )


def build_private_tenant_filter(
    user_id: str, collection_ids: List[str]
) -> Filter:
    """Mandatory tenant filter for the shared private Qdrant collection."""
    must: List[FieldCondition] = [
        FieldCondition(key="user_id", match=MatchValue(value=user_id)),
    ]
    if len(collection_ids) == 1:
        must.append(
            FieldCondition(
                key="collection_id",
                match=MatchValue(value=collection_ids[0]),
            )
        )
    elif collection_ids:
        must.append(
            FieldCondition(
                key="collection_id",
                match=MatchAny(any=list(collection_ids)),
            )
        )
    return Filter(must=must)


def merge_must_filters(
    base: Optional[Filter], extra_must: List[Any]
) -> Optional[Filter]:
    """Attach extra ``must`` conditions onto an existing Qdrant filter."""
    if not extra_must and base is None:
        return None
    must: List[Any] = list(extra_must)
    should = None
    must_not = None
    min_should = None
    if base is not None:
        if getattr(base, "must", None):
            must.extend(list(base.must))
        should = getattr(base, "should", None)
        must_not = getattr(base, "must_not", None)
        min_should = getattr(base, "min_should", None)
    return Filter(
        must=must or None,
        should=should,
        must_not=must_not,
        min_should=min_should,
    )


def _is_already_exists_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "already exists" in msg or "already exist" in msg


def _is_missing_collection_error(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    if status == 404:
        return True
    if isinstance(exc, UnexpectedResponse) and getattr(exc, "status_code", None) == 404:
        return True
    text = str(exc).lower()
    return "not found" in text and "collection" in text


def _is_timeout_error(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    if "timeout" in name:
        return True
    return "timeout" in str(exc).lower()


def split_public_and_private_collections(
    collection_names: List[str],
    private_collections_map: Optional[Dict[str, str]] = None,
) -> Tuple[List[str], List[str]]:
    """Split request collection IDs into public Qdrant names vs private Mongo IDs."""
    public_labels = all_known_public_collection_labels()
    private_ids = set((private_collections_map or {}).keys())
    public_names: List[str] = []
    private_mongo_ids: List[str] = []
    seen_public: set[str] = set()
    seen_private: set[str] = set()

    for name in collection_names or []:
        if not name or name == PRIVATE_COLLECTION_NAME:
            continue
        if name in private_ids or (
            name not in public_labels and looks_like_mongo_id(name)
        ):
            if name not in seen_private:
                seen_private.add(name)
                private_mongo_ids.append(name)
            continue
        if name not in seen_public:
            seen_public.add(name)
            public_names.append(name)

    return public_names, private_mongo_ids


class VectorStoreManager:
    """
    Manages vector storage operations for RAG (Retrieval Augmented Generation).

    This class provides methods to create and manage document collections,
    store documents with their embeddings, and retrieve relevant documents
    based on semantic similarity to queries.

    Note: When initializing this class, use the same embedding model that
    was used to embed the collection you want to work with.
    """

    def __init__(
        self,
        embeddings_model: str = DEFAULT_EMBEDDING_MODEL,
        qdrant_url: str = QDRANT_URL,
        qdrant_api_key: str = QDRANT_API_KEY,
    ) -> None:
        """
        Initialize the VectorStoreManager with the specified embeddings model.

        Args:
            embeddings_model: The name of the embeddings model to use.
                Defaults to NASA's specialized model.
        """
        # Initialize Qdrant client with timeout configuration
        self.client = QdrantClient(
            qdrant_url,
            api_key=qdrant_api_key,
            timeout=120.0,  # 2 minutes timeout for operations
        )
        self.embeddings_model = embeddings_model
        self.embeddings_size = 2560
        self._env_payload_cache: Dict[str, bool] = {}
        logger.debug(f"Initialized VectorStoreManager with model: {embeddings_model}")

    def ensure_private_collection(self) -> None:
        """Create the shared private collection and tenant indexes if missing.

        Never recreates the collection — that would wipe all tenants' data.
        """
        try:
            if not self.client.collection_exists(PRIVATE_COLLECTION_NAME):
                created = self.client.create_collection(
                    collection_name=PRIVATE_COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=self.embeddings_size,
                        distance=Distance.COSINE,
                    ),
                    hnsw_config=models.HnswConfigDiff(payload_m=16, m=0),
                )
                if created is False:
                    raise RuntimeError(
                        f"Failed to create collection: {PRIVATE_COLLECTION_NAME}"
                    )
                logger.info(
                    "Collection '%s' created successfully", PRIVATE_COLLECTION_NAME
                )
        except Exception as e:
            if not self.client.collection_exists(PRIVATE_COLLECTION_NAME):
                logger.error(
                    "Failed to create collection '%s': %s",
                    PRIVATE_COLLECTION_NAME,
                    e,
                )
                raise RuntimeError(
                    f"Failed to create collection: {PRIVATE_COLLECTION_NAME}"
                ) from e
            logger.info(
                "Collection '%s' already exists", PRIVATE_COLLECTION_NAME
            )
            try:
                self.client.update_collection(
                    collection_name=PRIVATE_COLLECTION_NAME,
                    hnsw_config=models.HnswConfigDiff(payload_m=16, m=0),
                )
            except Exception as update_err:
                logger.warning(
                    "Could not update HNSW on '%s': %s",
                    PRIVATE_COLLECTION_NAME,
                    update_err,
                )

        self._ensure_private_payload_indexes()

    def _ensure_private_payload_indexes(self) -> None:
        """Idempotently create tenant and collection_id payload indexes."""
        index_specs = (
            (
                "user_id",
                models.KeywordIndexParams(
                    type=models.KeywordIndexType.KEYWORD,
                    is_tenant=True,
                ),
            ),
            (
                "collection_id",
                models.PayloadSchemaType.KEYWORD,
            ),
        )
        for field_name, field_schema in index_specs:
            try:
                self.client.create_payload_index(
                    collection_name=PRIVATE_COLLECTION_NAME,
                    field_name=field_name,
                    field_schema=field_schema,
                )
            except Exception as e:
                if _is_already_exists_error(e):
                    logger.debug(
                        "Payload index '%s' on '%s' already exists",
                        field_name,
                        PRIVATE_COLLECTION_NAME,
                    )
                    continue
                logger.error(
                    "Failed to create payload index '%s' on '%s': %s",
                    field_name,
                    PRIVATE_COLLECTION_NAME,
                    e,
                )
                raise RuntimeError(
                    f"Failed to create payload index '{field_name}' on "
                    f"'{PRIVATE_COLLECTION_NAME}'"
                ) from e

    def create_collection(self, collection_name: str) -> bool:
        """
        Create a new collection in the vector store.

        Private user collections no longer get their own Qdrant collection.
        Use ``ensure_private_collection`` instead.

        Args:
            collection_name: Name of the collection to create

        Returns:
            bool: True if collection creation was successful

        Raises:
            RuntimeError: If the collection creation fails
        """
        if collection_name == PRIVATE_COLLECTION_NAME:
            self.ensure_private_collection()
            return True

        vectors_config = VectorParams(
            size=self.embeddings_size,
            distance=Distance.COSINE,
        )

        try:
            if self.client.collection_exists(collection_name):
                logger.info("Collection '%s' already exists", collection_name)
                return True

            success = self.client.create_collection(
                collection_name=collection_name, vectors_config=vectors_config
            )

            if success is False:
                raise RuntimeError(f"Failed to create collection: {collection_name}")

            logger.info(f"Collection '{collection_name}' created successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to create collection '{collection_name}': {e}")
            raise RuntimeError(f"Failed to create collection: {str(e)}") from e

    def list_collections(self) -> types.CollectionsResponse:
        """
        Get all collections from the vector store.

        Returns:
            types.CollectionsResponse: Qdrant collections response
        """
        return self.client.get_collections()

    async def list_public_collections(
        self, page: int = 1, limit: Optional[int] = None
    ) -> Tuple[List[Dict[str, str]], int]:
        """
        Get public collections for the current environment.

        Staging includes prod-named collections as well as staging-only ones.
        The shared private collection is never listed.
        """
        alias_map: Dict[str, str] = {}
        try:
            aliases_response = self.client.get_aliases()
            alias_items = getattr(aliases_response, "aliases", aliases_response)
            for item in alias_items or []:
                alias_name = getattr(item, "alias_name", None)
                collection_name = getattr(item, "collection_name", None)
                if collection_name and alias_name:
                    alias_map.setdefault(collection_name, alias_name)
        except Exception as e:
            logger.warning("Failed to load Qdrant aliases for public collections: %s", e)

        public_collections = []
        for item in iter_public_catalog(is_prod=IS_PROD):
            name = item.get("name")
            if not name or name == PRIVATE_COLLECTION_NAME:
                continue
            public_collections.append(
                {
                    "name": name,
                    "alias": item.get("alias") or alias_map.get(name) or None,
                    "description": item.get("description")
                    or "Public Collection from ESA",
                }
            )

        total = len(public_collections)
        if limit is None:
            return public_collections, total
        start = (page - 1) * limit
        end = start + limit
        return public_collections[start:end], total

    def list_collections_names(self) -> List[str]:
        """
        Get the names of all collections in the vector store.

        Returns:
            List[str]: List of collection names

        Raises:
            RuntimeError: If listing collections fails
        """
        try:
            collections_list = []
            collections = self.client.get_collections()

            # Fix the collection name extraction
            if hasattr(collections, "collections"):
                for collection in collections.collections:
                    collections_list.append(collection.name)
            else:
                # Handle different response format
                for collection in collections:
                    if isinstance(collection, tuple) and len(collection) > 1:
                        for c in list(collection[1]):
                            collections_list.append(c.name)
                    else:
                        collections_list.append(collection.name)

            return collections_list

        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            raise RuntimeError(f"Failed to list collections: {str(e)}") from e

    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection from the vector store.

        Args:
            collection_name: Name of the collection to delete

        Returns:
            bool: True if deletion was successful

        Raises:
            ValueError: If the collection doesn't exist
            RuntimeError: If deletion fails for other reasons
        """
        if collection_name == PRIVATE_COLLECTION_NAME:
            raise RuntimeError(
                "Refusing to delete the shared private collection "
                f"'{PRIVATE_COLLECTION_NAME}'"
            )

        if collection_name not in self.list_collections_names():
            logger.warning(
                f"Attempted to delete non-existent collection '{collection_name}'"
            )
            raise ValueError(f"Collection '{collection_name}' does not exist")

        try:
            self.client.delete_collection(collection_name=collection_name)
            logger.info(f"Collection '{collection_name}' deleted successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to delete collection '{collection_name}': {e}")
            raise RuntimeError(f"Failed to delete collection: {str(e)}") from e

    def count_points_for_collection(self, user_id: str, collection_id: str) -> int:
        """Count points in the shared private collection for one logical collection."""
        try:
            result = self.client.count(
                collection_name=PRIVATE_COLLECTION_NAME,
                count_filter=build_private_tenant_filter(user_id, [collection_id]),
                exact=True,
            )
            return int(getattr(result, "count", 0) or 0)
        except Exception as e:
            logger.warning(
                "Failed to count Qdrant points for collection %s: %s",
                collection_id,
                e,
            )
            return 0

    def delete_points_for_collection(self, user_id: str, collection_id: str) -> int:
        """Delete all private points for a user collection. Does not drop the Qdrant collection."""
        result = self.delete_private_docs(
            user_id=user_id,
            collection_id=collection_id,
            metadata=None,
        )
        return int(getattr(result, "deleted", 0) or 0)

    def delete_private_docs(
        self,
        user_id: str,
        collection_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Delete private points matching tenant keys plus optional metadata."""
        extra = self._qdrant_filter_from_dict(metadata)
        extra_must = list(getattr(extra, "must", None) or []) if extra else []
        filter_obj = merge_must_filters(
            build_private_tenant_filter(user_id, [collection_id]),
            extra_must,
        )
        try:
            count_result = self.client.count(
                collection_name=PRIVATE_COLLECTION_NAME,
                count_filter=filter_obj,
                exact=True,
            )
            count_before = count_result.count

            self.client.delete(
                collection_name=PRIVATE_COLLECTION_NAME,
                points_selector=filter_obj,
            )

            count_result_after = self.client.count(
                collection_name=PRIVATE_COLLECTION_NAME,
                count_filter=filter_obj,
                exact=True,
            )
            deleted_count = count_before - count_result_after.count
            logger.info(
                "Deleted %s documents from '%s' (user_id=%s collection_id=%s)",
                deleted_count,
                PRIVATE_COLLECTION_NAME,
                user_id,
                collection_id,
            )

            class DeleteResult:
                def __init__(self, deleted_count):
                    self.deleted = deleted_count

            return DeleteResult(deleted_count)
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise RuntimeError(f"Failed to delete documents: {str(e)}") from e

    async def add_document_list(
        self,
        document_list: List[Document],
        *,
        user_id: str,
        collection_id: str,
    ) -> List[str]:
        """
        Add a list of documents to the shared private collection.

        Args:
            document_list: List of documents to add
            user_id: Owner user ID written on every point
            collection_id: Logical Mongo collection ID written on every point

        Returns:
            List[str]: List of UUIDs for the added documents
        """
        if not document_list:
            logger.warning("Empty document list provided, nothing to add")
            return []
        if not user_id or not collection_id:
            raise ValueError("user_id and collection_id are required to upsert private points")

        self.ensure_private_collection()
        uuids = [str(uuid4()) for _ in range(len(document_list))]

        try:
            await self._add_documents_in_batches(
                document_list,
                uuids,
                user_id=user_id,
                collection_id=collection_id,
                batch_size=32,
            )
            logger.info(
                "Added %s documents to '%s' (collection_id=%s)",
                len(document_list),
                PRIVATE_COLLECTION_NAME,
                collection_id,
            )
            return uuids

        except Exception as e:
            logger.error(
                "Error adding documents to '%s': %s", PRIVATE_COLLECTION_NAME, e
            )
            raise RuntimeError(
                f"Failed to add documents to '{PRIVATE_COLLECTION_NAME}': {str(e)}"
            ) from e

    async def _add_documents_in_batches(
        self,
        documents: List[Document],
        uuids: List[str],
        *,
        user_id: str,
        collection_id: str,
        batch_size: int = 32,
    ) -> None:
        """
        Add documents to Qdrant in batches to prevent timeout issues.
        """
        total_documents = len(documents)
        logger.info(f"Adding {total_documents} documents in batches of {batch_size}")

        for i in range(0, total_documents, batch_size):
            batch_end = min(i + batch_size, total_documents)
            batch_documents = documents[i:batch_end]
            batch_uuids = uuids[i:batch_end]

            batch_num = (i // batch_size) + 1
            total_batches = (total_documents + batch_size - 1) // batch_size

            logger.info(
                f"Processing batch {batch_num}/{total_batches} ({len(batch_documents)} documents)"
            )

            try:
                await self._add_documents_directly(
                    batch_documents,
                    batch_uuids,
                    user_id=user_id,
                    collection_id=collection_id,
                )
                logger.info(f"Successfully added batch {batch_num}/{total_batches}")

                if batch_num < total_batches:
                    await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(
                    f"Failed to add batch {batch_num}/{total_batches}: {str(e)}"
                )
                raise RuntimeError(f"Failed to add batch {batch_num}: {str(e)}") from e

    async def _add_documents_directly(
        self,
        documents: List[Document],
        uuids: List[str],
        *,
        user_id: str,
        collection_id: str,
    ) -> None:
        """Upsert chunk points into the shared private collection."""
        from qdrant_client.models import PointStruct

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        embeddings, _ = await self.generate_batch_embeddings(
            texts, self.embeddings_model
        )
        points = []
        for text, metadata, embedding, uuid in zip(
            texts, metadatas, embeddings, uuids
        ):
            payload_metadata = dict(metadata or {})
            point = PointStruct(
                id=uuid,
                vector=embedding,
                payload={
                    "text": text,
                    "metadata": payload_metadata,
                    "user_id": user_id,
                    "collection_id": collection_id,
                },
            )
            points.append(point)

        await asyncio.to_thread(
            self.client.upsert,
            collection_name=PRIVATE_COLLECTION_NAME,
            points=points,
            wait=True,
        )

    def _qdrant_filter_from_dict(
        self, filter_dict: Optional[Dict[str, Any]]
    ) -> Optional[Filter]:
        """
        Convert a simple dictionary (key -> value) to a Qdrant filter object,
        applying conditions against payload.metadata.* keys by default.

        This is kept for backwards compatibility where callers pass a flat dict.
        """
        if not filter_dict:
            return None

        return Filter(
            must=[
                condition
                for key, value in filter_dict.items()
                for condition in self._build_condition(key, value)
            ]
        )

    @staticmethod
    def _to_generic_result(result: Any, collection_name: str) -> Optional[Any]:
        """
        Convert a Qdrant result (e.g., ScoredPoint) into a simple object with
        stable attributes consumed downstream, including collection_name.

        Returns None if conversion fails.
        """
        try:
            rid = getattr(result, "id", None)
            score = getattr(result, "score", None) or getattr(result, "distance", None)
            payload = (
                getattr(result, "payload", None)
                or getattr(result, "document", None)
                or {}
            )
            if payload.get("title") == "None":
                payload["title"] = None
            text = getattr(result, "text", None)
            metadata = getattr(result, "metadata", None)

            if text is None and isinstance(payload, dict):
                text = payload.get("text") or payload.get("content")
            if metadata is None and isinstance(payload, dict):
                metadata = payload.get("metadata")

            version = getattr(result, "version", None)

            return SimpleNamespace(
                id=rid,
                version=version,
                score=score,
                payload=payload if isinstance(payload, dict) else {},
                text=text or "",
                metadata=metadata if isinstance(metadata, dict) else {},
                collection_name=collection_name,
            )
        except Exception:
            return None

    def _collection_has_env_payload(self, collection_name: str) -> bool:
        """True when Qdrant lists an ``env`` payload field on the collection.

        Uses ``CollectionInfo.payload_schema``, which reports indexed payload
        fields. Collections without an ``env`` index are left unfiltered.
        See https://qdrant.tech/documentation/concepts/collections/
        """
        cache = getattr(self, "_env_payload_cache", None)
        if cache is None:
            cache = {}
            self._env_payload_cache = cache
        cached = cache.get(collection_name)
        if cached is not None:
            return cached
        has_env = False
        try:
            info = self.client.get_collection(collection_name)
            schema = getattr(info, "payload_schema", None) or {}
            has_env = "env" in schema
        except Exception as e:
            logger.warning(
                "Could not inspect payload schema for '%s'; skipping env filter: %s",
                collection_name,
                e,
            )
            has_env = False
        cache[collection_name] = has_env
        return has_env

    def _search_across_collections(
        self,
        collection_names: List[str],
        query_vector: List[float],
        score_threshold: float,
        query_filter: Optional[Filter],
        limit_per_collection: int,
        private_collections_map: Optional[Dict[str, str]] = None,
        user_id: Optional[str] = None,
    ) -> List[Any]:
        """
        Search public named collections and the shared private collection.

        Public ``env`` filtering is skipped for Wiley and for collections that
        do not advertise an ``env`` payload field in Qdrant.
        """
        logger.debug("private_collections_map: %s", private_collections_map)
        alias_map: Dict[str, str] = {}
        try:
            aliases_response = self.client.get_aliases()
            alias_items = getattr(aliases_response, "aliases", aliases_response)
            for item in alias_items or []:
                alias_name = getattr(item, "alias_name", None)
                collection_name = getattr(item, "collection_name", None)
                if collection_name and alias_name:
                    alias_map.setdefault(collection_name, alias_name)
        except Exception as e:
            logger.warning("Failed to load Qdrant aliases during search: %s", e)

        public_names, private_ids = split_public_and_private_collections(
            collection_names, private_collections_map
        )
        aggregated_results: List[Any] = []

        for collection_name in public_names:
            client_filter = (
                query_filter if is_eve_public_collection(collection_name) else None
            )
            extra_must: List[Any] = []
            if (
                not is_wiley_public_collection(collection_name)
                and self._collection_has_env_payload(collection_name)
            ):
                extra_must.append(build_public_env_filter(IS_PROD))
            collection_query_filter = merge_must_filters(client_filter, extra_must)
            try:
                qp_response = self.client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    limit=limit_per_collection,
                    score_threshold=score_threshold,
                    query_filter=collection_query_filter,
                    search_params=models.SearchParams(
                        quantization=models.QuantizationSearchParams(
                            ignore=False,
                            rescore=True,
                            oversampling=2.0,
                        )
                    ),
                    timeout=120,
                )
                results = getattr(qp_response, "points", []) or []
                display_name = alias_map.get(collection_name, collection_name)
                for scored in results:
                    conv = self._to_generic_result(scored, display_name)
                    if conv is not None:
                        aggregated_results.append(conv)
            except Exception as e:
                if _is_missing_collection_error(e) or _is_timeout_error(e):
                    raise RuntimeError(
                        f"Failed to search collection '{collection_name}': {e}"
                    ) from e
                logger.warning(f"Failed to search collection '{collection_name}': {e}")
                continue

        if private_ids:
            if not user_id:
                logger.warning(
                    "Skipping private collection search: user_id is required for tenant filter"
                )
            else:
                try:
                    self.ensure_private_collection()
                except Exception as e:
                    logger.error(
                        "Skipping private collection search; could not ensure '%s': %s",
                        PRIVATE_COLLECTION_NAME,
                        e,
                    )
                else:
                    name_map = private_collections_map or {}
                    for collection_id in private_ids:
                        try:
                            private_filter = build_private_tenant_filter(
                                user_id, [collection_id]
                            )
                            qp_response = self.client.query_points(
                                collection_name=PRIVATE_COLLECTION_NAME,
                                query=query_vector,
                                limit=limit_per_collection,
                                score_threshold=score_threshold,
                                query_filter=private_filter,
                                search_params=models.SearchParams(
                                    quantization=models.QuantizationSearchParams(
                                        ignore=False,
                                        rescore=True,
                                        oversampling=2.0,
                                    )
                                ),
                                timeout=120,
                            )
                            results = getattr(qp_response, "points", []) or []
                            for scored in results:
                                payload = getattr(scored, "payload", None) or {}
                                logical_id = (
                                    payload.get("collection_id")
                                    if isinstance(payload, dict)
                                    else None
                                )
                                display_name = name_map.get(
                                    logical_id, logical_id or PRIVATE_COLLECTION_NAME
                                )
                                conv = self._to_generic_result(scored, display_name)
                                if conv is not None:
                                    aggregated_results.append(conv)
                        except Exception as e:
                            if _is_timeout_error(e):
                                raise RuntimeError(
                                    f"Failed to search private collection "
                                    f"'{PRIVATE_COLLECTION_NAME}' "
                                    f"(collection_id={collection_id}): {e}"
                                ) from e
                            logger.warning(
                                "Failed to search private collection '%s' "
                                "(collection_id=%s): %s",
                                PRIVATE_COLLECTION_NAME,
                                collection_id,
                                e,
                            )
                            continue

        logger.info(
            "Retrieved %s total documents from %s public collections and %s private collections",
            len(aggregated_results),
            len(public_names),
            len(private_ids),
        )
        return aggregated_results

    @staticmethod
    def _sort_by_score_desc(results: List[Any]) -> List[Any]:
        """
        Sort a list of scored results in descending order of score.

        Args:
            results: List of results with a 'score' attribute

        Returns:
            Sorted list by score (highest first)
        """
        try:
            return sorted(results, key=lambda x: x.score, reverse=True)
        except Exception:
            # If objects don't have score attribute, return as-is
            return results

    def _build_condition(self, key: str, value: Any) -> List[FieldCondition]:
        """
        Build Qdrant field conditions from keys and values.

        Handles nested structures recursively.

        Args:
            key: The metadata field key
            value: The value to match

        Returns:
            List[FieldCondition]: List of generated field conditions
        """
        conditions = []

        if isinstance(value, dict):
            for _key, _value in value.items():
                conditions.extend(self._build_condition(f"{key}.{_key}", _value))

        elif isinstance(value, list):
            for _value in value:
                if isinstance(_value, dict):
                    conditions.extend(self._build_condition(f"{key}[]", _value))
                else:
                    conditions.extend(self._build_condition(f"{key}", _value))

        else:
            conditions.append(
                FieldCondition(
                    key=f"{key}",
                    match=MatchValue(value=value),
                )
            )

        return conditions

    def delete_docs_by_metadata_filter(
        self, collection_name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Delete documents that match a metadata filter.

        Args:
            collection_name: Name of the collection
            metadata: Metadata filter to select documents for deletion

        Returns:
            Dict[str, Any]: Result of the deletion operation with deleted count

            Raises:
            RuntimeError: If deletion fails
        """
        if collection_name == PRIVATE_COLLECTION_NAME:
            raise RuntimeError(
                "Refusing unscoped metadata delete on "
                f"'{PRIVATE_COLLECTION_NAME}'; use delete_private_docs"
            )
        try:
            # Get the count of documents before deletion using the count method
            filter_obj = self._qdrant_filter_from_dict(metadata)
            count_result = self.client.count(
                collection_name=collection_name,
                count_filter=filter_obj,
                exact=True,  # For an exact count
            )
            count_before = count_result.count

            # Perform the deletion
            result = self.client.delete(
                collection_name=collection_name,
                points_selector=filter_obj,
            )

            # Get the count of documents after deletion
            count_result_after = self.client.count(
                collection_name=collection_name,
                count_filter=filter_obj,
                exact=True,  # For an exact count
            )
            count_after = count_result_after.count

            # Calculate the actual number of deleted documents
            deleted_count = count_before - count_after

            logger.info(f"Deleted {deleted_count} documents from '{collection_name}'")

            # Create a result object with the deleted count
            class DeleteResult:
                def __init__(self, deleted_count):
                    self.deleted = deleted_count

            return DeleteResult(deleted_count)

        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise RuntimeError(f"Failed to delete documents: {str(e)}") from e

    async def generate_query_vector(
        self, query: str, embeddings_model: str
    ) -> Tuple[List[float], Optional[str]]:
        """
        Generate an embedding vector for a query.

        Args:
            query: The query text
            embeddings_model: Model to use for embedding generation

        Returns:
            List[float]: Vector representation of the query

        Raises:
            RuntimeError: If embedding generation fails
        """
        try:
            openai = OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_URL)
            response = openai.embeddings.create(input=query, model=embeddings_model)
            return response.data[0].embedding, None

        except Exception as e:
            logger.error(f"Failed to generate query vector from main model: {e}")
            error_logger = get_error_logger()
            await error_logger.log_error_sync(
                error=e,
                component=Component.RETRIEVAL,
                pipeline_stage=PipelineStage.RETRIEVAL,
                description="Failed to generate query vector from main model",
                error_type=type(e).__name__,
            )
            try:
                openai = OpenAI(
                    api_key=EMBEDDING_FALLBACK_API_KEY, base_url=EMBEDDING_FALLBACK_URL
                )
                response = openai.embeddings.create(input=query, model=embeddings_model)
                return response.data[0].embedding, str(e)
            except Exception as e:
                logger.error(
                    f"Failed to generate query vector from fallback model: {e}"
                )
                await error_logger.log_error_sync(
                    error=e,
                    component=Component.RETRIEVAL_FALLBACK,
                    pipeline_stage=PipelineStage.RETRIEVAL,
                    description="Failed to generate query vector from fallback model",
                    error_type=type(e).__name__,
                )

    async def generate_batch_embeddings(
        self, texts: List[str], embeddings_model: str
    ) -> Tuple[List[List[float]], Optional[str]]:
        """
        Generate embedding vectors for multiple texts in batch.

        Args:
            texts: List of texts to embed
            embeddings_model: Model to use for embedding generation

        Returns:
            Tuple[List[List[float]], Optional[str]]: List of embedding vectors and optional error message

        Raises:
            RuntimeError: If embedding generation fails
        """
        if not texts:
            return [], None

        try:
            openai = OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_URL)
            response = openai.embeddings.create(input=texts, model=embeddings_model)
            embeddings = [item.embedding for item in response.data]
            return embeddings, None

        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            openai = OpenAI(
                api_key=EMBEDDING_FALLBACK_API_KEY, base_url=EMBEDDING_FALLBACK_URL
            )
            response = openai.embeddings.create(input=texts, model=embeddings_model)
            embeddings = [item.embedding for item in response.data]
            return embeddings, str(e)

    async def retrieve_documents_from_query(
        self,
        collection_names: List[str],
        query: str,
        k: int = 5,
        score_threshold: float = 0.7,
        embeddings_model: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        private_collections_map: Optional[Dict[str, str]] = None,
        user_id: Optional[str] = None,
    ) -> List[Any]:
        """
        Retrieve relevant documents for a given query from multiple collections.

        Args:
            collection_names: List of names of the collections to search
            query: The query text
            year: List with two values [start_year, end_year] to filter by publication year.
            keywords: List of keywords to filter by title.
            k: Number of documents to retrieve
            score_threshold: Minimum similarity score (0-1)
            embeddings_model: Optional custom embedding model to use
                              (defaults to the model used at initialization)

        Returns:
            List[Any]: List of relevant documents with similarity scores

        Raises:
            RuntimeError: If retrieval fails
        """
        model = embeddings_model or self.embeddings_model

        try:
            # Generate embedding vector for the query
            query_vector, _ = await self.generate_query_vector(query, model)
            query_filter = Filter(**filters) if filters else None

            # Retrieve k per collection (caller may further rerank/filter)
            all_results = await asyncio.to_thread(
                self._search_across_collections,
                collection_names,
                query_vector,
                score_threshold,
                query_filter,
                k,
                private_collections_map,
                user_id,
            )

            logger.info(
                f"Retrieved {len(all_results)} documents from {len(collection_names)} collections "
                f"(filtered from {len(all_results)} total matches)"
            )
            return all_results

        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            raise RuntimeError(f"Failed to retrieve documents: {str(e)}") from e

    # RAG decision moved to LLMManager.should_use_rag

    async def retrieve_documents_with_latencies(
        self,
        collection_names: List[str],
        query: str,
        k: int = 5,
        score_threshold: float = 0.7,
        embeddings_model: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        private_collections_map: Optional[Dict[str, str]] = None,
        user_id: Optional[str] = None,
    ) -> tuple[List[Any], Dict[str, Optional[float]]]:
        """
        Retrieve relevant documents and measure query embedding and Qdrant retrieval latencies.

        Returns a tuple of (results, latencies)
        where latencies contains keys: "query_embedding_latency", "qdrant_retrieval_latency".
        """
        import time

        model = embeddings_model or self.embeddings_model

        embedding_latency: Optional[float] = None
        retrieval_latency: Optional[float] = None

        try:
            # Generate embedding vector for the query
            t0 = time.perf_counter()
            query_vector, _ = await self.generate_query_vector(query, model)
            embedding_latency = time.perf_counter() - t0

            query_filter = Filter(**filters) if filters else None

            # Search across collections
            t1 = time.perf_counter()
            all_results = await asyncio.to_thread(
                self._search_across_collections,
                collection_names,
                query_vector,
                score_threshold,
                query_filter,
                k,
                private_collections_map,
                user_id,
            )
            retrieval_latency = time.perf_counter() - t1

            logger.info(
                f"Retrieved {len(all_results)} documents from {len(collection_names)} collections "
                f"(filtered from {len(all_results)} total matches)"
            )

            latencies: Dict[str, Optional[float]] = {
                "query_embedding_latency": embedding_latency,
                "qdrant_retrieval_latency": retrieval_latency,
            }
            return all_results, latencies

        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            raise RuntimeError(f"Failed to retrieve documents: {str(e)}") from e

    def sync_retrieve_documents_from_query(
        self,
        collection_name: str,
        query: str,
        k: int = 5,
        score_threshold: float = 0.7,
        embeddings_model: Optional[str] = None,
    ) -> List[Any]:
        """
        Synchronous wrapper for retrieve_documents_from_query.

        This method allows calling the async retrieval method from sync contexts.
        """
        return asyncio.run(
            self.retrieve_documents_from_query(
                collection_names=[collection_name],
                query=query,
                k=k,
                score_threshold=score_threshold,
                embeddings_model=embeddings_model,
            )
        )

    # RAG decision moved to LLMManager.should_use_rag


if __name__ == "__main__":
    # Configure logging when run as a script
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
