"""In-memory stand-in for StorageService, shared by any test that touches artifacts."""

import uuid

from src.services.storage import ObjectNotFoundError


class _FakeBody:
    """Stand-in for a boto3 StreamingBody used by the fake storage."""

    def __init__(self, data: bytes):
        self.data = data

    def iter_chunks(self, chunk_size):
        yield self.data

    def close(self):
        pass


class FakeStorage:
    """In-memory storage backend mirroring the StorageService interface."""

    def __init__(self):
        self.objects = {}

    def build_user_key(self, user_id: str, ext: str, prefix: str = None) -> str:
        clean_ext = ext.lstrip(".").lower()
        clean_prefix = prefix.strip("/") if prefix else None
        base = f"users/{user_id}/{clean_prefix}" if clean_prefix else f"users/{user_id}"
        return f"{base}/{uuid.uuid4().hex}.{clean_ext}"

    async def put_object(self, key, body, content_type):
        self.objects[key] = {"body": body, "content_type": content_type}

    async def get_object(self, key):
        # Mirror StorageService, which turns a botocore NoSuchKey into this.
        if key not in self.objects:
            raise ObjectNotFoundError(key)
        obj = self.objects[key]
        return {
            "Body": _FakeBody(obj["body"]),
            "ETag": '"fake-etag"',
            "ContentLength": len(obj["body"]),
        }

    async def stream_body(self, body, chunk_size=65536):
        yield body.data

    async def delete_object(self, key):
        self.objects.pop(key, None)

    async def generate_presigned_get(self, key, expires_in=None):
        return f"http://minio.local/{key}?signed=1"

    async def generate_presigned_put(self, key, content_type=None, expires_in=None):
        return f"http://minio.local/{key}?signed=1"
