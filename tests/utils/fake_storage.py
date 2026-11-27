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

    async def exists(self, key):
        return key in self.objects

    async def get_object(self, key):
        # Mirror StorageService, which turns a botocore NoSuchKey into this.
        if key not in self.objects:
            raise ObjectNotFoundError(key)
        obj = self.objects[key]
        return {
            "Body": _FakeBody(obj["body"]),
            "ETag": '"fake-etag"',
            "ContentLength": len(obj["body"]),
            "ContentType": obj["content_type"],
        }

    async def stream_body(self, body, chunk_size=65536):
        yield body.data

    async def copy_object(self, source_key, dest_key):
        if source_key not in self.objects:
            raise ObjectNotFoundError(source_key)
        self.objects[dest_key] = dict(self.objects[source_key])

    async def delete_object(self, key):
        self.objects.pop(key, None)

    async def object_size(self, key):
        obj = self.objects.get(key)
        if obj is None:
            return 0
        return len(obj["body"])

    def list_key_records(self, prefix, max_keys, *, scan_limit=1000):
        rows = [
            {"key": key, "size": len(obj["body"]), "last_modified": None}
            for key, obj in self.objects.items()
            if key.startswith(prefix) and not key.endswith("/")
        ]
        return rows[:max_keys], len(rows) > max_keys

    async def delete_prefix(self, prefix, max_keys=10_000):
        doomed = [k for k in self.objects if k.startswith(prefix)][:max_keys]
        total = 0
        for key in doomed:
            total += len(self.objects[key]["body"])
            del self.objects[key]
        return len(doomed), total

    async def generate_presigned_get(
        self,
        key,
        expires_in=None,
        *,
        response_content_type=None,
        response_content_disposition=None,
    ):
        self.last_presign = {
            "key": key,
            "response_content_type": response_content_type,
            "response_content_disposition": response_content_disposition,
        }
        return f"http://minio.local/{key}?signed=1"
