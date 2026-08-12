"""Tests for the BYOK envelope cipher and its wiring into storage/migration.

Covers both DEK-wrapping backends (local KEK and KMS, the latter against an
in-memory fake client -- no real AWS calls), context-binding (wrong context
must fail closed), the legacy Secrets Manager read/delete fallback in
src/services/custom_model_secrets.py, and the one-time migration command.
"""

import base64
import json
import os

import pytest
from botocore.exceptions import ClientError
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

import src.commands.migrate_custom_model_secrets as migrate_command
from src.commands.migrate_custom_model_secrets import migrate_custom_model_secrets
from src.database.models.user_custom_model import UserCustomModel
import src.services.custom_model_cipher as custom_model_cipher
from src.services.custom_model_cipher import (
    CipherConfigError,
    CipherContextError,
    SecretCipher,
)
import src.services.custom_model_secrets as custom_model_secrets
from src.services.custom_model_secrets import (
    clear_secret_cache_for_tests,
    create_custom_model_secret,
    delete_custom_model_secret,
    get_custom_model_api_key,
)
from tests.utils.cleaner import cleanup_models

LOCAL_KEK_B64 = base64.b64encode(b"\x11" * 32).decode()
CONTEXT_A = {"user_id": "user-a", "provider_id": "openai", "model_id": "model-1"}
CONTEXT_B = {"user_id": "user-b", "provider_id": "openai", "model_id": "model-2"}


class FakeKmsClient:
    """In-memory stand-in for boto3's KMS client.

    Wraps/unwraps with a fixed master key via AES-256-GCM, using the
    EncryptionContext as AAD -- enough to exercise our EncryptionContext
    plumbing and error mapping without touching real AWS.
    """

    KEY_ID = "arn:aws:kms:eu-central-1:123456789012:key/fake-test-key"

    def __init__(self):
        self._master_key = AESGCM.generate_key(bit_length=256)

    @staticmethod
    def _aad(encryption_context: dict) -> bytes:
        return json.dumps(encryption_context, sort_keys=True, separators=(",", ":")).encode()

    def encrypt(self, *, KeyId, Plaintext, EncryptionContext):
        assert KeyId == self.KEY_ID
        nonce = os.urandom(12)
        ciphertext = AESGCM(self._master_key).encrypt(
            nonce, Plaintext, self._aad(EncryptionContext)
        )
        return {"CiphertextBlob": nonce + ciphertext, "KeyId": self.KEY_ID}

    def decrypt(self, *, KeyId, CiphertextBlob, EncryptionContext):
        nonce, ciphertext = CiphertextBlob[:12], CiphertextBlob[12:]
        try:
            plaintext = AESGCM(self._master_key).decrypt(
                nonce, ciphertext, self._aad(EncryptionContext)
            )
        except Exception as exc:
            raise ClientError(
                {"Error": {"Code": "InvalidCiphertextException", "Message": str(exc)}},
                "Decrypt",
            ) from exc
        return {"Plaintext": plaintext, "KeyId": self.KEY_ID}


def _decode_blob(blob: str) -> dict:
    return json.loads(base64.b64decode(blob))


class AsyncMockReturning:
    """Tiny async-callable spy, avoids pulling in unittest.mock.AsyncMock just for call assertions."""

    def __init__(self, return_value):
        self._return_value = return_value
        self.calls: list[tuple] = []

    async def __call__(self, *args, **kwargs):
        self.calls.append(args)
        return self._return_value

    def assert_called_once_with(self, *args):
        assert self.calls == [args], f"expected single call with {args}, got {self.calls}"

    def assert_not_called(self):
        assert self.calls == [], f"expected no calls, got {self.calls}"


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_secret_cache_for_tests()
    yield
    clear_secret_cache_for_tests()


@pytest.fixture
def local_backend(monkeypatch):
    monkeypatch.setattr(custom_model_cipher, "CUSTOM_MODEL_KMS_KEY_ID", None)
    monkeypatch.setattr(custom_model_cipher, "BYOK_LOCAL_KEK", LOCAL_KEK_B64)


@pytest.fixture
def kms_backend(monkeypatch):
    fake_client = FakeKmsClient()
    monkeypatch.setattr(custom_model_cipher, "CUSTOM_MODEL_KMS_KEY_ID", fake_client.KEY_ID)
    monkeypatch.setattr(custom_model_cipher, "BYOK_LOCAL_KEK", None)
    monkeypatch.setattr(custom_model_cipher, "_kms_client", lambda: fake_client)
    return fake_client


@pytest.mark.no_db
def test_local_backend_round_trip(local_backend):
    cipher = SecretCipher()
    blob = cipher.encrypt("sk-super-secret", CONTEXT_A)
    assert cipher.decrypt(blob, CONTEXT_A) == "sk-super-secret"
    assert _decode_blob(blob)["backend"] == "local"


@pytest.mark.no_db
def test_kms_backend_round_trip(kms_backend):
    cipher = SecretCipher()
    blob = cipher.encrypt("sk-super-secret", CONTEXT_A)
    assert cipher.decrypt(blob, CONTEXT_A) == "sk-super-secret"
    assert _decode_blob(blob)["backend"] == "kms"


@pytest.mark.no_db
def test_kms_preferred_when_both_backends_configured(monkeypatch, kms_backend):
    monkeypatch.setattr(custom_model_cipher, "BYOK_LOCAL_KEK", LOCAL_KEK_B64)
    cipher = SecretCipher()
    blob = cipher.encrypt("sk-super-secret", CONTEXT_A)
    assert _decode_blob(blob)["backend"] == "kms"
    assert cipher.decrypt(blob, CONTEXT_A) == "sk-super-secret"


@pytest.mark.no_db
def test_local_backend_wrong_context_fails_closed(local_backend):
    cipher = SecretCipher()
    blob = cipher.encrypt("sk-super-secret", CONTEXT_A)
    with pytest.raises(CipherContextError):
        cipher.decrypt(blob, CONTEXT_B)


@pytest.mark.no_db
def test_kms_backend_wrong_context_fails_closed(kms_backend):
    cipher = SecretCipher()
    blob = cipher.encrypt("sk-super-secret", CONTEXT_A)
    with pytest.raises(CipherContextError):
        cipher.decrypt(blob, CONTEXT_B)


@pytest.mark.no_db
def test_no_backend_configured_raises_config_error(monkeypatch):
    monkeypatch.setattr(custom_model_cipher, "CUSTOM_MODEL_KMS_KEY_ID", None)
    monkeypatch.setattr(custom_model_cipher, "BYOK_LOCAL_KEK", None)
    cipher = SecretCipher()
    with pytest.raises(CipherConfigError):
        cipher.encrypt("sk-super-secret", CONTEXT_A)


@pytest.mark.no_db
def test_encrypt_rejects_incomplete_context(local_backend):
    cipher = SecretCipher()
    with pytest.raises(ValueError):
        cipher.encrypt("sk-super-secret", {"user_id": "u1", "provider_id": "openai"})


@pytest.mark.asyncio
async def test_create_and_get_custom_model_api_key_uses_encrypted_key(
    local_backend, monkeypatch
):
    model = await UserCustomModel.create(
        user_id="user-1",
        display_name="Local key",
        provider_id="openai",
        catalog_model_id="gpt-5.4-2026-03-05",
        model_name="gpt-5.4-2026-03-05",
    )
    try:
        model.encrypted_key = await create_custom_model_secret(
            user_id=model.user_id,
            provider_id=model.provider_id,
            model_id=model.id,
            api_key="sk-live-key",
        )
        await model.save()
        assert model.secret_arn is None

        assert await get_custom_model_api_key(model) == "sk-live-key"

        # Second read must hit the 60s cache, not the cipher, so breaking the
        # cipher here proves the cache (not a lucky re-decrypt) served the value.
        monkeypatch.setattr(
            custom_model_secrets._cipher,
            "decrypt",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("cache miss")),
        )
        assert await get_custom_model_api_key(model) == "sk-live-key"
    finally:
        await cleanup_models([model])


@pytest.mark.asyncio
async def test_get_custom_model_api_key_legacy_fallback(monkeypatch):
    model = await UserCustomModel.create(
        user_id="user-1",
        display_name="Legacy key",
        provider_id="openai",
        catalog_model_id="gpt-5.4-2026-03-05",
        model_name="gpt-5.4-2026-03-05",
        secret_arn="arn:aws:secretsmanager:eu-central-1:123:secret:legacy",
    )
    try:
        read_legacy = AsyncMockReturning("sk-legacy-key")
        monkeypatch.setattr(custom_model_secrets, "read_legacy_secret_value", read_legacy)

        assert await get_custom_model_api_key(model) == "sk-legacy-key"
        read_legacy.assert_called_once_with("arn:aws:secretsmanager:eu-central-1:123:secret:legacy")
    finally:
        await cleanup_models([model])


@pytest.mark.asyncio
async def test_delete_custom_model_secret_cleans_up_legacy_row(monkeypatch):
    model = await UserCustomModel.create(
        user_id="user-1",
        display_name="Legacy key",
        provider_id="openai",
        catalog_model_id="gpt-5.4-2026-03-05",
        model_name="gpt-5.4-2026-03-05",
        secret_arn="arn:aws:secretsmanager:eu-central-1:123:secret:legacy",
    )
    try:
        delete_legacy = AsyncMockReturning(None)
        monkeypatch.setattr(custom_model_secrets, "delete_legacy_secret", delete_legacy)

        await delete_custom_model_secret(model)
        delete_legacy.assert_called_once_with("arn:aws:secretsmanager:eu-central-1:123:secret:legacy")
    finally:
        await cleanup_models([model])


@pytest.mark.asyncio
async def test_delete_custom_model_secret_is_noop_for_new_rows(monkeypatch):
    model = await UserCustomModel.create(
        user_id="user-1",
        display_name="New key",
        provider_id="openai",
        catalog_model_id="gpt-5.4-2026-03-05",
        model_name="gpt-5.4-2026-03-05",
        encrypted_key="irrelevant-for-this-test",
    )
    try:
        delete_legacy = AsyncMockReturning(None)
        monkeypatch.setattr(custom_model_secrets, "delete_legacy_secret", delete_legacy)

        await delete_custom_model_secret(model)
        delete_legacy.assert_not_called()
    finally:
        await cleanup_models([model])


@pytest.mark.asyncio
async def test_migrate_custom_model_secrets_reencrypts_and_deletes_legacy_secret(
    local_backend, monkeypatch
):
    model = await UserCustomModel.create(
        user_id="user-1",
        display_name="To migrate",
        provider_id="openai",
        catalog_model_id="gpt-5.4-2026-03-05",
        model_name="gpt-5.4-2026-03-05",
        secret_arn="arn:aws:secretsmanager:eu-central-1:123:secret:to-migrate",
    )
    try:
        read_legacy = AsyncMockReturning("sk-migrated-key")
        delete_legacy = AsyncMockReturning(None)
        monkeypatch.setattr(migrate_command, "read_legacy_secret_value", read_legacy)
        monkeypatch.setattr(migrate_command, "delete_legacy_secret", delete_legacy)

        summary = await migrate_custom_model_secrets()

        assert summary == {"migrated": 1, "failed": 0, "would_migrate": 0}
        read_legacy.assert_called_once_with("arn:aws:secretsmanager:eu-central-1:123:secret:to-migrate")
        delete_legacy.assert_called_once_with("arn:aws:secretsmanager:eu-central-1:123:secret:to-migrate")

        refreshed = await UserCustomModel.find_by_id(model.id)
        assert refreshed.secret_arn is None
        assert refreshed.encrypted_key
        assert await get_custom_model_api_key(refreshed) == "sk-migrated-key"

        # Re-running is a no-op: the row no longer matches the migration query.
        read_legacy.calls.clear()
        second_summary = await migrate_custom_model_secrets()
        assert second_summary == {"migrated": 0, "failed": 0, "would_migrate": 0}
        read_legacy.assert_not_called()
    finally:
        await cleanup_models([model])


@pytest.mark.asyncio
async def test_migrate_custom_model_secrets_retries_cleanup_without_reencrypting(
    local_backend, monkeypatch
):
    """Simulates a prior run that saved encrypted_key but crashed before
    deleting the legacy secret: re-running must not re-encrypt, only finish
    the cleanup."""
    model = await UserCustomModel.create(
        user_id="user-1",
        display_name="Partially migrated",
        provider_id="openai",
        catalog_model_id="gpt-5.4-2026-03-05",
        model_name="gpt-5.4-2026-03-05",
        secret_arn="arn:aws:secretsmanager:eu-central-1:123:secret:partial",
    )
    try:
        model.encrypted_key = await create_custom_model_secret(
            user_id=model.user_id,
            provider_id=model.provider_id,
            model_id=model.id,
            api_key="sk-already-encrypted",
        )
        await model.save()

        read_legacy = AsyncMockReturning("should-not-be-used")
        delete_legacy = AsyncMockReturning(None)
        monkeypatch.setattr(migrate_command, "read_legacy_secret_value", read_legacy)
        monkeypatch.setattr(migrate_command, "delete_legacy_secret", delete_legacy)

        summary = await migrate_custom_model_secrets()

        assert summary == {"migrated": 1, "failed": 0, "would_migrate": 0}
        read_legacy.assert_not_called()
        delete_legacy.assert_called_once_with("arn:aws:secretsmanager:eu-central-1:123:secret:partial")

        refreshed = await UserCustomModel.find_by_id(model.id)
        assert refreshed.secret_arn is None
        assert await get_custom_model_api_key(refreshed) == "sk-already-encrypted"
    finally:
        await cleanup_models([model])


@pytest.mark.asyncio
async def test_migrate_custom_model_secrets_dry_run_does_not_mutate(monkeypatch):
    model = await UserCustomModel.create(
        user_id="user-1",
        display_name="Dry run",
        provider_id="openai",
        catalog_model_id="gpt-5.4-2026-03-05",
        model_name="gpt-5.4-2026-03-05",
        secret_arn="arn:aws:secretsmanager:eu-central-1:123:secret:dry-run",
    )
    try:
        read_legacy = AsyncMockReturning("sk-should-not-be-read-value")
        monkeypatch.setattr(migrate_command, "read_legacy_secret_value", read_legacy)

        summary = await migrate_custom_model_secrets(dry_run=True)

        assert summary["would_migrate"] >= 1
        assert summary["migrated"] == 0
        read_legacy.assert_not_called()

        refreshed = await UserCustomModel.find_by_id(model.id)
        assert refreshed.secret_arn == "arn:aws:secretsmanager:eu-central-1:123:secret:dry-run"
        assert not refreshed.encrypted_key
    finally:
        await cleanup_models([model])
