"""Envelope encryption for user custom model API keys.

Each API key is AES-256-GCM encrypted under a fresh, random 256-bit data
encryption key (DEK). The DEK itself is "wrapped" (encrypted) by a longer-lived
key-encryption key so only the wrapped DEK and the ciphertext need to be
stored; this is the standard envelope-encryption construction (see
https://docs.aws.amazon.com/kms/latest/developerguide/kms-cryptography.html#enveloping).

Two backends can perform the wrap/unwrap step, selected by config:

* ``kms`` -- the DEK is wrapped with a single AWS KMS CMK
  (``CUSTOM_MODEL_KMS_KEY_ID``) via the KMS ``Encrypt``/``Decrypt`` APIs.
  Keys are well under the 4096-byte limit for direct KMS encryption, so no
  ``GenerateDataKey`` round trip is needed
  (https://docs.aws.amazon.com/kms/latest/APIReference/API_Encrypt.html).
  An ``EncryptionContext`` binds the wrap operation to the owning
  user/provider/model; KMS refuses to decrypt under a mismatched context
  (https://docs.aws.amazon.com/kms/latest/APIReference/API_Decrypt.html).
* ``local`` -- the DEK is wrapped with a static 256-bit KEK from config
  (``BYOK_LOCAL_KEK``), also via AES-256-GCM. This lets BYOK work in local
  compose / CI with no AWS account at all.

Both backends additionally authenticate the *data* ciphertext with the same
context, passed as AES-GCM associated data (AAD), so a blob from one
user/provider/model row can never be decrypted under another's context, even
if an attacker could swap ciphertexts between rows in the database.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from typing import Mapping, Optional

import boto3
from botocore.exceptions import ClientError
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from src.config import AWS_ENDPOINT_URL, AWS_REGION, BYOK_LOCAL_KEK, CUSTOM_MODEL_KMS_KEY_ID

logger = logging.getLogger(__name__)

_BLOB_VERSION = 1
_ALG = "AES-256-GCM"
_NONCE_BYTES = 12
_CONTEXT_KEYS = ("user_id", "provider_id", "model_id")


class CipherConfigError(RuntimeError):
    """Neither a KMS key nor a local KEK is configured."""


class CipherContextError(ValueError):
    """A blob failed to decrypt: tampered, wrong context, or malformed."""


def _normalize_context(context: Mapping[str, str]) -> dict:
    missing = [key for key in _CONTEXT_KEYS if not context.get(key)]
    if missing:
        raise ValueError(f"Cipher context missing required keys: {missing}")
    return {key: str(context[key]) for key in _CONTEXT_KEYS}


def _context_aad(normalized_context: dict) -> bytes:
    # sort_keys makes this deterministic regardless of dict insertion order,
    # so the same logical context always produces the same AAD bytes.
    return json.dumps(normalized_context, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )


def _select_backend() -> str:
    if CUSTOM_MODEL_KMS_KEY_ID:
        return "kms"
    if BYOK_LOCAL_KEK:
        return "local"
    raise CipherConfigError(
        "No BYOK key-wrapping backend configured: set CUSTOM_MODEL_KMS_KEY_ID "
        "(KMS, cloud) or BYOK_LOCAL_KEK (static local KEK, compose/dev) before "
        "storing or reading custom model keys."
    )


def _kms_client():
    kwargs = {"region_name": AWS_REGION}
    if AWS_ENDPOINT_URL:
        kwargs["endpoint_url"] = AWS_ENDPOINT_URL
    return boto3.client("kms", **kwargs)


def _decode_local_kek() -> bytes:
    raw = (BYOK_LOCAL_KEK or "").strip()
    if not raw:
        raise CipherConfigError("BYOK_LOCAL_KEK is not configured")
    for decoder in (
        lambda v: base64.b64decode(v, validate=True),
        bytes.fromhex,
    ):
        try:
            decoded = decoder(raw)
        except Exception:
            continue
        if len(decoded) == 32:
            return decoded
    raise CipherConfigError(
        "BYOK_LOCAL_KEK must decode to exactly 32 bytes (base64 or hex encoded)"
    )


def _kms_wrap(dek: bytes, encryption_context: dict) -> tuple[bytes, str]:
    key_id = CUSTOM_MODEL_KMS_KEY_ID
    try:
        response = _kms_client().encrypt(
            KeyId=key_id,
            Plaintext=dek,
            EncryptionContext=encryption_context,
        )
    except ClientError:
        logger.exception("KMS Encrypt failed while wrapping a custom model DEK")
        raise
    return response["CiphertextBlob"], response.get("KeyId", key_id)


def _kms_unwrap(
    wrapped_dek: bytes, encryption_context: dict, key_ref: Optional[str]
) -> bytes:
    key_id = key_ref or CUSTOM_MODEL_KMS_KEY_ID
    try:
        response = _kms_client().decrypt(
            KeyId=key_id,
            CiphertextBlob=wrapped_dek,
            EncryptionContext=encryption_context,
        )
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code == "InvalidCiphertextException":
            raise CipherContextError(
                "Custom model secret failed to decrypt under the given context"
            ) from exc
        logger.exception("KMS Decrypt failed while unwrapping a custom model DEK")
        raise
    return response["Plaintext"]


def _local_wrap(dek: bytes, aad: bytes) -> tuple[bytes, str]:
    kek = _decode_local_kek()
    nonce = os.urandom(_NONCE_BYTES)
    ciphertext = AESGCM(kek).encrypt(nonce, dek, aad)
    # Not needed to unwrap (the KEK is chosen by config, not by key_ref); kept
    # only as a diagnostic fingerprint so a KEK rotation shows up clearly in
    # decrypt errors/logs instead of silently trying the wrong key.
    key_ref = hashlib.sha256(kek).hexdigest()[:16]
    return nonce + ciphertext, key_ref


def _local_unwrap(wrapped_dek: bytes, aad: bytes) -> bytes:
    kek = _decode_local_kek()
    nonce, ciphertext = wrapped_dek[:_NONCE_BYTES], wrapped_dek[_NONCE_BYTES:]
    try:
        return AESGCM(kek).decrypt(nonce, ciphertext, aad)
    except InvalidTag as exc:
        raise CipherContextError(
            "Custom model secret failed to decrypt under the given context"
        ) from exc


class SecretCipher:
    """Envelope-encrypts/decrypts a single API key string at a time."""

    def encrypt(self, plaintext: str, context: Mapping[str, str]) -> str:
        """Encrypt ``plaintext``, bound to ``context``. Returns a storable blob."""
        if not plaintext:
            raise ValueError("Cannot encrypt an empty API key")

        normalized = _normalize_context(context)
        aad = _context_aad(normalized)
        backend = _select_backend()

        dek = bytearray(AESGCM.generate_key(bit_length=256))
        try:
            data_nonce = os.urandom(_NONCE_BYTES)
            ciphertext = AESGCM(bytes(dek)).encrypt(
                data_nonce, plaintext.encode("utf-8"), aad
            )

            if backend == "kms":
                wrapped_dek, key_ref = _kms_wrap(bytes(dek), normalized)
            else:
                wrapped_dek, key_ref = _local_wrap(bytes(dek), aad)

            blob = {
                "v": _BLOB_VERSION,
                "alg": _ALG,
                "nonce": base64.b64encode(data_nonce).decode("ascii"),
                "ct": base64.b64encode(ciphertext).decode("ascii"),
                "wrapped_dek": base64.b64encode(wrapped_dek).decode("ascii"),
                "backend": backend,
                "key_ref": key_ref,
            }
            payload = json.dumps(blob, separators=(",", ":")).encode("utf-8")
            return base64.b64encode(payload).decode("ascii")
        finally:
            for i in range(len(dek)):
                dek[i] = 0

    def decrypt(self, blob: str, context: Mapping[str, str]) -> str:
        """Decrypt a blob produced by :meth:`encrypt`. Fails closed on wrong context."""
        normalized = _normalize_context(context)
        aad = _context_aad(normalized)

        try:
            parsed = json.loads(base64.b64decode(blob, validate=True))
            version = parsed["v"]
            alg = parsed["alg"]
            backend = parsed["backend"]
            data_nonce = base64.b64decode(parsed["nonce"])
            ciphertext = base64.b64decode(parsed["ct"])
            wrapped_dek = base64.b64decode(parsed["wrapped_dek"])
            key_ref = parsed.get("key_ref")
        except Exception as exc:
            raise CipherContextError("Malformed custom model secret blob") from exc

        if version != _BLOB_VERSION or alg != _ALG:
            raise CipherContextError(
                f"Unsupported custom model secret blob (v={version!r}, alg={alg!r})"
            )

        dek = bytearray(32)
        try:
            if backend == "kms":
                dek[:] = _kms_unwrap(wrapped_dek, normalized, key_ref)
            elif backend == "local":
                dek[:] = _local_unwrap(wrapped_dek, aad)
            else:
                raise CipherContextError(f"Unknown cipher backend {backend!r}")

            try:
                plaintext = AESGCM(bytes(dek)).decrypt(data_nonce, ciphertext, aad)
            except InvalidTag as exc:
                raise CipherContextError(
                    "Custom model secret failed to decrypt under the given context"
                ) from exc
            return plaintext.decode("utf-8")
        finally:
            for i in range(len(dek)):
                dek[i] = 0
