"""Storage helpers for user custom model API keys.

API keys live as an envelope-encrypted blob (``UserCustomModel.encrypted_key``)
on the existing DocumentDB row instead of one Secrets Manager secret per user
key (the previous scheme costs ~$0.40/secret/month and doesn't scale).
Encryption itself is delegated to :mod:`src.services.custom_model_cipher`;
this module wires the cipher to the model row and keeps the 60s in-process
plaintext cache the Secrets Manager version used.

Rows created before this change only have ``secret_arn`` set. ``get`` falls
back to reading the legacy Secrets Manager secret, and ``delete`` still cleans
it up, so un-migrated rows keep working until
``python -m src.commands.migrate_custom_model_secrets`` sweeps them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

import boto3
from botocore.exceptions import ClientError

from src.config import AWS_ENDPOINT_URL, AWS_REGION
from src.database.models.user_custom_model import UserCustomModel
from src.services.custom_model_cipher import SecretCipher

logger = logging.getLogger(__name__)

_SECRET_CACHE_TTL_SECONDS = 60.0
_secret_cache: dict[str, tuple[str, float]] = {}

_cipher = SecretCipher()


def _sm_client():
    kwargs = {"region_name": AWS_REGION}
    if AWS_ENDPOINT_URL:
        kwargs["endpoint_url"] = AWS_ENDPOINT_URL
    return boto3.client("secretsmanager", **kwargs)


def _decode_legacy_secret_payload(secret_string: str) -> str:
    try:
        payload = json.loads(secret_string)
    except json.JSONDecodeError as exc:
        raise ValueError("Custom model secret payload is not valid JSON") from exc
    api_key = payload.get("api_key")
    if not api_key or not str(api_key).strip():
        raise ValueError("Custom model secret is missing api_key")
    return str(api_key).strip()


def _cipher_context(*, user_id: str, provider_id: str, model_id: str) -> dict:
    return {"user_id": user_id, "provider_id": provider_id, "model_id": model_id}


async def create_custom_model_secret(
    *, user_id: str, provider_id: str, model_id: str, api_key: str
) -> str:
    """Encrypt an API key into the blob to store on the model row."""
    context = _cipher_context(user_id=user_id, provider_id=provider_id, model_id=model_id)
    blob = await asyncio.to_thread(_cipher.encrypt, api_key, context)
    logger.info("Encrypted custom model secret for model_id=%s", model_id)
    return blob


async def update_custom_model_secret(
    *, user_id: str, provider_id: str, model_id: str, api_key: str
) -> str:
    """Rotate the API key, returning a fresh encrypted blob."""
    context = _cipher_context(user_id=user_id, provider_id=provider_id, model_id=model_id)
    blob = await asyncio.to_thread(_cipher.encrypt, api_key, context)
    _secret_cache.pop(model_id, None)
    logger.info("Rotated custom model secret model_id=%s", model_id)
    return blob


async def get_custom_model_api_key(model: UserCustomModel) -> str:
    """Fetch (and cache) the plaintext API key for a custom model row."""
    now = time.monotonic()
    cached = _secret_cache.get(model.id)
    if cached and now - cached[1] < _SECRET_CACHE_TTL_SECONDS:
        return cached[0]

    if model.encrypted_key:
        context = _cipher_context(
            user_id=model.user_id, provider_id=model.provider_id, model_id=model.id
        )
        api_key = await asyncio.to_thread(_cipher.decrypt, model.encrypted_key, context)
    elif model.secret_arn:
        api_key = await read_legacy_secret_value(model.secret_arn)
    else:
        raise ValueError("Custom model has no stored credentials")

    _secret_cache[model.id] = (api_key, now)
    return api_key


async def delete_custom_model_secret(model: UserCustomModel) -> None:
    """Drop the cached key and, for legacy rows, the Secrets Manager entry.

    New rows have no external secret to delete; only the in-process cache is
    cleared for them.
    """
    _secret_cache.pop(model.id, None)
    if model.secret_arn:
        await delete_legacy_secret(model.secret_arn)
    logger.info("Deleted custom model secret model_id=%s", model.id)


async def read_legacy_secret_value(secret_arn: str) -> str:
    """Read a pre-migration Secrets Manager secret's plaintext API key."""

    def _get() -> str:
        response = _sm_client().get_secret_value(SecretId=secret_arn)
        secret_string = response.get("SecretString")
        if not secret_string:
            raise ValueError("Custom model secret has no SecretString payload")
        return _decode_legacy_secret_payload(secret_string)

    return await asyncio.to_thread(_get)


async def delete_legacy_secret(secret_arn: str) -> None:
    """Delete a legacy Secrets Manager secret. Missing secrets are ignored."""

    def _delete() -> None:
        try:
            _sm_client().delete_secret(SecretId=secret_arn, ForceDeleteWithoutRecovery=True)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in {"ResourceNotFoundException", "InvalidRequestException"}:
                return
            raise

    await asyncio.to_thread(_delete)


def clear_secret_cache_for_tests() -> None:
    """Test helper to reset the in-process secret cache."""
    _secret_cache.clear()
