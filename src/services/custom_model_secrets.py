"""AWS Secrets Manager helpers for user custom model API keys."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from src.config import (
    AWS_ENDPOINT_URL,
    AWS_REGION,
    CUSTOM_MODEL_SECRET_PREFIX,
)

logger = logging.getLogger(__name__)

_SECRET_CACHE_TTL_SECONDS = 60.0
_secret_cache: dict[str, tuple[str, float]] = {}


def _client():
    kwargs = {"region_name": AWS_REGION}
    if AWS_ENDPOINT_URL:
        kwargs["endpoint_url"] = AWS_ENDPOINT_URL
    return boto3.client("secretsmanager", **kwargs)


def _secret_name(user_id: str, model_id: str) -> str:
    return f"{CUSTOM_MODEL_SECRET_PREFIX}/users/{user_id}/models/{model_id}"


def _encode_secret_payload(api_key: str) -> str:
    return json.dumps({"api_key": api_key})


def _decode_secret_payload(secret_string: str) -> str:
    try:
        payload = json.loads(secret_string)
    except json.JSONDecodeError as exc:
        raise ValueError("Custom model secret payload is not valid JSON") from exc
    api_key = payload.get("api_key")
    if not api_key or not str(api_key).strip():
        raise ValueError("Custom model secret is missing api_key")
    return str(api_key).strip()


async def create_custom_model_secret(
    *, user_id: str, model_id: str, api_key: str
) -> str:
    """Create a Secrets Manager secret and return its ARN."""

    def _create() -> str:
        client = _client()
        name = _secret_name(user_id, model_id)
        response = client.create_secret(
            Name=name,
            SecretString=_encode_secret_payload(api_key),
            Tags=[
                {"Key": "user_id", "Value": user_id},
                {"Key": "model_id", "Value": model_id},
                {"Key": "service", "Value": "eve-custom-models"},
            ],
        )
        return response["ARN"]

    arn = await asyncio.to_thread(_create)
    logger.info("Created custom model secret for model_id=%s", model_id)
    return arn


async def update_custom_model_secret(*, secret_arn: str, api_key: str) -> None:
    """Rotate the API key stored in an existing secret."""

    def _update() -> None:
        _client().put_secret_value(
            SecretARN=secret_arn,
            SecretString=_encode_secret_payload(api_key),
        )

    await asyncio.to_thread(_update)
    _secret_cache.pop(secret_arn, None)
    logger.info("Updated custom model secret arn=%s", secret_arn)


async def get_custom_model_api_key(secret_arn: str) -> str:
    """Fetch the API key for a custom model secret ARN."""

    now = time.monotonic()
    cached = _secret_cache.get(secret_arn)
    if cached and now - cached[1] < _SECRET_CACHE_TTL_SECONDS:
        return cached[0]

    def _get() -> str:
        response = _client().get_secret_value(SecretId=secret_arn)
        secret_string = response.get("SecretString")
        if not secret_string:
            raise ValueError("Custom model secret has no SecretString payload")
        return _decode_secret_payload(secret_string)

    api_key = await asyncio.to_thread(_get)
    _secret_cache[secret_arn] = (api_key, now)
    return api_key


async def delete_custom_model_secret(secret_arn: str) -> None:
    """Delete a custom model secret. Missing secrets are ignored."""

    def _delete() -> None:
        try:
            _client().delete_secret(SecretId=secret_arn, ForceDeleteWithoutRecovery=True)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in {"ResourceNotFoundException", "InvalidRequestException"}:
                return
            raise

    await asyncio.to_thread(_delete)
    _secret_cache.pop(secret_arn, None)
    logger.info("Deleted custom model secret arn=%s", secret_arn)


def clear_secret_cache_for_tests() -> None:
    """Test helper to reset the in-process secret cache."""
    _secret_cache.clear()
