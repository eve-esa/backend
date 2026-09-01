"""Provider-neutral validation of OIDC access tokens.

One verification path serves every identity provider EVE runs against: Keycloak
in local compose, Cognito in the cloud. The application knows an issuer, an
expected audience and a JWT, never a product name.

Two rules shape the implementation and are worth stating before the code:

* the JOSE header is parsed locally before anything touches the network. A
  malformed token, an ``alg`` that is not an asymmetric one the provider
  advertises, or a missing ``kid`` is rejected without a single outbound
  request, so garbage sent to this process cannot be turned into traffic.
* the discovery document is trusted only after its own ``issuer`` field matches
  ``AUTH_ISSUER`` exactly. That check is what makes the ``jwks_uri`` it carries
  safe to fetch (OpenID Connect Discovery 1.0, section 4.3): without it a
  redirected or poisoned discovery response chooses the signing keys.

Failures are split in two on purpose. Anything the caller did wrong raises
``PermissionError`` and becomes a 401. Anything the provider did wrong, or the
network did, raises ``IdentityProviderUnavailable`` and becomes a 503: a
provider outage is not a bad credential and must not read like one.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional
from urllib.parse import urlsplit

import httpx
import jwt
from jwt import PyJWKSet
from jwt.exceptions import PyJWTError

from src.config import (
    AUTH_ALLOW_INSECURE_HTTP,
    AUTH_AUDIENCE,
    AUTH_DISCOVERY_URL,
    AUTH_ISSUER,
    AUTH_JWKS_CACHE_TTL_SECONDS,
)

logger = logging.getLogger(__name__)

# Every outbound call in this module. A provider that hangs must not hold a
# request worker: the caller gets a 503 in five seconds instead.
FETCH_TIMEOUT_SECONDS = 5.0

# Signature algorithms this service will consider at all, checked against the
# raw header before any network call. Symmetric algorithms are absent by
# construction, which is what closes the "HS256 signed with the public key"
# confusion, and ``none`` can never appear in a set that lists real ones.
# The intersection with what the provider advertises is applied afterwards.
ASYMMETRIC_ALGORITHMS = frozenset(
    {
        "RS256",
        "RS384",
        "RS512",
        "PS256",
        "PS384",
        "PS512",
        "ES256",
        "ES384",
        "ES512",
        "EdDSA",
    }
)

# Clock skew tolerated on exp/nbf/iat. Both providers stamp seconds.
CLOCK_SKEW_LEEWAY_SECONDS = 30

# A token signed with a key rotated in since the last fetch is legitimate, so an
# unknown kid triggers one refetch. The cooldown is what stops a stream of
# forged kids from turning into a stream of requests to the provider.
_UNKNOWN_KID_REFETCH_COOLDOWN_SECONDS = 30.0

_discovery_cache: Optional[tuple[float, dict[str, Any]]] = None
_jwks_cache: Optional[tuple[float, dict[str, Any]]] = None
_last_jwks_fetch_monotonic: float = 0.0
_fetch_lock = asyncio.Lock()


class IdentityProviderUnavailable(RuntimeError):
    """The identity provider could not be reached or answered nonsense.

    Distinct from ``PermissionError`` so dispatchers can answer 503 rather than
    presenting a provider outage as a rejected credential.
    """


def clear_oidc_caches() -> None:
    """Drop the cached discovery document and JWKS (tests, hot reload)."""
    global _discovery_cache, _jwks_cache, _last_jwks_fetch_monotonic
    _discovery_cache = None
    _jwks_cache = None
    _last_jwks_fetch_monotonic = 0.0


def _assert_transport_allowed(url: str, label: str) -> None:
    """Refuse plaintext http to the provider unless local dev opted in."""
    scheme = urlsplit(url).scheme.lower()
    if scheme == "https":
        return
    if AUTH_ALLOW_INSECURE_HTTP:
        return
    raise IdentityProviderUnavailable(
        f"{label} must use https (set AUTH_ALLOW_INSECURE_HTTP=true for local development)"
    )


async def _fetch_json(url: str, label: str) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=FETCH_TIMEOUT_SECONDS) as client:
            response = await client.get(url)
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError as exc:
        raise IdentityProviderUnavailable(f"Could not fetch {label}") from exc
    except ValueError as exc:
        raise IdentityProviderUnavailable(f"{label} is not JSON") from exc
    if not isinstance(payload, dict):
        raise IdentityProviderUnavailable(f"{label} is not a JSON object")
    return payload


def _discovery_url() -> str:
    if AUTH_DISCOVERY_URL:
        return AUTH_DISCOVERY_URL
    if not AUTH_ISSUER:
        raise IdentityProviderUnavailable("AUTH_ISSUER is not configured")
    return f"{AUTH_ISSUER.rstrip('/')}/.well-known/openid-configuration"


async def get_discovery_document() -> dict[str, Any]:
    """Return the provider's discovery document, cached for the JWKS TTL.

    The document is rejected unless it declares ``AUTH_ISSUER`` as its own
    issuer, so a wrong or intercepted document can never nominate the keys.
    """
    global _discovery_cache

    cached = _discovery_cache
    if cached is not None and time.monotonic() < cached[0]:
        return cached[1]

    url = _discovery_url()
    _assert_transport_allowed(url, "AUTH_DISCOVERY_URL")
    document = await _fetch_json(url, "the OIDC discovery document")

    declared_issuer = document.get("issuer")
    if declared_issuer != AUTH_ISSUER:
        raise IdentityProviderUnavailable(
            "Discovery document issuer does not match AUTH_ISSUER"
        )

    _discovery_cache = (time.monotonic() + AUTH_JWKS_CACHE_TTL_SECONDS, document)
    return document


async def _fetch_jwks() -> dict[str, Any]:
    document = await get_discovery_document()
    jwks_uri = document.get("jwks_uri")
    if not isinstance(jwks_uri, str) or not jwks_uri:
        raise IdentityProviderUnavailable("Discovery document has no jwks_uri")
    _assert_transport_allowed(jwks_uri, "jwks_uri")
    payload = await _fetch_json(jwks_uri, "the JWKS")

    try:
        key_set = PyJWKSet.from_dict(payload)
    except PyJWTError as exc:
        raise IdentityProviderUnavailable("JWKS contains no usable key") from exc

    keys: dict[str, Any] = {}
    for entry in key_set.keys:
        if entry.key_id:
            keys[entry.key_id] = entry
    if not keys:
        raise IdentityProviderUnavailable("JWKS contains no key with a kid")
    return keys


async def _get_signing_key(kid: str) -> Any:
    """Resolve a ``kid`` to a signing key, refetching once for a new key.

    The refetch is what keeps a provider key rotation from locking every user
    out until the TTL lapses; the cooldown is what keeps forged kids from
    turning that recovery into an amplifier.
    """
    global _jwks_cache, _last_jwks_fetch_monotonic

    cached = _jwks_cache
    if cached is not None and time.monotonic() < cached[0] and kid in cached[1]:
        return cached[1][kid]

    async with _fetch_lock:
        cached = _jwks_cache
        now = time.monotonic()
        if cached is not None and now < cached[0] and kid in cached[1]:
            return cached[1][kid]

        fresh_cache_missing_kid = cached is not None and now < cached[0]
        if (
            fresh_cache_missing_kid
            and now - _last_jwks_fetch_monotonic
            < _UNKNOWN_KID_REFETCH_COOLDOWN_SECONDS
        ):
            raise PermissionError("Token signing key is not known to the provider")

        keys = await _fetch_jwks()
        _last_jwks_fetch_monotonic = time.monotonic()
        _jwks_cache = (
            _last_jwks_fetch_monotonic + AUTH_JWKS_CACHE_TTL_SECONDS,
            keys,
        )

        key = keys.get(kid)
        if key is None:
            raise PermissionError("Token signing key is not known to the provider")
        return key


def parse_token_header(token: str) -> dict[str, Any]:
    """Read and vet the JOSE header without touching the network.

    Raises ``PermissionError`` on a malformed token, a symmetric or ``none``
    algorithm, or a missing ``kid``. Callers rely on this running first: a
    rejected token here costs zero outbound requests.
    """
    if not isinstance(token, str) or not token:
        raise PermissionError("Invalid token")
    try:
        header = jwt.get_unverified_header(token)
    except PyJWTError as exc:
        raise PermissionError("Invalid token") from exc

    algorithm = header.get("alg")
    if algorithm not in ASYMMETRIC_ALGORITHMS:
        raise PermissionError("Unsupported token signature algorithm")

    kid = header.get("kid")
    if not isinstance(kid, str) or not kid:
        raise PermissionError("Token has no key id")

    return header


def _assert_audience(claims: dict[str, Any]) -> None:
    """Apply the one audience rule that fits both providers.

    Keycloak puts the client in ``aud`` (the realm ships an audience mapper for
    it, otherwise the only audience is ``account`` and this refuses). Cognito
    access tokens carry no ``aud`` at all and name the client in ``client_id``.
    """
    expected = AUTH_AUDIENCE
    if not expected:
        raise PermissionError("Token audience cannot be verified")

    audience = claims.get("aud")
    if audience is not None:
        values = [audience] if isinstance(audience, str) else list(audience)
        if expected not in values:
            raise PermissionError("Token audience mismatch")
        return

    if claims.get("client_id") != expected:
        raise PermissionError("Token audience mismatch")


async def verify_access_token(token: str) -> dict[str, Any]:
    """Verify an OIDC access token and return its claims.

    Raises ``PermissionError`` when the token is not acceptable and
    ``IdentityProviderUnavailable`` when the provider could not be consulted.
    """
    header = parse_token_header(token)
    algorithm: str = header["alg"]

    document = await get_discovery_document()
    advertised = document.get("id_token_signing_alg_values_supported")
    if isinstance(advertised, list) and algorithm not in advertised:
        raise PermissionError("Unsupported token signature algorithm")

    signing_key = await _get_signing_key(header["kid"])

    try:
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=[algorithm],
            issuer=AUTH_ISSUER,
            leeway=CLOCK_SKEW_LEEWAY_SECONDS,
            # The audience rule below is provider-shaped and cannot be expressed
            # as a single ``audience=`` argument, so PyJWT's own check is off and
            # _assert_audience owns it.
            options={
                "verify_aud": False,
                "require": ["exp", "iat", "sub"],
            },
        )
    except PyJWTError as exc:
        raise PermissionError("Invalid token") from exc

    token_use = claims.get("token_use")
    if token_use is not None and token_use != "access":
        raise PermissionError("Token is not an access token")

    _assert_audience(claims)

    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject:
        raise PermissionError("Token has no subject")

    return claims


async def fetch_userinfo(token: str) -> dict[str, Any]:
    """Read the profile the provider will vouch for, using the caller's token.

    Cognito access tokens carry no email at all, so this is the only place the
    address and its verification state can come from on a first sign-in.
    """
    document = await get_discovery_document()
    endpoint = document.get("userinfo_endpoint")
    if not isinstance(endpoint, str) or not endpoint:
        raise IdentityProviderUnavailable("Discovery document has no userinfo_endpoint")
    _assert_transport_allowed(endpoint, "userinfo_endpoint")

    try:
        async with httpx.AsyncClient(timeout=FETCH_TIMEOUT_SECONDS) as client:
            response = await client.get(
                endpoint, headers={"Authorization": f"Bearer {token}"}
            )
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in (401, 403):
            raise PermissionError("Provider rejected the token at userinfo") from exc
        raise IdentityProviderUnavailable("Could not fetch userinfo") from exc
    except httpx.HTTPError as exc:
        raise IdentityProviderUnavailable("Could not fetch userinfo") from exc
    except ValueError as exc:
        raise IdentityProviderUnavailable("userinfo is not JSON") from exc

    if not isinstance(payload, dict):
        raise IdentityProviderUnavailable("userinfo is not a JSON object")
    return payload
