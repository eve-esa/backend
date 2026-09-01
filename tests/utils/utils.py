"""Authenticated-caller fixtures for the suite.

Every test that calls an endpoint needs a token the application will accept, and
the application now accepts exactly one kind: an access token signed by the
configured identity provider. So the suite runs a provider of its own, in
process: one RSA key, minted tokens, and the OIDC caches pre-seeded with the
matching JWKS. Nothing here reaches the network, and a test that somehow tried
would hit AUTH_DISCOVERY_URL, which CI points at an unroutable address.

The factory also inserts the ``external_identities`` row. Without it every
authenticated test would take the first-sign-in path and try to call ``userinfo``
at a provider that is not there.
"""

import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Tuple

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt import PyJWK
from jwt.algorithms import RSAAlgorithm

from src.config import AUTH_CLIENT_ID, AUTH_ISSUER
from src.database.models.external_identity import ExternalIdentity
from src.database.models.user import User
from src.database.mongo import async_mongo_manager
from src.services import oidc

TEST_KEY_ID = "eve-test-signing-key"
TEST_JWKS_URI = "https://test-issuer.invalid/protocol/openid-connect/certs"
TEST_USERINFO_ENDPOINT = "https://test-issuer.invalid/protocol/openid-connect/userinfo"
# Prefix on every subject this module mints, so the conftest teardown can find
# and drop the identity rows the suite leaves behind.
TEST_SUBJECT_PREFIX = "test-subject-"

_CACHE_TTL_SECONDS = 3600.0

_private_key: Optional[rsa.RSAPrivateKey] = None


def test_private_key() -> rsa.RSAPrivateKey:
    """The suite's one signing key. Generated once: 2048-bit RSA is not cheap."""
    global _private_key
    if _private_key is None:
        _private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return _private_key


def test_public_jwk(kid: str = TEST_KEY_ID) -> dict[str, Any]:
    """The public half of the suite's key, as a JWKS entry."""
    jwk = RSAAlgorithm.to_jwk(test_private_key().public_key(), as_dict=True)
    jwk.update({"kid": kid, "use": "sig", "alg": "RS256"})
    return jwk


def install_test_jwks(
    *,
    issuer: Optional[str] = None,
    kid: str = TEST_KEY_ID,
    algorithms: Optional[list[str]] = None,
) -> None:
    """Seed the OIDC caches so token verification never leaves the process.

    Writes the module globals directly rather than mocking a transport: the
    caches are exactly what a successful fetch would have produced, so the code
    under test runs its real verification path from the signature onwards.
    """
    expiry = time.monotonic() + _CACHE_TTL_SECONDS
    oidc._discovery_cache = (
        expiry,
        {
            "issuer": issuer or AUTH_ISSUER,
            "jwks_uri": TEST_JWKS_URI,
            "userinfo_endpoint": TEST_USERINFO_ENDPOINT,
            "id_token_signing_alg_values_supported": algorithms or ["RS256"],
        },
    )
    oidc._jwks_cache = (expiry, {kid: PyJWK.from_dict(test_public_jwk(kid))})
    oidc._last_jwks_fetch_monotonic = time.monotonic()


def mint_access_token(
    subject: str,
    *,
    issuer: Optional[str] = None,
    audience: Any = None,
    expires_in: int = 3600,
    key_id: str = TEST_KEY_ID,
    algorithm: str = "RS256",
    key: Any = None,
    extra_claims: Optional[dict[str, Any]] = None,
    omit_claims: tuple[str, ...] = (),
) -> str:
    """Mint an access token the way the provider would.

    The keyword arguments exist so the validation tests can produce a token that
    is wrong in exactly one way.
    """
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "iss": AUTH_ISSUER if issuer is None else issuer,
        "sub": subject,
        "aud": [AUTH_CLIENT_ID] if audience is None else audience,
        "iat": now,
        "exp": now + timedelta(seconds=expires_in),
        "token_use": "access",
    }
    if extra_claims:
        payload.update(extra_claims)
    for claim in omit_claims:
        payload.pop(claim, None)

    return jwt.encode(
        payload,
        key if key is not None else test_private_key(),
        algorithm=algorithm,
        headers={"kid": key_id},
    )


async def create_test_user_and_token(
    *,
    email: Optional[str] = None,
    first_name: str = "Test",
    last_name: str = "User",
) -> Tuple[User, str]:
    """Return a persisted user and an access token that resolves to them.

    The user is persisted in MongoDB so the API calls FastAPI executes can fetch
    it through the regular data-access helpers (``User.find_by_id`` et al.), and
    an ``external_identities`` row binds the token's subject to it, which is what
    keeps the resolver on the cache/lookup path instead of first sign-in.
    """

    email = email or f"{uuid.uuid4().hex[:8]}@example.com"

    if async_mongo_manager.database is None:
        await async_mongo_manager.connect()

    existing = await User.find_one({"email": email})
    # this should never happen, but just in case
    if existing:
        await existing.delete()

    user = await User.create(
        email=email,
        first_name=first_name,
        last_name=last_name,
    )

    subject = f"{TEST_SUBJECT_PREFIX}{uuid.uuid4().hex}"
    await ExternalIdentity.create(
        user_id=user.id,
        issuer=AUTH_ISSUER,
        subject=subject,
        email=email,
    )

    install_test_jwks()
    return user, mint_access_token(subject)
