"""Token validation, against a provider that exists only in this process.

The suite runs its own identity provider: one RSA key, a discovery document and
a JWKS served through a mocked httpx transport. Everything from the JOSE header
to the claim checks is the real code path, and the transport counts requests so
a test can assert that a rejection cost nothing.

The two provider shapes are both covered on purpose. Keycloak puts the client in
``aud`` and only because the realm ships an audience mapper; Cognito sends no
``aud`` at all and names the client in ``client_id``. A check that passes one and
fails the other is the failure mode this file exists to catch, which is why
``aud: ["account"]`` (Keycloak with the mapper missing) has a test of its own.
"""

import base64
import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from src.middlewares.auth import resolve_principal_from_bearer_token
from src.services import oidc
from src.services.oidc import IdentityProviderUnavailable, verify_access_token
from tests.utils.cleaner import cleanup_models
from tests.utils.utils import (
    TEST_KEY_ID,
    create_test_user_and_token,
    mint_access_token,
    test_private_key,
    test_public_jwk,
)

ISSUER = "https://idp.test.invalid/realms/eve"
DISCOVERY_URL = "https://idp.test.invalid/realms/eve/.well-known/openid-configuration"
JWKS_URI = "https://idp.test.invalid/realms/eve/protocol/openid-connect/certs"
USERINFO_URL = "https://idp.test.invalid/realms/eve/protocol/openid-connect/userinfo"
AUDIENCE = "eve-frontend-test"

# Captured at import, before any test patches httpx.AsyncClient. The patch is
# global, so a factory that looked the class up at call time would recurse into
# itself instead of building a client.
REAL_ASYNC_CLIENT = httpx.AsyncClient


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _segment(payload: dict) -> str:
    return _b64(json.dumps(payload, separators=(",", ":")).encode())


class FakeProvider:
    """Serves discovery and JWKS, and remembers how often it was asked."""

    def __init__(self):
        self.requests: list[str] = []
        self.discovery = {
            "issuer": ISSUER,
            "jwks_uri": JWKS_URI,
            "userinfo_endpoint": USERINFO_URL,
            "id_token_signing_alg_values_supported": ["RS256", "ES256"],
        }
        self.jwks = {"keys": [test_public_jwk()]}

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        if url == DISCOVERY_URL:
            return httpx.Response(200, json=self.discovery)
        if url == JWKS_URI:
            return httpx.Response(200, json=self.jwks)
        return httpx.Response(404, json={"error": "not found"})

    @property
    def request_count(self) -> int:
        return len(self.requests)


@pytest.fixture
def provider(monkeypatch):
    """Point the OIDC service at the in-process provider."""
    fake = FakeProvider()

    def client_factory(*args, **kwargs):
        return REAL_ASYNC_CLIENT(
            transport=httpx.MockTransport(fake.handler),
            timeout=kwargs.get("timeout", 5.0),
        )

    monkeypatch.setattr(oidc.httpx, "AsyncClient", client_factory)
    monkeypatch.setattr(oidc, "AUTH_ISSUER", ISSUER)
    monkeypatch.setattr(oidc, "AUTH_DISCOVERY_URL", DISCOVERY_URL)
    monkeypatch.setattr(oidc, "AUTH_AUDIENCE", AUDIENCE)
    return fake


def keycloak_token(**overrides) -> str:
    """A token shaped the way Keycloak issues them: the client is in ``aud``."""
    params = {"issuer": ISSUER, "audience": [AUDIENCE, "account"], "subject": "kc-subject"}
    params.update(overrides)
    subject = params.pop("subject")
    return mint_access_token(subject, **params)


def cognito_token(**overrides) -> str:
    """A token shaped the way Cognito issues them: no ``aud``, a ``client_id``."""
    params = {
        "issuer": ISSUER,
        "audience": None,
        "subject": "cognito-subject",
        "extra_claims": {"client_id": AUDIENCE},
        "omit_claims": ("aud",),
    }
    params.update(overrides)
    subject = params.pop("subject")
    return mint_access_token(subject, **params)


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_valid_keycloak_token_is_accepted(provider):
    claims = await verify_access_token(keycloak_token())
    assert claims["sub"] == "kc-subject"
    assert claims["iss"] == ISSUER


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_valid_cognito_shaped_token_is_accepted(provider):
    """No aud at all: the client_id claim carries the audience instead."""
    claims = await verify_access_token(cognito_token())
    assert claims["sub"] == "cognito-subject"
    assert "aud" not in claims


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_expired_token_is_rejected(provider):
    with pytest.raises(PermissionError):
        await verify_access_token(keycloak_token(expires_in=-3600))


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_wrong_issuer_is_rejected(provider):
    with pytest.raises(PermissionError):
        await verify_access_token(keycloak_token(issuer="https://evil.invalid/realms/eve"))


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_wrong_audience_is_rejected(provider):
    with pytest.raises(PermissionError):
        await verify_access_token(keycloak_token(audience=["some-other-client"]))


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_account_only_audience_is_rejected(provider):
    """Keycloak without the audience protocol mapper.

    Its access tokens then carry `aud: ["account"]` and nothing else. Accepting
    that would accept any token from the realm, issued to any client.
    """
    with pytest.raises(PermissionError):
        await verify_access_token(keycloak_token(audience=["account"]))


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_cognito_shaped_token_with_wrong_client_id_is_rejected(provider):
    with pytest.raises(PermissionError):
        await verify_access_token(
            cognito_token(extra_claims={"client_id": "another-app"})
        )


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_token_use_other_than_access_is_rejected(provider):
    """A Cognito id token reaching the API is not an access token."""
    with pytest.raises(PermissionError):
        await verify_access_token(keycloak_token(extra_claims={"token_use": "id"}))


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_keycloak_id_token_is_rejected(provider):
    """The one an audience check alone cannot catch.

    A Keycloak id token carries the client in aud exactly like its access token,
    and Keycloak stamps no token_use to tell them apart. The browser is holding
    one of these, so "it verifies" is not good enough: what separates them is
    scope, which every Keycloak access token has and no id token does.
    """
    with pytest.raises(PermissionError):
        await verify_access_token(
            keycloak_token(omit_claims=("token_use",), extra_claims={"typ": "ID"})
        )


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_keycloak_access_token_without_token_use_is_accepted(provider):
    """The real shape: no token_use anywhere, scope always present."""
    claims = await verify_access_token(
        keycloak_token(
            omit_claims=("token_use",),
            extra_claims={"scope": "openid profile email"},
        )
    )
    assert claims["sub"] == "kc-subject"
    assert "token_use" not in claims


@pytest.mark.no_db
@pytest.mark.asyncio
@pytest.mark.parametrize("bad_audience", [42, {"aud": "eve"}, 3.5, True])
async def test_malformed_audience_is_a_permission_error(provider, bad_audience):
    """A string or an array of strings, per RFC 7519, and nothing else.

    Coercing the value instead of checking it raises TypeError, which none of
    the dispatchers handle: it would leave as a 500 carrying the raw text.
    """
    with pytest.raises(PermissionError):
        await verify_access_token(keycloak_token(audience=bad_audience))


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_missing_subject_is_rejected(provider):
    with pytest.raises(PermissionError):
        await verify_access_token(keycloak_token(omit_claims=("sub",)))


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_unknown_kid_is_rejected_after_one_refetch(provider):
    """A kid the provider does not publish is a forged or rotated-out key."""
    with pytest.raises(PermissionError):
        await verify_access_token(keycloak_token(key_id="not-a-real-kid"))
    assert JWKS_URI in provider.requests


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_token_signed_by_a_foreign_key_is_rejected(provider):
    """Right kid, wrong key: only the signature check can catch this one."""
    foreign_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    with pytest.raises(PermissionError):
        await verify_access_token(keycloak_token(key=foreign_key))


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_hs256_signed_with_the_public_key_is_rejected(provider):
    """The classic algorithm-confusion forgery.

    The public key is public, so if HS256 were ever accepted anybody could mint
    a valid token. Built by hand because PyJWT itself refuses to use an
    asymmetric key as an HMAC secret: the forgery has to come from outside the
    library for the test to mean anything. The header check refuses symmetric
    algorithms outright, and does so before any network call.
    """
    public_pem = test_private_key().public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    expires = int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp())
    signing_input = ".".join(
        [
            _segment({"alg": "HS256", "typ": "JWT", "kid": TEST_KEY_ID}),
            _segment(
                {
                    "iss": ISSUER,
                    "sub": "attacker",
                    "aud": [AUDIENCE],
                    "iat": expires - 3600,
                    "exp": expires,
                }
            ),
        ]
    )
    signature = hmac.new(public_pem, signing_input.encode(), hashlib.sha256).digest()
    forged = f"{signing_input}.{_b64(signature)}"

    with pytest.raises(PermissionError):
        await verify_access_token(forged)
    assert provider.request_count == 0


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_alg_none_is_rejected(provider):
    """An unsigned token, hand-built because no library will mint one."""
    unsigned = ".".join(
        [
            _segment({"alg": "none", "typ": "JWT", "kid": TEST_KEY_ID}),
            _segment(
                {"iss": ISSUER, "sub": "attacker", "aud": [AUDIENCE], "exp": 9999999999}
            ),
            "",
        ]
    )

    with pytest.raises(PermissionError):
        await verify_access_token(unsigned)
    assert provider.request_count == 0


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_malformed_token_makes_zero_network_calls(provider):
    """The header is parsed locally first, so garbage cannot generate traffic."""
    for garbage in ("", "not-a-token", "a.b", "a.b.c", "Bearer something"):
        with pytest.raises(PermissionError):
            await verify_access_token(garbage)
    assert provider.request_count == 0


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_token_without_kid_makes_zero_network_calls(provider):
    """No kid means no way to choose a key, so there is nothing to fetch."""
    signing_input = ".".join(
        [
            _segment({"alg": "RS256", "typ": "JWT"}),
            _segment({"iss": ISSUER, "sub": "nobody", "aud": [AUDIENCE], "exp": 9999999999}),
        ]
    )
    signature = test_private_key().sign(
        signing_input.encode(), padding.PKCS1v15(), hashes.SHA256()
    )

    with pytest.raises(PermissionError):
        await verify_access_token(f"{signing_input}.{_b64(signature)}")
    assert provider.request_count == 0


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_discovery_issuer_mismatch_is_refused(provider):
    """The check that makes jwks_uri safe to fetch.

    A document that does not declare our issuer does not get to nominate the
    signing keys, so this must fail before the JWKS is ever requested.
    """
    provider.discovery = {**provider.discovery, "issuer": "https://someone-else.invalid"}
    with pytest.raises(IdentityProviderUnavailable):
        await verify_access_token(keycloak_token())
    assert JWKS_URI not in provider.requests


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_unreachable_provider_is_not_a_permission_error(provider, monkeypatch):
    """A provider outage must not read as a bad credential."""

    def failing_client(*args, **kwargs):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        return REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(oidc.httpx, "AsyncClient", failing_client)
    with pytest.raises(IdentityProviderUnavailable):
        await verify_access_token(keycloak_token())


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_plaintext_http_provider_is_refused_by_default(provider, monkeypatch):
    monkeypatch.setattr(oidc, "AUTH_ALLOW_INSECURE_HTTP", False)
    monkeypatch.setattr(oidc, "AUTH_DISCOVERY_URL", "http://idp.test.invalid/discovery")
    with pytest.raises(IdentityProviderUnavailable):
        await verify_access_token(keycloak_token())


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_discovery_is_fetched_once_and_cached(provider):
    await verify_access_token(keycloak_token())
    first = provider.request_count
    await verify_access_token(keycloak_token())
    assert provider.request_count == first


@pytest.mark.asyncio
async def test_resolved_principal_is_oidc():
    """The resolver reports how the caller authenticated, not just that they did.

    Asserting a 200 would pass even if the token had been accepted by some
    leftover path; asserting the auth_type is what pins it to this one.
    """
    user, token = await create_test_user_and_token()
    try:
        principal = await resolve_principal_from_bearer_token(token)
        assert principal.auth_type == "oidc"
        assert principal.user_id == user.id
        assert principal.api_key_id is None
        assert principal.caller_type() == "login"
    finally:
        await cleanup_models([user])
