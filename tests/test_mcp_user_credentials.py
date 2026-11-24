"""User credentials for toolkits: spec parsing, validation, the live check and storage.

The live check runs against a mocked httpx transport (same pattern as
``tests/test_oidc_validation.py``). Storage runs against the real test database
and the local KEK the gate sets, so the encryption path is the one production
uses minus KMS.
"""

import asyncio
import base64
import json
import logging

import httpx
import pytest
import pytest_asyncio
from bson import ObjectId
from pymongo.errors import DuplicateKeyError

from src.database.models.mcp_server import MCPServer, UserCredentialSpec
from src.database.models.mcp_server_credential import MCPServerCredential
from src.services.mcp import user_credentials as uc
from src.services.mcp.user_credentials import (
    CredentialsCheckUnavailable,
    CredentialsConflict,
    CredentialsInvalid,
    CredentialsRejected,
    CredentialsUnreadable,
)

# Captured at import, before any test patches httpx.AsyncClient.
REAL_ASYNC_CLIENT = httpx.AsyncClient

TOKEN_URL = "https://auth.example.test/oauth/token"
CLIENT_ID = "client-id-plain-1234"
CLIENT_SECRET = "s3cr3t-value-never-logged-9876"

SPEC_RAW = {
    "fields": [
        {"name": "client_id", "label": "Client ID", "header": "X-Client-Id", "secret": False},
        {"name": "client_secret", "label": "Client secret", "header": "X-Client-Secret"},
    ],
    "check": {
        "type": "oauth2_client_credentials",
        "token_url": TOKEN_URL,
        "client_id_field": "client_id",
        "client_secret_field": "client_secret",
    },
}

SPEC_NO_CHECK_RAW = {
    "fields": [{"name": "api_key", "label": "API key", "header": "X-Api-Key"}],
}


def _spec(raw=SPEC_RAW) -> UserCredentialSpec:
    return UserCredentialSpec.model_validate(raw)


def _values():
    return {"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET}


def _server(**overrides) -> MCPServer:
    data = {
        "name": "toolkit-a",
        "config": {"url": "https://mcp.example.test/mcp"},
        "user_credentials": SPEC_RAW,
    }
    data.update(overrides)
    return MCPServer(**data)


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setattr("src.config.FEATURE_TOOLKIT_CREDENTIALS", True)


@pytest.fixture
def fresh_warnings(monkeypatch):
    monkeypatch.setattr(uc, "_warned_invalid_specs", set())


@pytest.fixture
def token_endpoint(monkeypatch):
    """Route every httpx.AsyncClient to a scripted handler and record what was sent."""

    state = {"handler": None, "requests": [], "client_kwargs": []}

    def handler(request: httpx.Request) -> httpx.Response:
        state["requests"].append(request)
        return state["handler"](request)

    def client_factory(*args, **kwargs):
        state["client_kwargs"].append(kwargs)
        return REAL_ASYNC_CLIENT(*args, transport=httpx.MockTransport(handler), **kwargs)

    async def resolve(host, port):
        state["resolved"].append(host)
        return state["addresses"]

    state["addresses"] = ["93.184.215.14"]
    state["resolved"] = []
    monkeypatch.setattr(uc.httpx, "AsyncClient", client_factory)
    monkeypatch.setattr(uc, "_resolve_addresses", resolve)
    return state


@pytest_asyncio.fixture
async def clean_credentials():
    collection = MCPServerCredential.get_collection()
    await collection.delete_many({})
    yield collection
    await collection.delete_many({})


# effective_spec


@pytest.mark.no_db
def test_effective_spec_is_none_when_the_flag_is_off(monkeypatch):
    monkeypatch.setattr("src.config.FEATURE_TOOLKIT_CREDENTIALS", False)
    assert uc.effective_spec(_server()) is None


@pytest.mark.no_db
def test_effective_spec_is_none_for_a_user_owned_row(flag_on):
    assert uc.effective_spec(_server(user_id="u1")) is None
    assert uc.effective_spec({"name": "x", "user_id": "u1", "user_credentials": SPEC_RAW}) is None


@pytest.mark.no_db
def test_effective_spec_is_none_when_the_field_is_absent(flag_on):
    assert uc.effective_spec(_server(user_credentials=None)) is None
    assert uc.effective_spec({"name": "x"}) is None


@pytest.mark.no_db
def test_effective_spec_warns_once_on_garbage(flag_on, fresh_warnings, caplog):
    garbage = {"fields": [{"name": "Bad Name", "label": "", "header": "Authorization"}]}
    caplog.set_level(logging.WARNING, logger=uc.__name__)
    assert uc.effective_spec(_server(user_credentials=garbage)) is None
    assert uc.effective_spec({"name": "toolkit-a", "user_credentials": garbage}) is None
    assert uc.effective_spec(_server(user_credentials="not a spec")) is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "toolkit-a" in warnings[0].getMessage()
    # The operator value itself stays out of the log, only error types go in.
    assert "Bad Name" not in caplog.text
    assert "Authorization" not in caplog.text


@pytest.mark.no_db
def test_effective_spec_parses_a_valid_row_and_a_raw_dict(flag_on):
    spec = uc.effective_spec(_server())
    assert isinstance(spec, UserCredentialSpec)
    assert [f.name for f in spec.fields] == ["client_id", "client_secret"]
    raw = uc.effective_spec({"name": "toolkit-a", "user_id": None, "user_credentials": SPEC_RAW})
    assert raw == spec


# validate_values


@pytest.mark.no_db
def test_validate_strips_and_returns_exactly_the_declared_fields():
    clean = uc.validate_values(_spec(), {"client_id": f"  {CLIENT_ID} ", "client_secret": CLIENT_SECRET})
    assert clean == _values()


@pytest.mark.no_db
@pytest.mark.parametrize(
    "values, field",
    [
        ({"client_id": CLIENT_ID}, "client_secret"),
        ({**_values(), "extra": "x"}, "extra"),
        ({"client_id": "   ", "client_secret": CLIENT_SECRET}, "client_id"),
        ({"client_id": "", "client_secret": CLIENT_SECRET}, "client_id"),
        ({"client_id": CLIENT_ID, "client_secret": "a" * 1025}, "client_secret"),
        ({"client_id": CLIENT_ID, "client_secret": "abc\r\ndef"}, "client_secret"),
        ({"client_id": CLIENT_ID, "client_secret": "abc\ndef"}, "client_secret"),
        ({"client_id": CLIENT_ID, "client_secret": "abc\tdef"}, "client_secret"),
        ({"client_id": CLIENT_ID, "client_secret": "abc\x00def"}, "client_secret"),
        ({"client_id": CLIENT_ID, "client_secret": "abc\x7fdef"}, "client_secret"),
        ({"client_id": CLIENT_ID, "client_secret": "café-secret"}, "client_secret"),
        ({"client_id": CLIENT_ID, "client_secret": 12345}, "client_secret"),
    ],
)
def test_validate_rejects(values, field):
    with pytest.raises(CredentialsInvalid) as info:
        uc.validate_values(_spec(), values)
    assert info.value.field == field
    secret = values.get("client_secret")
    if isinstance(secret, str) and secret.strip():
        assert secret not in str(info.value)
        assert secret not in info.value.message


@pytest.mark.no_db
def test_validate_accepts_1024_printable_chars():
    value = "~" * 1024
    assert uc.validate_values(_spec(SPEC_NO_CHECK_RAW), {"api_key": value}) == {"api_key": value}


@pytest.mark.no_db
def test_validate_rejects_a_non_mapping():
    with pytest.raises(CredentialsInvalid):
        uc.validate_values(_spec(), ["client_id"])


# verify_values


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_verify_returns_false_without_a_check(token_endpoint):
    assert await uc.verify_values(_spec(SPEC_NO_CHECK_RAW), {"api_key": "k" * 20}) is False
    assert token_endpoint["requests"] == []


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_verify_passes_on_a_token_and_sends_the_form(token_endpoint):
    token_endpoint["handler"] = lambda r: httpx.Response(200, json={"access_token": "tok", "token_type": "Bearer"})
    assert await uc.verify_values(_spec(), _values()) is True
    request = token_endpoint["requests"][0]
    assert request.method == "POST"
    assert str(request.url) == TOKEN_URL
    form = dict(httpx.QueryParams(request.content.decode()))
    assert form == {"grant_type": "client_credentials", "client_id": CLIENT_ID, "client_secret": CLIENT_SECRET}
    kwargs = token_endpoint["client_kwargs"][0]
    assert kwargs["follow_redirects"] is False
    assert kwargs["timeout"] == 10


@pytest.mark.no_db
@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401])
async def test_verify_raises_rejected_on_400_and_401(token_endpoint, status):
    token_endpoint["handler"] = lambda r: httpx.Response(status, json={"error": "invalid_client"})
    with pytest.raises(CredentialsRejected):
        await uc.verify_values(_spec(), _values())


@pytest.mark.no_db
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(302, headers={"Location": "https://evil.example.test/"}),
        httpx.Response(403, json={"error": "forbidden"}),
        httpx.Response(500, text="boom"),
        httpx.Response(503),
        httpx.Response(200, text=f"error: bad secret {CLIENT_SECRET}"),
        httpx.Response(200, json={"token_type": "Bearer"}),
        httpx.Response(200, json={"access_token": ""}),
        httpx.Response(200, json={"access_token": 42}),
        httpx.Response(200, json=["access_token"]),
    ],
    ids=["302", "403", "500", "503", "text-200", "no-token", "empty-token", "int-token", "list"],
)
async def test_verify_raises_unavailable(token_endpoint, response, caplog):
    caplog.set_level(logging.DEBUG)
    token_endpoint["handler"] = lambda r: response
    with pytest.raises(CredentialsCheckUnavailable) as info:
        await uc.verify_values(_spec(), _values())
    assert CLIENT_SECRET not in caplog.text
    assert CLIENT_SECRET not in str(info.value)
    # A redirect is never followed.
    assert len(token_endpoint["requests"]) == 1


@pytest.mark.no_db
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [httpx.ReadTimeout("timed out"), httpx.ConnectTimeout("timed out"), httpx.ConnectError("refused")],
)
async def test_verify_raises_unavailable_on_transport_errors(token_endpoint, error, caplog):
    caplog.set_level(logging.DEBUG)

    def handler(request):
        raise error

    token_endpoint["handler"] = handler
    with pytest.raises(CredentialsCheckUnavailable) as info:
        await uc.verify_values(_spec(), _values())
    assert info.value.__cause__ is None
    assert CLIENT_SECRET not in caplog.text


# Storage


USER = str(ObjectId())
OTHER_USER = str(ObjectId())
SERVER = str(ObjectId())


@pytest.mark.asyncio
async def test_save_stores_no_plaintext_and_hints_only_non_secret_fields(clean_credentials, token_endpoint):
    token_endpoint["handler"] = lambda r: httpx.Response(200, json={"access_token": "tok"})
    saved = await uc.save(user_id=USER, server_id=SERVER, spec=_spec(), values=_values())
    assert isinstance(saved, MCPServerCredential)
    assert saved.verified_at is not None
    assert saved.hints == {"client_id": "…" + CLIENT_ID[-4:]}

    raw = await clean_credentials.find_one({"user_id": USER, "server_id": SERVER})
    dump = json.dumps(raw, default=str)
    assert CLIENT_SECRET not in dump
    assert CLIENT_ID not in dump


@pytest.mark.asyncio
async def test_save_without_check_leaves_verified_at_empty_and_short_hint(clean_credentials):
    spec = _spec({"fields": [{"name": "region", "label": "Region", "header": "X-Region", "secret": False}]})
    saved = await uc.save(user_id=USER, server_id=SERVER, spec=spec, values={"region": "eu-west1"})
    assert saved.verified_at is None
    assert saved.hints == {"region": "…"}


@pytest.mark.asyncio
async def test_save_rejected_stores_nothing(clean_credentials, token_endpoint):
    token_endpoint["handler"] = lambda r: httpx.Response(401, json={"error": "invalid_client"})
    with pytest.raises(CredentialsRejected):
        await uc.save(user_id=USER, server_id=SERVER, spec=_spec(), values=_values())
    assert await clean_credentials.count_documents({}) == 0


@pytest.mark.asyncio
async def test_save_twice_keeps_one_row_and_created_at(clean_credentials, token_endpoint):
    token_endpoint["handler"] = lambda r: httpx.Response(200, json={"access_token": "tok"})
    first = await uc.save(user_id=USER, server_id=SERVER, spec=_spec(), values=_values())
    second = await uc.save(
        user_id=USER, server_id=SERVER, spec=_spec(), values={**_values(), "client_secret": "another-secret-5555"}
    )
    assert await clean_credentials.count_documents({"user_id": USER, "server_id": SERVER}) == 1
    assert second.id == first.id
    assert second.created_at == first.created_at
    assert second.updated_at >= first.updated_at
    headers = await uc.headers_for(user_id=USER, server_id=SERVER, spec=_spec())
    assert headers["X-Client-Secret"] == "another-secret-5555"


@pytest.mark.asyncio
async def test_delete(clean_credentials):
    spec = _spec(SPEC_NO_CHECK_RAW)
    await uc.save(user_id=USER, server_id=SERVER, spec=spec, values={"api_key": "k" * 20})
    assert await uc.delete(user_id=USER, server_id=SERVER) is True
    assert await uc.delete(user_id=USER, server_id=SERVER) is False
    assert await uc.headers_for(user_id=USER, server_id=SERVER, spec=spec) is None


@pytest.mark.asyncio
async def test_status_for_returns_only_the_callers_rows(clean_credentials):
    spec = _spec(SPEC_NO_CHECK_RAW)
    other_server = str(ObjectId())
    await uc.save(user_id=USER, server_id=SERVER, spec=spec, values={"api_key": "k" * 20})
    await uc.save(user_id=USER, server_id=other_server, spec=spec, values={"api_key": "j" * 20})
    await uc.save(user_id=OTHER_USER, server_id=SERVER, spec=spec, values={"api_key": "m" * 20})

    status = await uc.status_for(user_id=USER, server_ids=[SERVER, other_server, str(ObjectId())])
    assert set(status) == {SERVER, other_server}
    assert all(row.user_id == USER for row in status.values())
    assert await uc.status_for(user_id=USER, server_ids=[]) == {}


@pytest.mark.asyncio
async def test_headers_for_maps_fields_to_headers(clean_credentials, token_endpoint):
    token_endpoint["handler"] = lambda r: httpx.Response(200, json={"access_token": "tok"})
    await uc.save(user_id=USER, server_id=SERVER, spec=_spec(), values=_values())
    headers = await uc.headers_for(user_id=USER, server_id=SERVER, spec=_spec())
    assert headers == {"X-Client-Id": CLIENT_ID, "X-Client-Secret": CLIENT_SECRET}
    assert await uc.headers_for(user_id=OTHER_USER, server_id=SERVER, spec=_spec()) is None


@pytest.mark.asyncio
async def test_headers_for_is_none_when_the_spec_now_declares_a_new_field(clean_credentials, caplog):
    await uc.save(user_id=USER, server_id=SERVER, spec=_spec(SPEC_NO_CHECK_RAW), values={"api_key": "k" * 20})
    grown = _spec(
        {
            "fields": [
                {"name": "api_key", "label": "API key", "header": "X-Api-Key"},
                {"name": "tenant", "label": "Tenant", "header": "X-Tenant"},
            ]
        }
    )
    caplog.set_level(logging.WARNING, logger=uc.__name__)
    assert await uc.headers_for(user_id=USER, server_id=SERVER, spec=grown) is None
    assert f"mcp.user_credentials.stale server_id={SERVER}" in caplog.text


def _flip_ciphertext(blob: str) -> str:
    parsed = json.loads(base64.b64decode(blob))
    ct = bytearray(base64.b64decode(parsed["ct"]))
    ct[0] ^= 0x01
    parsed["ct"] = base64.b64encode(bytes(ct)).decode()
    return base64.b64encode(json.dumps(parsed).encode()).decode()


@pytest.mark.asyncio
@pytest.mark.parametrize("tamper", [_flip_ciphertext, lambda blob: "not-a-blob"], ids=["flipped", "garbage"])
async def test_tampered_blob_raises_unreadable(clean_credentials, caplog, tamper):
    secret = "k" * 12 + "TAMPER"
    spec = _spec(SPEC_NO_CHECK_RAW)
    saved = await uc.save(user_id=USER, server_id=SERVER, spec=spec, values={"api_key": secret})
    await clean_credentials.update_one(
        {"user_id": USER, "server_id": SERVER},
        {"$set": {"encrypted_values": tamper(saved.encrypted_values)}},
    )
    caplog.set_level(logging.DEBUG)
    with pytest.raises(CredentialsUnreadable) as info:
        await uc.headers_for(user_id=USER, server_id=SERVER, spec=spec)
    assert info.value.__cause__ is None
    errors = [r for r in caplog.records if r.levelno == logging.ERROR and r.name == uc.__name__]
    assert len(errors) == 1
    message = errors[0].getMessage()
    assert USER in message and SERVER in message
    assert secret not in caplog.text


@pytest.mark.asyncio
async def test_blob_copied_to_another_users_row_fails(clean_credentials):
    spec = _spec(SPEC_NO_CHECK_RAW)
    saved = await uc.save(user_id=USER, server_id=SERVER, spec=spec, values={"api_key": "k" * 20})
    await clean_credentials.insert_one(
        {
            "user_id": OTHER_USER,
            "server_id": SERVER,
            "encrypted_values": saved.encrypted_values,
            "hints": {},
            "field_names": ["api_key"],
        }
    )
    with pytest.raises(CredentialsUnreadable):
        await uc.headers_for(user_id=OTHER_USER, server_id=SERVER, spec=spec)


@pytest.mark.asyncio
async def test_no_submitted_value_reaches_any_log(clean_credentials, token_endpoint, caplog):
    caplog.set_level(logging.DEBUG)
    responses = iter(
        [
            httpx.Response(200, json={"access_token": "tok"}),
            httpx.Response(401, text=f"bad secret {CLIENT_SECRET}"),
            httpx.Response(500, text=f"bad secret {CLIENT_SECRET}"),
        ]
    )
    token_endpoint["handler"] = lambda r: next(responses)
    await uc.save(user_id=USER, server_id=SERVER, spec=_spec(), values=_values())
    for expected in (CredentialsRejected, CredentialsCheckUnavailable):
        with pytest.raises(expected):
            await uc.save(user_id=USER, server_id=SERVER, spec=_spec(), values=_values())
    with pytest.raises(CredentialsInvalid):
        await uc.save(
            user_id=USER, server_id=SERVER, spec=_spec(), values={"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET + "\n x"}
        )
    await uc.headers_for(user_id=USER, server_id=SERVER, spec=_spec())
    assert CLIENT_SECRET not in caplog.text
    assert CLIENT_ID not in caplog.text


# Hardening


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_verify_bounds_the_whole_request_with_one_deadline(token_endpoint, monkeypatch):
    monkeypatch.setattr(uc, "CHECK_TIMEOUT_S", 0.05)

    async def slow(request):
        await asyncio.sleep(1)
        return httpx.Response(200, json={"access_token": "tok"})

    token_endpoint["handler"] = slow
    with pytest.raises(CredentialsCheckUnavailable) as info:
        await uc.verify_values(_spec(), _values())
    assert info.value.__cause__ is None


@pytest.mark.no_db
@pytest.mark.asyncio
@pytest.mark.parametrize("token_url", ["https://exa\x00mple.test/token", "https://[::1/token", "https:///token"])
async def test_verify_invalid_url_is_unavailable(token_endpoint, token_url):
    spec = _spec({**SPEC_RAW, "check": {**SPEC_RAW["check"], "token_url": token_url}})
    with pytest.raises(CredentialsCheckUnavailable):
        await uc.verify_values(spec, _values())
    assert token_endpoint["requests"] == []


@pytest.mark.no_db
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "addresses",
    [["127.0.0.1"], ["10.0.0.5"], ["169.254.169.254"], ["93.184.215.14", "192.168.1.1"], ["::1"], []],
)
async def test_verify_refuses_a_token_host_that_is_not_public(token_endpoint, addresses, caplog):
    caplog.set_level(logging.WARNING, logger=uc.__name__)
    token_endpoint["addresses"] = addresses
    token_endpoint["handler"] = lambda r: httpx.Response(200, json={"access_token": "tok"})
    with pytest.raises(CredentialsCheckUnavailable):
        await uc.verify_values(_spec(), _values())
    assert token_endpoint["requests"] == []
    assert "mcp.user_credentials.check_non_public host=auth.example.test" in caplog.text
    assert CLIENT_SECRET not in caplog.text


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_verify_refuses_a_literal_private_ip_through_the_real_resolver(monkeypatch):
    calls = []
    monkeypatch.setattr(uc.httpx, "AsyncClient", lambda *a, **k: calls.append(k))
    spec = _spec({**SPEC_RAW, "check": {**SPEC_RAW["check"], "token_url": "https://127.0.0.1/token"}})
    with pytest.raises(CredentialsCheckUnavailable):
        await uc.verify_values(spec, _values())
    assert calls == []


class _CollideOnce:
    """Delegates to the real collection, the first update_one raises DuplicateKeyError."""

    def __init__(self, real):
        self.real = real
        self.updates = 0

    async def update_one(self, *args, **kwargs):
        self.updates += 1
        if self.updates == 1:
            raise DuplicateKeyError("E11000 duplicate key")
        return await self.real.update_one(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.real, name)


@pytest.mark.asyncio
async def test_save_retries_once_on_a_duplicate_key(clean_credentials, monkeypatch):
    wrapper = _CollideOnce(MCPServerCredential.get_collection())
    monkeypatch.setattr(MCPServerCredential, "get_collection", classmethod(lambda cls: wrapper))
    saved = await uc.save(user_id=USER, server_id=SERVER, spec=_spec(SPEC_NO_CHECK_RAW), values={"api_key": "k" * 20})
    assert wrapper.updates == 2
    assert saved.field_names == ["api_key"]


@pytest.mark.asyncio
async def test_save_raises_conflict_when_the_row_is_gone_before_the_read_back(clean_credentials, monkeypatch):
    class _Gone:
        async def find_one(self, *args, **kwargs):
            return None

    monkeypatch.setattr(MCPServerCredential, "get_primary_collection", classmethod(lambda cls: _Gone()))
    with pytest.raises(CredentialsConflict):
        await uc.save(user_id=USER, server_id=SERVER, spec=_spec(SPEC_NO_CHECK_RAW), values={"api_key": "k" * 20})


@pytest.mark.asyncio
async def test_status_and_headers_read_the_primary(clean_credentials, monkeypatch):
    spec = _spec(SPEC_NO_CHECK_RAW)
    await uc.save(user_id=USER, server_id=SERVER, spec=spec, values={"api_key": "k" * 20})
    real = MCPServerCredential.get_primary_collection.__func__
    calls = []

    def spy(cls):
        calls.append(cls)
        return real(cls)

    monkeypatch.setattr(MCPServerCredential, "get_primary_collection", classmethod(spy))
    assert set(await uc.status_for(user_id=USER, server_ids=[SERVER])) == {SERVER}
    assert len(calls) == 1
    assert await uc.headers_for(user_id=USER, server_id=SERVER, spec=spec) == {"X-Api-Key": "k" * 20}
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_save_stores_sorted_field_names(clean_credentials, token_endpoint):
    token_endpoint["handler"] = lambda r: httpx.Response(200, json={"access_token": "tok"})
    saved = await uc.save(user_id=USER, server_id=SERVER, spec=_spec(), values=_values())
    assert saved.field_names == ["client_id", "client_secret"]
    raw = await clean_credentials.find_one({"user_id": USER, "server_id": SERVER})
    assert raw["field_names"] == ["client_id", "client_secret"]


@pytest.mark.no_db
def test_is_complete():
    spec = _spec()
    row = {"field_names": ["client_id", "client_secret"]}
    assert uc.is_complete(row, spec) is True
    assert uc.is_complete({"field_names": ["client_id"]}, spec) is False
    assert uc.is_complete({}, spec) is False
    model = MCPServerCredential(user_id=USER, server_id=SERVER, encrypted_values="x", field_names=["client_id", "client_secret"])
    assert uc.is_complete(model, spec) is True
    assert uc.is_complete(MCPServerCredential(user_id=USER, server_id=SERVER, encrypted_values="x"), spec) is False


@pytest.mark.asyncio
async def test_headers_for_is_none_for_a_row_saved_before_field_names(clean_credentials, caplog):
    spec = _spec(SPEC_NO_CHECK_RAW)
    await uc.save(user_id=USER, server_id=SERVER, spec=spec, values={"api_key": "k" * 20})
    await clean_credentials.update_one({"user_id": USER, "server_id": SERVER}, {"$unset": {"field_names": ""}})
    caplog.set_level(logging.WARNING, logger=uc.__name__)
    assert await uc.headers_for(user_id=USER, server_id=SERVER, spec=spec) is None
    assert f"mcp.user_credentials.stale server_id={SERVER}" in caplog.text
