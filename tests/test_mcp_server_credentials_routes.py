"""Routes for a user's own toolkit credentials: list status, save and delete.

The provider check is mocked at ``verify_values``. Storage runs against the real
test database and the local KEK the gate sets.
"""

import logging
from typing import Optional
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from botocore.exceptions import ClientError

from src.database.models.mcp_server import MCPServer, ToolConfig, ToolTransport
from src.database.models.mcp_server_credential import MCPServerCredential
from src.services.custom_model_cipher import CipherConfigError
from src.services.mcp import user_credentials as uc
from tests.utils.cleaner import cleanup_models
from tests.utils.utils import create_test_user_and_token

CLIENT_ID = "client-id-plain-1234"
CLIENT_SECRET = "s3cr3t-value-never-logged-9876"

SPEC_RAW = {
    "fields": [
        {
            "name": "client_id",
            "label": "Client ID",
            "header": "X-Client-Id",
            "secret": False,
            "help_text": "From the provider console",
        },
        {"name": "client_secret", "label": "Client secret", "header": "X-Client-Secret"},
    ],
    "required": True,
    "help_url": "https://provider.example.test/keys",
    "check": {
        "type": "oauth2_client_credentials",
        "token_url": "https://auth.example.test/oauth/token",
        "client_id_field": "client_id",
        "client_secret_field": "client_secret",
    },
}

_SAVE = "src.services.mcp.user_credentials.save"
_VERIFY = "src.services.mcp.user_credentials.verify_values"


def _values():
    return {"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET}


def _server(*, user_id: Optional[str] = None, name: str = "creds-toolkit", spec=SPEC_RAW):
    return MCPServer(
        user_id=user_id,
        name=name,
        config=ToolConfig(url="https://example.com/mcp", transport=ToolTransport.STREAMABLE_HTTP),
        user_credentials=spec,
    )


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _no_secret_anywhere(text: str) -> None:
    assert CLIENT_SECRET not in text
    assert "X-Client-Secret" not in text
    assert "X-Client-Id" not in text
    assert "auth.example.test" not in text


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setattr("src.config.FEATURE_TOOLKIT_CREDENTIALS", True)


@pytest.fixture
def flag_off(monkeypatch):
    monkeypatch.setattr("src.config.FEATURE_TOOLKIT_CREDENTIALS", False)


@pytest_asyncio.fixture
async def clean_credentials():
    collection = MCPServerCredential.get_collection()
    await collection.delete_many({})
    yield collection
    await collection.delete_many({})


@pytest_asyncio.fixture
async def world(clean_credentials):
    me, my_token = await create_test_user_and_token()
    other, other_token = await create_test_user_and_token()
    server = _server()
    await server.save()
    owned = _server(user_id=me.id, name="my-own-toolkit")
    await owned.save()
    yield {
        "me": me,
        "token": my_token,
        "other": other,
        "other_token": other_token,
        "server": server,
        "owned": owned,
    }
    await cleanup_models([server, owned, me, other])


async def _list_item(async_client, token: str, name: str) -> dict:
    response = await async_client.get("/mcp-servers?limit=100", headers=_auth(token))
    assert response.status_code == 200
    _no_secret_anywhere(response.text)
    return next(i for i in response.json()["data"] if i["name"] == name)


async def _put(async_client, token: str, server_id: str, body):
    return await async_client.put(
        f"/mcp-servers/{server_id}/credentials", headers=_auth(token), json=body
    )


# Flag and visibility


@pytest.mark.asyncio
async def test_flag_off_hides_everything(async_client, world, flag_off):
    sid = world["server"].id
    put = await _put(async_client, world["token"], sid, {"values": _values()})
    assert put.status_code == 404
    delete = await async_client.delete(
        f"/mcp-servers/{sid}/credentials", headers=_auth(world["token"])
    )
    assert delete.status_code == 404
    item = await _list_item(async_client, world["token"], "creds-toolkit")
    assert item["user_credentials"] is None


@pytest.mark.asyncio
async def test_user_owned_row_with_a_spec_is_ignored(async_client, world, flag_on):
    sid = world["owned"].id
    put = await _put(async_client, world["token"], sid, {"values": _values()})
    assert put.status_code == 404
    delete = await async_client.delete(
        f"/mcp-servers/{sid}/credentials", headers=_auth(world["token"])
    )
    assert delete.status_code == 404
    item = await _list_item(async_client, world["token"], "my-own-toolkit")
    assert item["user_credentials"] is None


@pytest.mark.asyncio
async def test_unknown_or_invisible_ids_answer_404(async_client, world, flag_on):
    other_owned = _server(user_id=world["other"].id, name="theirs")
    await other_owned.save()
    try:
        for sid in ("not-an-id", "0" * 24, other_owned.id):
            put = await _put(async_client, world["token"], sid, {"values": _values()})
            assert put.status_code == 404
    finally:
        await cleanup_models([other_owned])


@pytest.mark.asyncio
async def test_list_and_detail_describe_fields_without_internals(async_client, world, flag_on):
    item = await _list_item(async_client, world["token"], "creds-toolkit")
    creds = item["user_credentials"]
    assert creds["configured"] is False
    assert creds["required"] is True
    assert creds["help_url"] == "https://provider.example.test/keys"
    assert creds["hints"] == {}
    assert creds["verified_at"] is None
    assert creds["fields"] == [
        {
            "name": "client_id",
            "label": "Client ID",
            "secret": False,
            "help_text": "From the provider console",
        },
        {"name": "client_secret", "label": "Client secret", "secret": True, "help_text": None},
    ]
    with patch(
        "src.routers.mcp_server._load_mcp_tools_for_servers", new=AsyncMock(return_value=[])
    ):
        detail = await async_client.get(
            f"/mcp-servers/{world['server'].id}", headers=_auth(world["token"])
        )
    assert detail.status_code == 200
    _no_secret_anywhere(detail.text)
    assert detail.json()["user_credentials"]["configured"] is False


# Save


@pytest.mark.asyncio
async def test_save_happy_path(async_client, world, flag_on, caplog):
    caplog.set_level(logging.DEBUG)
    with patch(_VERIFY, new=AsyncMock(return_value=True)):
        response = await _put(async_client, world["token"], world["server"].id, {"values": _values()})
    assert response.status_code == 200, response.text
    _no_secret_anywhere(response.text)
    assert CLIENT_ID not in response.text
    body = response.json()
    assert body["configured"] is True
    assert body["hints"] == {"client_id": "…1234"}
    assert body["verified_at"] is not None
    assert body["updated_at"] is not None
    assert [f["name"] for f in body["fields"]] == ["client_id", "client_secret"]

    item = await _list_item(async_client, world["token"], "creds-toolkit")
    assert item["user_credentials"]["configured"] is True
    assert item["user_credentials"]["hints"] == {"client_id": "…1234"}

    # Another user sees the same toolkit unconfigured.
    theirs = await _list_item(async_client, world["other_token"], "creds-toolkit")
    assert theirs["user_credentials"]["configured"] is False

    assert "mcp.user_credentials.saved" in caplog.text
    assert CLIENT_SECRET not in caplog.text
    assert CLIENT_ID not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error, status, code",
    [
        (uc.CredentialsInvalid("client_id", "must not be empty"), 422, "credentials_invalid"),
        (uc.CredentialsRejected("no"), 422, "credentials_rejected"),
        (uc.CredentialsCheckUnavailable("down"), 503, "credentials_check_unavailable"),
        (CipherConfigError("no key"), 503, "credentials_storage_unavailable"),
        (
            ClientError({"Error": {"Code": "AccessDenied", "Message": "x"}}, "Encrypt"),
            503,
            "credentials_storage_unavailable",
        ),
    ],
)
async def test_save_error_mapping(async_client, world, flag_on, caplog, error, status, code):
    caplog.set_level(logging.DEBUG)
    with patch(_SAVE, new=AsyncMock(side_effect=error)):
        response = await _put(async_client, world["token"], world["server"].id, {"values": _values()})
    assert response.status_code == status, response.text
    detail = response.json()["detail"]
    assert detail["code"] == code
    assert detail["message"]
    if code == "credentials_invalid":
        assert detail["field"] == "client_id"
    if code == "credentials_rejected":
        assert detail["message"] == "The provider rejected these credentials."
    if code == "credentials_check_unavailable":
        assert detail["message"] == (
            "The provider could not be reached to check these credentials."
        )
    if code == "credentials_storage_unavailable":
        assert type(error).__name__ in caplog.text
    _no_secret_anywhere(response.text)
    assert CLIENT_SECRET not in caplog.text


@pytest.mark.asyncio
async def test_invalid_values_from_the_real_validation(async_client, world, flag_on):
    values = {"client_id": CLIENT_ID, "client_secret": "bad\nvalue-" + CLIENT_SECRET}
    response = await _put(async_client, world["token"], world["server"].id, {"values": values})
    assert response.status_code == 422
    assert response.json()["detail"]["field"] == "client_secret"
    _no_secret_anywhere(response.text)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {"values": {"client_id": CLIENT_ID, "client_secret": 12345678987654321}},
        {"values": CLIENT_SECRET},
        {"values": {f"k{i}": CLIENT_SECRET for i in range(6)}},
        {"values": _values(), "extra": CLIENT_SECRET},
        [CLIENT_SECRET],
        CLIENT_SECRET,
    ],
)
async def test_malformed_body_does_not_echo_values(async_client, world, flag_on, caplog, body):
    caplog.set_level(logging.DEBUG)
    response = await _put(async_client, world["token"], world["server"].id, body)
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "credentials_invalid"
    _no_secret_anywhere(response.text)
    assert "12345678987654321" not in response.text
    assert CLIENT_SECRET not in caplog.text


@pytest.mark.asyncio
async def test_body_that_is_not_json_does_not_echo(async_client, world, flag_on):
    response = await async_client.put(
        f"/mcp-servers/{world['server'].id}/credentials",
        headers={**_auth(world["token"]), "Content-Type": "application/json"},
        content=b'{"values": {"client_secret": "' + CLIENT_SECRET.encode(),
    )
    assert response.status_code == 422
    _no_secret_anywhere(response.text)


# Delete


@pytest.mark.asyncio
async def test_delete_then_list_is_unconfigured_and_idempotent(
    async_client, world, flag_on, caplog
):
    caplog.set_level(logging.INFO)
    with patch(_VERIFY, new=AsyncMock(return_value=True)):
        saved = await _put(async_client, world["token"], world["server"].id, {"values": _values()})
    assert saved.status_code == 200
    url = f"/mcp-servers/{world['server'].id}/credentials"
    first = await async_client.delete(url, headers=_auth(world["token"]))
    assert first.status_code == 204
    assert first.content == b""
    again = await async_client.delete(url, headers=_auth(world["token"]))
    assert again.status_code == 204
    item = await _list_item(async_client, world["token"], "creds-toolkit")
    assert item["user_credentials"]["configured"] is False
    assert "mcp.user_credentials.deleted" in caplog.text
    assert CLIENT_SECRET not in caplog.text


@pytest.mark.asyncio
async def test_delete_does_not_touch_another_users_row(async_client, world, flag_on):
    with patch(_VERIFY, new=AsyncMock(return_value=True)):
        saved = await _put(
            async_client, world["other_token"], world["server"].id, {"values": _values()}
        )
    assert saved.status_code == 200
    response = await async_client.delete(
        f"/mcp-servers/{world['server'].id}/credentials", headers=_auth(world["token"])
    )
    assert response.status_code == 204
    theirs = await _list_item(async_client, world["other_token"], "creds-toolkit")
    assert theirs["user_credentials"]["configured"] is True
