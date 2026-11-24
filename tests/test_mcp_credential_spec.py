"""Shape rules for the operator-declared toolkit credential fields."""

import pytest
from pydantic import ValidationError

from src.database.models.mcp_server import MCPServer, UserCredentialSpec

pytestmark = pytest.mark.no_db

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/"
    "openid-connect/token"
)


def spec(**overrides):
    base = {
        "fields": [
            {
                "name": "client_id",
                "label": "Client ID",
                "header": "X-CDSE-Client-Id",
                "secret": False,
            },
            {
                "name": "client_secret",
                "label": "Client secret",
                "header": "X-CDSE-Client-Secret",
            },
        ],
        "check": {
            "type": "oauth2_client_credentials",
            "token_url": TOKEN_URL,
            "client_id_field": "client_id",
            "client_secret_field": "client_secret",
        },
    }
    base.update(overrides)
    return base


def field(**overrides):
    base = {"name": "api_key", "label": "API key", "header": "X-Api-Key"}
    base.update(overrides)
    return base


def test_valid_spec_parses():
    parsed = UserCredentialSpec.model_validate(spec())
    assert [f.secret for f in parsed.fields] == [False, True]
    assert parsed.required is False
    assert parsed.check.token_url == TOKEN_URL


@pytest.mark.parametrize("name", ["Client", "1id", "has-dash", "", "a" * 33])
def test_bad_field_name_is_rejected(name):
    with pytest.raises(ValidationError):
        UserCredentialSpec.model_validate(spec(fields=[field(name=name)], check=None))


@pytest.mark.parametrize("header", ["Authorization", "x-eve-token", "Cookie", "bad header"])
def test_denied_or_malformed_header_is_rejected(header):
    with pytest.raises(ValidationError):
        UserCredentialSpec.model_validate(
            spec(fields=[field(header=header)], check=None)
        )


def test_duplicate_header_in_other_case_is_rejected():
    fields = [field(name="a", header="X-Key"), field(name="b", header="x-key")]
    with pytest.raises(ValidationError):
        UserCredentialSpec.model_validate(spec(fields=fields, check=None))


def test_duplicate_name_is_rejected():
    fields = [field(header="X-One"), field(header="X-Two")]
    with pytest.raises(ValidationError):
        UserCredentialSpec.model_validate(spec(fields=fields, check=None))


@pytest.mark.parametrize("count", [0, 6])
def test_field_count_bounds(count):
    fields = [field(name=f"f{i}", header=f"X-F{i}") for i in range(count)]
    with pytest.raises(ValidationError):
        UserCredentialSpec.model_validate(spec(fields=fields, check=None))


def test_http_token_url_is_rejected():
    check = spec()["check"] | {"token_url": TOKEN_URL.replace("https", "http")}
    with pytest.raises(ValidationError):
        UserCredentialSpec.model_validate(spec(check=check))


def test_check_naming_an_undeclared_field_is_rejected():
    check = spec()["check"] | {"client_secret_field": "missing"}
    with pytest.raises(ValidationError):
        UserCredentialSpec.model_validate(spec(check=check))


def test_http_help_url_is_rejected():
    with pytest.raises(ValidationError):
        UserCredentialSpec.model_validate(spec(help_url="http://example.com/help"))


@pytest.mark.parametrize("garbage", ["nope", 42, {"fields": "x"}, [1, 2]])
def test_garbage_user_credentials_still_loads_the_row(garbage):
    row = {
        "name": "effis",
        "config": {"url": "https://example.com/mcp"},
        "user_credentials": garbage,
    }
    assert MCPServer.model_validate(row).user_credentials == garbage
